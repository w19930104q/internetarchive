"""Local HTTP-server tests for atomic verified downloads.

These tests exercise the full network path of ``File.download(atomic=True)``
(and the Item/API wrappers) against a real ``ThreadingHTTPServer`` on
localhost -- no mocks -- covering interrupted streams and in-process retries,
cross-process resume, remote identity changes, and atomic replacement
failures. Every assertion checks the on-disk digest, the preservation of a
pre-existing target, and the reported error/exit status.
"""

import hashlib
import os
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import pytest
import responses
from requests.exceptions import HTTPError, ReadTimeout

from internetarchive import File, Item, get_session
from internetarchive import download as api_download
from internetarchive.exceptions import (
    InvalidChecksumError,
    MetadataValidationError,
)
from internetarchive.files import _parse_content_range, _state_matches
from internetarchive.utils import json as ia_json
from tests.conftest import NASA_METADATA_PATH


def md5_hex(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


class _State:
    """Thread-safe mutable server state shared with the test."""

    def __init__(self, content: bytes):
        self.lock = threading.Lock()
        self.content = content
        # When > 0, the next N requests abort the response after this many
        # bytes (or immediately at 0) to simulate a truncated stream.
        self.fail_after = 0
        self.fail_count = 0
        # When True, ignore Range request headers and answer with 200 + full
        # body (some real servers do this).
        self.ignore_range = False
        # When True, answer 206 responses with a lying Content-Range header.
        self.bad_content_range = False
        # When > 0, answer the next N requests with this status instead of data.
        self.error_status = 0
        self.error_count = 0
        self.requests = []  # (range_header, path)


def make_server(content: bytes):
    """Start a loopback HTTP server serving one mutable in-memory file.

    :returns: A ``(server, state)`` tuple; the server runs in a daemon thread.
    """
    state = _State(content)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
            pass

        def do_GET(self):
            with state.lock:
                state.requests.append(
                    (self.headers.get('Range'), urlparse(self.path).path)
                )
                if state.error_count > 0:
                    state.error_count -= 1
                    status = state.error_status
                    self.send_response(status)
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                body = state.content
                fail_after = state.fail_after if state.fail_count > 0 else None
                if fail_after is not None:
                    state.fail_count -= 1
                ignore_range = state.ignore_range
                bad_content_range = state.bad_content_range

            range_header = self.headers.get('Range')
            start = 0
            partial = False
            if range_header and not ignore_range:
                # Only the open-ended ``bytes=START-`` form is used by resume.
                spec = range_header.strip()
                assert spec.startswith('bytes=')
                start = int(spec[len('bytes=') :].rstrip('-'))
                partial = True

            if partial:
                payload = body[start:]
                self.send_response(206)
                self.send_header('Content-Type', 'application/octet-stream')
                if bad_content_range:
                    # Claim a different start than the bytes actually carry.
                    self.send_header(
                        'Content-Range',
                        f'bytes {start + 1}-{len(body) - 1}/{len(body)}',
                    )
                else:
                    self.send_header(
                        'Content-Range',
                        f'bytes {start}-{len(body) - 1}/{len(body)}',
                    )
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
            else:
                payload = body
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()

            try:
                if fail_after is not None:
                    # Truncate mid-stream and drop the connection.
                    self.wfile.write(payload[:fail_after])
                    self.wfile.flush()
                    return
                self.wfile.write(payload)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, state


class _FakeItem:
    """Minimal item stand-in: File only needs a session and item metadata."""

    def __init__(self, session, identifier, name, size, md5):
        self.session = session
        self.item_metadata = {
            'metadata': {'identifier': identifier},
            'files': [{'name': name, 'size': str(size), 'md5': md5}],
        }


@pytest.fixture
def atomic_context(tmp_path, monkeypatch):
    """Provide a running server, a File pointed at it, and the target path."""
    content = b'0123456789abcdef' * 16  # 256 bytes
    server, state = make_server(content)
    session = get_session(config={'general': {'secure': False}})
    session.host = f'127.0.0.1:{server.server_address[1]}'
    identifier = 'testitem'
    name = 'data.bin'
    item = _FakeItem(session, identifier, name, len(content), md5_hex(content))
    file_obj = File(item, name)
    # Neutralize retry sleeps.
    monkeypatch.setattr('internetarchive.files.sleep', lambda _s: None)
    monkeypatch.chdir(tmp_path)
    target = str(tmp_path / name)
    try:
        yield file_obj, target, state, content, tmp_path
    finally:
        server.shutdown()
        server.server_close()


def test_parse_content_range_helper():
    assert _parse_content_range('bytes 64-255/256') == (64, 256)
    assert _parse_content_range('bytes 0-1023/*') == (0, None)
    assert _parse_content_range('') == (None, None)
    assert _parse_content_range(None) == (None, None)
    assert _parse_content_range('not a range') == (None, None)


def test_state_identity_matching():
    state = {'identifier': 'it', 'name': 'f', 'size': 100, 'md5': 'abc'}
    assert _state_matches(state, 'it', 'f', 100, 'abc')
    # Any single field change invalidates the state.
    assert not _state_matches(state, 'other', 'f', 100, 'abc')
    assert not _state_matches(state, 'it', 'g', 100, 'abc')
    assert not _state_matches(state, 'it', 'f', 101, 'abc')
    assert not _state_matches(state, 'it', 'f', 100, 'fff')
    assert not _state_matches(None, 'it', 'f', 100, 'abc')
    assert not _state_matches('junk', 'it', 'f', 100, 'abc')


def test_item_atomic_rejects_streaming_options(tmp_path, nasa_item):
    """Item.download must reject streaming/partial combinations up front."""
    with pytest.raises(ValueError, match='atomic downloads only'):
        nasa_item.download(atomic=True, stdout=True)
    with pytest.raises(ValueError, match='atomic downloads only'):
        nasa_item.download(atomic=True, return_responses=True)
    with pytest.raises(ValueError, match='atomic downloads only'):
        nasa_item.download(atomic=True, range_jobs=[('nasa_meta.xml', 'bytes=0-9')])
    with pytest.raises(ValueError, match='atomic downloads only'):
        nasa_item.download(atomic=True, headers={'Range': 'bytes=0-9'})


def test_atomic_creates_nested_parent_directories(atomic_context):
    """A target inside a not-yet-existing subdirectory must still download."""
    file_obj, target, _state, content, _tmp = atomic_context
    nested = os.path.join(os.path.dirname(target), 'sub', 'dir')
    assert (
        file_obj.download(
            file_path=file_obj.name,
            destdir=nested,
            atomic=True,
            retries=0,
        )
        is True
    )
    nested_target = os.path.join(nested, file_obj.name)
    assert open(nested_target, 'rb').read() == content
    assert not os.path.exists(f'{nested_target}.ia-part')


def test_atomic_happy_path_writes_verified_target(atomic_context):
    file_obj, target, state, content, _tmp = atomic_context
    assert (
        file_obj.download(
            file_path=file_obj.name,
            destdir=os.path.dirname(target),
            atomic=True,
            retries=0,
        )
        is True
    )
    with open(target, 'rb') as fh:
        assert fh.read() == content
    assert md5_hex(open(target, 'rb').read()) == file_obj.md5
    # Intermediate state is cleaned up on success.
    assert not os.path.exists(f'{target}.ia-part')
    assert not os.path.exists(f'{target}.ia-state')
    # No resume Range on a fresh download; the full body was fetched once.
    assert [r[0] for r in state.requests] == [None]


def test_atomic_rejects_incompatible_options_before_request(atomic_context):
    file_obj, target, state, _content, _tmp = atomic_context
    match = 'atomic downloads only write to a target path'
    with pytest.raises(ValueError, match=match):
        file_obj.download(
            file_path=file_obj.name,
            destdir=os.path.dirname(target),
            atomic=True,
            stdout=True,
        )
    with pytest.raises(ValueError, match=match):
        file_obj.download(
            file_path=file_obj.name,
            destdir=os.path.dirname(target),
            atomic=True,
            fileobj=open(os.devnull, 'wb'),
        )
    with pytest.raises(ValueError, match=match):
        file_obj.download(
            file_path=file_obj.name,
            destdir=os.path.dirname(target),
            atomic=True,
            return_responses=True,
        )
    with pytest.raises(ValueError, match=match):
        file_obj.download(
            file_path=file_obj.name,
            destdir=os.path.dirname(target),
            atomic=True,
            headers={'Range': 'bytes=0-9'},
        )
    # Nothing was fetched or written.
    assert state.requests == []
    assert not os.path.exists(target)


def test_atomic_requires_size_and_md5(atomic_context):
    file_obj, target, _state, content, _tmp = atomic_context
    file_obj.size = 0
    with pytest.raises(MetadataValidationError):
        file_obj.download(
            file_path=file_obj.name, destdir=os.path.dirname(target), atomic=True
        )
    file_obj.size = len(content)
    file_obj.md5 = None
    with pytest.raises(MetadataValidationError):
        file_obj.download(
            file_path=file_obj.name, destdir=os.path.dirname(target), atomic=True
        )
    assert not os.path.exists(target)


def test_atomic_interrupted_stream_retries_and_resumes(atomic_context):
    """A truncated first attempt must be retried with a Range and still
    produce a byte-identical, verified target."""
    file_obj, target, state, content, _tmp = atomic_context
    state.fail_count = 1
    state.fail_after = 64  # first response dies after 64 of 256 bytes

    # A chunk smaller than the truncated prefix guarantees those bytes are
    # written to the part file before the connection drops.
    assert (
        file_obj.download(
            file_path=file_obj.name,
            destdir=os.path.dirname(target),
            atomic=True,
            retries=2,
            chunk_size=16,
        )
        is True
    )
    with open(target, 'rb') as fh:
        data = fh.read()
    assert data == content
    assert md5_hex(data) == file_obj.md5
    # First request was a full GET; the retry resumed at the partial offset.
    ranges = [r[0] for r in state.requests]
    assert ranges[0] is None
    assert ranges[1] == 'bytes=64-'
    # No target existed while the partial was still incomplete.
    assert not os.path.exists(f'{target}.ia-part')
    assert not os.path.exists(f'{target}.ia-state')


def test_atomic_retries_exhausted_keeps_target_and_surfaces_error(atomic_context):
    file_obj, target, state, content, _tmp_path = atomic_context
    state.fail_count = 5
    state.fail_after = 32
    with pytest.raises((ReadTimeout, OSError, ConnectionError)):
        file_obj.download(
            file_path=file_obj.name,
            destdir=os.path.dirname(target),
            atomic=True,
            retries=0,
            chunk_size=16,
        )
    # The final target was never created; the partial remains for a later resume.
    assert not os.path.exists(target)
    assert os.path.exists(f'{target}.ia-part')
    assert os.path.exists(f'{target}.ia-state')
    # Two complete 16-byte chunks were durable before the connection broke.
    persisted_offset = os.path.getsize(f'{target}.ia-part')
    assert persisted_offset == 32

    # A second, healthy invocation resumes from the partial and completes.
    state.fail_count = 0
    assert (
        file_obj.download(
            file_path=file_obj.name,
            destdir=os.path.dirname(target),
            atomic=True,
            retries=0,
            chunk_size=16,
        )
        is True
    )
    assert open(target, 'rb').read() == content
    assert not os.path.exists(f'{target}.ia-part')
    # The resume request started exactly at the persisted partial offset.
    assert state.requests[-1][0] == f'bytes={persisted_offset}-'


def test_atomic_resume_state_bound_to_identity_restarts_on_remote_change(
    atomic_context,
):
    """When the remote file changes (new size + MD5), an old partial and state
    must be discarded and the new content downloaded from byte zero."""
    file_obj, target, state, _content, _tmp = atomic_context
    state.fail_count = 1
    state.fail_after = 48
    with pytest.raises((ReadTimeout, OSError, ConnectionError)):
        file_obj.download(
            file_path=file_obj.name,
            destdir=os.path.dirname(target),
            atomic=True,
            retries=0,
            chunk_size=16,
        )
    persisted_offset = os.path.getsize(f'{target}.ia-part')
    assert persisted_offset == 48

    # Remote content changes: new bytes, new size and MD5.
    new_content = b'Z' * 128 + b'A' * 128
    with state.lock:
        state.content = new_content
    file_obj.size = len(new_content)
    file_obj.md5 = md5_hex(new_content)

    assert (
        file_obj.download(
            file_path=file_obj.name,
            destdir=os.path.dirname(target),
            atomic=True,
            retries=0,
            chunk_size=16,
        )
        is True
    )
    data = open(target, 'rb').read()
    assert data == new_content
    assert md5_hex(data) == file_obj.md5
    # The identity change forced a full re-fetch, never an append.
    assert [r[0] for r in state.requests][-1] is None
    assert not os.path.exists(f'{target}.ia-part')


def test_atomic_existing_target_verified_before_skip(atomic_context):
    file_obj, target, state, content, _tmp = atomic_context
    # A correct pre-existing target is verified and skipped without any request.
    with open(target, 'wb') as fh:
        fh.write(content)
    assert (
        file_obj.download(
            file_path=file_obj.name, destdir=os.path.dirname(target), atomic=True
        )
        is True
    )
    assert state.requests == []
    assert open(target, 'rb').read() == content

    # A corrupt same-size target is replaced with verified content.
    corrupt = b'X' * len(content)
    with open(target, 'wb') as fh:
        fh.write(corrupt)
    assert (
        file_obj.download(
            file_path=file_obj.name,
            destdir=os.path.dirname(target),
            atomic=True,
            retries=0,
        )
        is True
    )
    assert open(target, 'rb').read() == content
    assert [r[0] for r in state.requests] == [None]

    # A wrong-length target is likewise replaced.
    with open(target, 'wb') as fh:
        fh.write(b'short')
    assert (
        file_obj.download(
            file_path=file_obj.name,
            destdir=os.path.dirname(target),
            atomic=True,
            retries=0,
        )
        is True
    )
    assert open(target, 'rb').read() == content


def test_atomic_existing_target_preserved_on_failure(atomic_context):
    """Network failure must never clobber a pre-existing (even stale) target."""
    file_obj, target, state, _content, _tmp = atomic_context
    original = b'keep-me-intact'
    with open(target, 'wb') as fh:
        fh.write(original)
    state.error_status = 500
    state.error_count = 5
    with pytest.raises(HTTPError, match='500'):
        file_obj.download(
            file_path=file_obj.name,
            destdir=os.path.dirname(target),
            atomic=True,
            retries=1,
        )
    assert open(target, 'rb').read() == original
    # The partial/state are separate from the target.
    assert os.path.exists(f'{target}.ia-state')


def test_atomic_checksum_mismatch_fails_permanently(atomic_context):
    """If the server body's MD5 does not match the metadata, the file fails,
    no target is published, and retries cannot paper over the mismatch."""
    file_obj, target, _state, _content, _tmp = atomic_context
    file_obj.md5 = md5_hex(b'totally different expected md5')
    with pytest.raises(InvalidChecksumError):
        file_obj.download(
            file_path=file_obj.name,
            destdir=os.path.dirname(target),
            atomic=True,
            retries=2,
        )
    assert not os.path.exists(target)
    assert not os.path.exists(f'{target}.ia-part')
    # State is removed so a later run starts clean rather than resuming bad bytes.
    assert not os.path.exists(f'{target}.ia-state')


def test_atomic_server_ignoring_range_restarts_from_zero(atomic_context):
    """A 200 reply to a resume request must restart the file from the
    beginning instead of appending a full body onto the partial prefix."""
    file_obj, target, state, content, _tmp = atomic_context
    state.fail_count = 1
    state.fail_after = 32
    with pytest.raises((ReadTimeout, OSError, ConnectionError)):
        file_obj.download(
            file_path=file_obj.name,
            destdir=os.path.dirname(target),
            atomic=True,
            retries=0,
            chunk_size=16,
        )
    assert os.path.getsize(f'{target}.ia-part') == 32

    # Server now ignores Range headers and answers 200 with the whole body.
    state.ignore_range = True
    assert (
        file_obj.download(
            file_path=file_obj.name,
            destdir=os.path.dirname(target),
            atomic=True,
            retries=0,
        )
        is True
    )
    data = open(target, 'rb').read()
    assert data == content  # no duplicated/concatenated body
    assert md5_hex(data) == file_obj.md5


def test_atomic_bad_content_range_rejected(atomic_context):
    """A 206 whose Content-Range does not match the requested resume offset
    must fail rather than splice misaligned bytes into the partial."""
    file_obj, target, state, _content, _tmp = atomic_context
    state.fail_count = 1
    state.fail_after = 32
    with pytest.raises((ReadTimeout, OSError, ConnectionError)):
        file_obj.download(
            file_path=file_obj.name,
            destdir=os.path.dirname(target),
            atomic=True,
            retries=0,
            chunk_size=16,
        )
    assert os.path.getsize(f'{target}.ia-part') == 32

    state.bad_content_range = True
    with pytest.raises(MetadataValidationError, match='invalid Content-Range'):
        file_obj.download(
            file_path=file_obj.name,
            destdir=os.path.dirname(target),
            atomic=True,
            retries=0,
        )
    # No target was published from misaligned bytes.
    assert not os.path.exists(target)


def test_atomic_replace_failure_preserves_original_target(atomic_context, monkeypatch):
    file_obj, target, _state, content, _tmp = atomic_context
    original = b'original-target-bytes'
    with open(target, 'wb') as fh:
        fh.write(original)

    real_replace = os.replace

    def broken_replace(src, dst):
        if dst == target:
            raise OSError('simulated rename failure')
        return real_replace(src, dst)

    monkeypatch.setattr('internetarchive.files.os.replace', broken_replace)
    with pytest.raises(OSError, match='simulated rename failure'):
        file_obj.download(
            file_path=file_obj.name,
            destdir=os.path.dirname(target),
            atomic=True,
            retries=0,
        )
    # The old target is untouched; the verified partial still exists alongside.
    assert open(target, 'rb').read() == original
    assert os.path.exists(f'{target}.ia-part')
    assert os.path.getsize(f'{target}.ia-part') == len(content)

    # Once the replace works again, the same partial can complete the job.
    monkeypatch.undo()
    assert (
        file_obj.download(
            file_path=file_obj.name,
            destdir=os.path.dirname(target),
            atomic=True,
            retries=0,
        )
        is True
    )
    assert open(target, 'rb').read() == content


def test_atomic_item_batch_collects_errors(tmp_path, monkeypatch):
    """A failing file in a batch must be reported by name while other files
    succeed, and leave any pre-existing target intact."""
    content_a = b'file-a-content' * 8
    content_b = b'file-b-content' * 8
    server, _state = make_server(content_a)
    session = get_session(config={'general': {'secure': False}})
    session.host = f'127.0.0.1:{server.server_address[1]}'
    monkeypatch.setattr('internetarchive.files.sleep', lambda _s: None)

    identifier = 'batchitem'
    metadata = {
        'metadata': {'identifier': identifier},
        'files': [
            {'name': 'a.bin', 'size': str(len(content_a)), 'md5': md5_hex(content_a)},
            {'name': 'b.bin', 'size': str(len(content_b)), 'md5': md5_hex(content_b)},
        ],
    }
    item = Item(session, identifier, item_metadata=metadata)
    # Only a.bin is actually served; requests for b.bin get the same body and
    # therefore fail MD5 verification.
    target_a = tmp_path / identifier / 'a.bin'
    target_b = tmp_path / identifier / 'b.bin'
    original_b = b'pre-existing-b'
    target_b.parent.mkdir(parents=True, exist_ok=True)
    target_b.write_bytes(original_b)

    errors = item.download(
        files=['a.bin', 'b.bin'],
        destdir=str(tmp_path),
        atomic=True,
        retries=0,
        ignore_errors=True,
        no_directory=False,
    )
    assert errors == ['b.bin']
    assert target_a.read_bytes() == content_a
    assert target_b.read_bytes() == original_b
    server.shutdown()
    server.server_close()


def test_atomic_toplevel_api_forwards_flag(tmp_path, monkeypatch):
    """The top-level ``internetarchive.download`` convenience function must
    forward ``atomic`` through to Item/File and publish verified targets."""
    monkeypatch.chdir(tmp_path)
    body = b'toplevel-atomic-' * 16
    with open(NASA_METADATA_PATH) as fh:
        metadata = ia_json.load(fh)
    md5 = hashlib.md5(body).hexdigest()
    for f in metadata['files']:
        if f['name'] == 'nasa_meta.xml':
            f['size'] = str(len(body))
            f['md5'] = md5
    url_re = re.compile(r'https?://archive\.org/download/.*')
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        rsps.add(
            responses.GET,
            re.compile(r'https?://archive\.org/metadata/nasa'),
            body=ia_json.dumps(metadata),
            content_type='application/json',
        )
        rsps.add(responses.GET, url_re, body=body)
        errors = api_download(
            'nasa',
            files='nasa_meta.xml',
            destdir=str(tmp_path),
            no_directory=True,
            atomic=True,
        )
    assert errors == []
    target = tmp_path / 'nasa_meta.xml'
    assert target.read_bytes() == body
    assert not (tmp_path / 'nasa_meta.xml.ia-part').exists()


# A standalone script run in a subprocess: it expects the server address and
# destination directory as arguments and performs one atomic download attempt.
# It is deliberately self-contained so the child has no in-memory state to
# inherit -- resume can only work through the on-disk part/state files.
_SUBPROCESS_DOWNLOAD = r'''
import hashlib
import os
import sys

sys.path.insert(0, {repo!r})
from internetarchive import File, get_session

host_port, destdir = sys.argv[1], sys.argv[2]
content_hex = sys.argv[3]
content = bytes.fromhex(content_hex)
name = 'data.bin'
session = get_session(config={{'general': {{'secure': False}}}})
session.host = host_port
md5 = hashlib.md5(content).hexdigest()


class Item:
    def __init__(self):
        self.session = session
        self.item_metadata = {{
            'metadata': {{'identifier': 'testitem'}},
            'files': [{{'name': name, 'size': str(len(content)), 'md5': md5}}],
        }}


file_obj = File(Item(), name)
try:
    file_obj.download(
        file_path=name,
        destdir=destdir,
        atomic=True,
        retries=0,
        chunk_size=16,
    )
except Exception as exc:  # noqa: BLE001 - propagate any failure as exit 1
    print(type(exc).__name__, exc)
    sys.exit(1)
'''


def test_atomic_resume_across_processes(tmp_path, monkeypatch):
    """Two separate processes must hand off via the on-disk state: the first
    dies mid-stream leaving a partial, the second resumes with a Range and
    publishes a verified target."""
    content = b'0123456789abcdef' * 16  # 256 bytes
    server, state = make_server(content)
    host_port = f'127.0.0.1:{server.server_address[1]}'
    destdir = tmp_path / 'dl'
    destdir.mkdir()
    monkeypatch.chdir(tmp_path)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = _SUBPROCESS_DOWNLOAD.format(repo=repo_root)

    try:
        # First process: server truncates after 64 bytes, so the child exits 1
        # but leaves part/state behind.
        state.fail_count = 1
        state.fail_after = 64
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, '-c', script, host_port, str(destdir), content.hex()],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 1, proc.stderr
        part = destdir / 'data.bin.ia-part'
        state_file = destdir / 'data.bin.ia-state'
        assert part.exists()
        assert state_file.exists()
        assert part.stat().st_size == 64

        # Second process: healthy server; it must resume rather than restart.
        proc2 = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, '-c', script, host_port, str(destdir), content.hex()],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc2.returncode == 0, proc2.stderr
        target = destdir / 'data.bin'
        assert target.read_bytes() == content
        assert md5_hex(target.read_bytes()) == md5_hex(content)
        assert not part.exists()
        assert not state_file.exists()
        # The second process's request was a resume at the persisted offset.
        assert state.requests[-1][0] == 'bytes=64-'
    finally:
        server.shutdown()
        server.server_close()
