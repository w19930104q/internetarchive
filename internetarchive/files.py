#
# The internetarchive module is a Python/CLI interface to Archive.org.
#
# Copyright (C) 2012-2026 Internet Archive
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
internetarchive.files
~~~~~~~~~~~~~~~~~~~~~

:copyright: (C) 2012-2024 by Internet Archive.
:license: AGPL 3, see LICENSE for more details.
"""

import logging
import os
import re
import sys
import tempfile
from contextlib import nullcontext, suppress
from email.utils import parsedate_to_datetime
from pathlib import Path
from time import sleep
from urllib.parse import quote

from requests.exceptions import (
    ConnectionError,
    ConnectTimeout,
    HTTPError,
    ReadTimeout,
    RetryError,
)
from tqdm import tqdm

from internetarchive import auth, exceptions, iarequest, utils
from internetarchive.utils import json

log = logging.getLogger(__name__)

# Suffixes for the on-disk state of an atomic verified download. The partial
# bytes are written to ``<target>.ia-part`` and the resume state to
# ``<target>.ia-state``, both alongside the target so they stay on the same
# filesystem (required for the final atomic ``os.replace``).
ATOMIC_PART_SUFFIX = '.ia-part'
ATOMIC_STATE_SUFFIX = '.ia-state'


def _atomic_paths(file_path):
    """Return the ``(part_path, state_path)`` pair for an atomic download.

    :param file_path: The final target path of the download.
    :returns: A tuple of the partial-file path and the state-file path.
    """
    return (
        f'{file_path}{ATOMIC_PART_SUFFIX}',
        f'{file_path}{ATOMIC_STATE_SUFFIX}',
    )


def _write_atomic_state(state_path, state):
    """Write resume state to disk atomically, so a crashed process can never
    leave a torn state file behind.

    :param state_path: The state file path.
    :param state: The state mapping to persist as JSON.
    """
    directory = os.path.dirname(state_path) or '.'
    fd, tmp_path = tempfile.mkstemp(
        dir=directory,
        prefix=os.path.basename(state_path),
        suffix='.tmp',
    )
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            json.dump(state, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, state_path)
    except OSError:
        with suppress(OSError):
            os.remove(tmp_path)
        raise


def _read_atomic_state(state_path):
    """Read resume state written by :func:`_write_atomic_state`.

    :param state_path: The state file path.
    :returns: The state mapping, or ``None`` if the file is missing or unreadable.
    """
    try:
        with open(state_path, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _state_matches(state, identifier, name, size, md5):
    """Check whether persisted resume state still describes the current remote
    file. Any identity change (item, filename, size, or MD5) invalidates the
    partial download and forces a fresh start.

    :param state: The persisted state mapping.
    :param identifier: The expected item identifier.
    :param name: The expected remote filename.
    :param size: The expected remote file size.
    :param md5: The expected remote MD5.
    :returns: ``True`` if the state matches the remote identity.
    """
    if not isinstance(state, dict):
        return False
    return (
        state.get('identifier') == identifier
        and state.get('name') == name
        and state.get('size') == size
        and state.get('md5') == md5
    )


def _parse_content_range(content_range):
    """Parse the start offset and total length out of a ``Content-Range`` header.

    Accepts the RFC 7233 form ``bytes START-END/TOTAL`` (TOTAL may be ``*``).

    :param content_range: The raw header value.
    :returns: A ``(start, total)`` tuple; either element may be ``None`` when
              absent. Returns ``(None, None)`` if the header cannot be parsed.
    """
    if not content_range:
        return None, None
    match = re.match(r'bytes\s+(\d+)-\d+/(?:(\d+)|\*)', content_range.strip())
    if not match:
        return None, None
    start = int(match.group(1))
    total = int(match.group(2)) if match.group(2) else None
    return start, total


class BaseFile:
    """Base class for Archive.org file objects.

    This class provides common attributes for file objects. It extracts
    file metadata from item metadata and provides access to file properties
    like size, format, checksums, and modification time.

    Attributes:
        identifier: The item identifier this file belongs to.
        name: The filename.
        size: File size in bytes.
        source: File source (``'original'``, ``'derivative'``, ``'metadata'``).
        format: File format (e.g., ``'JPEG'``, ``'Ogg Vorbis'``).
        md5: MD5 checksum of the file.
        sha1: SHA1 checksum of the file.
        mtime: Modification time as a Unix timestamp.
        crc32: CRC32 checksum of the file.
        exists: ``True`` if the file exists in the item's metadata.
    """

    def __init__(self, item_metadata, name, file_metadata=None):
        if file_metadata is None:
            file_metadata = {}
        name = name.strip('/')
        if not file_metadata:
            for f in item_metadata.get('files', []):
                if f.get('name') == name:
                    file_metadata = f
                    break

        self.identifier = item_metadata.get('metadata', {}).get('identifier')
        self.name = name
        self.size = None
        self.source = None
        self.format = None
        self.md5 = None
        self.sha1 = None
        self.mtime = None
        self.crc32 = None

        self.exists = bool(file_metadata)

        for key in file_metadata:
            setattr(self, key, file_metadata[key])
        # An additional, more orderly way to access file metadata,
        # which avoids filtering the attributes.
        self.metadata = file_metadata
        self.mtime = float(self.mtime) if self.mtime else 0
        self.size = int(self.size) if self.size else 0


class File(BaseFile):
    """This class represents a file in an archive.org item. You
    can use this class to access the file metadata::

        >>> import internetarchive
        >>> item = internetarchive.Item('stairs')
        >>> file = internetarchive.File(item, 'stairs.avi')
        >>> print(f.format, f.size)
        ('Cinepack', '3786730')

    Or to download a file::

        >>> file.download()
        >>> file.download('fabulous_movie_of_stairs.avi')

    This class also uses IA's S3-like interface to delete a file
    from an item. You need to supply your IAS3 credentials in
    environment variables in order to delete::

        >>> file.delete(access_key='Y6oUrAcCEs4sK8ey',
        ...             secret_key='youRSECRETKEYzZzZ')

    You can retrieve S3 keys here: `https://archive.org/account/s3.php
    <https://archive.org/account/s3.php>`__

    """

    def __init__(self, item, name, file_metadata=None):
        """
        :param item: The item that the file is part of.
        :param name: The filename of the file.
        :param file_metadata: A dict of metadata for the given file.
        """
        super().__init__(item.item_metadata, name, file_metadata)
        self.item = item
        url_parts = {
            'protocol': item.session.protocol,
            'id': self.identifier,
            'name': quote(name.encode('utf-8')),
            'host': item.session.host,
        }
        self.url = '{protocol}//{host}/download/{id}/{name}'.format(**url_parts)
        if self.item.session.access_key and self.item.session.secret_key:
            self.auth = auth.S3Auth(
                self.item.session.access_key, self.item.session.secret_key
            )
        else:
            self.auth = None

    def __repr__(self):
        return (
            f'File(identifier={self.identifier!r}, '
            f'filename={self.name!r}, '
            f'size={self.size!r}, '
            f'format={self.format!r})'
        )

    def download(  # noqa: C901,PLR0911,PLR0912,PLR0915
        self,
        file_path=None,
        verbose=None,
        ignore_existing=None,
        checksum=None,
        checksum_archive=None,
        destdir=None,
        retries=None,
        ignore_errors=None,
        fileobj=None,
        return_responses=None,
        no_change_timestamp=None,
        params=None,
        chunk_size=None,
        stdout=None,
        ors=None,
        timeout=None,
        headers=None,
        count_views=False,
        atomic=None,
    ):
        """Download the file into the current working directory.

        :param file_path: Download file to the given file_path.
        :param verbose: Turn on verbose output.
        :param ignore_existing: Overwrite local files if they already exist.
        :param checksum: Skip downloading file based on checksum.
        :param checksum_archive: Skip downloading file based on checksum, and
                                 skip checksum validation if it already succeeded
                                 (will create and use _checksum_archive.txt).
        :param destdir: The directory to download files to.
        :param retries: The number of times to retry on failed requests.
        :param ignore_errors: Don't fail if a single file fails to
                              download, continue to download other files.
        :param fileobj: Write data to the given file-like object
                        (e.g. sys.stdout).
        :param return_responses: Rather than downloading files to disk, return
                                 a list of response objects.
        :param no_change_timestamp: If True, leave the time stamp as the
                                    current time instead of changing it to
                                    that given in the original archive.
        :param stdout: Print contents of file to stdout instead of
                       downloading to file.
        :param ors: Append a newline or $ORS to the end of file.
                    This is mainly intended to be used internally with `stdout`.
        :param params: URL parameters to send with download request.
                       By default the library injects ``cnt=0`` so downloads do
                       not count toward archive.org view counts; pass
                       ``count_views=True`` to omit it. An explicit ``cnt`` key
                       in ``params`` always wins.
        :param count_views: If True, omit the default ``cnt=0`` parameter so
                            the download counts toward archive.org view counts.
                            Has no effect if ``params`` already contains a
                            ``cnt`` key.
        :param headers: Extra HTTP headers to send with the download request.
                        Supplying a ``Range`` header (e.g.
                        ``{'Range': 'bytes=0-1023'}``) performs an intentional
                        partial fetch: the automatic resume behaviour and
                        full-file checksum validation are skipped, so the bytes
                        returned by the server are written as-is.
        :param atomic: If True, download to a temporary partial file and only
                       atomically replace the target once both its content
                       length and metadata MD5 have been verified. The final
                       target is never left partially written or corrupt:
                       network, checksum, write, or replace failures keep any
                       pre-existing target untouched. Resume state is persisted
                       next to the target (``<target>.ia-part`` and
                       ``<target>.ia-state``) and bound to the item identifier,
                       filename, size, and MD5, so downloads can resume across
                       processes and restart from scratch when the remote file
                       changes. Only downloads to a target path are supported;
                       combining this with ``stdout``, ``fileobj``,
                       ``return_responses``, or an explicit ``Range`` header
                       raises ``ValueError`` before any request is made. A
                       pre-existing target is itself verified before being
                       skipped, and the file fails if its metadata lacks a
                       valid size or MD5. Defaults to False.

        :returns: ``True`` if file was successfully downloaded.
        """
        verbose = False if verbose is None else verbose
        ignore_existing = False if ignore_existing is None else ignore_existing
        checksum = False if checksum is None else checksum
        checksum_archive = False if checksum_archive is None else checksum_archive
        atomic = False if atomic is None else atomic
        retries = 2 if retries is None else retries
        ignore_errors = ignore_errors or False
        return_responses = return_responses or False
        no_change_timestamp = no_change_timestamp or False
        params = dict(params) if params else {}
        if not count_views:
            params.setdefault('cnt', '0')
        timeout = 12 if not timeout else timeout
        headers = headers or {}
        retries_sleep = 3  # TODO: exponential sleep
        retrying = False  # for retry loop
        resume = False  # True when auto-resuming a partial local download
        # Whether the caller supplied an explicit Range header (case-insensitive).
        # An explicit range is an intentional partial fetch and must never trigger
        # auto-resume or full-file checksum validation.
        explicit_range = any(k.lower() == 'range' for k in headers)

        if atomic:
            incompatible = []
            if stdout:
                incompatible.append('stdout')
            if fileobj is not None:
                incompatible.append('fileobj')
            if return_responses:
                incompatible.append('return_responses')
            if explicit_range:
                incompatible.append('a Range header')
            if incompatible:
                raise ValueError(
                    'atomic downloads only write to a target path; they cannot '
                    f'be combined with {", ".join(incompatible)}'
                )

        self.item.session.mount_http_adapter(max_retries=retries)
        file_path = file_path or self.name

        if os.name == 'nt' and not return_responses:
            file_path, _ = utils.sanitize_windows_relpath(
                file_path,
                verbose=bool(verbose),
                printer=lambda m: print(m, file=sys.stderr),
            )

        if destdir:
            if not (return_responses or stdout):
                try:
                    os.makedirs(destdir, exist_ok=True)
                except OSError:
                    pass
            if os.path.isfile(destdir):
                raise OSError(f'{destdir} is not a directory!')
            file_path = os.path.join(destdir, file_path)

        # Windows sanitization handled earlier; legacy comment removed.

        # Directory traversal guard (all platforms). Ensure target path is inside destdir
        # (or cwd if none provided). Determine intended base directory.
        intended_base = destdir if destdir else os.getcwd()
        try:
            if not utils.is_path_within_directory(
                intended_base, os.path.abspath(file_path)
            ):
                raise exceptions.DirectoryTraversalError(
                    f'Unsafe file path resolved outside destination directory: {file_path}'
                )
        except AttributeError:
            # Fallback if DirectoryTraversalError not present (older versions); re-raise generic.
            raise OSError(
                f'Unsafe file path resolved outside destination directory: {file_path}'
            )

        parent_dir = os.path.dirname(file_path)

        # Warn (not fail) if path length may cause Windows issues (>240 chars typical safe limit)
        if os.name == 'nt' and len(os.path.abspath(file_path)) > 240:
            log.warning('Long path may cause issues: %s', file_path)
            if verbose:
                print(
                    f' warning: long path may cause issues: {file_path}',
                    file=sys.stderr,
                )

        # Atomic verified download: own code path with an independent retry loop,
        # persistent resume state, and an os.replace that only happens after the
        # full content passes length and MD5 verification. The legacy path below
        # (and all of its behaviour) is untouched when atomic is False.
        if atomic:
            try:
                self._download_atomic(
                    file_path,
                    parent_dir,
                    verbose=verbose,
                    retries=retries,
                    params=params,
                    chunk_size=chunk_size,
                    timeout=timeout,
                    headers=headers,
                    no_change_timestamp=no_change_timestamp,
                )
            except (ValueError, exceptions.MetadataValidationError):
                # Programming/metadata errors must not be swallowed as retriable
                # download failures.
                raise
            except Exception as exc:
                if verbose:
                    print(f' error downloading {file_path}: {exc}', file=sys.stderr)
                if ignore_errors:
                    return False
                raise
            return True

        # Check if we should skip... (never when streaming to stdout: the
        # local filesystem is irrelevant to a stdout download).
        if (
            not return_responses
            and not stdout
            and os.path.exists(file_path.encode('utf-8'))
        ):
            if checksum_archive:
                checksum_archive_filename = '_checksum_archive.txt'
                if not os.path.exists(checksum_archive_filename):
                    with open(checksum_archive_filename, 'w', encoding='utf-8') as f:
                        pass
                with open(checksum_archive_filename, encoding='utf-8') as f:
                    checksum_archive_data = f.read().splitlines()
                if file_path in checksum_archive_data:
                    msg = (
                        f'skipping {file_path}, '
                        f'file already exists based on checksum_archive.'
                    )
                    log.info(msg)
                    if verbose:
                        print(f' {msg}', file=sys.stderr)
                    return
            if ignore_existing:
                msg = f'skipping {file_path}, file already exists.'
                log.info(msg)
                if verbose:
                    print(f' {msg}', file=sys.stderr)
                return
            elif checksum or checksum_archive:
                with open(file_path, 'rb') as fp:
                    md5_sum = utils.get_md5(fp)

                if md5_sum == self.md5:
                    msg = (
                        f'skipping {file_path}, file already exists based on checksum.'
                    )
                    log.info(msg)
                    if verbose:
                        print(f' {msg}', file=sys.stderr)
                    if checksum_archive:
                        # add file to checksum_archive to skip it next time
                        with open(
                            checksum_archive_filename, 'a', encoding='utf-8'
                        ) as f:
                            f.write(f'{file_path}\n')
                    return

        # Retry loop
        while True:
            try:
                if parent_dir != '' and not (return_responses or stdout):
                    os.makedirs(parent_dir, exist_ok=True)

                if (
                    not return_responses
                    and not stdout
                    and not ignore_existing
                    and self.name != f'{self.identifier}_files.xml'
                    and os.path.exists(file_path.encode('utf-8'))
                ):
                    st = os.stat(file_path.encode('utf-8'))
                    # Only auto-resume when the caller has not supplied an explicit
                    # Range header (e.g. via the ``--range`` CLI flag). An explicit
                    # range is an intentional partial fetch and must not trigger the
                    # resume seek/append or the full-file checksum validation below.
                    # (Resume is also skipped for stdout, since seeking a pipe fails
                    # and there is no local partial file to append to.)
                    if (
                        st.st_size != self.size
                        and not (checksum or checksum_archive)
                        and not explicit_range
                    ):
                        # Recompute the resume Range from the current file size on
                        # every attempt so it stays aligned with the seek offset
                        # below; a stale Range left over from a prior attempt would
                        # re-fetch already-written bytes and corrupt the file.
                        # Preserve any caller-supplied headers; only set Range.
                        headers = {**headers, "Range": f"bytes={st.st_size}-"}
                        resume = True

                response = self.item.session.get(
                    self.url,
                    stream=True,
                    timeout=timeout,
                    auth=self.auth,
                    params=params,
                    headers=headers,
                )
                # Get timestamp from Last-Modified header
                last_mod_header = response.headers.get('Last-Modified')
                if last_mod_header:
                    dt = parsedate_to_datetime(last_mod_header)
                    last_mod_mtime = dt.timestamp()
                else:
                    last_mod_mtime = self.mtime

                response.raise_for_status()

                # Check if we should skip based on last modified time...
                if (
                    not fileobj
                    and not return_responses
                    and not stdout
                    and os.path.exists(file_path.encode('utf-8'))
                ):
                    st = os.stat(file_path.encode('utf-8'))
                    if st.st_mtime == last_mod_mtime:
                        if self.name == f'{self.identifier}_files.xml' or (
                            st.st_size == self.size
                        ):
                            msg = (
                                f'skipping {file_path}, file already exists based on '
                                'length and date.'
                            )
                            log.info(msg)
                            if verbose:
                                print(f' {msg}', file=sys.stderr)
                            return

                elif return_responses:
                    return response

                if verbose:
                    total = int(response.headers.get('content-length', 0)) or None
                    progress_bar = tqdm(
                        desc=f' downloading {self.name}',
                        total=total,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                    )
                else:
                    progress_bar = nullcontext()

                if not chunk_size:
                    chunk_size = 1048576
                if stdout:
                    # stdout is its own sink; never fall back to a local file,
                    # even on a retry (which must keep writing to the pipe).
                    fileobj = os.fdopen(sys.stdout.fileno(), 'wb', closefd=False)
                elif not fileobj or retrying:
                    if resume:
                        fileobj = open(file_path.encode('utf-8'), 'rb+')
                    else:
                        fileobj = open(file_path.encode('utf-8'), 'wb')

                with fileobj, progress_bar as bar:
                    if resume:
                        fileobj.seek(st.st_size)
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            size = fileobj.write(chunk)
                            if bar is not None:
                                bar.update(size)
                    if ors:
                        fileobj.write(os.environ.get("ORS", "\n").encode("utf-8"))

                if resume:
                    with open(file_path, 'rb') as fh:
                        local_checksum = utils.get_md5(fh)
                    try:
                        assert local_checksum == self.md5
                    except AssertionError:
                        msg = (
                            f"\"{file_path}\" corrupt, "
                            "checksums do not match. "
                            "Remote file may have been modified, "
                            "retry download."
                        )
                        os.remove(file_path.encode('utf-8'))
                        raise exceptions.InvalidChecksumError(msg)
                break
            except (
                RetryError,
                HTTPError,
                ConnectTimeout,
                OSError,
                ReadTimeout,
                exceptions.InvalidChecksumError,
            ) as exc:
                # A 416 (Range Not Satisfiable) is a permanent response to an
                # explicit range request -- retrying cannot help, so fail fast.
                resp = getattr(exc, 'response', None)
                unsatisfiable = getattr(resp, 'status_code', None) == 416
                if retries > 0 and not unsatisfiable:
                    retrying = True
                    retries -= 1
                    msg = (
                        'download failed, sleeping for '
                        f'{retries_sleep} seconds and retrying. '
                        f'{retries} retries left.'
                    )
                    log.warning(msg)
                    sleep(retries_sleep)
                    continue
                if unsatisfiable:
                    valid = resp.headers.get('Content-Range', '')
                    rng = headers.get('Range', '')
                    msg = (
                        f'error downloading {file_path}: requested range '
                        f'{rng!r} not satisfiable'
                        + (f' (file is {valid})' if valid else '')
                    )
                else:
                    msg = f'error downloading file {file_path}, exception raised: {exc}'
                log.error(msg)
                # Never touch the local filesystem for a stdout download.
                if not stdout:
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass
                if verbose:
                    print(f' {msg}', file=sys.stderr)
                if ignore_errors:
                    return False
                else:
                    raise exc

        # Set mtime with timestamp from Last-Modified header
        if not no_change_timestamp:
            # If we want to set the timestamp to that of the original archive...
            with suppress(OSError):  # Probably file-like object, e.g. sys.stdout.
                os.utime(file_path.encode('utf-8'), (0, last_mod_mtime))

        msg = f'downloaded {self.identifier}/{self.name} to {file_path}'
        log.info(msg)
        return True

    def _download_atomic(
        self,
        file_path,
        parent_dir,
        verbose=False,
        retries=2,
        params=None,
        chunk_size=None,
        timeout=12,
        headers=None,
        no_change_timestamp=False,
    ):
        """Download the file atomically with full verification and cross-process resume.

        Bytes are accumulated in ``<target>.ia-part`` with resume metadata in
        ``<target>.ia-state``. The target is created or replaced via
        :func:`os.replace` only once the assembled content matches the remote
        size and metadata MD5; every failure path leaves a pre-existing target
        untouched.

        :param file_path: The final target path.
        :param parent_dir: The directory containing the target.
        :param verbose: Print progress and status messages to stderr.
        :param retries: Number of retries for network/write failures.
        :param params: URL query parameters for the download request.
        :param chunk_size: Streaming chunk size in bytes.
        :param timeout: Request timeout.
        :param headers: Extra request headers (never a ``Range`` -- rejected upstream).
        :param no_change_timestamp: If True, leave the target's mtime as the
                                    current time instead of applying Last-Modified.
        :raises MetadataValidationError: If the file metadata lacks a valid size
                                         or MD5, or the downloaded content fails
                                         verification.
        :raises OSError: On unrecovered network, write, or replace failures.
        """
        headers = dict(headers or {})
        params = params or {}

        # Verified downloads cannot work without the remote identity and a
        # complete-content checksum. Fail before touching the network or any
        # local file.
        if not self.size or not isinstance(self.size, int):
            msg = (
                f'cannot download {self.identifier}/{self.name!r} atomically: '
                'metadata has no valid file size'
            )
            raise exceptions.MetadataValidationError(msg)
        if not self.md5 or not isinstance(self.md5, str):
            msg = (
                f'cannot download {self.identifier}/{self.name!r} atomically: '
                'metadata has no valid MD5 checksum'
            )
            raise exceptions.MetadataValidationError(msg)

        remote_size = self.size
        remote_md5 = self.md5
        part_path, state_path = _atomic_paths(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        identity = {
            'identifier': self.identifier,
            'name': self.name,
            'size': remote_size,
            'md5': remote_md5,
        }

        # An existing target is trusted only after its own size and MD5 match
        # the metadata; otherwise it must be replaced, never skipped blindly.
        if os.path.exists(file_path.encode('utf-8')):
            with open(file_path, 'rb') as fh:
                local_md5 = utils.get_md5(fh)
            if os.path.getsize(file_path) == remote_size and local_md5 == remote_md5:
                msg = f'skipping {file_path}, existing file already verified.'
                log.info(msg)
                if verbose:
                    print(f' {msg}', file=sys.stderr)
                # Remove stale partial/state files from an interrupted earlier run.
                with suppress(OSError):
                    os.remove(part_path)
                with suppress(OSError):
                    os.remove(state_path)
                return
            msg = (
                f'existing {file_path} failed verification '
                '(size or MD5 mismatch); re-downloading atomically.'
            )
            log.warning(msg)
            if verbose:
                print(f' {msg}', file=sys.stderr)

        if not chunk_size:
            chunk_size = 1048576
        attempts_left = retries
        last_mod_mtime = self.mtime
        while True:
            resume_offset = 0
            state = _read_atomic_state(state_path)
            if state is not None and _state_matches(
                state, self.identifier, self.name, remote_size, remote_md5
            ):
                try:
                    resume_offset = os.path.getsize(part_path)
                except OSError:
                    resume_offset = 0
                if resume_offset >= remote_size:
                    # A full-size partial never got verified/replaced. Appending
                    # another body cannot help, so restart from scratch.
                    with suppress(OSError):
                        os.remove(part_path)
                    resume_offset = 0
            else:
                # No state, unreadable state, or a remote identity change: any
                # partial bytes belong to a different download and must be
                # discarded before starting fresh.
                with suppress(OSError):
                    os.remove(part_path)
                resume_offset = 0
                _write_atomic_state(state_path, identity)

            request_headers = headers
            if resume_offset and resume_offset < remote_size:
                request_headers = {
                    **headers,
                    'Range': f'bytes={resume_offset}-',
                }

            try:
                response = self.item.session.get(
                    self.url,
                    stream=True,
                    timeout=timeout,
                    auth=self.auth,
                    params=params,
                    headers=request_headers,
                )
                last_mod_header = response.headers.get('Last-Modified')
                if last_mod_header:
                    last_mod_mtime = parsedate_to_datetime(last_mod_header).timestamp()
                response.raise_for_status()

                if resume_offset and resume_offset < remote_size:
                    if response.status_code != 206:
                        # Server ignored our Range: restart from scratch so
                        # partial prefix and full body are never concatenated.
                        log.warning(
                            'server returned %s for a resume request to %s; '
                            'restarting download from the beginning',
                            response.status_code,
                            self.url,
                        )
                        resume_offset = 0
                        with suppress(OSError):
                            os.remove(part_path)
                    else:
                        range_start, range_total = _parse_content_range(
                            response.headers.get('Content-Range')
                        )
                        if range_start != resume_offset or (
                            range_total is not None and range_total != remote_size
                        ):
                            msg = (
                                f'invalid Content-Range '
                                f'{response.headers.get("Content-Range")!r} for '
                                f'resume at byte {resume_offset} of {remote_size}'
                            )
                            raise exceptions.MetadataValidationError(msg)

                if resume_offset == 0:
                    # Full (or restarted) fetch: truncate any stale bytes.
                    part_file = open(part_path, 'wb')
                else:
                    part_file = open(part_path, 'ab')
                with part_file:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            part_file.write(chunk)
                    part_file.flush()
                    os.fsync(part_file.fileno())

                # Length verification before the more expensive hash.
                actual_size = os.path.getsize(part_path)
                if actual_size != remote_size:
                    with suppress(OSError):
                        os.remove(part_path)
                    msg = (
                        f'"{file_path}" incomplete: expected {remote_size} '
                        f'bytes, got {actual_size}'
                    )
                    raise exceptions.InvalidChecksumError(msg)
                with open(part_path, 'rb') as fh:
                    actual_md5 = utils.get_md5(fh)
                if actual_md5 != remote_md5:
                    with suppress(OSError):
                        os.remove(part_path)
                    msg = (
                        f'"{file_path}" corrupt, checksums do not match '
                        f'(expected {remote_md5}, got {actual_md5}). Remote file '
                        'may have been modified.'
                    )
                    raise exceptions.InvalidChecksumError(msg)

                # Both checks passed: publish atomically. The target (including
                # any pre-existing file) is replaced in one rename; if this
                # raises, the target is untouched and the partial stays on disk
                # for diagnosis/retry.
                os.replace(part_path, file_path)
                with suppress(OSError):
                    os.remove(state_path)
                if not no_change_timestamp:
                    with suppress(OSError):
                        os.utime(file_path.encode('utf-8'), (0, last_mod_mtime))
                msg = (
                    f'downloaded and verified {self.identifier}/{self.name} '
                    f'to {file_path}'
                )
                log.info(msg)
                if verbose:
                    print(f' {msg}', file=sys.stderr)
                return
            except (
                RetryError,
                HTTPError,
                ConnectionError,
                ConnectTimeout,
                ReadTimeout,
                OSError,
            ) as exc:
                resp = getattr(exc, 'response', None)
                unsatisfiable = getattr(resp, 'status_code', None) == 416
                if attempts_left > 0 and not unsatisfiable:
                    attempts_left -= 1
                    msg = (
                        'atomic download failed, sleeping for 3 seconds and '
                        f'retrying. {attempts_left} retries left. ({exc})'
                    )
                    log.warning(msg)
                    if verbose:
                        print(f' {msg}', file=sys.stderr)
                    sleep(3)
                    continue
                # Exhausted (or permanent failure). Keep the original target
                # exactly as it was; only partial/state artifacts are ours to
                # remove. Leave them in place too so a later invocation can
                # resume -- but surface the error.
                log.error('error downloading file %s: %s', file_path, exc)
                raise
            except exceptions.InvalidChecksumError:
                # Content mismatch is permanent for the current metadata:
                # retrying the same bytes cannot help. State is invalid and the
                # partial was already removed; drop the state file as well.
                with suppress(OSError):
                    os.remove(state_path)
                raise

    def delete(
        self,
        cascade_delete=None,
        access_key=None,
        secret_key=None,
        verbose=None,
        debug=None,
        retries=None,
        headers=None,
    ):
        """Delete a file from the Archive.

        Note: Some files -- such as ``<itemname>_meta.xml`` -- cannot be deleted.

        :param cascade_delete: Delete all files associated with the specified
                               file, including upstream derivatives and the original.
        :param access_key: IA-S3 access_key to use when making the given request.
        :param secret_key: IA-S3 secret_key to use when making the given request.
        :param verbose: Print actions to stdout.
        :param debug: Set to True to print headers to stdout and exit
                      without sending the delete request.
        """
        cascade_delete = '0' if not cascade_delete else '1'
        access_key = self.item.session.access_key if not access_key else access_key
        secret_key = self.item.session.secret_key if not secret_key else secret_key
        debug = debug or False
        verbose = verbose or False
        max_retries = retries or 2
        headers = headers or {}

        if 'x-archive-cascade-delete' not in headers:
            headers['x-archive-cascade-delete'] = cascade_delete

        url = f'{self.item.session.protocol}//s3.us.archive.org/{self.identifier}/{quote(self.name)}'
        self.item.session.mount_http_adapter(
            max_retries=max_retries, status_forcelist=[503], host='s3.us.archive.org'
        )
        request = iarequest.S3Request(
            method='DELETE',
            url=url,
            headers=headers,
            access_key=access_key,
            secret_key=secret_key,
        )
        if debug:
            return request
        else:
            if verbose:
                msg = f' deleting: {self.name}'
                if cascade_delete == '1':
                    msg += ' and all derivative files.'
                print(msg, file=sys.stderr)
            prepared_request = self.item.session.prepare_request(request)

            try:
                resp = self.item.session.send(prepared_request)
                resp.raise_for_status()
            except (RetryError, HTTPError, ConnectTimeout, OSError, ReadTimeout) as exc:
                error_msg = f'Error deleting {url}, {exc}'
                log.error(error_msg)
                raise
            else:
                return resp
            finally:
                # The retry adapter is mounted to the session object.
                # Make sure to remove it after delete, so it isn't
                # mounted if and when the session object is used for an
                # upload. This is important because we use custom retry
                # handling for IA-S3 uploads.
                url_prefix = f'{self.item.session.protocol}//s3.us.archive.org'
                del self.item.session.adapters[url_prefix]


class OnTheFlyFile(File):
    """A file that is generated on-the-fly by Archive.org (e.g., EPUB, MOBI)."""

    def __init__(self, item, name):
        """
        :param item: The item that the file is part of.
        :param name: The filename of the file.
        """
        super().__init__(item.item_metadata, name)
