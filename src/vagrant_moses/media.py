"""Native extraction of C64 files from release media.

:func:`iter_prgs` walks zip/gzip nesting down to disk images, ``.t64`` tapes and
``.p00`` wrappers. Disk images are recognised by size (35/40/42-track d64, d71, d81,
each with or without an error-info block) and read by one CBM DOS directory walk.
"""

from __future__ import annotations

import gzip
import io
import zipfile
from dataclasses import dataclass, field
from itertools import accumulate
from typing import Callable, Dict, Iterator, List, Optional, Tuple

SECTOR_SIZE = 256
FILE_TYPES = ("DEL", "SEQ", "PRG", "USR", "REL")
TYPE_PRG = 2
_CLOSED, _LOCKED = 0x80, 0x40
_ZONES_1541 = (21,) * 17 + (19,) * 7 + (18,) * 6 + (17,) * 5

ErrorHandler = Optional[Callable[[str, Exception], None]]


class MediaError(ValueError):
    """Malformed media: a bad track/sector link, a chain loop, or a short image."""


@dataclass(frozen=True)
class Geometry:
    """A CBM DOS image layout: sectors per track (track 1 first) and directory start."""

    name: str
    sectors: Tuple[int, ...]
    dir_ts: Tuple[int, int]
    starts: Tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "starts", (0, 0, *accumulate(self.sectors)))

    @property
    def blocks(self) -> int:
        """Total sector count."""
        return self.starts[-1]

    def offset(self, track: int, sector: int) -> int:
        """Byte offset of ``track``/``sector``; :class:`MediaError` if off the disk."""
        if not 1 <= track <= len(self.sectors):
            raise MediaError(f"{self.name}: track out of range: {track}")
        if not 0 <= sector < self.sectors[track - 1]:
            raise MediaError(f"{self.name}: sector out of range T{track}S{sector}")
        return (self.starts[track] + sector) * SECTOR_SIZE


GEOMETRIES = (
    Geometry("d64", _ZONES_1541, (18, 1)),
    Geometry("d64-40", _ZONES_1541 + (17,) * 5, (18, 1)),
    Geometry("d64-42", _ZONES_1541 + (17,) * 7, (18, 1)),
    Geometry("d71", _ZONES_1541 * 2, (18, 1)),
    Geometry("d81", (40,) * 80, (40, 3)),
)
# An error-info block appends one status byte per sector.
GEOMETRY_BY_SIZE: Dict[int, Geometry] = {
    g.blocks * n: g for g in GEOMETRIES for n in (SECTOR_SIZE, SECTOR_SIZE + 1)
}
DISK_EXTS = (".d64", ".d71", ".d81")


@dataclass(frozen=True)
class MediaFile:
    """One file recovered from media. ``data`` of a PRG starts with its load address.

    ``error`` is set when a broken sector link cut the file short, as a drive would.
    """

    name: str
    data: bytes
    ftype: int = TYPE_PRG
    locked: bool = False
    error: str = ""

    @property
    def type_name(self) -> str:
        """CBM DOS file type mnemonic."""
        return FILE_TYPES[self.ftype]


def _petscii_name(raw: bytes) -> str:
    return raw.rstrip(b"\xa0").decode("latin-1")


def _chain(data: bytes, geom: Geometry, track: int, sector: int):
    """``(sectors, error)`` along a T/S link chain, stopping at a bad or looping link."""
    blocks: List[bytes] = []
    seen = set()
    while track:
        if (track, sector) in seen:
            return blocks, f"sector chain loops at T{track}S{sector}"
        seen.add((track, sector))
        try:
            off = geom.offset(track, sector)
        except MediaError as exc:
            return blocks, str(exc)
        blocks.append(data[off : off + SECTOR_SIZE])
        track, sector = blocks[-1][0], blocks[-1][1]
    return blocks, ""


def _payload(block: bytes) -> bytes:
    """A sector's data; a terminal link names its last used index, below 2 meaning all."""
    return block[2 : block[1] + 1] if block[0] == 0 and block[1] > 1 else block[2:]


def read_disk(data: bytes, geom: Optional[Geometry] = None) -> List[MediaFile]:
    """Every closed file in a disk image, in directory order.

    ``geom`` defaults to the geometry matching ``len(data)``. A broken file chain ends
    the file and is noted in :attr:`MediaFile.error`; an unreadable directory raises.
    """
    geom = geom or GEOMETRY_BY_SIZE.get(len(data))
    if geom is None or len(data) < geom.blocks * SECTOR_SIZE:
        raise MediaError(f"not a disk image: {len(data)} bytes")
    dirs, error = _chain(data, geom, *geom.dir_ts)
    if not dirs:
        raise MediaError(f"no directory: {error}")
    files: List[MediaFile] = []
    for block in dirs:
        for slot in range(2, SECTOR_SIZE, 32):
            entry = block[slot : slot + 30]
            ftype = entry[0] & 0x0F
            if not entry[0] & _CLOSED or not 0 < ftype < len(FILE_TYPES):
                continue
            blocks, error = _chain(data, geom, entry[1], entry[2])
            body = b"".join(map(_payload, blocks))
            name = _petscii_name(entry[3:19])
            files.append(MediaFile(name, body, ftype, bool(entry[0] & _LOCKED), error))
    return files


def read_t64(data: bytes) -> List[MediaFile]:
    """Every normal file on a ``.t64`` tape image, load address prepended.

    End addresses are unreliable in the wild (0, or a fixed bogus value), so a
    file's length is clamped to the gap before the next file's offset.
    """
    if not data.startswith(b"C64") or len(data) < 0x40:
        raise MediaError("not a t64: bad signature")
    slots = min(data[0x22] | data[0x23] << 8, (len(data) - 0x40) // 32)
    entries = [data[0x40 + 32 * i : 0x60 + 32 * i] for i in range(slots)]
    entries = [e for e in entries if e[0] == 1]
    ends = sorted({int.from_bytes(e[8:12], "little") for e in entries} | {len(data)})
    files = []
    for e in entries:
        start, end = e[2] | e[3] << 8, e[4] | e[5] << 8
        off = int.from_bytes(e[8:12], "little")
        room = next((o for o in ends if o > off), off) - off
        size = (end - start) & 0xFFFF
        size = size if 0 < size <= room else room
        name = _petscii_name(e[16:32]).rstrip(" ")
        files.append(MediaFile(name, e[2:4] + data[off : off + size]))
    return files


def read_p00(data: bytes) -> MediaFile:
    """The file inside a PC64 ``.p00`` wrapper."""
    if not data.startswith(b"C64File\0") or len(data) < 26:
        raise MediaError("not a p00: bad signature")
    return MediaFile(data[8:24].rstrip(b"\0").decode("latin-1"), data[26:])


def iter_prgs(
    name: str, data: bytes, on_error: ErrorHandler = None, disks: bool = True
) -> Iterator[Tuple[str, bytes]]:
    """``(path, bytes)`` for every leaf reached through gzip/zip/disk/t64/p00 nesting.

    Containers are detected by content (disk images by size, or by extension when
    truncated); paths join names with ``/``; ``disks=False`` yields disk images whole.
    ``on_error(path, exc)`` receives unreadable members and truncated disk files, and
    iteration continues; without it, only unreadable members raise.
    """
    low = name.lower()
    try:
        if data[:2] == b"\x1f\x8b":
            inner = name[:-3] if low.endswith(".gz") else name
            yield from iter_prgs(inner, gzip.decompress(data), on_error, disks)
        elif data[:4] == b"PK\x03\x04":
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for info in zf.infolist():
                    if not info.is_dir():
                        member = f"{name}/{info.filename}"
                        yield from iter_prgs(member, zf.read(info), on_error, disks)
        elif not disks and (len(data) in GEOMETRY_BY_SIZE or low.endswith(DISK_EXTS)):
            yield name, data
        elif len(data) in GEOMETRY_BY_SIZE or low.endswith(DISK_EXTS):
            for f in read_disk(data):
                if f.error and on_error:
                    on_error(f"{name}/{f.name}", MediaError(f.error))
                if f.ftype == TYPE_PRG and (f.data or not f.error):
                    yield f"{name}/{f.name}", f.data
        elif data.startswith(b"C64File\0"):
            f = read_p00(data)
            yield f"{name}/{f.name}", f.data
        elif data.startswith(b"C64"):
            for f in read_t64(data):
                yield f"{name}/{f.name}", f.data
        else:
            yield name, data
    except (MediaError, OSError, EOFError, zipfile.BadZipFile) as exc:
        if on_error is None:
            raise
        on_error(name, exc)
