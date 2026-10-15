"""The C64 ROMs: sidplayfp's ROM-less stand-ins, and the real ones fetched on demand.

The real ROMs are Commodore copyright and never shipped: :func:`fetch_rom` caches
them, SHA-256 verified. Without them, :func:`kernal_stub` is libsidplayfp's ROM-less
KERNAL, and BASIC and the character ROM read zero, as sidplayfp runs a tune.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple

from . import memmap as mm

# VICE's own C64 ROM set at an immutable mirror tag; ``$C64_ROM_MIRROR`` overrides.
DEFAULT_ROM_MIRROR = (
    "https://raw.githubusercontent.com/VICE-Team/svn-mirror/r46192/vice/data/C64"
)


class Rom(NamedTuple):
    """One fetched asset: mirror filename, mapped base, size and SHA-256.

    ``url`` pins the asset to one upstream location instead of the ROM mirror.
    """

    filename: str
    base: int
    size: int
    sha256: str
    url: Optional[str] = None


ROMS: Dict[str, Rom] = {
    "kernal": Rom(
        "kernal-901227-03.bin",
        mm.KERNAL_BASE,
        0x2000,
        "83c60d47047d7beab8e5b7bf6f67f80daa088b7a6a27de0d7e016f6484042721",
    ),
    "basic": Rom(
        "basic-901226-01.bin",
        mm.BASIC_BASE,
        0x2000,
        "89878cea0a268734696de11c4bae593eaaa506465d2029d619c0e0cbccdfa62d",
    ),
    "chargen": Rom(
        "chargen-901225-01.bin",
        mm.CHARGEN_BASE,
        0x1000,
        "fd0d53b8480e86163ac98998976c72cc58d5dd8eb824ed7b829774e74213b420",
    ),
}

# libsidplayfp's RLE image of the RAM a booted KERNAL leaves at $0000-$03FF.
ASSETS: Dict[str, Rom] = {
    "poweron": Rom(
        "poweron.bin",
        0x0000,
        1777,
        "dcb750bd025598645b89d8fc2840eaa809d0c662bbbf77aa6965442a2d3113fb",
        "https://raw.githubusercontent.com/libsidplayfp/libsidplayfp/"
        "9c464110fd471e3989a6c794c0dbf8a218727e13/src/poweron.bin",
    ),
}

# CHARGEN excluded: at $D000 it aliases the I/O page a flat image must keep.
DEFAULT_ROMS = ("kernal", "basic")

KERNAL_BOOT = (0xFD15, 0xFDA3, 0xFF5B)  # RESTOR, IOINIT, CINT
BASIC_BOOT = (0xE453, 0xE3BF)  # BASIC vectors, BASIC RAM
BASIC_RUN = 0xA7B1  # BASIC interpreter main loop, where RUN enters

# KERNAL jump-table entries and the routines they call, each an RTS in the stub.
KERNAL_CALLS = (
    0xFF81, 0xFF5B, 0xFF84, 0xFDA3, 0xFF87, 0xFD50, 0xFF8A, 0xFD15, 0xFF8D, 0xFD1A,
    0xFF90, 0xFE18, 0xFF93, 0xEDB9, 0xFF96, 0xEDC7, 0xFF99, 0xFE25, 0xFF9C, 0xFE34,
    0xFF9F, 0xEA87, 0xFFA2, 0xFE21, 0xFFA5, 0xEE13, 0xFFA8, 0xEDDD, 0xFFAB, 0xEDEF,
    0xFFAE, 0xEDFE, 0xFFB1, 0xED0C, 0xFFB4, 0xED09, 0xFFB7, 0xFE07, 0xFFBA, 0xFE00,
    0xFFBD, 0xFDF9, 0xFFC0, 0xF34A, 0xFFC3, 0xF291, 0xFFC6, 0xF20E, 0xFFC9, 0xF250,
    0xFFCC, 0xF333, 0xFFCF, 0xF157, 0xFFD2, 0xF1CA, 0xFFD5, 0xF49E, 0xFFD8, 0xF5DD,
    0xFFDB, 0xF6E4, 0xFFDE, 0xF6DD, 0xFFE1, 0xF6ED, 0xFFE4, 0xF13E, 0xFFE7, 0xF32F,
    0xFFEA, 0xF69B, 0xFFED, 0xE505, 0xFFF0, 0xE50A, 0xFFF3, 0xE500,
)  # fmt: skip

# The KERNAL's RAM vectors $0314-$0333 as RESTOR leaves them, as ``(addr, bytes)``.
KERNAL_VECTORS = (
    (
        mm.RAM_IRQ_VECTOR_LO,
        b"".join(
            v.to_bytes(2, "little")
            for v in (0xEA31, 0xFE66, 0xFE47, 0xF34A, 0xF291, 0xF20E, 0xF250, 0xF333)
            + (0xF157, 0xF1CA, 0xF6ED, 0xF13E, 0xF32F, 0xFE66, 0xF4A5, 0xF5ED)
        ),
    ),
)

_SAVE, _RESTORE = b"\x48\x8a\x48\x98\x48", b"\x68\xa8\x68\xaa\x68"
_STUB = (
    (0xEA31, b"\x4c\x7e\xea"),  # JMP $EA7E
    (0xEA7E, b"\x0c\x0d\xdc"),  # NOP $DC0D: the CIA #1 acknowledge as a dummy read
    (0xEA81, _RESTORE + b"\x40"),
    (0xFCE2, b"\x02"),  # RESET jams
    (0xFE43, b"\x78\x6c\x18\x03"),  # NMI: SEI; JMP ($0318)
    (0xFE47, _SAVE),
    (0xFEBC, _RESTORE + b"\x40"),
    (0xFF48, _SAVE + b"\x6c\x14\x03"),  # IRQ: save A/X/Y; JMP ($0314)
    (0xFFFA, bytes((0x43, 0xFE, 0xE2, 0xFC, 0x48, 0xFF))),
)


class RomFetchError(RuntimeError):
    """A ROM is unavailable, or the fetched bytes failed their checksum."""


def kernal_stub() -> bytes:
    """sidplayfp's ROM-less KERNAL (``SystemROMBanks.h``): NOPs and a few stubs."""
    rom = bytearray(b"\xea" * 0x2000)
    for addr, code in _STUB:
        rom[addr - mm.KERNAL_BASE : addr - mm.KERNAL_BASE + len(code)] = code
    for addr in KERNAL_CALLS:
        rom[addr - mm.KERNAL_BASE] = 0x60
    return bytes(rom)


def rom_less() -> Dict[int, bytes]:
    """sidplayfp's ROM-less ROM set by base: the KERNAL stub, zero BASIC and CHAR."""
    return {
        mm.BASIC_BASE: bytes(0x2000),
        mm.CHARGEN_BASE: bytes(0x1000),
        mm.KERNAL_BASE: kernal_stub(),
    }


def default_rom_cache() -> Path:
    """``$VAGRANT_MOSES_ROMCACHE``, else ``$XDG_CACHE_HOME/vagrant-moses/roms``."""
    env = os.environ.get("VAGRANT_MOSES_ROMCACHE")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(xdg) / "vagrant-moses" / "roms"


def _spec(name: str) -> Rom:
    spec = ROMS.get(name) or ASSETS.get(name)
    if spec is None:
        known = ", ".join(sorted({**ROMS, **ASSETS}))
        raise RomFetchError(f"unknown ROM {name!r} (known: {known})")
    return spec


def _verified(spec: Rom, data: bytes) -> bool:
    return len(data) == spec.size and hashlib.sha256(data).hexdigest() == spec.sha256


def _download(url: str, *, retries: int = 4) -> bytes:
    """``url``'s bytes, retrying transient failures with exponential backoff."""
    req = urllib.request.Request(url, headers={"User-Agent": "vagrant-moses/fetch"})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:  # nosec B310
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise RomFetchError(f"{url}: not found") from exc
            last = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
        if attempt + 1 < retries:
            time.sleep(min(2**attempt, 5))
    raise RomFetchError(f"{url}: unreachable after {retries} attempts ({last})")


def _atomic_write(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    with os.fdopen(fd, "wb") as out:
        out.write(data)
    os.replace(tmp, dest)


def fetch_rom(
    name: str, *, cache_dir=None, mirror: str = DEFAULT_ROM_MIRROR, force=False
) -> Path:
    """Fetch the ``name`` ROM into ``cache_dir``; return its verified path.

    A cached copy whose SHA-256 matches is returned untouched; otherwise the
    asset's pinned URL, else ``mirror``/``filename`` (``$C64_ROM_MIRROR``
    overrides), is downloaded, verified and written atomically.

    Raises:
      RomFetchError: unknown name, unreachable mirror, or a checksum mismatch.
    """
    spec = _spec(name)
    dest = Path(cache_dir or default_rom_cache()) / spec.filename
    if not force and dest.exists() and _verified(spec, dest.read_bytes()):
        return dest
    base = os.environ.get("C64_ROM_MIRROR", mirror).rstrip("/")
    data = _download(spec.url or f"{base}/{spec.filename}")
    if not _verified(spec, data):
        digest = hashlib.sha256(data).hexdigest()
        raise RomFetchError(
            f"{name}: checksum mismatch ({len(data)} bytes, sha256 {digest})"
        )
    _atomic_write(dest, data)
    return dest


def resolve_rom(name: str, *, cache_dir=None, local_env: str = "C64_ROMS"):
    """Path to the ``name`` ROM, or ``None`` when it is genuinely unavailable.

    Checks the local directory ``$C64_ROMS`` (the VICE filename or the plain
    name), then the cache, then the mirror.
    """
    spec = _spec(name)
    local = os.environ.get(local_env)
    if local:
        for cand in (Path(local) / spec.filename, Path(local) / name):
            if cand.exists() and _verified(spec, cand.read_bytes()):
                return cand
    try:
        return fetch_rom(name, cache_dir=cache_dir)
    except RomFetchError:
        return None


def load_roms(names: Iterable[str] = DEFAULT_ROMS, **kwargs) -> Dict[int, bytes]:
    """``{base: bytes}`` of the named ROMs; ``kwargs`` go to :func:`resolve_rom`.

    Raises:
      RomFetchError: a ROM is unavailable (offline and uncached).
    """
    out = {}
    for name in names:
        path = resolve_rom(name, **kwargs)
        if path is None:
            raise RomFetchError(f"{name}: unavailable (offline, not cached)")
        out[_spec(name).base] = path.read_bytes()
    return out


def apply_roms(
    mem: bytearray,
    names: Iterable[str] = DEFAULT_ROMS,
    *,
    bank: Optional[int] = mm.BANK_DEFAULT,
    **kwargs,
) -> bytearray:
    """Overlay the named ROMs on the flat address-space image ``mem``; return it.

    ``bank`` (``None`` to skip) goes to ``$01`` with ``DDR_DEFAULT`` in ``$00``.
    A flat image has no RAM under its ROMs; :class:`~vagrant_moses.bus.Bus` banks.
    """
    if len(mem) != 0x10000:
        raise RomFetchError(f"image is {len(mem)} bytes, expected 65536")
    for base, data in load_roms(names, **kwargs).items():
        mem[base : base + len(data)] = data
    if bank is not None:
        mem[mm.CPU_DDR], mem[mm.CPU_PORT] = mm.DDR_DEFAULT, bank
    return mem


def rom_image(names: Iterable[str] = DEFAULT_ROMS, **kwargs) -> bytearray:
    """A zeroed address-space image with the named ROMs mapped (:func:`apply_roms`)."""
    return apply_roms(bytearray(0x10000), names, **kwargs)


def roms_available(names: Iterable[str] = DEFAULT_ROMS, **kwargs) -> bool:
    """Every named ROM resolves (local, cached or fetchable)."""
    return all(resolve_rom(n, **kwargs) is not None for n in names)


def decode_poweron(text: str) -> List[Tuple[int, bytes]]:
    """libsidplayfp's ``poweron.bin`` source as ``(address, bytes)`` runs.

    Each run (``copyPoweronPattern``) is an offset byte; when its bit 7 is set, a
    count follows (bit 7: one repeated value); then the data.
    """
    stream = [int(v, 16) for v in re.findall(r"0x([0-9a-fA-F]{2})", text)]
    runs, addr, i = [], 0, 0
    while i < len(stream):
        off, count, repeat = stream[i], 0, False
        i += 1
        if off & 0x80:
            off, count, repeat = off & 0x7F, stream[i] & 0x7F, bool(stream[i] & 0x80)
            i += 1
        addr += off
        count += 1
        size = 1 if repeat else count
        data = bytes(stream[i : i + size])
        runs.append((addr, data * count if repeat else data))
        i += size
        addr += count
    return runs


def poweron_writes(**kwargs) -> Optional[List[Tuple[int, bytes]]]:
    """The booted-KERNAL RAM image as ``(address, bytes)`` runs; ``None`` offline."""
    path = resolve_rom("poweron", **kwargs)
    return None if path is None else decode_poweron(path.read_text("ascii"))


__all__ = [
    "ASSETS",
    "BASIC_BOOT",
    "BASIC_RUN",
    "DEFAULT_ROMS",
    "DEFAULT_ROM_MIRROR",
    "KERNAL_BOOT",
    "KERNAL_CALLS",
    "KERNAL_VECTORS",
    "ROMS",
    "Rom",
    "RomFetchError",
    "apply_roms",
    "decode_poweron",
    "default_rom_cache",
    "fetch_rom",
    "kernal_stub",
    "load_roms",
    "poweron_writes",
    "resolve_rom",
    "rom_image",
    "rom_less",
    "roms_available",
]
