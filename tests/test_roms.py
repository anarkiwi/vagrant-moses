"""ROM fetch/cache/mapping (network monkeypatched) and sidplayfp's ROM-less set."""

from __future__ import annotations

import hashlib
import urllib.error

import pytest

from vagrant_moses import roms
from vagrant_moses.roms import (
    ROMS,
    RomFetchError,
    apply_roms,
    decode_poweron,
    default_rom_cache,
    fetch_rom,
    kernal_stub,
    resolve_rom,
    rom_image,
    rom_less,
    roms_available,
)

FAKE = {
    name: bytes((i + 1,)) * spec.size for i, (name, spec) in enumerate(ROMS.items())
}


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    for var in ("C64_ROMS", "C64_ROM_MIRROR", "VAGRANT_MOSES_ROMCACHE"):
        monkeypatch.delenv(var, raising=False)


def _offline(url, **_kw):
    raise RomFetchError(f"{url}: offline")


@pytest.fixture
def served(monkeypatch):
    """Serve the fake ROM bytes under their real checksums; record the URLs."""
    for name, data in FAKE.items():
        digest = hashlib.sha256(data).hexdigest()
        monkeypatch.setitem(ROMS, name, ROMS[name]._replace(sha256=digest))
    calls = []

    def fake(url, **_kw):
        calls.append(url)
        return next(FAKE[n] for n, s in ROMS.items() if url.endswith(s.filename))

    monkeypatch.setattr(roms, "_download", fake)
    return calls


def test_kernal_stub_is_sidplayfps_rom_less_kernal():
    rom = kernal_stub()

    def word(addr):
        return rom[addr - 0xE000] | rom[addr - 0xDFFF] << 8

    assert (word(0xFFFA), word(0xFFFC), word(0xFFFE)) == (0xFE43, 0xFCE2, 0xFF48)
    assert rom[0xEA31 - 0xE000 : 0xEA34 - 0xE000] == b"\x4c\x7e\xea"
    assert rom[0xEA7E - 0xE000 : 0xEA81 - 0xE000] == b"\x0c\x0d\xdc"
    assert rom[0xFF48 - 0xE000 : 0xFF50 - 0xE000] == bytes.fromhex("488a4898486c1403")
    assert rom[0xFFD2 - 0xE000] == rom[0xF1CA - 0xE000] == 0x60
    assert rom[0xE100 - 0xE000] == 0xEA
    assert rom_less()[0xE000] == rom and not any(rom_less()[0xA000])


def test_kernal_vectors_are_restors_table():
    ((addr, data),) = roms.KERNAL_VECTORS
    assert addr == 0x0314 and len(data) == 0x20
    assert data[:6] == bytes.fromhex("31ea66fe47fe")


def test_rom_map_is_the_c64_layout():
    assert [(r.base, r.size) for r in ROMS.values()] == [
        (0xE000, 0x2000),
        (0xA000, 0x2000),
        (0xD000, 0x1000),
    ]
    assert roms.DEFAULT_ROMS == ("kernal", "basic")


def test_default_rom_cache_env_and_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert default_rom_cache() == tmp_path / "vagrant-moses" / "roms"
    monkeypatch.setenv("VAGRANT_MOSES_ROMCACHE", str(tmp_path / "r"))
    assert default_rom_cache() == tmp_path / "r"


def test_unknown_rom_raises():
    with pytest.raises(RomFetchError, match="unknown ROM"):
        fetch_rom("charen")


def test_fetch_rom_downloads_then_caches(served, tmp_path):
    dest = fetch_rom("kernal", cache_dir=tmp_path)
    assert dest.read_bytes() == FAKE["kernal"]
    assert fetch_rom("kernal", cache_dir=tmp_path) == dest
    assert len(served) == 1
    fetch_rom("kernal", cache_dir=tmp_path, force=True)
    assert len(served) == 2


def test_fetch_rom_refetches_a_corrupt_cache_entry(served, tmp_path):
    fetch_rom("basic", cache_dir=tmp_path).write_bytes(bytes(ROMS["basic"].size))
    assert fetch_rom("basic", cache_dir=tmp_path).read_bytes() == FAKE["basic"]
    assert len(served) == 2


def test_fetch_rom_checksum_mismatch_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(roms, "_download", lambda url, **kw: b"\xea" * 0x2000)
    with pytest.raises(RomFetchError, match="checksum mismatch"):
        fetch_rom("kernal", cache_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_fetch_rom_honours_the_mirror_env(served, monkeypatch, tmp_path):
    monkeypatch.setenv("C64_ROM_MIRROR", "https://mirror.example/roms/")
    fetch_rom("kernal", cache_dir=tmp_path)
    assert served == ["https://mirror.example/roms/kernal-901227-03.bin"]


def test_resolve_rom_prefers_a_verified_local_dir(served, monkeypatch, tmp_path):
    local = tmp_path / "vice"
    local.mkdir()
    (local / ROMS["kernal"].filename).write_bytes(FAKE["kernal"])
    (local / "basic").write_bytes(FAKE["basic"])
    (local / "chargen").write_bytes(bytes(0x1000))
    monkeypatch.setenv("C64_ROMS", str(local))
    assert resolve_rom("kernal", cache_dir=tmp_path).parent == local
    assert resolve_rom("basic", cache_dir=tmp_path).name == "basic"
    assert resolve_rom("chargen", cache_dir=tmp_path).parent == tmp_path
    assert len(served) == 1


def test_resolve_rom_none_when_offline(monkeypatch, tmp_path):
    monkeypatch.setattr(roms, "_download", _offline)
    assert resolve_rom("kernal", cache_dir=tmp_path) is None
    assert roms_available(cache_dir=tmp_path) is False
    assert roms.poweron_writes(cache_dir=tmp_path) is None
    with pytest.raises(RomFetchError, match="unavailable"):
        rom_image(cache_dir=tmp_path)


@pytest.mark.usefixtures("served")
def test_rom_image_maps_kernal_and_basic(tmp_path):
    mem = rom_image(cache_dir=tmp_path)
    assert bytes(mem[0xE000:]) == FAKE["kernal"]
    assert bytes(mem[0xA000:0xC000]) == FAKE["basic"]
    assert (mem[0], mem[1]) == (0x2F, 0x37) and not any(mem[0xD000:0xE000])
    assert roms_available(cache_dir=tmp_path) is True


@pytest.mark.usefixtures("served")
def test_apply_roms_overlays_an_existing_image(tmp_path):
    mem = bytearray(b"\xaa" * 0x10000)
    assert apply_roms(mem, ("chargen",), cache_dir=tmp_path, bank=None) is mem
    assert bytes(mem[0xD000:0xE000]) == FAKE["chargen"]
    assert mem[1] == mem[0x9FFF] == 0xAA


def test_apply_roms_rejects_a_wrong_sized_image():
    with pytest.raises(RomFetchError, match="expected"):
        apply_roms(bytearray(16))


def test_poweron_decodes_libsidplayfps_rle(monkeypatch, tmp_path):
    text = (
        "/* addr, off, rle, values */\n"
        "/*$0003*/ 0x83, 0x01, 0xaa, 0xbb,\n"
        "/*$0008*/ 0x03,       0x4c,\n"
        "/*$000c*/ 0x83, 0x82, 0x00,\n"
    )
    assert decode_poweron(text) == [(3, b"\xaa\xbb"), (8, b"\x4c"), (12, bytes(3))]
    path = tmp_path / "poweron.bin"
    path.write_text(text)
    monkeypatch.setattr(roms, "resolve_rom", lambda *a, **k: path)
    assert roms.poweron_writes() == decode_poweron(text)


def test_poweron_is_pinned_to_the_oracles_libsidplayfp():
    spec = roms.ASSETS["poweron"]
    assert spec.url.endswith(f"/{spec.filename}") and "9c464110" in spec.url


class _Resp:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.data


def test_download_retries_then_reports(monkeypatch):
    attempts = []

    def flaky(req, **_kw):
        attempts.append(req.full_url)
        if len(attempts) < 2:
            raise urllib.error.URLError("reset")
        return _Resp(b"ok")

    monkeypatch.setattr(roms.time, "sleep", lambda s: None)
    monkeypatch.setattr(roms.urllib.request, "urlopen", flaky)
    assert roms._download("https://x/y") == b"ok" and len(attempts) == 2

    def down(req, **_kw):
        raise urllib.error.HTTPError(req.full_url, 503, "busy", None, None)

    monkeypatch.setattr(roms.urllib.request, "urlopen", down)
    with pytest.raises(RomFetchError, match="unreachable after 3"):
        roms._download("https://x/y", retries=3)

    def missing(req, **_kw):
        raise urllib.error.HTTPError(req.full_url, 404, "gone", None, None)

    monkeypatch.setattr(roms.urllib.request, "urlopen", missing)
    with pytest.raises(RomFetchError, match="not found"):
        roms._download("https://x/y")


@pytest.mark.roms
def test_real_roms_fetch_verify_and_map(tmp_path):
    """Fetch the real ROMs and check they are the C64 ones (network)."""
    mem = rom_image(tuple(ROMS), cache_dir=tmp_path, bank=None)
    assert mem[0xFFFC] | mem[0xFFFD] << 8 == 0xFCE2
    assert mem[0xFFFE] | mem[0xFFFF] << 8 == 0xFF48
    assert mem[0xA000] | mem[0xA001] << 8 == 0xE394
    assert bytes(mem[0xD000:0xD008]) == bytes.fromhex("3c666e6e60623c00")
    ram = bytearray(0x400)
    for addr, data in roms.poweron_writes(cache_dir=tmp_path):
        ram[addr : addr + len(data)] = data
    assert ram[0x314:0x334] == roms.KERNAL_VECTORS[0][1]
