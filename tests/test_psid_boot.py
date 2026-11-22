"""The PSID/RSID boot RAM and driver banking against deity-informant's machine start.

``deity_informant.c64.boot`` is the cycle-exact start these helpers were taken from;
each image here must equal the RAM it boots, byte for byte.
"""

from __future__ import annotations

import random

import pytest

from vagrant_moses import c64, roms

dic64 = pytest.importorskip("deity_informant.c64")
SENTINEL = pytest.importorskip("deity_informant.smc.machine").SENTINEL

#: A booted-KERNAL stand-in: runs across zero page, the stack, the vectors and a tune.
RUNS = [(0x0002, b"\x11\x22\x33"), (0x0100, bytes(range(64))), (0x0300, b"\xa5" * 0x60)]
RUNS += [(0x1010, b"\x5a\x5b"), (0xA000, b"\xee" * 16)]


def _image(seed):
    return bytes(random.Random(seed).randbytes(0x10000))


@pytest.fixture(name="booted")
def _booted(monkeypatch):
    monkeypatch.setattr(c64, "poweron_writes", lambda: RUNS)
    monkeypatch.setattr(dic64, "poweron_writes", lambda: RUNS)


@pytest.mark.parametrize(
    "addr,port",
    [(0x0000, 0x37), (0x9FFF, 0x37), (0xA000, 0x36), (0xCFFF, 0x36)]
    + [(0xD000, 0x34), (0xDFFF, 0x34), (0xE000, 0x35), (0xFFFF, 0x35)],
)
def test_iomap_bands(addr, port):
    assert c64.psid_iomap(addr) == port


def test_iomap_matches_deity_informant_everywhere():
    assert all(c64.psid_iomap(a) == dic64.iomap(a) for a in range(0x10000))


def test_iomap_banks_init_and_play_roms():
    init, play = 0x1000, 0xA800  # init under BASIC ROM, play in RAM behind it
    assert c64.psid_iomap(init) & 0x03 == 0x03  # BASIC and KERNAL in
    assert c64.psid_iomap(play) & 0x03 == 0x02  # BASIC out, KERNAL in


CASES = [
    ("PSID", 0x1000, 0x1060, 0x1000, 1, False, True),
    ("PSID", 0x1000, 0x1060, 0x1003, 2, True, False),
    ("PSID", 0xFFF0, 0x10000, 0xFFF0, 3, True, False),
    ("PSID", 0x0200, 0x0400, 0x0300, 1, False, True),
    ("PSID", 0xCF00, 0xD000, 0xCF10, 4, False, True),
    ("PSID", 0x0002, 0x00C0, 0x0010, 1, True, True),
    ("RSID", 0x0801, 0x0900, 0x080D, 1, False, True),
    ("RSID", 0x1000, 0x2000, 0x1000, 5, True, True),
    ("RSID", 0x0300, 0x0320, 0x0300, 1, False, True),
    ("BASIC", 0x0801, 0x0880, 0, 1, False, True),
    ("BASIC", 0x0801, 0x1000, 0, 7, True, True),
]


@pytest.mark.parametrize("seed,case", list(enumerate(CASES)))
@pytest.mark.usefixtures("booted")
def test_boot_image_is_deity_informants_ram(seed, case):
    kind, load, end, init, song, ntsc, cia = case
    image = _image(seed)
    boot = dic64.boot(image, load, end, init, song, ntsc, kind, cia, roms.rom_less())
    got = c64.psid_boot_image(kind, load, end, init, ntsc, image, SENTINEL)
    assert isinstance(got, bytearray)
    assert got == boot.image
    code, ret = c64.psid_driver_tail(kind, init, SENTINEL)
    assert ret == boot.returns
    if kind == "PSID":
        assert boot.init == (SENTINEL - len(code), song - 1)
        assert boot.port == c64.psid_iomap(init)


@pytest.mark.usefixtures("booted")
def test_boot_image_without_sentinel_leaves_the_driver_out():
    image = _image(99)
    boot = dic64.boot(image, 0x1000, 0x1100, 0x1000, 1, kind="PSID")
    got = c64.psid_boot_image("PSID", 0x1000, 0x1100, 0x1000, image=image)
    tail = len(c64.psid_driver_tail("PSID", 0x1000, SENTINEL)[0])
    assert got[SENTINEL - tail : SENTINEL] == image[SENTINEL - tail : SENTINEL]
    got[SENTINEL - tail : SENTINEL] = boot.image[SENTINEL - tail : SENTINEL]
    assert got == boot.image


def test_boot_image_defaults_to_poweron_ram():
    got = c64.psid_boot_image("PSID", 0x1000, 0x1000, 0x1000, ntsc=True)
    ram = c64.poweron_ram()
    assert got[: c64.PSID_CLEARED] != ram[: c64.PSID_CLEARED]
    assert got[0x0400:] == ram[0x0400:]
    assert got[c64.PAL_FLAG] == 0
    assert c64.read_vector(got, c64.NMI_VEC[0]) == c64.PSID_JAM
    assert [c64.read_vector(got, a) for a in c64.BASIC_END_POINTERS] == [0x1000] * 4


def test_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="MUS"):
        c64.psid_boot_image("MUS", 0x1000, 0x1010, 0x1000)
    with pytest.raises(ValueError):
        c64.psid_driver_tail("psid", 0x1000, SENTINEL)


def test_rsid_offline_raises(monkeypatch):
    monkeypatch.setattr(c64, "poweron_writes", lambda: None)
    with pytest.raises(roms.RomFetchError):
        c64.psid_boot_image("RSID", 0x1000, 0x1010, 0x1000)
    assert c64.psid_boot_image("PSID", 0x1000, 0x1010, 0x1000)[c64.PAL_FLAG] == 1


@pytest.mark.roms
@pytest.mark.parametrize("kind", ["RSID", "BASIC"])
def test_rsid_boot_on_the_real_kernal_ram(kind):
    image = _image(7)
    boot = dic64.boot(
        image, 0x0801, 0x0A00, 0x0801, 1, True, kind, True, roms.rom_less()
    )
    assert c64.psid_boot_image(kind, 0x0801, 0x0A00, 0x0801, True, image, SENTINEL) == (
        boot.image
    )
