"""Release-media extraction: disk geometries, chains, tapes, nesting, and the CLI."""

from __future__ import annotations

import dataclasses
import gzip
import io
import os
import random
import shutil
import subprocess
import zipfile

import pytest

from mediagen import LOCKED_PRG, PRG, SEQ, build_disk, build_t64
from vagrant_moses import cli
from vagrant_moses.media import (
    GEOMETRIES,
    GEOMETRY_BY_SIZE,
    SECTOR_SIZE,
    MediaError,
    MediaFile,
    iter_prgs,
    read_disk,
    read_p00,
    read_t64,
)

GEOM = {g.name: g for g in GEOMETRIES}
C1541 = shutil.which("c1541")


def prg(load: int, size: int, seed: int) -> bytes:
    rng = random.Random(seed)
    return load.to_bytes(2, "little") + bytes(rng.randrange(256) for _ in range(size))


def zipped(members) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("sub/", b"")
        for name, data in members:
            zf.writestr(name, data)
    return buf.getvalue()


SAMPLE = [
    ("TUNE", prg(0x1000, 3000, 1), PRG),
    ("EDITOR V3.04", prg(0x0801, 254 * 3 - 2, 2), LOCKED_PRG),
    ("ONE", b"\x01", PRG),
    ("NOTES", b"seq data", SEQ),
] + [(f"F{i}", prg(0xC000 + i, i * 10, 10 + i), PRG) for i in range(7)]


@pytest.mark.parametrize(
    "size,name",
    [
        (174848, "d64"),
        (175531, "d64"),
        (196608, "d64-40"),
        (197376, "d64-40"),
        (205312, "d64-42"),
        (206114, "d64-42"),
        (349696, "d71"),
        (351062, "d71"),
        (819200, "d81"),
        (822400, "d81"),
    ],
)
def test_geometry_by_size(size, name):
    assert GEOMETRY_BY_SIZE[size].name == name


def test_offsets():
    d71, d81 = GEOM["d71"], GEOM["d81"]
    assert d71.offset(36, 0) == 683 * SECTOR_SIZE
    assert d81.offset(40, 3) == (39 * 40 + 3) * SECTOR_SIZE
    for bad in ((0, 0), (36, 0), (1, 21), (35, 17)):
        with pytest.raises(MediaError):
            GEOM["d64"].offset(*bad)


@pytest.mark.parametrize("geom", GEOMETRIES, ids=lambda g: g.name)
@pytest.mark.parametrize("error_block", [False, True])
def test_read_disk_roundtrip(geom, error_block):
    img, firsts = build_disk(geom, SAMPLE, error_block)
    assert firsts[0][0] == len(geom.sectors)
    files = read_disk(bytes(img))
    assert [(f.name, f.data) for f in files] == [(n, d) for n, d, _ in SAMPLE]
    assert [f.locked for f in files][:2] == [False, True]
    assert files[3].type_name == "SEQ"
    prgs = dict(iter_prgs("x.d64", bytes(img)))
    assert "x.d64/NOTES" not in prgs and prgs["x.d64/EDITOR V3.04"] == SAMPLE[1][1]


def test_open_and_deleted_entries_skipped():
    img, _ = build_disk(GEOM["d64"], [("OPEN", b"ab", 0x02), ("DEL", b"cd", 0x80)])
    assert not read_disk(bytes(img))


def test_short_terminal_sector_reads_whole():
    img, firsts = build_disk(GEOM["d64"], [("Z", b"", PRG)])
    assert read_disk(bytes(img))[0].data == bytes(254)
    off = GEOM["d64"].offset(*firsts[0])
    img[off + 1] = 0
    assert read_disk(bytes(img))[0].data == bytes(254)


def test_broken_chains_truncate():
    geom = GEOM["d64"]
    a, b = prg(1, 600, 3), prg(2, 10, 4)
    img, firsts = build_disk(geom, [("A", a, PRG), ("B", b, PRG), ("C", b, PRG)])
    second = geom.offset(*firsts[0])
    third = geom.offset(*img[second : second + 2])
    img[third : third + 2] = img[second : second + 2]
    files = read_disk(bytes(img))
    assert files[0].data == a[:508] and "loops at" in files[0].error
    img[third : third + 2] = b"\x24\x00"
    entry = geom.offset(18, 1) + 2 + 64
    img[entry + 1 : entry + 3] = b"\x99\x00"
    errors = []
    got = dict(iter_prgs("r.d64", bytes(img), lambda *e: errors.append(e)))
    assert got == {"r.d64/A": a[:508], "r.d64/B": b}
    assert [e[0] for e in errors] == ["r.d64/A", "r.d64/C"]
    assert "track out of range: 36" in str(errors[0][1])
    assert list(iter_prgs("r.d64", bytes(img))) == list(got.items())
    img[entry - 64 - 2 : entry - 64] = b"\x12\x01"
    assert len(read_disk(bytes(img))) == 3
    with pytest.raises(MediaError, match="no directory"):
        read_disk(bytes(img), dataclasses.replace(geom, dir_ts=(40, 0)))


def test_not_a_disk():
    with pytest.raises(MediaError, match="not a disk image"):
        read_disk(bytes(1000))
    errors = []
    assert not list(iter_prgs("short.d64", bytes(1000), lambda *a: errors.append(a)))
    assert errors[0][0] == "short.d64"
    with pytest.raises(MediaError):
        list(iter_prgs("short.D81", bytes(1000)))


def test_t64():
    a, b, c = prg(0x0801, 500, 5), prg(0x1000, 40, 6), prg(0x2000, 30, 7)
    tape = bytearray(build_t64([("A", a, -1), ("B", b, 0), ("C", c, 0xC3C6)], slots=4))
    tape[0x40 + 3 * 32] = 0
    files = read_t64(bytes(tape))
    assert [(f.name, f.data) for f in files] == [("A", a), ("B", b), ("C", c)]
    assert dict(iter_prgs("t.T64", bytes(tape)))["t.T64/B"] == b
    with pytest.raises(MediaError):
        read_t64(b"NOPE" + bytes(100))


def test_p00():
    payload = prg(0x0801, 20, 8)
    p00 = b"C64File\0" + b"HELLO".ljust(16, b"\0") + b"\0\0" + payload
    assert read_p00(p00) == MediaFile("HELLO", payload)
    assert list(iter_prgs("hello.p00", p00)) == [("hello.p00/HELLO", payload)]
    with pytest.raises(MediaError):
        read_p00(b"C64File")


def test_nested():
    img, _ = build_disk(GEOM["d64-40"], SAMPLE[:2], error_block=True)
    tape = build_t64([("T", prg(0x0801, 9, 9), 0)])
    inner = zipped([("disk.d64.gz", gzip.compress(bytes(img))), ("x.t64", tape)])
    outer = zipped([("release/inner.zip", inner), ("loose.prg", b"\x01\x08xy")])
    got = dict(iter_prgs("csdb.zip", outer))
    assert got == {
        "csdb.zip/release/inner.zip/disk.d64/TUNE": SAMPLE[0][1],
        "csdb.zip/release/inner.zip/disk.d64/EDITOR V3.04": SAMPLE[1][1],
        "csdb.zip/release/inner.zip/x.t64/T": prg(0x0801, 9, 9),
        "csdb.zip/loose.prg": b"\x01\x08xy",
    }
    whole = dict(iter_prgs("csdb.zip", outer, disks=False))
    assert whole["csdb.zip/release/inner.zip/disk.d64"] == img
    assert list(iter_prgs("a", gzip.compress(b"z"))) == [("a", b"z")]


def test_bad_containers():
    errors = []
    bad = zipped([("ok.prg", b"ok"), ("bad.gz", b"\x1f\x8bjunk")])
    got = list(iter_prgs("r.zip", bad, lambda *a: errors.append(a)))
    assert got == [("r.zip/ok.prg", b"ok")] and errors[0][0] == "r.zip/bad.gz"
    with pytest.raises(zipfile.BadZipFile):
        list(iter_prgs("r.zip", b"PK\x03\x04junk"))


def run_c1541(*args, cwd):
    env = dict(os.environ, HOME=str(cwd))
    subprocess.run([C1541, *args], cwd=cwd, env=env, capture_output=True, check=True)


C1541_FILES = {f"f{i}": prg(0x0800 + i, 1500 * i + 3, 20 + i) for i in range(1, 9)}
needs_c1541 = pytest.mark.skipif(C1541 is None, reason="VICE c1541 not installed")


@needs_c1541
@pytest.mark.parametrize("kind", ["d64", "d71", "d81"])
def test_reads_c1541_images(tmp_path, kind):
    """Images formatted and written by VICE read back natively."""
    writes = []
    for name, data in C1541_FILES.items():
        (tmp_path / name).write_bytes(data)
        writes += ["-write", name, name]
    run_c1541("-format", "native,id", kind, "img", *writes, cwd=tmp_path)
    img = (tmp_path / "img").read_bytes()
    assert {f.name.lower(): f.data for f in read_disk(img)} == C1541_FILES


@needs_c1541
@pytest.mark.parametrize("geom", GEOMETRIES, ids=lambda g: g.name)
@pytest.mark.parametrize("error_block", [False, True])
def test_c1541_reads_ours(tmp_path, geom, error_block):
    """VICE extracts the same files from the synthetic images as the native reader."""
    files = [(n.upper(), d, PRG) for n, d in C1541_FILES.items()]
    img, _ = build_disk(geom, files, error_block)
    ext = geom.name[:3]
    (tmp_path / f"img.{ext}").write_bytes(img)
    out = tmp_path / "out"
    out.mkdir()
    run_c1541("-attach", f"../img.{ext}", "-extract", cwd=out)
    vice = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    assert vice == {f.name.lower(): f.data for f in read_disk(bytes(img))}
    assert vice == C1541_FILES


def test_cli(tmp_path, capsys):
    img, _ = build_disk(
        GEOM["d81"], [("A", b"\x00\x10abc", PRG), ("A", b"\x00\x20", PRG)]
    )
    src = tmp_path / "dl" / "250000"
    src.mkdir(parents=True)
    (src / "rel.zip").write_bytes(zipped([("x/../a b.d81", bytes(img))]))
    (tmp_path / "one").write_bytes(b"\x01")
    assert (
        cli.main(
            [
                "media",
                "extract",
                str(tmp_path / "dl"),
                str(tmp_path / "one"),
                "-o",
                str(tmp_path / "prg"),
            ]
        )
        == 0
    )
    lines = capsys.readouterr().out.splitlines()
    assert [ln.split()[1:] for ln in lines] == [
        ["5", "$1000", "250000/rel.zip/x/_/a_b.d81/A"],
        ["2", "$2000", "250000/rel.zip/x/_/a_b.d81/A~2"],
        ["1", "-", "one"],
    ]
    assert (tmp_path / "prg/250000/rel.zip/x/_/a_b.d81/A~2").read_bytes() == b"\x00\x20"
    (src / "bad.d64").write_bytes(b"short")
    assert cli.main(["media", "extract", str(src), "-o", str(tmp_path / "o2")]) == 1
    assert "ERROR bad.d64" in capsys.readouterr().err
