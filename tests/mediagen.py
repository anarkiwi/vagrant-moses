"""Synthetic CBM disk and tape images for media tests (no BAM, directory and chains only)."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from vagrant_moses.media import SECTOR_SIZE, Geometry

PRG, LOCKED_PRG, SEQ = 0x82, 0xC2, 0x81


def build_disk(
    geom: Geometry,
    files: Sequence[Tuple[str, bytes, int]],
    error_block: bool = False,
) -> Tuple[bytearray, List[Tuple[int, int]]]:
    """A ``geom`` image holding ``(name, data, type byte)`` files and each file's first T/S.

    Data sectors are allocated from the last track down, so high tracks (40-track,
    d71 side 2, d81) are exercised; the directory fills the directory track in order.
    """
    img = bytearray(geom.blocks * SECTOR_SIZE)
    dir_track, dir_sector = geom.dir_ts
    free = [
        (t, s)
        for t in range(len(geom.sectors), 0, -1)
        if t != dir_track
        for s in range(geom.sectors[t - 1])
    ]

    def put(ts, link, payload):
        off = geom.offset(*ts)
        img[off : off + 2] = bytes(link)
        img[off + 2 : off + 2 + len(payload)] = payload

    firsts = []
    for _, data, _ in files:
        chunks = [data[i : i + 254] for i in range(0, len(data), 254)] or [b""]
        sectors = [free.pop(0) for _ in chunks]
        firsts.append(sectors[0])
        for i, chunk in enumerate(chunks):
            last = i == len(chunks) - 1
            put(sectors[i], (0, len(chunk) + 1) if last else sectors[i + 1], chunk)
    put((dir_track, 0), geom.dir_ts, b"")
    nblocks = max(1, -(-len(files) // 8))
    dirs = [(dir_track, dir_sector + i) for i in range(nblocks)]
    for b, ts in enumerate(dirs):
        link = dirs[b + 1] if b + 1 < len(dirs) else (0, 0xFF)
        body = bytearray(SECTOR_SIZE - 2)
        for slot, idx in enumerate(range(8 * b, min(8 * b + 8, len(files)))):
            name, data, ftype = files[idx]
            entry = body[slot * 32 : slot * 32 + 30]
            entry[0], (entry[1], entry[2]) = ftype, firsts[idx]
            entry[3:19] = name.encode("latin-1").ljust(16, b"\xa0")
            nsect = max(1, -(-len(data) // 254))
            entry[28:30] = nsect.to_bytes(2, "little")
            body[slot * 32 : slot * 32 + 30] = entry
        put(ts, link, body)
    if error_block:
        img += bytes(geom.blocks)
    return img, firsts


def build_t64(entries: Sequence[Tuple[str, bytes, int]], slots: int = 0) -> bytes:
    """A ``.t64`` of ``(name, prg, end override or -1)`` entries; ``slots`` pads the directory."""
    slots = max(slots, len(entries))
    header = bytearray(0x40 + 32 * slots)
    header[0:19] = b"C64 tape image file"
    header[0x22:0x24] = slots.to_bytes(2, "little")
    header[0x24:0x26] = len(entries).to_bytes(2, "little")
    body, off = bytearray(), len(header)
    for i, (name, prg, end) in enumerate(entries):
        start = prg[0] | prg[1] << 8
        end = (start + len(prg) - 2) & 0xFFFF if end < 0 else end
        e = 0x40 + 32 * i
        header[e : e + 2] = b"\x01\x82"
        header[e + 2 : e + 6] = prg[:2] + end.to_bytes(2, "little")
        header[e + 8 : e + 12] = (off + len(body)).to_bytes(4, "little")
        header[e + 16 : e + 32] = name.encode("latin-1").ljust(16, b" ")
        body += prg[2:]
    return bytes(header + body)
