# Release media

`vagrant_moses.media` turns csdb-style downloads into the C64 files inside them,
natively and without VICE.

## Walking nested media

`iter_prgs(name, data, on_error=None, disks=True) -> Iterator[(path, bytes)]`
recurses through any nesting of:

| container | detected by | yields |
|---|---|---|
| gzip | `1f 8b` magic | the decompressed member (`.gz` dropped from the name) |
| zip | `PK\3\4` magic | every non-directory member |
| disk image | size (see below), or a `.d64`/`.d71`/`.d81` extension | every closed PRG |
| `.t64` tape | `C64` signature | every normal tape file, load address prepended |
| PC64 `.p00` | `C64File\0` signature | the wrapped file |

Anything else is a leaf and is yielded as-is. Paths join container names with
`/`, e.g. `release.zip/disk.d64.gz` → `release.zip/disk.d64/TUNE`.
`disks=False` yields disk images whole instead of opening them.

Errors: an unreadable member (bad zip/gzip, truncated image, missing
directory) goes to `on_error(path, exc)` and iteration continues; without a
handler it raises. A disk file cut short by a broken sector link is still
yielded and is also passed to `on_error`.

## Disk images

`read_disk(data, geom=None) -> [MediaFile]` lists every closed file (DEL, SEQ,
PRG, USR, REL) in directory order, with its `name`, `data`, `ftype`, `locked`
bit and `error`. The geometry comes from the image size; an error-info block
adds one byte per sector.

| geometry | tracks | sectors per track | directory | sizes |
|---|---|---|---|---|
| `d64` | 35 | 21/19/18/17 zones | 18/1 | 174848, 175531 |
| `d64-40` | 40 | zones, 36–40 at 17 | 18/1 | 196608, 197376 |
| `d64-42` | 42 | zones, 36–42 at 17 | 18/1 | 205312, 206114 |
| `d71` | 70 | zones repeated for side 2 (36–70) | 18/1 | 349696, 351062 |
| `d81` | 80 | 40 | 40/3 | 819200, 822400 |

The chain rules follow the drive (VICE `vdrive`):

- A terminal sector (link track 0) holds data up to the index in its second link
  byte. An index below 2 means the whole sector.
- A link to an impossible track or sector, or back to a sector already read, ends
  the file after the sectors read so far. `MediaFile.error` names the cause.
- The BAM is not consulted.

## Tapes

`read_t64(data)` reads the directory's start address and data offset for each
entry. End addresses are often 0 or a fixed wrong value in the wild, so the
length is clamped to the gap before the next file's offset (or the end of the
image).

## Command line

```bash
pip install "vagrant-moses[cli]"
vagrant-moses media extract dl/ -o prg/ > prg_index.txt
```

Each source file, or every file under a source directory, is extracted under
`-o`. Path components are reduced to `[A-Za-z0-9._+-]`, and repeated paths get a
`~N` suffix. Each written file prints `sha256 size $load path`. Errors go to
stderr, and the exit status is 1 if there were any.

## Checking against VICE

`scripts/c1541_crosscheck.py SRC...` extracts every disk image found under SRC
with both the native reader and `c1541 -extract`, and prints each file that only
one of them found. The remaining differences come from c1541 itself:

- it cannot write host files whose names contain `/` or `:`, or duplicate names;
- it aborts with a false "cyclic T/S chain" error on long or cross-linked chains.
