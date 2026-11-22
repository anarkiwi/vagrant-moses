"""C64 machine facts a host needs to drive a playroutine's interrupts faithfully.

An interpreter already models the 6510's volatile I/O (VIC, CIA, SID); this
supplies the surroundings -- power-on RAM, the RAM a PSID/RSID boots onto,
installed-vector discovery, and the ROM-free IRQ dispatch path that makes a
handler entry a subroutine call.
"""

from __future__ import annotations

from . import memmap as mm
from .roms import RomFetchError, poweron_writes

IRQ_VEC = (mm.RAM_IRQ_VECTOR_LO, mm.RAM_IRQ_VECTOR_HI)  # CINV (KERNAL A/X/Y save)
NMI_VEC = (mm.RAM_NMI_VECTOR_LO, mm.RAM_NMI_VECTOR_HI)  # NMINV
HW_IRQ_VEC = (mm.CPU_IRQ_VECTOR, mm.CPU_IRQ_VECTOR + 1)  # hardware IRQ/BRK vector
HW_NMI_VEC = (mm.CPU_NMI_VECTOR, mm.CPU_NMI_VECTOR + 1)  # hardware NMI vector

# KERNAL IRQ epilogue: PLA;TAY;PLA;TAX;PLA;RTI ($EA81, NMI $FEBC)
_EPILOGUE = bytes((0x68, 0xA8, 0x68, 0xAA, 0x68, 0x40))
_STUBS = (
    (0xEA31, b"\xea" * (0xEA81 - 0xEA31)),  # handler body: any exit entry NOPs to $EA81
    (0xEA81, _EPILOGUE),
    (0xFEBC, _EPILOGUE),
)
IRQ_DISPATCH = 0xFF33  # entry: set I, push the 6510 IRQ frame, enter the vector
IRQ_RETURN = 0xFF41  # the interrupted program: its RTS balances the caller's frame
KERNAL_IRQ = 0xFF48  # KERNAL hardware-IRQ prologue: save A/X/Y, JMP (CINV)
_P_IRQ = 0x20  # interrupted-program status pushed for the RTI (I clear)


def poweron_ram():
    """C64 power-on RAM fill (libsidplayfp ``SystemRAMBank::reset``), as bytes.

    Each quarter of RAM alternates 0x00/0xFF, with stripes of four opposite bytes
    every eight from offset 2; never-written RAM reads these on hardware.
    """
    ram = bytearray(0x10000)
    byte = 0x00
    for j in range(0, 0x10000, 0x4000):
        ram[j : j + 0x4000] = bytes((byte,)) * 0x4000
        byte ^= 0xFF
        stripe = bytes((byte,)) * 4
        for i in range(0x02, 0x4000, 0x08):
            ram[j + i : j + i + 4] = stripe
    return bytes(ram)


def read_vector(mem, lo):
    """Little-endian 16-bit vector at ``lo`` (``lo``/``lo+1``)."""
    return mem[lo] | (mem[(lo + 1) & 0xFFFF] << 8)


def installed_vector(mem, written, img):
    """Installed interrupt vector ``(vector addr, uses_kernal_cinv)``, or ``None``.

    Prefer a written vector (CINV/NMINV/hardware ``$FFFE``), else a CINV lifted
    from the load image ``img``; ``written`` is the observed write-address set.
    """
    for pair, kernal in ((IRQ_VEC, True), (HW_IRQ_VEC, False), (NMI_VEC, False)):
        if pair[0] in written or pair[1] in written:
            return (pair[0], kernal)
    lo, hi = img
    if lo <= IRQ_VEC[0] < hi and lo <= IRQ_VEC[1] < hi and read_vector(mem, IRQ_VEC[0]):
        return (IRQ_VEC[0], True)
    return None


def installed_handler(mem, written, img):
    """Installed interrupt handler ``(addr, uses_kernal_cinv)``, or ``None``."""
    found = installed_vector(mem, written, img)
    return None if found is None else (read_vector(mem, found[0]), found[1])


def dispatch_vector(bits, nmi):
    """``(vector, saves_registers)`` an IRQ or NMI is dispatched through.

    With the KERNAL banked in (``bits & HIRAM``) its IRQ entry saves A/X/Y and
    jumps through CINV, its NMI entry through NMINV; else the CPU vector is it.
    """
    if bits & mm.HIRAM:
        return (NMI_VEC[0], False) if nmi else (IRQ_VEC[0], True)
    return (HW_NMI_VEC[0] if nmi else HW_IRQ_VEC[0]), False


def install_kernal_irq_stubs(vm):
    """Write the KERNAL IRQ epilogue ($EA31 body NOPs to $EA81: pull Y/X/A, RTI).

    ``vm`` needs only a writable C64 main-memory ``vm.mem``.
    """
    for addr, code in _STUBS:
        vm.mem[addr : addr + len(code)] = code


def irq_stubs(vec, kernal):
    """``[(addr, code)]`` of the ROM-free dispatch path through vector ``vec``.

    ``$FF33`` pushes the 6510 IRQ frame (return pc ``$FF41``, then P) and enters
    the vector via the ``$FF48`` prologue when ``kernal``; ``$FF41`` is the RTS.
    """
    tail = (
        (0x4C, KERNAL_IRQ & 0xFF, KERNAL_IRQ >> 8)
        if kernal
        else (0x6C, vec & 0xFF, vec >> 8)
    )
    frame = (
        0x78,
        0xA9,
        IRQ_RETURN >> 8,
        0x48,
        0xA9,
        IRQ_RETURN & 0xFF,
        0x48,
        0xA9,
        _P_IRQ,
        0x48,
    )
    stubs = [(IRQ_DISPATCH, bytes(frame + tail)), (IRQ_RETURN, b"\x60")]
    if kernal:
        prologue = (0x48, 0x8A, 0x48, 0x98, 0x48, 0x6C, vec & 0xFF, vec >> 8)
        stubs.append((KERNAL_IRQ, bytes(prologue)))
        stubs.extend(_STUBS)
    return stubs


def install_irq_entry(vm, vec, kernal, img=(0, 0)):
    """Install :func:`irq_stubs` into ``vm``; return the per-frame entry pc.

    ``vm`` needs a writable C64 main-memory ``vm.mem``; a ``ValueError`` when
    image ``img`` claims a stub address, the tune's bytes living there.
    """
    lo, hi = img
    for addr, code in irq_stubs(vec, kernal):
        if lo < addr + len(code) and addr < hi:
            raise ValueError(
                "load image $%04X-$%04X covers the IRQ dispatch stub at $%04X"
                % (lo, hi, addr)
            )
        vm.mem[addr : addr + len(code)] = code
    return IRQ_DISPATCH


#: Tune kinds :func:`psid_boot_image` boots: a PSID, an RSID, an RSID with the C64 BASIC flag.
PSID_KINDS = ("PSID", "RSID", "BASIC")
#: ``psiddrv.a65``'s ``$0000-$03FE`` clear (``0x3FF`` bytes).
PSID_CLEARED = 0x3FF
#: The KERNAL's PAL/NTSC flag (``$02A6``: 1 PAL, 0 NTSC).
PAL_FLAG = 0x02A6
#: The KERNAL IRQ's exit (acknowledge CIA 1, restore A/X/Y, RTI): ``psiddrv.a65``'s
#: ``irqret``, where sidplayfp points CINV for an RSID without the BASIC flag or a PSID.
PSID_IRQ_EXIT = 0xEA7E
#: The ROM-less KERNAL's reset entry, a JAM (``KernalRomBank::set``): ``psiddrv.a65``'s
#: ``exception``, where sidplayfp points a PSID's BRK (CBINV) and NMI (NMINV) vectors.
PSID_JAM = 0xFCE2
#: The BASIC ``LOAD`` pointers ``SidTuneBase::placeSidTuneInC64mem`` sets: the load
#: end in ``$2D``/``$2F``/``$31`` (variables / arrays / strings start) and ``$AE``,
#: the load start in ``$AC``.
BASIC_END_POINTERS, BASIC_START_POINTER = (0x2D, 0x2F, 0x31, 0xAE), 0xAC


def psid_iomap(addr):
    """The ``$01`` value libsidplayfp's driver banks in for PSID code at ``addr``.

    ``psiddrv::iomap``: ``$37`` (BASIC, KERNAL, I/O) below ``$A000``, ``$36``
    (KERNAL, I/O) in ``$A000-$CFFF``, ``$34`` (RAM only) in ``$D000-$DFFF`` and
    ``$35`` (I/O only) from ``$E000``.
    """
    if addr < mm.BASIC_BASE:
        return 0x37
    if addr < mm.CHARGEN_BASE:
        return 0x36
    return 0x35 if addr >= mm.KERNAL_BASE else 0x34


def _kind(kind):
    if kind not in PSID_KINDS:
        raise ValueError(f"unknown tune kind {kind!r} (known: {', '.join(PSID_KINDS)})")
    return kind != "PSID"


def psid_driver_tail(kind, init, sentinel):
    """``(code, returns)``: the driver code ending at ``sentinel`` and where init returns.

    A host that traps at ``sentinel`` runs this in place of ``psiddrv.a65``'s own
    code, cycle for cycle, and places it at ``[sentinel - len(code), sentinel)``.
    An RSID's init returns to ``lda``/``beq idle``; a PSID's (``kind`` ``"PSID"``)
    enters at the code's start by ``sei``/``jmp init`` (``plp``'s I flag,
    ``jmp (initvec)``) and returns to ``lda``/``beq``/``lda``/``beq run``/``cli``;
    each ``lda`` reads a byte of the code.
    """
    s = sentinel
    if _kind(kind):
        return bytes([0xAD, *(s - 1).to_bytes(2, "little"), 0xF0, 0x00]), s - 5
    tail = bytes([0xAD, *(s - 11).to_bytes(2, "little"), 0xF0, 0x00])
    tail += bytes([0xAD, *(s - 2).to_bytes(2, "little"), 0xF0, 0x00, 0x58])
    return bytes([0x78, 0x4C, *init.to_bytes(2, "little")]) + tail, s - len(tail)


def psid_boot_image(kind, load, end, init, ntsc=False, image=None, sentinel=None):
    """The 64 KiB RAM a tune's ``init`` starts on under sidplayfp, as a ``bytearray``.

    ``psiddrv.a65``'s cold start and ``SidTuneBase::placeSidTuneInC64mem``:
    ``image`` (default :func:`poweron_ram`) with ``$0000-$03FE`` cleared, for an
    RSID (``kind`` ``"RSID"`` or ``"BASIC"``, the C64 BASIC flag) the booted
    KERNAL's RAM (:func:`~vagrant_moses.roms.poweron_writes`) over it, ``$02A6``
    1 for PAL or 0 for ``ntsc``, CINV at :data:`PSID_IRQ_EXIT` (an RSID without
    the BASIC flag) or CINV, CBINV and NMINV at :data:`PSID_IRQ_EXIT`,
    :data:`PSID_JAM`, :data:`PSID_JAM` (a PSID), the BASIC ``LOAD`` pointers
    (:data:`BASIC_END_POINTERS` at ``end``, :data:`BASIC_START_POINTER` at
    ``load``, each mod ``$10000``), then the tune's bytes ``image[load:end]``
    over all of it. With a ``sentinel``, :func:`psid_driver_tail` of ``kind`` and
    ``init`` (a PSID's init address; unused for an RSID) ends at it, last.

    Raises:
      ValueError: ``kind`` is not one of :data:`PSID_KINDS`.
      vagrant_moses.roms.RomFetchError: an RSID's booted-KERNAL RAM is unavailable.
    """
    rsid = _kind(kind)
    src = poweron_ram() if image is None else image
    img = bytearray(src)
    img[:PSID_CLEARED] = bytes(PSID_CLEARED)
    if rsid:
        runs = poweron_writes()
        if runs is None:
            raise RomFetchError("poweron: unavailable (offline, not cached)")
        for addr, data in runs:
            img[addr : addr + len(data)] = data
    img[PAL_FLAG] = 0 if ntsc else 1
    vectors = {"BASIC": (), "RSID": (PSID_IRQ_EXIT,)}.get(
        kind, (PSID_IRQ_EXIT, PSID_JAM, PSID_JAM)
    )
    vec = b"".join(v.to_bytes(2, "little") for v in vectors)
    img[mm.RAM_IRQ_VECTOR_LO : mm.RAM_IRQ_VECTOR_LO + len(vec)] = vec
    pointers = [(a, end) for a in BASIC_END_POINTERS] + [(BASIC_START_POINTER, load)]
    for addr, v in pointers:
        img[addr : addr + 2] = (v & 0xFFFF).to_bytes(2, "little")
    img[load:end] = src[load:end]
    if sentinel is not None:
        code = psid_driver_tail(kind, init, sentinel)[0]
        img[sentinel - len(code) : sentinel] = code
    return img
