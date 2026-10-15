"""C64 machine facts a host needs to drive a playroutine's interrupts faithfully.

An interpreter already models the 6510's volatile I/O (VIC, CIA, SID); this
supplies the surroundings -- power-on RAM, installed-vector discovery, and the
ROM-free IRQ dispatch path that makes a handler entry a subroutine call.
"""

from __future__ import annotations

from . import memmap as mm

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
