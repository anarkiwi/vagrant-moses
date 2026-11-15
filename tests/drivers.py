"""Small 6502 games read through CIA 1's keyboard matrix, and the driver programs that
play them: steer an angle, press a key inside a scan with retries, walk a cursor."""

from __future__ import annotations

from vagrant_moses import Machine, Program, R

from asm import assemble

ORG = 0x0800
ANGLE, LIMIT = 0xC000, 0xC002
COUNT, FRAME = 0xC010, 0xC011
X, Y, COLS = 0xC020, 0xC021, 0xC022
INC_KEY, DEC_KEY, PRESS_KEY = (1, 2), (1, 3), (0, 1)
RIGHT, LEFT, DOWN, UP = (2, 0), (2, 1), (2, 2), (2, 3)
SETUP = """
  SEI
  LDA #$FF
  STA $DC02
  LDA #$00
  STA $DC03
"""
WAIT = """
  LDY #$18
wait: DEY
  BNE wait
"""
GAMES = {
    "angle": SETUP
    + """
pass: LDA #$FD
  STA $DC00
  LDA $DC01
  TAX
  AND #$04
  BNE noinc
  LDA $C000
  CMP $C002
  BEQ noinc
  INC $C000
noinc: TXA
  AND #$08
  BNE moved
  DEC $C000
moved: NOP
"""
    + WAIT
    + "  JMP pass\n",
    "scan": SETUP
    + """
main: INC $C011
scan: LDA #$FE
  STA $DC00
  LDA $DC01
  AND #$02
done: BNE skip
  LDA $C011
  AND #$03
  BNE skip
  INC $C010
skip: NOP
"""
    + WAIT
    + "  JMP main\n",
    "cursor": SETUP
    + """
pass: LDA #$FB
  STA $DC00
  LDA $DC01
  STA $C022
  AND #$01
  BNE nr
  INC $C020
commitr: NOP
nr: LDA $C022
  AND #$02
  BNE nl
  DEC $C020
commitl: NOP
nl: LDA $C022
  AND #$04
  BNE nd
  INC $C021
commitd: NOP
nd: LDA $C022
  AND #$08
  BNE nu
  DEC $C021
commitu: NOP
nu: NOP
"""
    + WAIT
    + "  JMP pass\n",
}


def game(name: str, backend: str, **ram):
    """``(machine, labels)``: game ``name`` loaded and reset on ``backend``, with the RAM
    bytes ``ram`` (``angle``, ``limit``, ``count``, ``frame``, ``x``, ``y``) set."""
    machine = Machine(backend=backend)
    code, labels = assemble(GAMES[name], ORG)
    machine.ram[ORG : ORG + len(code)] = code
    where = {"angle": ANGLE, "limit": LIMIT, "count": COUNT, "frame": FRAME}
    for key, value in ram.items():
        machine.ram[{**where, "x": X, "y": Y}[key]] = value
    machine.reset(ORG)
    return machine, labels


def steer(labels, target: int, stall: int = 4, budget: int = 2_000_000) -> Program:
    """Hold the key that turns the angle towards ``target`` by the shorter way round,
    checking at each pass; halt 0 there, 1 once it has not moved for ``stall`` passes.
    """
    p = Program(budget, marker=labels["pass"])
    p.mov(R[2], 256).mov(R[3], stall)
    p.label("loop").run_to(labels["pass"], leave=True).ld(R[0], ANGLE)
    p.beq(R[0], target, "done").bne(R[0], R[2], "moved")
    p.djnz(R[3], "steer").clear().halt(1)
    p.label("moved").mov(R[3], stall)
    p.label("steer").mov(R[2], R[0]).mov(R[4], target).sub8(R[4], R[0]).clear()
    p.bmi8(R[4], "dec").key(*INC_KEY).jmp("loop")
    p.label("dec").key(*DEC_KEY).jmp("loop")
    p.label("done").clear().frames(R[5]).out(R[5]).halt(0)
    return p


def press(labels, tries: int, budget: int = 2_000_000) -> Program:
    """Press the key only between the scan's start and its end, then check the press
    counted; retry up to ``tries`` times; halt 0 (outputs: tries left) or 1."""
    p = Program(budget)
    p.ldr(R[1], COUNT).mov(R[2], tries)
    p.label("try").run_to(labels["scan"], leave=True).key(*PRESS_KEY)
    p.run_to(labels["done"], leave=True).key(*PRESS_KEY, down=False)
    p.run_to(labels["skip"], leave=True).ld(R[0], COUNT)
    p.bne(R[0], R[1], "ok").djnz(R[2], "try").halt(1)
    p.label("ok").out(R[2]).halt(0)
    return p


def walk(labels, tx: int, ty: int, steps: int = 64, budget: int = 4_000_000):
    """Walk the cursor to ``(tx, ty)``: per pass up to one x and one y key, run to the
    commit of the last direction pressed and one instruction on; halt 0 there, else 1.
    """
    p = Program(budget)
    p.mov(R[5], steps)
    p.label("loop").ld(R[0], X).ld(R[1], Y).clear().mov(R[3], 0)
    p.beq(R[0], tx, "y").blt(R[0], tx, "right").key(*LEFT).mov(R[3], 1).jmp("y")
    p.label("right").key(*RIGHT).mov(R[3], 1)
    p.label("y").beq(R[1], ty, "x").blt(R[1], ty, "down").key(*UP).jmp("ycommit")
    p.label("down").key(*DOWN)
    p.label("ycommit").run_to({labels["commitd"], labels["commitu"]}, leave=True)
    p.step().jmp("next")
    p.label("x").beq(R[3], 0, "done")
    p.run_to({labels["commitr"], labels["commitl"]}, leave=True).step()
    p.label("next").djnz(R[5], "loop").clear().halt(1)
    p.label("done").clear().out(R[5]).halt(0)
    return p
