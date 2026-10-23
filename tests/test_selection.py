import math
import platform
import random
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from asmevo.schemas import Candidate, candidate_id
from asmevo.search.selection import asm_hash, best, normalize_asm, survive, tournament

BASELINE_ASM = """\
	.file	"p.c"
	.text
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC0:
	.string	"%ld # not a comment\\n"
	.text
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB0:
	.cfi_startproc
	subq	$8, %rsp	# reserve
	.cfi_def_cfa_offset 16
	xorl	%eax, %eax
	addq	$8, %rsp
	ret
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.ident	"GCC: (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0"
	.section	.note.GNU-stack,"",@progbits
"""

# Same program: other comments and spacing, statements joined with ';', no cfi/ident/size.
REFORMATTED_ASM = """\
/* rewritten by the model,
   same code */
.text
.section .rodata.str1.1, "aMS", @progbits, 1
.LC0:
    .string "%ld # not a comment\\n"
.text
.p2align 4
.globl main
main:
.LFB0:
    subq $8,%rsp ; xorl %eax,%eax   # both on one line
    addq $8, %rsp
    ret
.LFE0:
.section .note.GNU-stack,"",@progbits
"""


def make(n: int, fitness: float, asm: str | None = None, generation: int = 1) -> Candidate:
    asm = asm if asm is not None else f"movl ${n}, %eax\nret\n"
    return Candidate(
        id=candidate_id(1, "evolution", 0, n), problem_id=1, method="evolution", seed=0,
        generation=generation,
        parent_ids=() if generation == 0 else (candidate_id(1, "evolution", 0, 0),),
        prompt_id=None if generation == 0 else "mutate-v1", asm=asm, asm_hash=asm_hash(asm),
        result=None, fitness=fitness, feedback="", llm_calls=(),
    )


BASELINE = make(0, 1.0, asm=BASELINE_ASM, generation=0)
INVALID = -math.inf


def ids(pop):
    return [c.id.rsplit("-", 1)[1] for c in pop]


def test_hash_ignores_layout_comments_and_inert_directives():
    assert asm_hash(REFORMATTED_ASM) == asm_hash(BASELINE_ASM)


@pytest.mark.parametrize("old, new", [
    ("xorl\t%eax, %eax", "movl\t$0, %eax"),         # different instruction
    ("not a comment", "not a comment!"),            # string data
    ('"%ld # not', '"%ld  # not'),                  # whitespace inside a string
    (".p2align 4", ".p2align 5"),                   # code layout
    (".globl\tmain", ".globl\tmain\n\t.local\tfoo"),  # .local is not .loc
])
def test_hash_tells_different_programs_apart(old, new):
    assert old in BASELINE_ASM
    assert asm_hash(BASELINE_ASM.replace(old, new)) != asm_hash(BASELINE_ASM)


def test_tournament_never_picks_invalid():
    pop = [BASELINE, make(1, INVALID), make(2, INVALID), make(3, INVALID)]
    rng = random.Random(0)
    assert all(tournament(pop, 2, rng) is BASELINE for _ in range(50))


def test_survive_keeps_baseline_and_fittest():
    offspring = [make(n, 1.0 + n / 10) for n in range(1, 6)]
    assert ids(survive([BASELINE], offspring, size=4)) == ["0", "5", "4", "3"]


def test_survive_dedups_keeping_the_earliest_copy():
    member = make(1, 1.2, asm="movl $1, %eax\nret\n")
    retimed = make(2, 1.9, asm="movl $1,%eax   # same code\n  ret\n")
    baseline_copy = make(3, 1.05, asm=REFORMATTED_ASM)
    assert ids(survive([BASELINE, member], [retimed, baseline_copy])) == ["0", "1"]


def test_best_prefers_fitness_then_earliest():
    tie = make(1, 1.0)
    assert best([BASELINE, tie, make(2, INVALID)]) is BASELINE
    assert best([BASELINE, tie, make(3, 1.3)]).id.endswith("-3")
    with pytest.raises(ValueError):
        best([make(4, INVALID)])


# Strings with comment and separator characters, a jump table, and initialized data.
C_SOURCE = r"""
#include <stdio.h>
static const char *names[] = {"zero # not a comment", "one; two", "three /* not */"};
int table[64] = {1, 2, 3};
int pick(int x) {
    switch (x & 7) {
    case 0: return 11; case 1: return 23; case 2: return 37; case 3: return 41;
    case 4: return 59; case 5: return 61; default: return 7;
    }
}
int main(void) {
    int n;
    if (scanf("%d", &n) != 1) return 1;
    for (int i = 0; i < 64; i++) table[i] += pick(i * n);
    printf("%s %d\n", names[n % 3], table[n & 63]);
    return 0;
}
"""
INERT_SECTIONS = {".eh_frame", ".comment", ".note.GNU-stack"}


def assembled(asm: str, path: Path) -> dict[str, str]:
    """Contents and relocations of each section of ``asm`` assembled, unwind info aside."""
    path.with_suffix(".s").write_text(asm + "\n")
    subprocess.run(["gcc", "-c", str(path.with_suffix(".s")), "-o", str(path)], check=True)
    out = {}
    for flag in ("-s", "-r"):
        dump = subprocess.run(["objdump", flag, str(path)], check=True, capture_output=True,
                              text=True).stdout
        for part in re.split(r"\n(?=Contents of section |RELOCATION RECORDS FOR )", dump):
            m = re.match(r"(?:Contents of section (\S+):|RELOCATION RECORDS FOR \[(\S+)\]:)",
                         part)
            if m and (m[1] or m[2]) not in INERT_SECTIONS:
                out[f"{flag} {m[1] or m[2]}"] = part.split("\n", 1)[1].strip()
    return out


@pytest.mark.skipif(
    not (platform.system() == "Linux" and platform.machine() in ("x86_64", "amd64")
         and shutil.which("gcc") and shutil.which("objdump")),
    reason="needs Linux x86-64 with gcc and objdump: run in the docker image",
)
def test_normalized_asm_assembles_to_the_same_program(tmp_path):
    src = tmp_path / "p.c"
    src.write_text(C_SOURCE)
    asm = subprocess.run(["gcc", "-O3", "-S", "-o", "-", str(src)], check=True,
                         capture_output=True, text=True).stdout
    original = assembled(asm, tmp_path / "original.o")
    assert {".text", ".rodata.str1.1", ".data"} <= {k.split()[1] for k in original}
    assert assembled(normalize_asm(asm), tmp_path / "normalized.o") == original
