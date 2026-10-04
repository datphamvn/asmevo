"""Parent selection, survivor selection (elitism), and duplicate detection for search.

All functions are pure: the same inputs and ``rng`` state give the same result, so a search
run is reproducible from its seed. Invalid candidates (fitness -inf) are never picked as a
parent, a survivor, or the selected result.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Iterable, Sequence

from asmevo.schemas import Candidate

# Directives that only feed debug info, unwind tables or the symbol table, so the bytes the
# CPU runs are the same without them. Section, alignment and data directives are kept: two
# programs that differ there are different programs (alignment alone can change timing).
_IGNORED_DIRECTIVES = frozenset({".file", ".ident", ".loc", ".loc_mark_labels", ".size", ".type"})
_STRING_RE = re.compile(r'("(?:\\.|[^"\\\n])*")')  # odd items of re.split are literals


def _strip_comments(asm: str) -> str:
    """Drop ``#`` and ``/* */`` comments outside string literals, keeping line breaks."""
    out: list[str] = []
    i, n, in_string = 0, len(asm), False
    while i < n:
        ch = asm[i]
        if in_string:
            if ch == "\\":
                out.append(asm[i : i + 2])
                i += 2
                continue
            in_string = ch not in '"\n'
            out.append(ch)
            i += 1
        elif ch == '"':
            in_string = True
            out.append(ch)
            i += 1
        elif asm.startswith("/*", i):
            end = asm.find("*/", i + 2)
            end = n if end < 0 else end + 2
            out.append("\n" * asm.count("\n", i, end) or " ")
            i = end
        elif ch == "#":
            end = asm.find("\n", i)
            i = n if end < 0 else end
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def normalize_asm(asm: str) -> str:
    """``asm`` as one statement per line, without comments, blank lines, layout whitespace,
    or directives that do not change the program. String literals are left untouched."""
    statements = []
    for line in _strip_comments(asm).splitlines():
        parts = _STRING_RE.split(line)
        for i in range(0, len(parts), 2):
            code = re.sub(r"[ \t]+", " ", parts[i].replace(";", "\n"))
            parts[i] = re.sub(r" ?, ?", ",", code)
        for stmt in "".join(parts).split("\n"):
            stmt = stmt.strip()
            word = stmt.split(" ", 1)[0]
            if stmt and word not in _IGNORED_DIRECTIVES and not word.startswith(".cfi_"):
                statements.append(stmt)
    return "\n".join(statements)


def asm_hash(asm: str) -> str:
    """SHA-256 of ``normalize_asm(asm)``: programs that differ only in comments, whitespace
    or ignored directives share it, so the search builds and times each program once."""
    return hashlib.sha256(normalize_asm(asm).encode()).hexdigest()


def _dedup_key(c: Candidate) -> str:
    if c.asm_hash is not None:
        return c.asm_hash
    return asm_hash(c.asm) if c.asm is not None else c.id


def tournament(pop: Sequence[Candidate], k: int, rng: random.Random) -> Candidate:
    """Fittest of ``k`` valid members drawn without replacement (all of them if fewer are
    valid); ties go to the one drawn first."""
    if k < 1:
        raise ValueError(f"tournament size must be >= 1, got {k}")
    valid = [c for c in pop if c.valid]
    if not valid:
        raise ValueError("no valid candidate to select a parent from")
    return max(rng.sample(valid, min(k, len(valid))), key=lambda c: c.fitness)


def survive(
    pop: Sequence[Candidate], offspring: Sequence[Candidate], size: int = 4
) -> tuple[Candidate, ...]:
    """Next population: the -O3 baseline, then the ``size - 1`` fittest distinct valid
    candidates of ``pop`` and ``offspring`` together, so the best one found is never lost.

    ``pop`` must hold the baseline (generation 0). Duplicates by ``asm_hash`` keep their
    earliest copy, so re-timing the same program cannot displace a member, and copies of the
    baseline are dropped. Ties keep ``pop`` order, then ``offspring`` order.
    """
    if size < 1:
        raise ValueError(f"population size must be >= 1, got {size}")
    baselines = [c for c in pop if c.generation == 0]
    if len(baselines) != 1:
        raise ValueError(f"pop must hold exactly one -O3 baseline, found {len(baselines)}")
    baseline = baselines[0]
    seen = {_dedup_key(baseline)}
    pool = []
    for c in (*pop, *offspring):
        key = _dedup_key(c)
        if c is not baseline and c.valid and key not in seen:
            seen.add(key)
            pool.append(c)
    pool.sort(key=lambda c: c.fitness, reverse=True)  # stable: ties keep first-seen order
    return (baseline, *pool[: size - 1])


def best(candidates: Iterable[Candidate]) -> Candidate:
    """Fittest valid candidate, the earliest on ties (so a candidate merely as fast as the
    baseline does not replace it); every method picks ``SearchTrace.selected_id`` this way."""
    valid = [c for c in candidates if c.valid]
    if not valid:
        raise ValueError("no valid candidate to select")
    return max(valid, key=lambda c: c.fitness)
