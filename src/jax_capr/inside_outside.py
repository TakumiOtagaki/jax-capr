from __future__ import annotations

from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Tuple

import numpy as np

from jax_rnafold.common.partition import psum_hairpin
from jax_rnafold.common.utils import NBPS, NTS, MAX_LOOP, get_bp_bases
from jax_rnafold.d0 import energy

from .jax_inside import InsideTables, compute_inside_tables

State = Tuple[str, Tuple[int, ...]]


@dataclass
class Contribution:
    factor: float
    children: Tuple[State, ...]


@dataclass
class InsideOutsideResult:
    partition: float
    sequence: str
    inside: InsideTables
    outside: Dict[State, float]
    bpp: np.ndarray


def _state_key(state_type: str, *indices: int) -> State:
    return (state_type, tuple(indices))


def _get_state_value(state: State, tables: InsideTables) -> float:
    kind, idx = state
    if kind == "E":
        return tables.E[idx[0]]
    if kind == "P":
        bp, i, j = idx
        return tables.P[bp, i, j]
    if kind == "ML":
        nb, i, j = idx
        return tables.ML[nb, i, j]
    if kind == "MB":
        i, j = idx
        return tables.MB[i, j]
    if kind == "OMM":
        i, j = idx
        return tables.OMM[i, j]
    raise KeyError(f"Unknown state: {state}")


def _verify_contributions(
    contributions: Dict[State, List[Contribution]],
    tables: InsideTables,
    atol: float = 1e-6,
    rtol: float = 1e-4,
) -> None:
    for state, contribs in contributions.items():
        target = _get_state_value(state, tables)
        if abs(target) < atol:
            continue
        total = 0.0
        for contrib in contribs:
            val = contrib.factor
            for child in contrib.children:
                val *= _get_state_value(child, tables)
            total += val
        if not np.isfinite(total) or not np.allclose(total, target, atol=atol, rtol=rtol):
            raise ValueError(
                f"Contribution mismatch for state {state}: expected {target}, got {total}"
            )


def compute_inside_outside(
    seq: str,
    em: energy.Model,
    max_loop: int = None,
    scale: float | None = None,
) -> InsideOutsideResult:
    inside = compute_inside_tables(
        seq,
        em,
        max_loop=max_loop if max_loop is not None else MAX_LOOP,
        # scale=scale,
    )

    E = inside.E
    P = inside.P
    ML = inside.ML
    MB = inside.MB
    OMM = inside.OMM
    padded = inside.padded_p_seq

    n = inside.seq_len
    hairpin_min = em.hairpin
    two_loop_length = inside.two_loop_length

    contributions: DefaultDict[State, List[Contribution]] = DefaultDict(list)

    def add(state: State, factor: float, *children: State) -> None:
        if factor == 0.0:
            return
        contributions[state].append(Contribution(float(factor), tuple(children)))

    for i in range(n - 1, -1, -1):
        for bp in range(NBPS):
            bi, bj = get_bp_bases(bp)
            j_start = i + hairpin_min + 1
            for j in range(j_start, n):
                state = _state_key("P", bp, i, j)

                hairpin = psum_hairpin(padded, em, bi, bj, i, j)
                add(state, hairpin)

                for bpkl in range(NBPS):
                    bk, bl = get_bp_bases(bpkl)

                    for offset in range(two_loop_length):
                        l = j - 2 - offset
                        if l < i + 2:
                            break
                        factor_r = (
                            padded[i + 1, bk]
                            * padded[l, bl]
                            * em.en_bulge(bi, bj, bk, bl, j - l - 1)
                        )
                        add(state, factor_r, _state_key("P", bpkl, i + 1, l))

                        k = i + 2 + offset
                        if k >= j - 1:
                            continue
                        factor_l = (
                            padded[k, bk]
                            * padded[j - 1, bl]
                            * em.en_bulge(bi, bj, bk, bl, k - i - 1)
                        )
                        add(state, factor_l, _state_key("P", bpkl, k, j - 1))

                    mmij = 0.0
                    for bip1 in range(NTS):
                        for bjm1 in range(NTS):
                            mmij += (
                                padded[i + 1, bip1]
                                * padded[j - 1, bjm1]
                                * em.en_il_inner_mismatch(bi, bj, bip1, bjm1)
                            )

                    for bip1 in range(NTS):
                        for bjm1 in range(NTS):
                            pr_ij_mm = (
                                padded[i + 1, bip1] * padded[j - 1, bjm1]
                            )

                            if j - i >= 4:
                                factor_11 = (
                                    pr_ij_mm
                                    * padded[i + 2, bk]
                                    * padded[j - 2, bl]
                                    * em.en_internal(
                                        bi,
                                        bj,
                                        bk,
                                        bl,
                                        bip1,
                                        bjm1,
                                        bip1,
                                        bjm1,
                                        1,
                                        1,
                                    )
                                )
                                add(state, factor_11, _state_key("P", bpkl, i + 2, j - 2))

                            for offset in range(two_loop_length):
                                z = i + 3 + offset
                                if z >= j - 2:
                                    break
                                for b in range(NTS):
                                    rup = j - z - 1
                                    if rup >= 1:
                                        factor_right = (
                                            pr_ij_mm
                                            * padded[i + 2, bk]
                                            * padded[z, bl]
                                            * padded[z + 1, b]
                                            * em.en_internal(
                                                bi,
                                                bj,
                                                bk,
                                                bl,
                                                bip1,
                                                bjm1,
                                                bip1,
                                                b,
                                                1,
                                                rup,
                                            )
                                        )
                                        add(state, factor_right, _state_key("P", bpkl, i + 2, z))

                                    lup = z - i - 1
                                    if lup >= 1:
                                        factor_left = (
                                            pr_ij_mm
                                            * padded[z, bk]
                                            * padded[j - 2, bl]
                                            * padded[z - 1, b]
                                            * em.en_internal(
                                                bi,
                                                bj,
                                                bk,
                                                bl,
                                                bip1,
                                                bjm1,
                                                b,
                                                bjm1,
                                                lup,
                                                1,
                                            )
                                        )
                                        add(state, factor_left, _state_key("P", bpkl, z, j - 2))

                    for k in range(i + 2, min(i + 2 + two_loop_length, j - 1)):
                        for l in range(max(k + 1, j - two_loop_length - 1), j - 1):
                            if l <= k:
                                continue
                            lup = k - i - 1
                            rup = j - l - 1
                            if lup <= 0 or rup <= 0:
                                continue

                            if (
                                (lup == 2 and rup == 2)
                                or (lup == 2 and rup == 3)
                                or (lup == 3 and rup == 2)
                            ):
                                for bip1 in range(NTS):
                                    for bjm1 in range(NTS):
                                        for bkm1 in range(NTS):
                                            for blp1 in range(NTS):
                                                factor_special = (
                                                    padded[k, bk]
                                                    * padded[l, bl]
                                                    * padded[k - 1, bkm1]
                                                    * padded[l + 1, blp1]
                                                    * padded[i + 1, bip1]
                                                    * padded[j - 1, bjm1]
                                                    * em.en_internal(
                                                        bi,
                                                        bj,
                                                        bk,
                                                        bl,
                                                        bip1,
                                                        bjm1,
                                                        bkm1,
                                                        blp1,
                                                        lup,
                                                        rup,
                                                        )
                                                )
                                                add(
                                                    state,
                                                    factor_special,
                                                    _state_key("P", bpkl, k, l),
                                                )
                            else:
                                factor_general = (
                                    mmij
                                    * em.en_internal_init(lup + rup)
                                    * em.en_internal_asym(lup, rup)
                                )
                                add(
                                    state,
                                    factor_general,
                                    _state_key("OMM", k, l),
                                )

                if j - i > 2:
                    for bpkl in range(NBPS):
                        bk, bl = get_bp_bases(bpkl)
                        factor_stack = (
                            padded[i + 1, bk]
                            * padded[j - 1, bl]
                            * em.en_stack(bi, bj, bk, bl)
                        )
                        add(state, factor_stack, _state_key("P", bpkl, i + 1, j - 1))

                    add(
                        state,
                        em.en_multi_closing(bi, bj),
                        _state_key("ML", 2, i + 1, j - 1),
                    )

        for j in range(i, n):
            state = _state_key("OMM", i, j)
            for bp_idx in range(NBPS):
                bi, bj = get_bp_bases(bp_idx)
                for bim1 in range(NTS):
                    for bjp1 in range(NTS):
                        factor = (
                            em.en_il_outer_mismatch(bi, bj, bim1, bjp1)
                            * padded[i - 1, bim1]
                            * padded[j + 1, bjp1]
                            * padded[i, bi]
                            * padded[j, bj]
                        )
                        add(state, factor, _state_key("P", bp_idx, i, j))

        for j in range(i, n + 1):
            state = _state_key("MB", i, j)
            for bp_idx in range(NBPS):
                bi, bj = get_bp_bases(bp_idx)
                factor = (
                    em.en_multi_branch(bi, bj)
                    * padded[i, bi]
                    * padded[j, bj]
                )
                add(state, factor, _state_key("P", bp_idx, i, j))

        for nb in range(3):
            for j in range(i, n):
                state = _state_key("ML", nb, i, j)
                add(state, 1.0, _state_key("ML", nb, i + 1, j))
                idx = max(0, nb - 1)
                for k in range(i, j + 1):
                    add(
                        state,
                        1.0,
                        _state_key("ML", idx, k + 1, j),
                        _state_key("MB", i, k),
                    )

        state_E = _state_key("E", i)
        add(state_E, 1.0, _state_key("E", i + 1))
        for j in range(i + 1, n):
            for bp_idx in range(NBPS):
                bi, bj = get_bp_bases(bp_idx)
                factor = (
                    padded[i, bi]
                    * padded[j, bj]
                    * em.en_ext_branch(bi, bj)
                )
                add(
                    state_E,
                    factor,
                    _state_key("E", j + 1),
                    _state_key("P", bp_idx, i, j),
                )

    _verify_contributions(contributions, inside)

    indegree: DefaultDict[State, int] = DefaultDict(int)
    children_map: DefaultDict[State, List[State]] = DefaultDict(list)

    for parent, contribs in contributions.items():
        seen = set()
        for contrib in contribs:
            for child in contrib.children:
                if child not in seen:
                    indegree[child] += 1
                    seen.add(child)
        children_map[parent] = list(seen)
        indegree[parent] = indegree[parent]

    outside: DefaultDict[State, float] = DefaultDict(float)
    root = _state_key("E", 0)
    outside[root] = 1.0
    queue: List[State] = [root]
    processed = set()

    while queue:
        state = queue.pop()
        if state in processed:
            continue
        processed.add(state)
        outs = outside[state]
        if outs != 0.0:
            contribs = contributions.get(state, ())
            for contrib in contribs:
                child_values = [
                    _get_state_value(child, inside) for child in contrib.children
                ]
                for idx, child in enumerate(contrib.children):
                    other_prod = contrib.factor
                    for k, val in enumerate(child_values):
                        if k == idx:
                            continue
                        other_prod *= val
                    outside[child] += outs * other_prod
        for child in children_map.get(state, ()):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    bpp = np.zeros((n, n), dtype=np.float64)
    Z = inside.E[0]
    for i in range(n):
        for j in range(i + 1, n):
            prob = 0.0
            for bp in range(NBPS):
                state = _state_key("P", bp, i, j)
                prob += inside.P[bp, i, j] * outside.get(state, 0.0)
            if Z != 0.0:
                prob /= Z
            bpp[i, j] = prob
            bpp[j, i] = prob

    return InsideOutsideResult(
        partition=float(Z),
        sequence=seq,
        inside=inside,
        outside=dict(outside),
        bpp=bpp,
    )


def compute_bpp_matrix(seq: str, em: energy.Model) -> np.ndarray:
    return compute_inside_outside(seq, em).bpp
