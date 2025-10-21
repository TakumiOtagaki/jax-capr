from __future__ import annotations

from dataclasses import dataclass
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from jax_rnafold.common.partition import psum_hairpin
from jax_rnafold.common.utils import NBPS, NTS, RNA_ALPHA, get_bp_bases
from jax_rnafold.d0 import energy

State = Tuple[str, Tuple[int, ...]]


@dataclass
class Contribution:
    """Represents a single additive contribution to a DP state."""

    factor: float
    children: Tuple[State, ...]


@dataclass
class InsideOutsideResult:
    partition: float
    sequence: str
    E: np.ndarray
    P: np.ndarray
    ML: np.ndarray
    MB: np.ndarray
    OMM: np.ndarray
    outside: Dict[State, float]
    bpp: np.ndarray


def _sequence_to_one_hot(seq: str, dtype=np.float64) -> np.ndarray:
    mapping = {base: idx for idx, base in enumerate(RNA_ALPHA)}
    n = len(seq)
    arr = np.zeros((n, len(RNA_ALPHA)), dtype=dtype)
    for i, base in enumerate(seq):
        idx = mapping.get(base.upper())
        if idx is not None:
            arr[i, idx] = 1.0
        else:
            arr[i, :] = 0.25
    return arr


def _state_key(state_type: str, *indices: int) -> State:
    return (state_type, tuple(indices))


def compute_inside_outside(
    seq: str,
    em: energy.Model,
    dtype=np.float64,
) -> InsideOutsideResult:
    """
    Compute inside and outside tables for a fixed RNA sequence using the
    reference McCaskill recurrence from jax-rnafold (d0 reference implementation).
    """
    p_seq = _sequence_to_one_hot(seq, dtype=dtype)
    n = p_seq.shape[0]

    padded_p_seq = np.zeros((n + 1, 4), dtype=dtype)
    padded_p_seq[:n] = p_seq

    E = np.zeros((n + 1,), dtype=dtype)
    P = np.zeros((NBPS, n + 1, n + 1), dtype=dtype)
    ML = np.zeros((3, n + 1, n + 1), dtype=dtype)
    MB = np.zeros((n + 1, n + 1), dtype=dtype)
    OMM = np.zeros((n + 1, n + 1), dtype=dtype)

    E[n] = 1.0
    ML[0, :, :] = 1.0

    contributions: DefaultDict[State, List[Contribution]] = DefaultDict(list)

    def get_value(state: State) -> float:
        stype, idx = state
        if stype == "E":
            return E[idx[0]]
        if stype == "P":
            bp, i, j = idx
            return P[bp, i, j]
        if stype == "ML":
            nb, i, j = idx
            return ML[nb, i, j]
        if stype == "MB":
            i, j = idx
            return MB[i, j]
        if stype == "OMM":
            i, j = idx
            return OMM[i, j]
        raise KeyError(f"Unknown state {state}")

    def accumulate(state: State, contribs: Iterable[Contribution]) -> float:
        total = 0.0
        contrib_list: List[Contribution] = []
        for contrib in contribs:
            if contrib.factor == 0.0:
                continue
            term = contrib.factor
            skip = False
            for child in contrib.children:
                val = get_value(child)
                term *= val
                if term == 0.0:
                    skip = True
                    break
            if skip or term == 0.0:
                contrib_list.append(contrib)
                continue
            total += term
            contrib_list.append(contrib)
        contributions[state].extend(contrib_list)
        return total

    for i in range(n - 1, -1, -1):
        # fill P
        for bp in range(NBPS):
            bi, bj = get_bp_bases(bp)
            j_start = i + em.hairpin + 1
            for j in range(j_start, n):
                contribs: List[Contribution] = []

                hairpin = psum_hairpin(padded_p_seq, em, bi, bj, i, j)
                if hairpin != 0.0:
                    contribs.append(Contribution(hairpin, ()))

                for bpkl in range(NBPS):
                    bk, bl = get_bp_bases(bpkl)
                    # bulges
                    for kl in range(i + 2, j - 1):
                        factor_r = (
                            padded_p_seq[i + 1, bk]
                            * padded_p_seq[kl, bl]
                            * em.en_bulge(bi, bj, bk, bl, j - kl - 1)
                        )
                        if factor_r != 0.0:
                            contribs.append(
                                Contribution(
                                    factor_r,
                                    (_state_key("P", bpkl, i + 1, kl),),
                                )
                            )

                        factor_l = (
                            padded_p_seq[kl, bk]
                            * padded_p_seq[j - 1, bl]
                            * em.en_bulge(bi, bj, bk, bl, kl - i - 1)
                        )
                        if factor_l != 0.0:
                            contribs.append(
                                Contribution(
                                    factor_l,
                                    (_state_key("P", bpkl, kl, j - 1),),
                                )
                            )

                    # internal loops (1x1 and special cases)
                for bip1 in range(NTS):
                    for bjm1 in range(NTS):
                        pr_ij_mm = (
                            padded_p_seq[i + 1, bip1]
                            * padded_p_seq[j - 1, bjm1]
                        )
                        if j - i >= 4:
                            factor_11 = (
                                pr_ij_mm
                                * padded_p_seq[i + 2, bk]
                                * padded_p_seq[j - 2, bl]
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
                            if factor_11 != 0.0:
                                contribs.append(
                                    Contribution(
                                        factor_11,
                                        (_state_key("P", bpkl, i + 2, j - 2),),
                                    )
                                )

                        for z in range(i + 3, j - 2):
                            for b in range(NTS):
                                lup = 1
                                rup = j - z - 1
                                if rup < 1:
                                    continue
                                factor_right = (
                                    pr_ij_mm
                                    * padded_p_seq[i + 2, bk]
                                    * padded_p_seq[z, bl]
                                    * padded_p_seq[z + 1, b]
                                    * em.en_internal(
                                        bi,
                                        bj,
                                        bk,
                                        bl,
                                        bip1,
                                        bjm1,
                                        bip1,
                                        b,
                                        lup,
                                        rup,
                                    )
                                )
                                if factor_right != 0.0:
                                    contribs.append(
                                        Contribution(
                                            factor_right,
                                            (_state_key("P", bpkl, i + 2, z),),
                                        )
                                    )

                                lup = z - i - 1
                                rup = 1
                                if lup < 1:
                                    continue
                                factor_left = (
                                    pr_ij_mm
                                    * padded_p_seq[z, bk]
                                    * padded_p_seq[j - 2, bl]
                                    * padded_p_seq[z - 1, b]
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
                                        rup,
                                    )
                                )
                                if factor_left != 0.0:
                                    contribs.append(
                                        Contribution(
                                            factor_left,
                                            (_state_key("P", bpkl, z, j - 2),),
                                        )
                                    )

                    # other internal loops (2x2, 2x3, 3x2, general)
                mmij_total = 0.0
                for bip1 in range(NTS):
                    for bjm1 in range(NTS):
                        mmij_total += (
                            padded_p_seq[i + 1, bip1]
                            * padded_p_seq[j - 1, bjm1]
                            * em.en_il_inner_mismatch(bi, bj, bip1, bjm1)
                        )
                for k in range(i + 2, j - 2):
                    for l in range(k + 1, j - 1):
                        lup = k - i - 1
                        rup = j - l - 1
                        if lup <= 0 or rup <= 0:
                            continue

                        if (lup == 2 and rup == 2) or (
                            (lup == 2 and rup == 3) or (lup == 3 and rup == 2)
                        ):
                            for bip1 in range(NTS):
                                for bjm1 in range(NTS):
                                    for bkm1 in range(NTS):
                                        for blp1 in range(NTS):
                                            factor_special = (
                                                padded_p_seq[k, bk]
                                                * padded_p_seq[l, bl]
                                                * padded_p_seq[k - 1, bkm1]
                                                * padded_p_seq[l + 1, blp1]
                                                * padded_p_seq[i + 1, bip1]
                                                * padded_p_seq[j - 1, bjm1]
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
                                            if factor_special != 0.0:
                                                contribs.append(
                                                    Contribution(
                                                        factor_special,
                                                        (
                                                            _state_key(
                                                                "P",
                                                                bpkl,
                                                                k,
                                                                l,
                                                            ),
                                                        ),
                                                    )
                                                )
                        else:
                            init = (
                                em.en_internal_init(lup + rup)
                                * em.en_internal_asym(lup, rup)
                            )
                            factor_general = mmij_total * init
                            if factor_general != 0.0:
                                contribs.append(
                                    Contribution(
                                        factor_general,
                                        (_state_key("OMM", k, l),),
                                    )
                                )

                # Stacking
                for bpkl in range(NBPS):
                    bk, bl = get_bp_bases(bpkl)
                    factor_stack = (
                        padded_p_seq[i + 1, bk]
                        * padded_p_seq[j - 1, bl]
                        * em.en_stack(bi, bj, bk, bl)
                    )
                    if factor_stack != 0.0:
                        contribs.append(
                            Contribution(
                                factor_stack,
                                (_state_key("P", bpkl, i + 1, j - 1),),
                            )
                        )

                # Multiloop closure
                if j - i > 2:
                    factor_multi = em.en_multi_closing(bi, bj)
                    contribs.append(
                        Contribution(
                            factor_multi,
                            (_state_key("ML", 2, i + 1, j - 1),),
                        )
                    )

                state = _state_key("P", bp, i, j)
                P[bp, i, j] = accumulate(state, contribs)

        # fill OMM
        for j in range(i, n):
            contribs: List[Contribution] = []
            for bpij in range(NBPS):
                bi, bj = get_bp_bases(bpij)
                for bim1 in range(NTS):
                    for bjp1 in range(NTS):
                        factor = (
                            em.en_il_outer_mismatch(bi, bj, bim1, bjp1)
                            * padded_p_seq[i - 1, bim1]
                            * padded_p_seq[j + 1, bjp1]
                            * padded_p_seq[i, bi]
                            * padded_p_seq[j, bj]
                        )
                        if factor != 0.0:
                            contribs.append(
                                Contribution(
                                    factor,
                                    (_state_key("P", bpij, i, j),),
                                )
                            )
            state = _state_key("OMM", i, j)
            OMM[i, j] = accumulate(state, contribs)

        # fill MB
        for j in range(i, n + 1):
            contribs: List[Contribution] = []
            for bp in range(NBPS):
                bi, bj = get_bp_bases(bp)
                factor = (
                    em.en_multi_branch(bi, bj)
                    * padded_p_seq[i, bi]
                    * padded_p_seq[j, bj]
                )
                if factor != 0.0:
                    contribs.append(
                        Contribution(
                            factor,
                            (_state_key("P", bp, i, j),),
                        )
                    )
            state = _state_key("MB", i, j)
            MB[i, j] = accumulate(state, contribs)

        # fill ML
        for nb in range(3):
            for j in range(i, n):
                contribs: List[Contribution] = []
                contribs.append(
                    Contribution(1.0, (_state_key("ML", nb, i + 1, j),))
                )
                for k in range(i, j + 1):
                    idx = max(0, nb - 1)
                    contribs.append(
                        Contribution(
                            1.0,
                            (
                                _state_key("ML", idx, k + 1, j),
                                _state_key("MB", i, k),
                            ),
                        )
                    )
                state = _state_key("ML", nb, i, j)
                ML[nb, i, j] = accumulate(state, contribs)

        # fill E
        contribs: List[Contribution] = []
        contribs.append(Contribution(1.0, (_state_key("E", i + 1),)))
        for j in range(i + 1, n):
            for bp in range(NBPS):
                bi, bj = get_bp_bases(bp)
                factor = (
                    padded_p_seq[i, bi]
                    * padded_p_seq[j, bj]
                    * em.en_ext_branch(bi, bj)
                )
                if factor != 0.0:
                    contribs.append(
                        Contribution(
                            factor,
                            (
                                _state_key("E", j + 1),
                                _state_key("P", bp, i, j),
                            ),
                        )
                    )
        state = _state_key("E", i)
        E[i] = accumulate(state, contribs)

    Z = E[0]

    indegree: DefaultDict[State, int] = DefaultDict(int)
    children_map: DefaultDict[State, List[State]] = DefaultDict(list)
    for parent, contribs in contributions.items():
        seen_children = set()
        for contrib in contribs:
            for child in contrib.children:
                if child not in seen_children:
                    indegree[child] += 1
                    seen_children.add(child)
        children_map[parent] = list(seen_children)
        indegree[parent] = indegree[parent]

    outside: DefaultDict[State, float] = DefaultDict(float)
    root = _state_key("E", 0)
    outside[root] = 1.0

    queue: List[State] = [root]
    visited = set()

    while queue:
        state = queue.pop()
        if state in visited:
            continue
        visited.add(state)
        outs = outside[state]
        if outs == 0.0:
            children = children_map.get(state, ())
            for child in children:
                indegree[child] -= 1
                if indegree[child] == 0 and child in contributions:
                    queue.append(child)
            continue

        for contrib in contributions.get(state, ()):
            child_values = [get_value(child) for child in contrib.children]
            for idx, child in enumerate(contrib.children):
                other_prod = contrib.factor
                for k, val in enumerate(child_values):
                    if k == idx:
                        continue
                    other_prod *= val
                outside[child] += outs * other_prod

        for child in children_map.get(state, ()):
            indegree[child] -= 1
            if indegree[child] == 0 and child in contributions:
                queue.append(child)

    bpp = np.zeros((n, n), dtype=dtype)
    for i in range(n):
        for j in range(i + 1, n):
            prob = 0.0
            for bp in range(NBPS):
                state = _state_key("P", bp, i, j)
                prob += P[bp, i, j] * outside.get(state, 0.0)
            if Z != 0.0:
                prob /= Z
            bpp[i, j] = prob
            bpp[j, i] = prob

    return InsideOutsideResult(
        partition=Z,
        sequence=seq,
        E=E,
        P=P,
        ML=ML,
        MB=MB,
        OMM=OMM,
        outside=dict(outside),
        bpp=bpp,
    )


def compute_bpp_matrix(seq: str, em: energy.Model, dtype=np.float64) -> np.ndarray:
    return compute_inside_outside(seq, em, dtype=dtype).bpp
