"""Scaffolding for JAX-based outside dynamic programming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax.numpy as jnp

from jax_rnafold.common.utils import MAX_LOOP, NBPS, bp_bases


Array = jnp.ndarray


class InsideTablesLike(Protocol):
    """Protocol capturing the inside tables required for outside DP."""

    E: Array
    P: Array
    ML: Array
    MB: Array
    OMM: Array
    p_seq: Array
    s_table: Array
    scale: float


@dataclass(slots=True)
class OutsideComputation:
    """Container for outside DP tables."""

    bar_E: Array
    bar_P: Array
    bar_M: Array
    bar_MB: Array
    bar_OMM: Array


def initialize_tables(inside: InsideTablesLike) -> OutsideComputation:
    """Allocate zero-initialised outside tables with correct shapes."""

    bar_E = jnp.zeros_like(inside.E)
    bar_P = jnp.zeros_like(inside.P)
    bar_M = jnp.zeros_like(inside.ML)
    bar_MB = jnp.zeros_like(inside.MB)
    bar_OMM = jnp.zeros_like(inside.OMM)

    # Exterior outside base condition: partition function derivative w.r.t Z.
    bar_E = bar_E.at[0].set(1.0)

    return OutsideComputation(
        bar_E=bar_E,
        bar_P=bar_P,
        bar_M=bar_M,
        bar_MB=bar_MB,
        bar_OMM=bar_OMM,
    )


def compute_outside(inside: InsideTablesLike, model) -> OutsideComputation:
    """
    Run the outside recursion using precomputed inside tables.

    Implementation is staged: initially allocate tables, later fill with
    recurrence logic following the project requirements.
    """
    tables = initialize_tables(inside)

    seq_len = int(inside.p_seq.shape[0])
    padded_p_seq = jnp.zeros((seq_len + 1, 4), dtype=inside.p_seq.dtype)
    padded_p_seq = padded_p_seq.at[:seq_len].set(inside.p_seq)

    s_table = inside.s_table
    P = inside.P
    ML = inside.ML
    MB = inside.MB

    dtype = inside.E.dtype
    bp_array = jnp.array(bp_bases[:NBPS], dtype=jnp.int32)
    bi = bp_array[:, 0]
    bj = bp_array[:, 1]

    ext_branch_weights = jnp.array(
        [model.en_ext_branch(int(b0), int(b1)) for b0, b1 in zip(bi.tolist(), bj.tolist())],
        dtype=dtype,
    )
    multi_branch_weights = jnp.array(
        [model.en_multi_branch(int(b0), int(b1)) for b0, b1 in zip(bi.tolist(), bj.tolist())],
        dtype=dtype,
    )
    multi_closing_weights = jnp.array(
        [model.en_multi_closing(int(b0), int(b1)) for b0, b1 in zip(bi.tolist(), bj.tolist())],
        dtype=dtype,
    )
    stack_weights = jnp.array(
        [
            [
                model.en_stack(int(bi_out), int(bj_out), int(bi_in), int(bj_in))
                for bi_in, bj_in in zip(bi.tolist(), bj.tolist())
            ]
            for bi_out, bj_out in zip(bi.tolist(), bj.tolist())
        ],
        dtype=dtype,
    )

    bar_E = tables.bar_E
    bar_P = tables.bar_P
    bar_M = tables.bar_M
    bar_MB = tables.bar_MB
    bar_OMM = tables.bar_OMM

    two_loop_length = min(seq_len, MAX_LOOP)

    # Exterior loop outside propagation.
    for i in range(seq_len):
        e_i = float(bar_E[i])
        if e_i == 0.0:
            continue

        # Unpaired contribution.
        if i + 1 <= seq_len:
            bar_E = bar_E.at[i + 1].add(e_i * s_table[1])

        if i + 1 >= seq_len:
            continue

        js = jnp.arange(i + 1, seq_len)
        if js.size == 0:
            continue

        E_j1 = inside.E[js + 1]
        p_i_bi = padded_p_seq[i, bi]
        p_j_bj = padded_p_seq[js[:, None], bj[None, :]]

        # Contribution to bar_P.
        delta_bar_P = (
            e_i
            * s_table[2]
            * E_j1[:, None]
            * p_i_bi[None, :]
            * p_j_bj
            * ext_branch_weights[None, :]
        )
        bar_P = bar_P.at[:, i, js].add(delta_bar_P.T)

        # Contribution to bar_E[j+1].
        P_slice = P[:, i, js].T  # shape (len_js, NBPS)
        weighted = (
            P_slice
            * p_i_bi[None, :]
            * p_j_bj
            * ext_branch_weights[None, :]
        )
        delta_bar_E = e_i * s_table[2] * jnp.sum(weighted, axis=1)
        bar_E = bar_E.at[js + 1].add(delta_bar_E)

    # Multibranch helper (MB) outside propagation.
    for i in range(seq_len):
        k_range = jnp.arange(i, seq_len + 1)
        if k_range.size == 0:
            continue

        p_i_bi = padded_p_seq[i, bi]
        p_k_bj = padded_p_seq[k_range[:, None], bj[None, :]]
        weights = (
            p_i_bi[None, :]
            * p_k_bj
            * multi_branch_weights[None, :]
            * s_table[2]
        )
        MB_slice = MB[i, k_range]

        # Avoid work if bar_MB zero across range.
        bar_MB_row = bar_MB[i, k_range]
        if float(jnp.sum(bar_MB_row)) == 0.0:
            continue

        bar_P_updates = bar_MB_row[:, None] * weights
        bar_P = bar_P.at[:, i, k_range].add(bar_P_updates.T)

    # Multibranch main outside propagation (ML).
    for i in range(seq_len - 1, -1, -1):
        for nb in range(3):
            j_range = jnp.arange(i, seq_len)
            if j_range.size == 0:
                continue
            for j in j_range:
                bar_val = float(bar_M[nb, i, j])
                if bar_val == 0.0:
                    continue

                # Propagate through unpaired extension.
                if i + 1 <= seq_len:
                    bar_M = bar_M.at[nb, i + 1, j].add(bar_val * s_table[1])

                idx = nb - 1 if nb - 1 > 0 else 0

                k_start = i
                k_stop = j + 1
                if k_start >= k_stop:
                    continue

                k_vals = jnp.arange(k_start, k_stop)
                MB_vals = MB[i, k_vals]
                ML_vals = ML[idx, k_vals + 1, j]

                bar_M = bar_M.at[idx, k_vals + 1, j].add(bar_val * MB_vals)
                bar_MB = bar_MB.at[i, k_vals].add(bar_val * ML_vals)

    # Contributions from stacked pairs and multibranch closing terms in P recursion.
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            outer_bar = bar_P[:, i, j]
            if float(jnp.sum(outer_bar)) == 0.0:
                continue

            span = j - i
            if span >= 2:
                k = i + 1
                l = j - 1

                p_k_bi = padded_p_seq[k, bi]
                p_l_bj = padded_p_seq[l, bj]

                coeff = (
                    outer_bar[:, None]
                    * s_table[2]
                    * stack_weights
                    * p_k_bi[None, :]
                    * p_l_bj[None, :]
                )
                delta_inner = jnp.sum(coeff, axis=0)
                bar_P = bar_P.at[:, k, l].add(delta_inner)

                delta_ml = jnp.dot(outer_bar, multi_closing_weights)
                bar_M = bar_M.at[2, k, l].add(delta_ml)

    # Bulge and internal loop outside propagation.
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            if j - i < 2:
                continue

            for bp_idx in range(NBPS):
                outer = float(bar_P[bp_idx, i, j])
                if outer == 0.0:
                    continue

                bi_pair = int(bi[bp_idx])
                bj_pair = int(bj[bp_idx])
                bk = bi_pair
                bl = bj_pair

                # Bulges (right and left).
                for kl_offset in range(two_loop_length):
                    l = j - 2 - kl_offset
                    if l < i + 2:
                        break
                    bulge_len = j - l - 1
                    weight = (
                        padded_p_seq[i + 1, bk]
                        * padded_p_seq[l, bl]
                        * model.en_bulge(bi_pair, bj_pair, bk, bl, bulge_len)
                        * s_table[bulge_len + 2]
                    )
                    bar_P = bar_P.at[bp_idx, i + 1, l].add(outer * weight)

                for kl_offset in range(two_loop_length):
                    k = i + 2 + kl_offset
                    if k >= j - 1:
                        break
                    bulge_len = k - i - 1
                    weight = (
                        padded_p_seq[k, bk]
                        * padded_p_seq[j - 1, bl]
                        * model.en_bulge(bi_pair, bj_pair, bk, bl, bulge_len)
                        * s_table[bulge_len + 2]
                    )
                    bar_P = bar_P.at[bp_idx, k, j - 1].add(outer * weight)

                # Precompute mismatch term mmij.
                mmij = 0.0
                for bip1 in range(4):
                    pr_ip1 = float(padded_p_seq[i + 1, bip1])
                    if pr_ip1 == 0.0:
                        continue
                    for bjm1 in range(4):
                        pr_jm1 = float(padded_p_seq[j - 1, bjm1])
                        if pr_jm1 == 0.0:
                            continue
                        mmij += (
                            pr_ip1
                            * pr_jm1
                            * model.en_il_inner_mismatch(bi_pair, bj_pair, bip1, bjm1)
                        )

                # 1x1 internal loop.
                if i + 2 < j - 1:
                    for bip1 in range(4):
                        pr_ip1 = float(padded_p_seq[i + 1, bip1])
                        if pr_ip1 == 0.0:
                            continue
                        for bjm1 in range(4):
                            pr_jm1 = float(padded_p_seq[j - 1, bjm1])
                            if pr_jm1 == 0.0:
                                continue

                            pr_ij_mm = pr_ip1 * pr_jm1
                            k_inner = i + 2
                            l_inner = j - 2
                            weight_11 = (
                                padded_p_seq[k_inner, bk]
                                * padded_p_seq[l_inner, bl]
                                * pr_ij_mm
                                * model.en_internal(
                                    bi_pair,
                                    bj_pair,
                                    bk,
                                    bl,
                                    bip1,
                                    bjm1,
                                    bip1,
                                    bjm1,
                                    1,
                                    1,
                                )
                                * s_table[4]
                            )
                            bar_P = bar_P.at[bp_idx, k_inner, l_inner].add(
                                outer * weight_11
                            )

                            # 1xn and nx1 cases.
                            for z_offset in range(two_loop_length):
                                l_right = j - 3 - z_offset
                                if l_right < i + 3:
                                    break
                                bulge_r = j - l_right - 1
                                for b in range(4):
                                    pr_b = float(padded_p_seq[l_right + 1, b])
                                    if pr_b == 0.0:
                                        continue
                                    weight_r = (
                                        padded_p_seq[i + 2, bk]
                                        * padded_p_seq[l_right, bl]
                                        * pr_b
                                        * pr_ij_mm
                                        * model.en_internal(
                                            bi_pair,
                                            bj_pair,
                                            bk,
                                            bl,
                                            bip1,
                                            bjm1,
                                            bip1,
                                            b,
                                            1,
                                            bulge_r,
                                        )
                                        * s_table[bulge_r + 3]
                                    )
                                    bar_P = bar_P.at[bp_idx, i + 2, l_right].add(
                                        outer * weight_r
                                    )

                            for z_offset in range(two_loop_length):
                                k_left = i + 3 + z_offset
                                if k_left >= j - 2:
                                    break
                                bulge_l = k_left - i - 1
                                for b in range(4):
                                    pr_b = float(padded_p_seq[k_left - 1, b])
                                    if pr_b == 0.0:
                                        continue
                                    weight_l = (
                                        padded_p_seq[k_left, bk]
                                        * padded_p_seq[j - 2, bl]
                                        * pr_b
                                        * pr_ij_mm
                                        * model.en_internal(
                                            bi_pair,
                                            bj_pair,
                                            bk,
                                            bl,
                                            bip1,
                                            bjm1,
                                            b,
                                            bjm1,
                                            bulge_l,
                                            1,
                                        )
                                        * s_table[bulge_l + 3]
                                    )
                                    bar_P = bar_P.at[bp_idx, k_left, j - 2].add(
                                        outer * weight_l
                                    )

                # Special 2x2, 2x3, 3x2 loops.
                for k_offset in range(min(3, two_loop_length)):
                    k_inner = i + k_offset + 2
                    if k_inner >= j - 1:
                        break
                    for l_offset in range(min(3, two_loop_length)):
                        l_inner = j - l_offset - 2
                        if l_inner <= k_inner:
                            continue
                        lup = k_inner - i - 1
                        rup = j - l_inner - 1
                        if (lup, rup) not in {(2, 2), (2, 3), (3, 2)}:
                            continue

                        for bip1 in range(4):
                            pr_ip1 = float(padded_p_seq[i + 1, bip1])
                            if pr_ip1 == 0.0:
                                continue
                            for bjm1 in range(4):
                                pr_jm1 = float(padded_p_seq[j - 1, bjm1])
                                if pr_jm1 == 0.0:
                                    continue
                                for bkm1 in range(4):
                                    pr_km1 = float(padded_p_seq[k_inner - 1, bkm1])
                                    if pr_km1 == 0.0:
                                        continue
                                    for blp1 in range(4):
                                        pr_lp1 = float(padded_p_seq[l_inner + 1, blp1])
                                        if pr_lp1 == 0.0:
                                            continue
                                        weight_special = (
                                            padded_p_seq[k_inner, bk]
                                            * padded_p_seq[l_inner, bl]
                                            * pr_ip1
                                            * pr_jm1
                                            * pr_km1
                                            * pr_lp1
                                            * model.en_internal(
                                                bi_pair,
                                                bj_pair,
                                                bk,
                                                bl,
                                                bip1,
                                                bjm1,
                                                bkm1,
                                                blp1,
                                                lup,
                                                rup,
                                            )
                                            * s_table[lup + rup + 2]
                                        )
                                        bar_P = bar_P.at[bp_idx, k_inner, l_inner].add(
                                            outer * weight_special
                                        )

                # General loops via OMM.
                for k_offset in range(two_loop_length):
                    k_inner = i + k_offset + 2
                    if k_inner >= j - 1:
                        break
                    for l_offset in range(two_loop_length):
                        l_inner = j - l_offset - 2
                        if l_inner <= k_inner:
                            continue

                        lup = k_inner - i - 1
                        rup = j - l_inner - 1

                        if lup <= 1 or rup <= 1:
                            continue
                        if (lup, rup) in {(2, 2), (2, 3), (3, 2)}:
                            continue

                        coeff = (
                            mmij
                            * model.en_internal_init(lup + rup)
                            * model.en_internal_asym(lup, rup)
                            * s_table[lup + rup + 2]
                        )
                        bar_OMM = bar_OMM.at[k_inner, l_inner].add(outer * coeff)

    # Propagate bar_OMM back to bar_P using outer mismatch definition.
    for i in range(1, seq_len):
        for j in range(i, seq_len):
            bar_val = float(bar_OMM[i, j])
            if bar_val == 0.0:
                continue

            for bp_idx in range(NBPS):
                bi_pair = int(bi[bp_idx])
                bj_pair = int(bj[bp_idx])

                for a in range(4):
                    pr_im1 = float(padded_p_seq[i - 1, a]) if i - 1 >= 0 else 0.0
                    if pr_im1 == 0.0:
                        continue
                    for b in range(4):
                        pr_jp1 = float(padded_p_seq[j + 1, b]) if j + 1 <= seq_len else 0.0
                        if pr_jp1 == 0.0:
                            continue
                        weight = (
                            bar_val
                            * padded_p_seq[i, bi_pair]
                            * padded_p_seq[j, bj_pair]
                            * pr_im1
                            * pr_jp1
                            * model.en_il_outer_mismatch(bi_pair, bj_pair, a, b)
                        )
                        bar_P = bar_P.at[bp_idx, i, j].add(weight)

    tables = OutsideComputation(
        bar_E=bar_E,
        bar_P=bar_P,
        bar_M=bar_M,
        bar_MB=bar_MB,
        bar_OMM=bar_OMM,
    )
    return tables
