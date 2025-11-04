"""Scaffolding for JAX-based outside dynamic programming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax.numpy as jnp

from jax_rnafold.common.utils import NBPS, bp_bases


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

    tables = OutsideComputation(
        bar_E=bar_E,
        bar_P=bar_P,
        bar_M=bar_M,
        bar_MB=bar_MB,
        bar_OMM=tables.bar_OMM,
    )
    return tables
