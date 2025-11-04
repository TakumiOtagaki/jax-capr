"""Scaffolding for JAX-based outside dynamic programming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from jax import lax
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


def fill_bar_xi(
    bar_E: Array,
    inside_E: Array,
    inside_P: Array,
    padded_p_seq: Array,
    bi: Array,
    bj: Array,
    ext_branch_weights: Array,
    s_table: Array,
) -> Array:
    """Update `bar_E` (xi outside) without touching `bar_P`.

    Follows the exterior recurrence in the pseudocode: only the xi outside
    weights are updated here. Contributions to `bar_P` are handled later in the
    span-descending phase.
    """

    seq_len = padded_p_seq.shape[0] - 1
    js = jnp.arange(seq_len)
    p_j_bj = padded_p_seq[js[:, None], bj[None, :]]

    def body(bar_E_state, i):
        e_i = bar_E_state[i]

        def update(carry):
            bar_E_local = carry
            p_i_bi = padded_p_seq[i, bi]
            P_slice = inside_P[:, i, :].T

            mask = (js < i - 2).astype(bar_E_local.dtype) # TODO: i - 1 か i - 2 か。
            weighted = (
                P_slice
                * p_i_bi[None, :]
                * p_j_bj
                * ext_branch_weights[None, :]
            )
            delta_bar_E = e_i * s_table[2] * jnp.sum(weighted, axis=1) * mask

            bar_E_local = bar_E_local.at[i + 1].add(e_i * s_table[1])
            bar_E_local = bar_E_local.at[js + 1].add(delta_bar_E)
            return bar_E_local

        bar_E_next = lax.cond(
            e_i != 0.0,
            update,
            lambda carry: carry,
            bar_E_state,
        )
        return bar_E_next, None

    bar_E_final, _ = lax.scan(body, bar_E, jnp.arange(seq_len, dtype=jnp.int32))
    return bar_E_final

def compute_outside_tables(): # これから implement
    tables = OutsideComputation(
        bar_E=bar_E,
        bar_P=bar_P,
        bar_M=bar_M,
        bar_MB=bar_MB,
        bar_OMM=bar_OMM,
    )
    return tables
