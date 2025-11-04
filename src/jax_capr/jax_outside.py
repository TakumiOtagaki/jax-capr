"""Scaffolding for JAX-based outside dynamic programming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Callable
import functools

import jax
from jax import vmap, jit, lax
import jax.numpy as jnp


from jax_rnafold.d0 import energy
from jax_rnafold.common.utils import bp_bases, N4, NBPS, MAX_LOOP
from jax_rnafold.common.checkpoint import checkpoint_scan

from .jax_inside import InsideComputation

jax.config.update("jax_enable_x64", True)

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


@dataclass(slots=True)
class OutsideCarry:
    """State threaded through the span-descending outside scan."""

    bar_E: Array
    bar_P: Array
    bar_M: Array
    bar_MB: Array
    bar_OMM: Array
    bar_Pm: Array
    bar_Pm1: Array


def get_outside_partition_fn(em: energy.Model, seq_len: int, inside: InsideComputation, 
                             max_loop: int = MAX_LOOP,
                             checkpoint_every: int = 10) -> Callable:
    if checkpoint_every is None:
        scan = lax.scan
    else:
        scan = functools.partial(checkpoint_scan, checkpoint_every=checkpoint_every)

    two_loop_length = min(seq_len, max_loop)
    s_table = inside.s_table


    special_hairpin_lens = em.special_hairpin_lens
    max_sp_hairpin_len_up = special_hairpin_lens.max() - 2 # Subtract 2 for the paired nt
    special_hairpin_idxs = em.special_hairpin_idxs
    special_hairpin_start_pos = em.special_hairpin_start_pos
    n_special_hairpins = em.n_special_hairpins


    @jit
    def fill_bar_E(bar_E, E, P, p_seq, em, n):
        def body(i, carry):
            # i は 2..n
            # sum_{j=0..i-2} bar_xi[j] * P[j, i-1] = dot(bar_xi[:i-1], P[:i-1, i-1])
            # i==1 のときは空スライスの内積=0.0 になるのでそのままでOK
            contrib = jnp.dot(carry[:i-1], P[:i-1, i-1])
            incr = E[i-1] + contrib
            return carry.at[i].add(incr)

        # 1 から n まで JAX 制御フローで回す（Python ループは使わない）
        bar_xi_out = lax.fori_loop(2, n+1, body, bar_E)
        return bar_xi_out


    def outside_partition(p_seq: Array, inside: InsideComputation) -> tuple[Array, Array, Array]:
        seq_len = inside.E.shape[0]
        bar_E = jnp.zeros(seq_len, dtype=inside.E.dtype)
        bar_E.at[1].set(1.0)  # base case: bar_E[0] = 1 in 1-based indexing
        bar_P = jnp.zeros_like(inside.P)
        bar_M = jnp.zeros_like(inside.ML)
        bar_MB = jnp.zeros_like(inside.MB)
        bar_OMM = jnp.zeros_like(inside.OMM)
        bar_Pm = jnp.zeros_like(inside.P)
        bar_Pm1 = jnp.zeros_like(inside.P)

        # first off, fill bar_E.
        bar_E = fill_bar_E(bar_E, inside.E, inside.P, p_seq, em, seq_len)

        # filling the other tables
        def fill_tables(carry, i):
            bar_OMM, bar_P, bar_M, bar_MB, bar_E, bar_Pm, bar_Pm1 = carry

            bar_P = fill_bar_Ps(carry, inside, em, i)
            bar_OMM = fill_bar_OMM_from_P(carry, inside, em, i)
            bar_MB = fill_bar_MB(carry, inside, i)
            bar_M = fill_bar_M(carry, inside, i)

            return (bar_OMM, bar_P, bar_M, bar_MB, bar_E, bar_Pm, bar_Pm1), None
        (bar_OMM, bar_P, bar_M, bar_MB, bar_E, bar_Pm, bar_Pm1), _ = scan(fill_tables,
                                        (bar_OMM, bar_P, bar_M, bar_MB, bar_E, bar_Pm, bar_Pm1),
                                        jnp.arange(0, seq_len))
        return (bar_P, bar_M, bar_E)

    return outside_partition


def fill_bar_MB(carry: OutsideCarry, inside: InsideTablesLike, i: int) -> Array:
    """Propagate multibranch helper contributions at position i."""

    raise NotImplementedError


def fill_bar_M(carry: OutsideCarry, inside: InsideTablesLike, i: int) -> Array:
    """Propagate multibranch DP contributions at position i."""

    raise NotImplementedError


def fill_bar_Ps(
    carry: OutsideCarry,
    inside: InsideTablesLike,
    model,
    i: int,
) -> Array:
    """Propagate paired-state outside weights for span starting at i."""

    raise NotImplementedError


def fill_bar_OMM_from_P(
    carry: OutsideCarry,
    inside: InsideTablesLike,
    model,
    i: int,
) -> Array:
    """Accumulate general internal-loop contributions into bar_OMM."""

    raise NotImplementedError


