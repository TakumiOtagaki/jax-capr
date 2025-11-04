"""Scaffolding for JAX-based outside dynamic programming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Callable, cast
import functools

import jax
from jax import vmap, jit, lax
import jax.numpy as jnp


from jax_rnafold.d0 import energy
from jax_rnafold.common.utils import bp_bases, N4, NBPS, MAX_LOOP
from jax_rnafold.common.checkpoint import checkpoint_scan

from .jax_inside import InsideComputation

jax.config.update("jax_enable_x64", True)
f64 = jnp.float64
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
    def fill_bar_E(bar_E, bar_P, padded_p_seq, em, n):
        def body(i, current_bar_E):
            def get_j_bp_term(j, bp_idx):
                cond = (j < i - 1)
                bp = bp_bases[bp_idx]
                bj = bp[0]
                bim1 = bp[1]
                base_en = current_bar_E[j] * padded_p_seq[j, bj] * padded_p_seq[i-1, bim1]
                return jnp.where(cond, base_en * bar_P[bp_idx, j, i-1] * em.en_ext_branch(bj, bim1), 0.0)
            get_all_terms = vmap(vmap(get_j_bp_term, (0, None)), (None, 0))
            terms = cast(Array, get_all_terms(jnp.arange(n), jnp.arange(NBPS)))
            sm = s_table[1] * current_bar_E[i-1] + jnp.sum(terms)
            updated_bar_E = current_bar_E.at[i].set(sm)
            return updated_bar_E, None

        # 2 から n まで JAX 制御フローで回す（Python ループは使わない）
        bar_xi_out, _ = scan(body, bar_E, jnp.arange(2, n+1))
        return bar_xi_out


    def fill_bar_Ps(
        carry: OutsideCarry,
        inside: InsideTablesLike,
        model,
        i: int,
    ) -> Array:
        """Propagate paired-state outside weights for span starting at i."""

        raise NotImplementedError

    def fill_bar_MB(carry: OutsideCarry, inside: InsideTablesLike, i: int) -> Array:
        """Propagate multibranch helper contributions at position i."""

        raise NotImplementedError


    def fill_bar_M(carry: OutsideCarry, inside: InsideTablesLike, i: int) -> Array:
        """Propagate multibranch DP contributions at position i."""

        raise NotImplementedError


    def fill_bar_OMM(
        carry: OutsideCarry,
        inside: InsideTablesLike,
        model,
        i: int,
    ) -> Array:
        """Accumulate general internal-loop contributions into bar_OMM."""

        raise NotImplementedError


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

        padded_p_seq = jnp.zeros((seq_len+1, 4), dtype=f64)
        padded_p_seq = padded_p_seq.at[:seq_len].set(p_seq)

        # first off, fill bar_E.
        bar_E = fill_bar_E(bar_E, inside.P, padded_p_seq, em, seq_len)

        # filling the other tables
        def fill_tables(carry, i):
            bar_OMM, bar_P, bar_M, bar_MB, bar_E, bar_Pm, bar_Pm1 = carry

            bar_P = fill_bar_Ps(carry, inside, em, i)
            bar_OMM = fill_bar_OMM(carry, inside, em, i)
            bar_MB = fill_bar_MB(carry, inside, i)
            bar_M = fill_bar_M(carry, inside, i)

            return (bar_OMM, bar_P, bar_M, bar_MB, bar_E, bar_Pm, bar_Pm1), None
        (bar_OMM, bar_P, bar_M, bar_MB, bar_E, bar_Pm, bar_Pm1), _ = scan(fill_tables,
                                        (bar_OMM, bar_P, bar_M, bar_MB, bar_E, bar_Pm, bar_Pm1),
                                        jnp.arange(0, seq_len))
        return (bar_P, bar_M, bar_E)

    return outside_partition



