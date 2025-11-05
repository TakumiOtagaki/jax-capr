"""Scaffolding for JAX-based outside dynamic programming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Protocol, Tuple
import functools
import weakref

import jax
from jax import vmap, jit, lax
import jax.numpy as jnp

# (既存の import に加えて)
# from jax_rnafold.common.utils import int_inf
from jax_rnafold.d0 import energy
from jax_rnafold.common.utils import bp_bases, N4, NBPS, MAX_LOOP
from jax_rnafold.common.checkpoint import checkpoint_scan

jax.config.update("jax_enable_x64", True)
f64 = jnp.float64
Array = jnp.ndarray


# Cache compiled outside kernels keyed by model identity and static arguments.
CacheKey = Tuple[int, int, int, int | None]
_OUTSIDE_KERNEL_CACHE: Dict[CacheKey, Tuple[Callable, weakref.ReferenceType[energy.Model]]] = {}


def _as_int(x: Array) -> Array:
    """Convert tracer-friendly scalar to int32 for indexing."""
    return lax.convert_element_type(x, jnp.int32)

class InsideTablesLike(Protocol):
    """Protocol capturing the inside tables required for outside DP."""

    E: Array
    P: Array
    ML: Array
    p_seq: Array
    s_table: Array
    scale: float


@dataclass(slots=True)
class OutsideComputation:
    """Container for outside DP tables."""

    bar_E: Array
    bar_P: Array
    bar_M: Array


@dataclass(slots=True)
class OutsideCarry:
    """State threaded through the span-descending outside scan."""

    bar_E: Array
    bar_P: Array
    bar_M: Array
    bar_Pm: Array
    bar_Pm1: Array


def _construct_outside_partition_fn(
    em: energy.Model,
    seq_len: int,
    *,
    max_loop: int = MAX_LOOP,
    checkpoint_every: int | None = 10,
) -> Callable[[Array, Array, Array, Array, Array], tuple[Array, Array, Array]]:
    if checkpoint_every is None:
        scan = lax.scan
    else:
        scan = functools.partial(checkpoint_scan, checkpoint_every=checkpoint_every)

    two_loop_length = min(seq_len, max_loop)

    @jit
    def fill_bar_E(bar_E, P, padded_p_seq, s_table):
        def body(current_bar_E, i):
            def get_j_bp_term(j, bp_idx):
                cond = (j < i - 1)
                bp = bp_bases[bp_idx]
                bj = _as_int(bp[0])
                bim1 = _as_int(bp[1])
                base_en = current_bar_E[j] * padded_p_seq[j, bj] * padded_p_seq[i - 1, bim1]
                return jnp.where(
                    cond,
                    base_en * P[bp_idx, j, i - 1] * em.en_ext_branch(bj, bim1),
                    0.0,
                )

            get_all_terms = vmap(vmap(get_j_bp_term, (0, None)), (None, 0))
            terms = get_all_terms(jnp.arange(seq_len + 1), jnp.arange(NBPS))
            sm = s_table[1] * current_bar_E[i - 1] + jnp.sum(terms)
            updated_bar_E = current_bar_E.at[i].set(sm)
            return updated_bar_E, None

        bar_xi_out, _ = scan(body, bar_E, jnp.arange(1, seq_len + 1))
        return bar_xi_out

    @jit
    def psum_outer_bulges(bh, bl, h, l, padded_p_seq, bar_P, s_table):
        def get_bp_ij(bp_idx_ij, ij_offset):
            bp = bp_bases[bp_idx_ij]
            bi = _as_int(bp[0])
            bj = _as_int(bp[1])
            bp_ij_sm = jnp.zeros((), dtype=bar_P.dtype)

            # Right bulge, note i = h - 1
            j = l + 2 + ij_offset
            i = h - 1
            cond_ij = (j < seq_len + 1) & (0 <= i)
            right_cond = h < l
            bp_ij_sm += jnp.where(
                right_cond & cond_ij,
                bar_P[bp_idx_ij, i, j]
                * padded_p_seq[i, bi]
                * padded_p_seq[j, bj]
                * em.en_bulge(bi, bj, bh, bl, j - l - 1)
                * s_table[j - l + 1],
                0.0,
            )

            # Left bulge, note j = l + 1
            i = h - 2 - ij_offset
            j = l + 1
            cond_ij = (j < seq_len + 1) & (0 <= i)
            left_cond = h < l
            bp_ij_sm += jnp.where(
                left_cond & cond_ij,
                bar_P[bp_idx_ij, i, j]
                * padded_p_seq[i, bi]
                * padded_p_seq[j, bj]
                * em.en_bulge(bi, bj, bh, bl, h - i - 1)
                * s_table[h - i + 1],
                0.0,
            )

            return bp_ij_sm

        def get_bp_all_ij(bp_idx):
            all_ij_offsets = jnp.arange(two_loop_length)
            all_bp_ij_sms = vmap(get_bp_ij, (None, 0))(bp_idx, all_ij_offsets)
            return jnp.sum(all_bp_ij_sms)

        all_bp_sms = vmap(get_bp_all_ij)(jnp.arange(NBPS))
        return jnp.sum(all_bp_sms)

    @jit
    def psum_outer_internal_loops(
        bp_idx_hl: int,
        h: int,
        l: int,
        padded_p_seq: Array,
        bar_P: Array,
        s_table: Array,
    ) -> Array:
        """
        psum_internal_loops (inside) の逆写像（outside）を計算する。

        (h, l) [内側ペア] に対する寄与を、すべての (i, j) [外側ペア] から計算する。
        i, j は lup (h - i - 1) と rup (j - l - 1) によって決定される。

        ネスト順序: (lup, rup) -> (bp_idx_ij) -> (bip1, bjm1)
        """

        max_lup_rup = two_loop_length - 2
        lup_offsets = jnp.arange(max_lup_rup)
        rup_offsets = jnp.arange(max_lup_rup)

        bp_hl = bp_bases[bp_idx_hl]
        bh = _as_int(bp_hl[0])
        bl = _as_int(bp_hl[1])

        valid_outer = (h > 0) & (h < seq_len + 1) & (l + 1 < seq_len + 1)

        def compute_outer_mismatch(_):
            def row_mismatch(bim1):
                def col_mismatch(bjp1):
                    return (
                        padded_p_seq[h - 1, bim1]
                        * padded_p_seq[l + 1, bjp1]
                        * em.en_il_outer_mismatch(bh, bl, bim1, bjp1)
                    )

                return jnp.sum(vmap(col_mismatch)(N4))

            mismatch_sum = jnp.sum(vmap(row_mismatch)(N4))
            return padded_p_seq[h, bh] * padded_p_seq[l, bl] * mismatch_sum

        outer_mismatch_factor = lax.cond(
            valid_outer,
            compute_outer_mismatch,
            lambda _: 0.0,
            operand=None,
        )

        @jit
        def get_bp_idx_ij_hoff_loff_term(bp_idx_ij, lup_offset, rup_offset):
            lup = lup_offset + 1
            rup = rup_offset + 1
            i = h - lup - 1
            j = l + rup + 1
            ij_cond = (j < seq_len + 1) & (0 <= i)
            len_cond = (lup + rup + 2 <= two_loop_length)
            valid_ij = ij_cond & len_cond

            bp = bp_bases[bp_idx_ij]
            bi = _as_int(bp[0])
            bj = _as_int(bp[1])

            def get_mmij_term(bip1, bjm1):
                return (
                    padded_p_seq[i + 1, bip1]
                    * padded_p_seq[j - 1, bjm1]
                    * em.en_il_inner_mismatch(bi, bj, bip1, bjm1)
                )

            mmij_terms = vmap(vmap(get_mmij_term, (0, None)), (None, 0))(N4, N4)
            mmij = jnp.sum(mmij_terms)

            sm = 0.0

            @jit
            def get_bp_1n_sm(bip1, bjm1):
                bp_1n_sm = 0.0
                cond_11 = (lup == 1) | (rup == 1)

                pr_ij_mm = padded_p_seq[i + 1, bip1] * padded_p_seq[j - 1, bjm1]
                bp_1n_sm += jnp.where(
                    cond_11,
                    bar_P[bp_idx_ij, i, j]
                    * padded_p_seq[i, bi]
                    * padded_p_seq[j, bj]
                    * pr_ij_mm
                    * em.en_internal(bi, bj, bh, bl, bip1, bjm1, bip1, bjm1, 1, 1)
                    * s_table[4],
                    0.0,
                )

                def z_b_fn(b):
                    z_b_sm = 0.0
                    cond_1N = (h == i + 2) & (2 < j - l)

                    il_en = em.en_internal(bi, bj, bh, bl, bip1, bjm1, bip1, b, 1, j - l - 1)

                    right_term = (
                        bar_P[bp_idx_ij, i, j]
                        * padded_p_seq[i, bi]
                        * padded_p_seq[j, bj]
                        * padded_p_seq[l + 1, b]
                        * pr_ij_mm
                        * il_en
                        * s_table[j - l + 2]
                    )
                    z_b_sm += jnp.where(cond_1N, right_term, 0.0)

                    cond_N1 = (2 < h - i) & (j == l + 2)
                    il_en = em.en_internal(bi, bj, bh, bl, bip1, bjm1, b, bjm1, h - i - 1, 1)

                    left_term = (
                        bar_P[bp_idx_ij, i, j]
                        * padded_p_seq[h, bh]
                        * padded_p_seq[j - 2, bl]
                        * padded_p_seq[h - 1, b]
                        * pr_ij_mm
                        * il_en
                        * s_table[h - i + 2]
                    )
                    z_b_sm += jnp.where(cond_N1, left_term, 0.0)

                    return z_b_sm

                get_all_zb_terms = vmap(z_b_fn)(N4)

                bp_1n_sm += jnp.sum(get_all_zb_terms)
                return bp_1n_sm

            get_all_1n_terms = vmap(vmap(get_bp_1n_sm, (0, None)), (None, 0))
            sm += jnp.sum(get_all_1n_terms(N4, N4))

            def get_bp_22_23_32_summand(bip1, bjm1, bhm1, blp1):
                cond_lup_rup = ((lup == 2) & (rup == 2)) | ((lup == 2) & (rup == 3)) | ((lup == 3) & (rup == 2))
                cond_idx = (h < j - 2) & (l >= h + 1)
                cond_22_23_32 = cond_lup_rup & cond_idx
                return jnp.where(
                    cond_22_23_32,
                    bar_P[bp_idx_ij, i, j]
                    * padded_p_seq[i, bi]
                    * padded_p_seq[j, bj]
                    * em.en_internal(bi, bj, bh, bl, bip1, bjm1, bhm1, blp1, lup, rup)
                    * padded_p_seq[h - 1, bhm1]
                    * padded_p_seq[l + 1, blp1]
                    * padded_p_seq[i + 1, bip1]
                    * padded_p_seq[j - 1, bjm1]
                    * s_table[lup + rup + 2],
                    0.0,
                )

            get_all_summands = vmap(get_bp_22_23_32_summand, (None, None, None, 0))
            get_all_summands = vmap(get_all_summands, (None, None, 0, None))
            get_all_summands = vmap(get_all_summands, (None, 0, None, None))
            get_all_summands = vmap(get_all_summands, (0, None, None, None))

            sm += jnp.sum(get_all_summands(N4, N4, N4, N4))

            idx_cond = h < l
            is_not_n1 = (lup > 1) & (rup > 1)
            is_22_23_32 = ((lup == 2) & (rup == 2)) | ((lup == 2) & (rup == 3)) | ((lup == 3) & (rup == 2))
            cond_general_il = idx_cond & is_not_n1 & ~is_22_23_32

            general_term = (
                em.en_internal_init(lup + rup)
                * em.en_internal_asym(lup, rup)
                * mmij
                * s_table[lup + rup + 2]
                * bar_P[bp_idx_ij, i, j]
                * outer_mismatch_factor
            )

            sm += jnp.where(cond_general_il, general_term, 0.0)

            return jnp.where(valid_ij, sm, 0.0)

        get_all_bp_idx_ij = vmap(get_bp_idx_ij_hoff_loff_term, (0, None, None))
        get_all_bp_idx_ij = vmap(get_all_bp_idx_ij, (None, 0, None))
        get_all_bp_idx_ij = vmap(get_all_bp_idx_ij, (None, None, 0))
        all_terms = get_all_bp_idx_ij(jnp.arange(NBPS), lup_offsets, rup_offsets)
        return jnp.sum(all_terms)

    def fill_bar_P(
            d: int,
            padded_p_seq: Array,
            ML: Array,
            E: Array,
            bar_P: Array,
            bar_Pm: Array,
            bar_Pm1: Array,
            bar_E: Array,
            s_table: Array,
        ) -> Array:
        """Propagate paired-state outside weights for span starting at i."""

        def get_bp_stack(bp_idx_ij, l, bh, bl):
            h = l - d
            i = h - 1
            j = l + 1
            cond = (0 <= i) & (j <= seq_len)
            bp = bp_bases[bp_idx_ij]
            bi = bp[0]
            bj = bp[1]
            return jnp.where(
                cond,
                bar_P[bp_idx_ij, i, j]
                * padded_p_seq[i, bi]
                * padded_p_seq[j, bj]
                * em.en_stack(bi, bj, bh, bl),
                0.0,
            )

        def get_bp_l_multi_sm(l):
            h = l - d

            def get_multi_i_term(i):
                cond = i + 1 < h - 1
                return jnp.where(
                    cond,
                    (
                        s_table[1] * ML[1, i + 1, h - 1] * bar_Pm1[i, l]
                        + bar_Pm[i, l]
                        * (
                            s_table[1] * ML[1, i + 1, h - 1]
                            + lax.pow(
                                em.en_multi_unpaired(),
                                (h - i - 1))
                            ) * s_table[h - i]
                    ),
                    0.0,
                )

            all_i_terms = vmap(get_multi_i_term)(jnp.arange(seq_len + 1))
            return jnp.sum(jnp.asarray(all_i_terms))

        def get_bp_l_sm(bp_idx_hl, l):
            h = l - d
            bp = bp_bases[bp_idx_hl]
            bh = bp[0]
            bl = bp[1]
            sm = jnp.zeros((), dtype=bar_P.dtype)

            sm += psum_outer_bulges(bh, bl, h, l, padded_p_seq, bar_P, s_table)
            sm += psum_outer_internal_loops(bp_idx_hl, h, l, padded_p_seq, bar_P, s_table)

            stack_summands = vmap(get_bp_stack, (0, None, None, None))(jnp.arange(NBPS), l, bh, bl)
            sm += jnp.sum(stack_summands) * s_table[2]

            sm += get_bp_l_multi_sm(l)

            # TODO: 以下の式で padded_p_seq[h, bh] * padded_p_seq[l, bl] を残すのかどうか検討
            sm += bar_E[h] * E[l + 1] * em.en_ext_branch(bh, bl) * padded_p_seq[h, bh] * padded_p_seq[l, bl]

            cond_valid = (0 < h) & (l < seq_len)
            return jnp.where(cond_valid, sm, 0.0)

        get_bp_all_ls = vmap(vmap(get_bp_l_sm, (None, 0)), (0, None))

        all_bp_ls = get_bp_all_ls(jnp.arange(NBPS), jnp.arange(seq_len + 1))
        h_indices = jnp.arange(seq_len + 1)
        l_indices = h_indices + d
        updated_bar_P = bar_P.at[:, h_indices, l_indices].set(all_bp_ls, mode='drop')

        return updated_bar_P

    def fill_bar_M(
        d: int,
        bar_M: Array,
        bar_P: Array,
        P: Array,
        padded_p_seq: Array,
        s_table: Array,
    ) -> Array:
        r"""
        以下を全ての h, l (l - h = d) について計算する。
        bar_M(2, h, l) & := s(1) bar_M(2, h-1, l) B(M_u) + s(2) bar_P(h - 1, l + 1) em.en_multi_closing(bhm1, blp1) \\
        bar_M(1, h, l) & := s(1) bar_M(1, h-1, l) B(M_u)
        + \sum_{i < h-1}  P(i, h-1) em.en_multi_branch(bi, bhm1) * bar_M(2, i, l) \\
        bar_M(0, h, l) & := s(1) bar_M(0, h-1, l) B(M_u)
        + \sum_{i < h-1} P(i, h-1) em.en_multi_branch(bi, bhm1) * \left(
        bar_M(0, i, l) + bar_M(1, i, l)
        \right)
        """

        def accumulate_single_h(h):
            sm_M2 = 0.0
            sm_M1 = 0.0
            sm_M0 = 0.0

            l = h + d
            cond = (l < seq_len + 1)

            multi_unpaired_factor = s_table[1] * em.en_multi_unpaired()

            sm_M2 += bar_M[2, h - 1, l] * multi_unpaired_factor
            sm_M1 += bar_M[1, h - 1, l] * multi_unpaired_factor
            sm_M0 += bar_M[0, h - 1, l] * multi_unpaired_factor

            def get_bp_idx_hm1_lp1_term(bp_idx_hm1_lp1):
                bp_hm1_lp1 = bp_bases[bp_idx_hm1_lp1]
                hm1 = bp_hm1_lp1[0]
                lp1 = bp_hm1_lp1[1]
                return (
                    s_table[2]
                    * bar_P[bp_idx_hm1_lp1, h - 1, l + 1]
                    * padded_p_seq[h - 1, hm1]
                    * padded_p_seq[l + 1, lp1]
                    * em.en_multi_closing(hm1, lp1)
                )

            get_all_bp_terms = vmap(get_bp_idx_hm1_lp1_term)
            sm_M2 += jnp.sum(get_all_bp_terms(jnp.arange(NBPS)))

            def get_i_term(i):
                cond_i = i < h - 1
                ml_i_to_M1 = bar_M[2, i, l]
                ml_i_to_M0 = bar_M[0, i, l] + bar_M[1, i, l]

                def get_idx_bp_i_hm1(bp_idx_ihm1):
                    bp_ihm1 = bp_bases[bp_idx_ihm1]
                    bi = bp_ihm1[0]
                    bhm1 = bp_ihm1[1]
                    return (
                        P[bp_idx_ihm1, i, h - 1]
                        * em.en_multi_branch(bi, bhm1)
                        * padded_p_seq[i, bi]
                        * padded_p_seq[h - 1, bhm1]
                    )

                get_all_bp_i_hm1_terms = vmap(get_idx_bp_i_hm1)
                bp_sum_i = jnp.sum(get_all_bp_i_hm1_terms(jnp.arange(NBPS)))
                return (
                    jnp.where(cond_i, bp_sum_i * ml_i_to_M1, 0.0),
                    jnp.where(cond_i, bp_sum_i * ml_i_to_M0, 0.0),
                )

            m1_terms, m0_terms = vmap(get_i_term)(jnp.arange(seq_len + 1))
            sm_M1 += jnp.sum(m1_terms)
            sm_M0 += jnp.sum(m0_terms)

            return (
                jnp.where(cond, sm_M0, bar_M[0, h, l]),
                jnp.where(cond, sm_M1, bar_M[1, h, l]),
                jnp.where(cond, sm_M2, bar_M[2, h, l]),
            )

        h_indices = jnp.arange(seq_len + 1)
        l_indices = h_indices + d
        updates = vmap(accumulate_single_h)(h_indices)
        bar_M = bar_M.at[0, h_indices, l_indices].set(updates[0], mode="drop")
        bar_M = bar_M.at[1, h_indices, l_indices].set(updates[1], mode="drop")
        bar_M = bar_M.at[2, h_indices, l_indices].set(updates[2], mode="drop")
        return bar_M

    def fill_bar_Pm(
        d: int,
        padded_p_seq: Array,
        ML: Array,
        bar_P: Array,
        bar_Pm: Array,
        s_table: Array,
    ) -> Array:
        r"""
        Pm[i, l] = \sum_{j} (l < j) s_table[1] * em.en_multi_branch(bi, bl) * bar_P[i, j] * ML[1, l+1, j-1]
        を同じ対角線上の i,l 全てについて計算する: l - i = d
        """

        def accumulate_single_i(i):
            l = i + d

            def accumulate_single_j(j):
                cond_j = (l < j) & (j < seq_len + 1)

                def valid_branch(_):
                    ml_val = ML[1, l + 1, j - 1]
                    s1 = s_table[1]

                    def accumulate_bp(bp_idx):
                        bp_ij = bp_bases[bp_idx]
                        bi = bp_ij[0]
                        bj = bp_ij[1]
                        branch_penalty = em.en_multi_branch(bi, bj)
                        return (
                            s1
                            * bar_P[bp_idx, i, j]
                            * branch_penalty
                            * padded_p_seq[i, bi]
                            * padded_p_seq[j, bj]
                            * ml_val
                        )

                    return jnp.sum(vmap(accumulate_bp)(jnp.arange(NBPS)))

                return lax.cond(cond_j, valid_branch, lambda _: 0.0, operand=None)

            j_indices = jnp.arange(seq_len + 1)
            return jnp.sum(vmap(accumulate_single_j)(j_indices))

        i_indices = jnp.arange(seq_len + 1)
        l_indices = i_indices + d
        updates = vmap(accumulate_single_i)(i_indices)
        bar_Pm = bar_Pm.at[i_indices, l_indices].set(updates, mode="drop")
        return bar_Pm

    def fill_bar_Pm1(
        d: int,
        padded_p_seq: Array,
        bar_P: Array,
        bar_Pm1: Array,
        s_table: Array,
    ) -> Array:
        r"""
        Pm1[i, l] = \sum_{j} (l < j) (s_table[1] * em.en_multi_unpaired())**(j - l - 1) * s_table[1] * em.en_multi_branch(bi, bl) * bar_P[i, j]
        を同じ対角線上の i,l 全てについて計算する: l - i = d
        """

        base_multi_unpaired = s_table[1] * em.en_multi_unpaired()

        def accumulate_single_i(i):
            l = i + d

            def accumulate_single_j(j):
                cond_j = (l < j) & (j < seq_len)

                def valid_branch(_):
                    gap = j - l - 1
                    unpaired_factor = lax.pow(
                        base_multi_unpaired,
                        jnp.asarray(gap, dtype=bar_P.dtype),
                    ) * s_table[j - l]

                    def accumulate_bp(bp_idx_ij):
                        bp = bp_bases[bp_idx_ij]
                        bi = bp[0]
                        bj = bp[1]
                        branch_penalty = em.en_multi_branch(bi, bj)
                        return (
                            bar_P[bp_idx_ij, i, j]
                            * branch_penalty
                            * padded_p_seq[i, bi]
                            * padded_p_seq[j, bj]
                            * unpaired_factor
                        )

                    inner_sum = jnp.sum(vmap(accumulate_bp)(jnp.arange(NBPS)))
                    return s_table[1] * inner_sum

                return lax.cond(cond_j, valid_branch, lambda _: 0.0, operand=None)

            j_indices = jnp.arange(seq_len)
            total = jnp.sum(vmap(accumulate_single_j)(j_indices))
            return total

        i_indices = jnp.arange(seq_len + 1)
        l_indices = i_indices + d
        updates = vmap(accumulate_single_i)(i_indices)
        bar_Pm1 = bar_Pm1.at[i_indices, l_indices].set(updates, mode="drop")
        return bar_Pm1

    def outside_partition(
        p_seq: Array,
        P: Array,
        ML: Array,
        E: Array,
        s_table: Array,
    ) -> tuple[Array, Array, Array]:
        seq_len = int(p_seq.shape[0])
        bar_E = jnp.zeros_like(E)
        bar_E = bar_E.at[0].set(1.0)
        bar_P = jnp.zeros_like(P)
        bar_M = jnp.zeros_like(ML)
        bar_Pm = jnp.zeros((seq_len + 1, seq_len + 1), dtype=P.dtype)
        bar_Pm1 = jnp.zeros((seq_len + 1, seq_len + 1), dtype=P.dtype)

        padded_p_seq = jnp.zeros((seq_len + 1, 4), dtype=f64)
        padded_p_seq = padded_p_seq.at[:seq_len].set(p_seq)

        bar_E = fill_bar_E(bar_E, P, padded_p_seq, s_table)

        def fill_tables_by_step(carry, d):
            bar_P, bar_M, bar_E, bar_Pm, bar_Pm1 = carry

            bar_P = fill_bar_P(
                d,
                padded_p_seq,
                ML,
                E,
                bar_P,
                bar_Pm,
                bar_Pm1,
                bar_E,
                s_table,
            )
            bar_Pm = fill_bar_Pm(d, padded_p_seq, ML, bar_P, bar_Pm, s_table)
            bar_Pm1 = fill_bar_Pm1(d, padded_p_seq, bar_P, bar_Pm1, s_table)
            bar_M = fill_bar_M(d, bar_M, bar_P, P, padded_p_seq, s_table)

            return (bar_P, bar_M, bar_E, bar_Pm, bar_Pm1), None

        (bar_P, bar_M, bar_E, bar_Pm, bar_Pm1), _ = scan(
            fill_tables_by_step,
            (bar_P, bar_M, bar_E, bar_Pm, bar_Pm1),
            jnp.arange(seq_len - 1, -1, 0),
        )
        return (bar_P, bar_M, bar_E)

    return outside_partition


def get_outside_partition_fn(
    em: energy.Model,
    seq_len: int,
    *,
    max_loop: int = MAX_LOOP,
    checkpoint_every: int | None = 10,
) -> Callable[[Array, Array, Array, Array, Array], tuple[Array, Array, Array]]:
    """Return a cached outside kernel specialized to a given sequence length."""

    key = (int(id(em)), seq_len, max_loop, checkpoint_every)
    cached = _OUTSIDE_KERNEL_CACHE.get(key)
    if cached is not None:
        fn, em_ref = cached
        if em_ref() is em:
            return fn

    kernel = _construct_outside_partition_fn(
        em,
        seq_len,
        max_loop=max_loop,
        checkpoint_every=checkpoint_every,
    )
    _OUTSIDE_KERNEL_CACHE[key] = (kernel, weakref.ref(em))
    return kernel


def compute_outside(
    inside: InsideTablesLike,
    model: energy.Model,
    *,
    max_loop: int | None = None,
    checkpoint_every: int | None = 10,
) -> OutsideComputation:
    """Run the outside recursion using precomputed inside tables."""
    if max_loop is None:
        max_loop = MAX_LOOP

    if not hasattr(inside, "p_seq"):
        raise AttributeError("Inside tables must expose a `p_seq` field for outside recursion.")

    seq_len = int(jnp.asarray(inside.p_seq).shape[0])
    outside_partition = get_outside_partition_fn(
        model,
        seq_len,
        max_loop=max_loop,
        checkpoint_every=checkpoint_every,
    )
    bar_P, bar_M, bar_E = outside_partition(
        inside.p_seq,
        inside.P,
        inside.ML,
        inside.E,
        inside.s_table,
    )

    return OutsideComputation(
        bar_E=bar_E,
        bar_P=bar_P,
        bar_M=bar_M,
    )
