"""Scaffolding for JAX-based outside dynamic programming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Callable, cast
import functools

import jax
from jax import vmap, jit, lax
import jax.numpy as jnp

# (既存の import に加えて)
# from jax_rnafold.common.utils import int_inf
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


def get_outside_partition_fn(em: energy.Model, seq_len: int, inside: InsideComputation, 
                             max_loop: int = MAX_LOOP,
                             checkpoint_every: int = 10) -> Callable:
    if checkpoint_every is None:
        scan = lax.scan
    else:
        scan = functools.partial(checkpoint_scan, checkpoint_every=checkpoint_every)

    two_loop_length = min(seq_len, max_loop)
    s_table = inside.s_table

    @jit
    def fill_bar_E(bar_E, P, padded_p_seq):
        def body(i, current_bar_E):
            def get_j_bp_term(j, bp_idx):
                cond = (j < i - 1)
                bp = bp_bases[bp_idx]
                bj = bp[0]
                bim1 = bp[1]
                base_en = current_bar_E[j] * padded_p_seq[j, bj] * padded_p_seq[i-1, bim1]
                return jnp.where(cond, base_en * P[bp_idx, j, i-1] * em.en_ext_branch(bj, bim1), 0.0)
            get_all_terms = vmap(vmap(get_j_bp_term, (0, None)), (None, 0))
            terms = get_all_terms(jnp.arange(seq_len + 1), jnp.arange(NBPS))
            sm = s_table[1] * current_bar_E[i-1] + jnp.sum(terms)
            updated_bar_E = current_bar_E.at[i].set(sm)
            return updated_bar_E, None

        bar_xi_out, _ = scan(body, bar_E, jnp.arange(2, seq_len+1))
        return bar_xi_out


    @jit
    def psum_outer_bulges(bh, bl, h, l, padded_p_seq, bar_P):
        def get_bp_ij(bp_idx_ij, ij_offset):
            bp = bp_bases[bp_idx_ij]
            bi = int(bp[0])
            bj = int(bp[1])
            bp_ij_sm = jnp.zeros((), dtype=bar_P.dtype)

            # Right bulge, note i = h - 1
            j = l + 2 + ij_offset
            i = h - 1
            cond_ij = (j < seq_len + 1) & (0 <= i)
            right_cond = (h < l) # right bulge なので h = i + 1 (i = h - 1) であり、h + 1 < l は必須.
            bp_ij_sm += jnp.where(right_cond & cond_ij,
                                   bar_P[bp_idx_ij, i, j] * padded_p_seq[i, bi] * \
                                    padded_p_seq[j, bj] * em.en_bulge(bi, bj, bh, bl, j-l-1) * \
                                    s_table[j-l+1],
                                      0.0)

            # Left bulge, note j = l + 1
            i = l - 2 - ij_offset
            j = l + 1
            cond_ij = (j < seq_len + 1) & (0 <= i)
            left_cond = (h < l)
            # left_val = bar_P[bp_idx_ij, i, j] * padded_p_seq[i, bi] * \
            #     padded_p_seq[j, bj] * em.en_bulge(bi, bj, bh, bl, h-i-1) * \
            #     s_table[h-i+1]
            bp_ij_sm += jnp.where(left_cond & cond_ij,
                                    bar_P[bp_idx_ij, i, j] * padded_p_seq[i, bi] * \
                                    padded_p_seq[j, bj] * em.en_bulge(bi, bj, bh, bl, h-i-1) * \
                                    s_table[h-i+1],
                                      0.0)

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
        em: energy.Model, 
        two_loop_length: int
    ) -> Array:
        """
        psum_internal_loops (inside) の逆写像（outside）を計算する。
        
        (h, l) [内側ペア] に対する寄与を、すべての (i, j) [外側ペア] から計算する。
        i, j は lup (h - i - 1) と rup (j - l - 1) によって決定される。
        
        ネスト順序: (lup, rup) -> (bp_idx_ij) -> (bip1, bjm1)
        """

        max_lup_rup = two_loop_length - 2
        lup_offsets = jnp.arange(max_lup_rup) # 0, 1, ...
        rup_offsets = jnp.arange(max_lup_rup) # 0, 1, ...

        bp_hl = bp_bases[bp_idx_hl]
        bh = int(bp_hl[0])
        bl = int(bp_hl[1])

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
            lup = lup_offset + 1  # lup = 1, 2, ...
            rup = rup_offset + 1  # rup = 1, 2, ...
            i = h - lup - 1       # i は 1-based index
            j = l + rup + 1       # j は 1-based index
            ij_cond = (j < seq_len + 1) & (0 <= i) # j は seq_len 以下
            len_cond = (lup + rup + 2 <= two_loop_length)
            valid_ij = ij_cond & len_cond

            bp = bp_bases[bp_idx_ij]
            bi = int(bp[0])
            bj = int(bp[1])


            def get_mmij_term(bip1, bjm1):
                return padded_p_seq[i+1, bip1]*padded_p_seq[j-1, bjm1] * \
                    em.en_il_inner_mismatch(bi, bj, bip1, bjm1)
            mmij_terms = vmap(vmap(get_mmij_term, (0, None)), (None, 0))(N4, N4)
            mmij = jnp.sum(mmij_terms)

            sm = 0.0

            # Note: 1x1 and 1xN and Nx1. Not just 1xN.
            @jit
            def get_bp_1n_sm(bip1, bjm1): # bp_idx_hl はすでに決定済み
                bp_1n_sm = 0.0
                cond_11 = (lup == 1) | (rup == 1)
                # cond_11 is equal to "i = h - 2 and  j = l + 2"

                pr_ij_mm = padded_p_seq[i+1, bip1]*padded_p_seq[j-1, bjm1]
                # 1x1. Don't need safe_P since we pad on both sides.
                bp_1n_sm += jnp.where(cond_11,
                    bar_P[bp_idx_ij, i, j]*padded_p_seq[i, bi] \
                            * padded_p_seq[j, bj]*pr_ij_mm \
                            * em.en_internal(bi, bj, bh, bl, bip1, bjm1, bip1, bjm1, 1, 1) \
                            * s_table[4],
                    0.0
                )

                def z_b_fn(b):
                    # これは 1xN のときの右側の和、Nx1 のときの左側の和を計算する。
                    # b は N の方の 内側 (h, l のそば)の塩基
                    z_b_sm = 0.0
                    cond_1N = (h == i + 2) & (2 < j - l)

                    il_en = em.en_internal(
                        bi, bj, bh, bl, bip1, bjm1, bip1, b, 1, j-l-1)
                    right_term = bar_P[bp_idx_hl, i, j]*padded_p_seq[i, bi] \
                                * padded_p_seq[j, bj]*padded_p_seq[l+1, b]*pr_ij_mm*il_en \
                                * s_table[j-l+2]
                    z_b_sm += jnp.where(cond_1N, right_term, 0.0)

                    cond_N1 = (2 < h - i) & (j == l + 2)
                    il_en = em.en_internal(
                        bi, bj, bh, bl, bip1, bjm1, b, bjm1, h-i-1, 1)
                    left_term = bar_P[bp_idx_hl, i, j]*padded_p_seq[h, bh] \
                            * padded_p_seq[j-2, bl]*padded_p_seq[h-1, b]*pr_ij_mm*il_en \
                            * s_table[h-i+2]
                    z_b_sm += jnp.where(cond_N1, left_term, 0.0)

                    return z_b_sm

                get_all_zb_terms = vmap(vmap(z_b_fn, (0, None)), (None, 0))

                bp_1n_sm += jnp.sum(get_all_zb_terms(N4))
                return bp_1n_sm
            get_all_1n_terms = vmap(vmap(vmap(get_bp_1n_sm, (0, None, None)),
                                        (None, 0, None)), (None, None, 0))
            sm += jnp.sum(get_all_1n_terms(N4, N4))


            # 2x2, 3x2, 2x3
            def get_bp_22_23_32_summand(bip1, bjm1, bhm1, blp1):
                cond_lup_rup = ((lup == 2) & (rup == 2)) \
                | ((lup == 2) & (rup == 3)) \
                | ((lup == 3) & (rup == 2))
                cond_idx = (h < j-2) & (l >= h+1)
                cond_22_23_32 = cond_lup_rup & cond_idx
                return jnp.where(cond_22_23_32,
                    bar_P[bp_idx_ij, i, j]*padded_p_seq[i, bi]*padded_p_seq[j, bj] \
                    * em.en_internal(bi, bj, bh, bl, bip1, bjm1, bhm1, blp1, lup, rup) \
                    * padded_p_seq[h-1, bhm1]*padded_p_seq[l+1, blp1] \
                    * padded_p_seq[i+1, bip1]*padded_p_seq[j-1, bjm1] \
                    * s_table[lup+rup+2],
                    0.0)
            get_all_summands = vmap(get_bp_22_23_32_summand, (None, None, None, 0))
            get_all_summands = vmap(get_all_summands, (None, None, 0, None))
            get_all_summands = vmap(get_all_summands, (None, 0, None, None))
            get_all_summands = vmap(get_all_summands, (0, None, None, None))

            sm += jnp.sum(get_all_summands(N4, N4, N4, N4))

            # general internal loops
                # 複雑な bhm1, blp1 などを参照する必要がない。mmij が bip1, bjm1 について畳み込んでいる。
            # idx_cond = (k >= i+2) & (k < j-2) & (l >= k+1) & (l < j-1)
            idx_cond = (h < l)
            is_not_n1 = (lup > 1) & (rup > 1)
            is_22_23_32 = ((lup == 2) & (rup == 2)) \
                        | ((lup == 2) & (rup == 3)) \
                        | ((lup == 3) & (rup == 2))
            cond_general_il = idx_cond & is_not_n1 & ~is_22_23_32

            general_term = (
                em.en_internal_init(lup + rup)
                * em.en_internal_asym(lup, rup)
                * mmij
                * s_table[lup + rup + 2]
                * bar_P[bp_idx_ij, i, j]
                * outer_mismatch_factor  # inline outer mismatch weight (bar_OMM removed)
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
        ) -> Array:
        """Propagate paired-state outside weights for span starting at i."""

        def get_bp_stack(bp_idx_ij, l, bh, bl): # for bar_P[bp_idx_of_hl, h, l] の計算. 
            # bp_idx は bp_idx_of_ij となっていることに注意する。
            # l - h = d ゆえ h = l - d
            h = l - d
            i = h - 1
            j = l + 1
            cond = (0 < i) & (j < seq_len) # 0 origin であり,l + 1 <= j < seq_len を満たすべきだから
            bp = bp_bases[bp_idx_ij]
            bi = int(bp[0]) # bi; i = h - 1
            bj = int(bp[1]) # bj; j = l + 1
            return jnp.where(cond, 
                bar_P[bp_idx_ij, h-1, l+1]*padded_p_seq[h-1, bi] * \
                padded_p_seq[l+1, bj]*em.en_stack(bi, bj, bh, bl),
                0.0
            )

        def get_bp_l_multi_sm(l):
            # h, l, bh, bl が与えられた時の multiloop による寄与を計算する。
            h = l - d
            def get_multi_i_term(i): # bl は上で定義されている
                cond = (i < h)
                return jnp.where(cond,
                                 (s_table[1] * ML[1, i+1, h-1] * bar_Pm1[i, l] 
                                   + bar_Pm[i, l] * (s_table[1] * ML[1, i+1, h-1] 
                                    + (s_table[1] * em.en_multi_unpaired())**(h - i - 1) * s_table[1])),
                                 0.0)
            all_i_terms = vmap(get_multi_i_term)(jnp.arange(seq_len+1))
            return jnp.sum(jnp.asarray(all_i_terms))

        def get_bp_l_sm(bp_idx_hl, l): # ある l に対してそれに対応する summation を計算する。
            # bar_P(h, l) の計算をしている。sum_{i, j} B(f_2) * bar_P(i, j) の部分に該当する。
            h = l - d
            cond = (0 <= h) & (l <= seq_len)
            bp = bp_bases[bp_idx_hl]
            bh = int(bp[0]) # bi; i = h - 1
            bl = int(bp[1]) # bj; j = l + 1
            sm = jnp.zeros((), dtype=bar_P.dtype)

            sm += psum_outer_bulges(bh, bl, h, l, padded_p_seq, bar_P)
            sm += psum_outer_internal_loops(bp_idx_hl, h, l, padded_p_seq, bar_P, s_table, em, two_loop_length)

            # stacks
            stack_summands = vmap(get_bp_stack, (0, None, None, None))(jnp.arange(NBPS), l, bh, bl)
            sm += jnp.sum(stack_summands) * s_table[2]

            # Multi-loops
            sm += get_bp_l_multi_sm(l)

            # Exterior
            sm += bar_E[h] * E[l + 1]

            cond = (1 <= h) & (l < seq_len - 1) # 0 origin であり,l + 1 <= j < seq_len を満たすべきだから
            return jnp.where(cond, sm, bar_P[bp_idx_hl, h, l])


        def get_bp_all_ls(bp_idx_hl):
            ls = jnp.arange(seq_len + 1)
            return vmap(get_bp_l_sm, (None, 0))(bp_idx_hl, ls)

        all_bp_js = vmap(get_bp_all_ls)(jnp.arange(NBPS))
        hs = jnp.arange(seq_len + 1 - d)
        ls = hs + d
        bar_P = bar_P.at[:, hs, ls].set(all_bp_js)

        return bar_P


    def fill_bar_M(
        d: int,
        bar_M: Array,
        bar_P: Array,
        P: Array,
        padded_p_seq: Array,
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

            # 共通項
            sm_M2 += bar_M[2, h - 1, l] * multi_unpaired_factor
            sm_M1 += bar_M[1, h - 1, l] * multi_unpaired_factor
            sm_M0 += bar_M[0, h - 1, l] * multi_unpaired_factor

            # P 由来の項
            def get_bp_idx_hm1_lp1_term(bp_idx_hm1_lp1):
                bp_hm1_lp1 = bp_bases[bp_idx_hm1_lp1]
                hm1 = int(bp_hm1_lp1[0])
                lp1 = int(bp_hm1_lp1[1])
                return s_table[2] * bar_P[bp_idx_hm1_lp1, h - 1, l + 1] \
                    * padded_p_seq[h-1, hm1] * padded_p_seq[l + 1, lp1] * em.en_multi_closing(hm1, lp1)
            get_all_bp_terms = vmap(get_bp_idx_hm1_lp1_term)
            sm_M2 += jnp.sum(get_all_bp_terms(jnp.arange(NBPS)))

            # sum_i の項 only for the bar_M1 and bar_M0
            def get_i_term(i):
                cond = (i < h - 1)
                ml_i_to_M1 = bar_M[2, i, l]
                ml_i_to_M0 = bar_M[0, i, l] + bar_M[1, i, l]
                def get_idx_bp_i_hm1(bp_idx_ihm1):
                    bp_ihm1 = bp_bases[bp_idx_ihm1]
                    bi = int(bp_ihm1[0])
                    bhm1 = int(bp_ihm1[1])
                    return P[bp_idx_ihm1, i, h - 1] * em.en_multi_branch(bi, bhm1) * padded_p_seq[i, bi] * padded_p_seq[h - 1, bhm1]
                get_all_bp_i_hm1_terms = vmap(get_idx_bp_i_hm1)
                bp_sum_i = jnp.sum(get_all_bp_i_hm1_terms(jnp.arange(NBPS)))
                return jnp.where(cond, bp_sum_i * ml_i_to_M1, 0.0), jnp.where(cond, bp_sum_i * ml_i_to_M0, 0.0)

            m1_terms, m0_terms = vmap(get_i_term)(jnp.arange(seq_len + 1))
            sm_M1 += jnp.sum(m1_terms)
            sm_M0 += jnp.sum(m0_terms)
            return jnp.where(cond, sm_M0, bar_M[0, h, l]), jnp.where(cond, sm_M1, bar_M[1, h, l]), jnp.where(cond, sm_M2, bar_M[2, h, l])

        h_indices = jnp.arange(d + 1, seq_len + 1)
        l_indices = h_indices + d   
        updates = vmap(accumulate_single_h)(h_indices)
        bar_M = bar_M.at[0, h_indices, l_indices].set(updates[0])
        bar_M = bar_M.at[1, h_indices, l_indices].set(updates[1])
        bar_M = bar_M.at[2, h_indices, l_indices].set(updates[2])
        return bar_M


    def fill_bar_Pm(
        d: int,
        padded_p_seq: Array,
        ML: Array,
        bar_P: Array,
        bar_Pm: Array,
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
                        bi = int(bp_ij[0])
                        bj = int(bp_ij[1])
                        branch_penalty = em.en_multi_branch(bi, bj)
                        return s1 * bar_P[bp_idx, i, j] * branch_penalty * padded_p_seq[i, bi] * padded_p_seq[j, bj] * ml_val

                    return jnp.sum(vmap(accumulate_bp)(jnp.arange(NBPS)))

                return lax.cond(cond_j, valid_branch, lambda _: 0.0, operand=None)

            j_indices = jnp.arange(seq_len + 1)
            return jnp.sum(vmap(accumulate_single_j)(j_indices))

        i_indices = jnp.arange(seq_len - d + 1)
        l_indices = i_indices + d
        updates = vmap(accumulate_single_i)(i_indices)
        bar_Pm = bar_Pm.at[i_indices, l_indices].set(updates)
        return bar_Pm
    
    def fill_bar_Pm1(
        d: int,
        padded_p_seq: Array,
        bar_P: Array,
        bar_Pm1: Array,
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
                        bi = int(bp[0])
                        bj = int(bp[1])
                        branch_penalty = em.en_multi_branch(bi, bj)
                        return bar_P[bp_idx_ij, i, j] * branch_penalty * padded_p_seq[i, bi] * padded_p_seq[j, bj] * unpaired_factor

                    inner_sum = jnp.sum(vmap(accumulate_bp)(jnp.arange(NBPS)))
                    return s_table[1] * inner_sum

                return lax.cond(cond_j, valid_branch, lambda _: 0.0, operand=None)

            j_indices = jnp.arange(seq_len)
            total = jnp.sum(vmap(accumulate_single_j)(j_indices))
            return total

        i_indices = jnp.arange(seq_len - d + 1)
        l_indices = i_indices + d
        updates = vmap(accumulate_single_i)(i_indices)
        bar_Pm1 = bar_Pm1.at[i_indices, l_indices].set(updates)
        return bar_Pm1

    def outside_partition(p_seq: Array, inside: InsideComputation) -> tuple[Array, Array, Array]:
        seq_len = int(p_seq.shape[0])
        bar_E = jnp.zeros_like(inside.E)
        bar_E = bar_E.at[1].set(1.0)  # base case: bar_E[0] = 1 in 1-based indexing
        bar_P = jnp.zeros_like(inside.P)
        bar_M = jnp.zeros_like(inside.ML)
        bar_Pm = jnp.zeros((seq_len + 1, seq_len + 1), dtype=inside.P.dtype)
        bar_Pm1 = jnp.zeros((seq_len + 1, seq_len + 1), dtype=inside.P.dtype)

        padded_p_seq = jnp.zeros((seq_len + 1, 4), dtype=f64)
        padded_p_seq = padded_p_seq.at[:seq_len].set(p_seq)

        # first off, fill bar_E.
        bar_E = fill_bar_E(bar_E, inside.P, padded_p_seq)

        # filling the other tables
        def fill_tables_by_step(carry, d):
            bar_P, bar_M, bar_E, bar_Pm, bar_Pm1 = carry

            bar_P = fill_bar_P(
                d, padded_p_seq, inside.ML, inside.E, bar_P, bar_Pm, bar_Pm1, bar_E
            )
            bar_Pm = fill_bar_Pm(d, padded_p_seq, inside.ML, bar_P, bar_Pm)
            bar_Pm1 = fill_bar_Pm1(d, padded_p_seq, bar_P, bar_Pm1)
            bar_M = fill_bar_M(
                d, bar_M, bar_P, inside.P, padded_p_seq
            )

            return (bar_P, bar_M, bar_E, bar_Pm, bar_Pm1), None
        (bar_P, bar_M, bar_E, bar_Pm, bar_Pm1), _ = scan(fill_tables_by_step,
                                        (bar_P, bar_M, bar_E, bar_Pm, bar_Pm1),
                                        jnp.arange(seq_len-1, -1, -1))
        return (bar_P, bar_M, bar_E)

    return outside_partition


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
        inside,
        max_loop=max_loop,
        checkpoint_every=checkpoint_every,
    )
    bar_P, bar_M, bar_E = outside_partition(inside.p_seq, inside)

    return OutsideComputation(
        bar_E=bar_E,
        bar_P=bar_P,
        bar_M=bar_M,
    )
