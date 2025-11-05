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
    max_sp_hairpin_len_up = max(special_hairpin_lens) - 2 # Subtract 2 for the paired nt
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


    # ---------- copy-pasted from jax_rnafold.d0.ss line123 - 213 ----------
    @jit
    def psum_outer_bulges(bh, bl, h, l, padded_p_seq, bar_P):
        def get_bp_ij(bp_idx_ij, ij_offset):
            bp = bp_bases[bp_idx_ij]
            bi = int(bp[0])
            bj = int(bp[1])
            bp_ij_sm = jnp.zeros((), dtype=bar_P.dtype)

            # Right bulge, note i = h - 1
            j = l + 2 + ij_offset
            right_cond = (l >= h + 1) # right bulge なので h = i + 1 (i = h - 1) であり、h + 1 < l は必須.
            right_val = bar_P[bp_idx_ij, h-1, j] * padded_p_seq[h-1, bi] * \
                padded_p_seq[l, bl] * em.en_bulge(bi, bj, bh, bl, j-l-1) * \
                s_table[j-l+1]
            bp_ij_sm += jnp.where(right_cond, right_val, 0.0)

            # Left bulge, note j = l + 1
            i = l - 2 - ij_offset
            left_cond = (h < l)
            left_val = bar_P[bp_idx_ij, i, l+1] * padded_p_seq[i, bi] * \
                padded_p_seq[l+1, bj] * em.en_bulge(bi, bj, bh, bl, h-i-1) * \
                s_table[h-i+1]
            bp_ij_sm += jnp.where(left_cond, left_val, 0.0)

            return bp_ij_sm

        def get_bp_all_ij(bp_idx):
            all_ij_offsets = jnp.arange(two_loop_length)
            all_bp_ij_sms = vmap(get_bp_ij, (None, 0))(bp_idx, all_ij_offsets)
            return jnp.sum(all_bp_ij_sms)

        all_bp_sms = vmap(get_bp_all_ij)(jnp.arange(NBPS))
        return jnp.sum(all_bp_sms)


# jax_outside.py に配置

    @jit
    def psum_outer_internal_loops(
        bh: int, 
        bl: int, 
        h: int, 
        l: int, 
        padded_p_seq: Array, 
        bar_P: Array, 
        s_table: Array, 
        em: energy.Model, 
        two_loop_length: int
    ) -> tuple[Array, Array]:
        """
        psum_internal_loops (inside) の逆写像（outside）を計算する。
        
        (h, l) [内側ペア] に対する寄与を、すべての (i, j) [外側ペア] から計算する。
        i, j は lup (h - i - 1) と rup (j - l - 1) によって決定される。
        
        ネスト順序: (lup, rup) -> (bp_idx_ij) -> (bip1, bjm1)
        """
        seq_len = padded_p_seq.shape[0] - 1 # p_seq は 0-indexed

        # 1. ループサイズ (lup, rup) で vmap する
        # lup, rup は 1 以上 (Internal loop の定義より)
        max_lup_rup = two_loop_length - 2
        lup_offsets = jnp.arange(max_lup_rup) # 0, 1, ...
        rup_offsets = jnp.arange(max_lup_rup) # 0, 1, ...

        # vmap される内側のループで使用する定数
        bp_indices_ij = jnp.arange(NBPS)
        n4_indices = N4

        @jit
        def get_contribution_for_lup(lup_offset: int) -> tuple[Array, Array]:
            """lup を固定して rup で vmap"""
            lup = lup_offset + 1  # lup = 1, 2, ...
            i = h - lup - 1       # i は 1-based index
            i_cond = (i >= 1)     # i は 1-based なので 1 以上

            @jit
            def get_contribution_for_rup(rup_offset: int) -> tuple[Array, Array]:
                """lup, rup を固定して bp_idx_ij で vmap"""
                rup = rup_offset + 1  # rup = 1, 2, ...
                j = l + rup + 1       # j は 1-based index
                j_cond = (j <= seq_len) # j は seq_len 以下

                # ループ長の制約
                len_cond = (lup + rup + 2 <= two_loop_length)
                
                # (i, j) がペアとして有効か
                # (h < l は呼び出し元 fill_bar_P で保証)
                # i = h - lup - 1
                # j = l + rup + 1
                # h < l より、i < j は自動的に満たされる
                valid_ij = i_cond & j_cond & len_cond

                @jit
                def get_contribution_for_ij_bp_type(bp_idx_ij: int) -> tuple[Array, Array]:
                    """lup, rup, bp_idx_ij を固定してミスマッチ (bip1, bjm1) で vmap"""
                    bp_ij = bp_bases[bp_idx_ij]
                    bi = int(bp_ij[0])
                    bj = int(bp_ij[1])
                    
                    # (i, j) が有効な場合のみ bar_P[i, j] の値を使用する
                    # (vmap のためインデックス自体は常に有効である必要があるが、
                    #  i, j は計算された値なので、jnp.where で後からマスクする)
                    bar_P_ij = bar_P[bp_idx_ij, i, j]

                    @jit
                    def get_contribution_for_mismatch(bip1: int, bjm1: int) -> tuple[Array, Array]:
                        """
                        i, j, lup, rup, bp_idx_ij, bip1, bjm1 が全て確定
                        """
                        
                        # (i, j) のミスマッチ (i+1, j-1) の確率
                        # i, j は 1-based。padded_p_seq は 0-based。
                        # mismatch (i+1) -> p_seq[i]
                        # mismatch (j-1) -> p_seq[j-2]
                        pr_ij_mm = padded_p_seq[i, bip1] * padded_p_seq[j - 2, bjm1]
                        
                        # ターミナルミスマッチエネルギー
                        mmij = em.en_il_inner_mismatch(bi, bj, bip1, bjm1)
                        
                        # --- 1. d_bar_P への寄与 (特定ループ) ---
                        en_loop = em.en_internal(bi, bj, bh, bl, bip1, bjm1, bip1, bjm1, lup, rup)
                        
                        cond_1x1 = (lup == 1) & (rup == 1)
                        cond_1xN = (lup == 1) & (rup > 1)
                        cond_Nx1 = (lup > 1) & (rup == 1)
                        cond_2x2 = (lup == 2) & (rup == 2)
                        cond_2x3 = (lup == 2) & (rup == 3)
                        cond_3x2 = (lup == 3) & (rup == 2)
                        
                        # 1x1 以外は s_table スケーリングが乗る
                        # (ss.py の実装に合わせる)
                        # 1x1 のみ s_table[4]
                        
                        # s_table[lup+rup+2] が正しいスケーリング
                        s = s_table[lup + rup + 2]
                        
                        # 1x1
                        coeff_1x1 = pr_ij_mm * mmij * s * en_loop
                        d_bar_P_1x1 = jnp.where(cond_1x1, coeff_1x1, 0.0)

                        # 1xN, Nx1
                        coeff_1N_N1 = pr_ij_mm * mmij * s * en_loop
                        d_bar_P_1N_N1 = jnp.where(cond_1xN | cond_Nx1, coeff_1N_N1, 0.0)

                        # 2x2, 2x3, 3x2
                        cond_special_2N = cond_2x2 | cond_2x3 | cond_3x2
                        coeff_2N = pr_ij_mm * mmij * s * en_loop
                        d_bar_P_2N = jnp.where(cond_special_2N, coeff_2N, 0.0)
                        
                        d_bar_P_sum = (d_bar_P_1x1 + d_bar_P_1N_N1 + d_bar_P_2N) * bar_P_ij

                        # --- 2. d_bar_OMM への寄与 (一般内部ループ) ---
                        cond_general = (lup > 1) & (rup > 1) & (~cond_special_2N)
                        
                        en_gen = em.en_internal_init(lup+rup) \
                                + em.en_internal_asym(lup, rup)
                        
                        coeff_gen = pr_ij_mm * en_gen * mmij * s
                        d_bar_OMM_sum = jnp.where(cond_general, coeff_gen * bar_P_ij, 0.0)
                        
                        return d_bar_P_sum, d_bar_OMM_sum

                    # vmap over mismatch types (bip1, bjm1)
                    d_bar_P_mm, d_bar_OMM_mm = vmap(vmap(get_contribution_for_mismatch, (0, None)), (None, 0))(n4_indices, n4_indices)
                    return jnp.sum(d_bar_P_mm), jnp.sum(d_bar_OMM_mm)
                
                # vmap over outer base pair types (bp_idx_ij)
                d_bar_P_all, d_bar_OMM_all = vmap(get_contribution_for_ij_bp_type)(bp_indices_ij)
                
                # (i, j) が無効なインデックスだった場合は、この (lup, rup) の組合せの寄与は 0
                return jnp.where(valid_ij, d_bar_P_all, 0.0), jnp.where(valid_ij, d_bar_OMM_all, 0.0)

            # vmap over rup (right loop size)
            d_bar_P_rup, d_bar_OMM_rup = vmap(get_contribution_for_rup)(rup_offsets)
            return jnp.sum(d_bar_P_rup), jnp.sum(d_bar_OMM_rup)
        
        # vmap over lup (left loop size)
        d_bar_P_lup, d_bar_OMM_lup = vmap(get_contribution_for_lup)(lup_offsets)
        
        return jnp.sum(d_bar_P_lup), jnp.sum(d_bar_OMM_lup)

    # ----------------- end of copy-paste ----------------------------------


    def fill_bar_P(
            d: int,
            padded_p_seq: Array,
            OMM: Array,
            ML: Array,
            P: Array,
            bar_P: Array,
            bar_Pm: Array,
            bar_Pm1: Array,
            bar_E: Array,
        ) -> Array:
        """Propagate paired-state outside weights for span starting at i."""

        def get_bp_stack(bp_idx_ij, l, bh, bl): # for bar_P[bp_idx_of_hl, h, l] の計算. 
            # bp_idx は bp_idx_of_ij となっていることに注意する。
            # l - h = d ゆえ h = l - d
            h = l - d # TODO: 0 <= h の条件を入れたいが、jax で許される形にするにはどうしたら良いか？初めの時点で hs, ls のペアを渡しておくか？
            bp = bp_bases[bp_idx_ij]
            bhm1 = int(bp[0]) # bi; i = h - 1
            blp1 = int(bp[1]) # bj; j = l + 1
            return bar_P[bp_idx_ij, h-1, l+1]*padded_p_seq[h-1, bhm1] * \
                padded_p_seq[l+1, blp1]*em.en_stack(bhm1, blp1, bh, bl)

        def get_bp_l_multi_sm(l):
            # h, l, bh, bl が与えられた時の multiloop による寄与を計算する。
            h = l - d
            def get_multi_i_term(i): # bl は上で定義されている
                i_cond = (i < h)
                # MB がどう絡むのかわからないので調査する必要がある。
                    # MB[i, j] は i, j が multiloop を閉じる一つのペアであるときの
                    #  multi branch boltzman factor * P[bp_idx_ij, i, j] の和
                return jnp.where(i_cond,
                                 (s_table[1] * ML[1, i+1, h-1] * bar_Pm1[i, l] 
                                   + bar_Pm[i, l] * (s_table[1] * ML[1, i+1, h-1] 
                                    + (s_table[1] * em.en_multi_unpaired())**(h - i - 1) * s_table[1])),
                                 0.0)
            all_i_terms = vmap(get_multi_i_term)(jnp.arange(seq_len+1))
            return jnp.sum(jnp.asarray(all_i_terms))

        def get_bp_l_sm(bp_idx_hl, l): # ある l に対してそれに対応する summation を計算する。
            # bar_P(h, l) の計算をしている。sum_{i, j} B(f_2) * bar_P(i, j) の部分に該当する。
            h = l - d
            bp = bp_bases[bp_idx_hl]
            bh = int(bp[0]) # bi; i = h - 1
            bl = int(bp[1]) # bj; j = l + 1
            sm = jnp.zeros((), dtype=bar_P.dtype)

            sm += psum_outer_bulges(bh, bl, h, l, padded_p_seq, P)
            sm_to_bar_P, sm_to_bar_OMM = psum_outer_internal_loops(bh, bl, h, l, padded_p_seq, P, OMM)
            sm += sm_to_bar_P

            # stacks
            stack_summands = vmap(get_bp_stack, (0, None, None, None))(jnp.arange(NBPS), l, bh, bl)
            sm += jnp.sum(stack_summands) * s_table[2]

            # Multi-loops
            sm += get_bp_l_multi_sm(l)
            cond = (1 <= h) & (l < seq_len - 1) # 0 origin であり,l + 1 <= j < seq_len を満たすべきだから
            return jnp.where(cond, sm, P[bp_idx_hl, h, l])


        def get_bp_all_ls(bp_idx_hl):
            ls = jnp.arange(seq_len + 1)
            return vmap(get_bp_l_sm, (None, 0))(bp_idx_hl, ls)

        all_bp_js = vmap(get_bp_all_ls)(jnp.arange(NBPS))
        hs, ls = jnp.arange(seq_len + 1 - d), jnp.arange(d, seq_len + 1)
        bar_P = bar_P.at[:, hs, ls].set(all_bp_js)

        return bar_P

    # def fill_bar_MB(carry: OutsideCarry, inside: InsideTablesLike, i: int) -> Array:
    #     """Propagate multibranch helper contributions at position i."""

    #     raise NotImplementedError


    def fill_bar_M(
        d: int,
        bar_M: Array,
        bar_P: Array,
        padded_p_seq: Array,
        P: Array,
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

        table_len = bar_M.shape[1]
        multi_unpaired_factor = s_table[1] * em.en_multi_unpaired()
        closing_scale = s_table[2]

        def body(h: int, current_bar_M: Array) -> Array:
            l = h + d
            valid = (l < table_len) & (l >= 0)

            def update_bar_M(curr: Array) -> Array:
                prev_idx = h - 1
                has_prev = prev_idx >= 0

                def left_m_term(_) -> Array:
                    return multi_unpaired_factor * curr[2, prev_idx, l]

                left_m2 = lax.cond(has_prev, left_m_term, lambda _: jnp.zeros((), dtype=curr.dtype), operand=None)

                def closing_term(_) -> Array:
                    def bp_body(bp_idx: int, acc: Array) -> Array:
                        bp = bp_bases[bp_idx]
                        bi = int(bp[0])
                        bj = int(bp[1])
                        contrib = bar_P[bp_idx, prev_idx, l + 1] * em.en_multi_closing(bi, bj)
                        return acc + contrib

                    return lax.fori_loop(
                        0,
                        NBPS,
                        bp_body,
                        jnp.zeros((), dtype=curr.dtype),
                    )

                has_closing = has_prev & (l + 1 < table_len)
                closing_contrib = lax.cond(
                    has_closing,
                    closing_term,
                    lambda _: jnp.zeros((), dtype=curr.dtype),
                    operand=None,
                )
                new_m2 = left_m2 + closing_scale * closing_contrib

                def left_m_state(order: int) -> Array:
                    return lax.cond(
                        has_prev,
                        lambda _: multi_unpaired_factor * curr[order, prev_idx, l],
                        lambda _: jnp.zeros((), dtype=curr.dtype),
                        operand=None,
                    )

                left_m1 = left_m_state(1)
                left_m0 = left_m_state(0)

                def sum_body(i: int, acc: tuple[Array, Array]) -> tuple[Array, Array]:
                    sum_m1_val, sum_m0_val = acc
                    cond = has_prev & (i < prev_idx)

                    def accumulate(_) -> tuple[Array, Array]:
                        def bp_body(bp_idx: int, acc_bp: Array) -> Array:
                            bp = bp_bases[bp_idx]
                            bi = int(bp[0])
                            bj = int(bp[1])
                            left_prob = padded_p_seq[i, bi]
                            right_prob = padded_p_seq[prev_idx, bj]
                            contrib = (
                                P[bp_idx, i, prev_idx]
                                * em.en_multi_branch(bi, bj)
                                * left_prob
                                * right_prob
                            )
                            return acc_bp + contrib

                        weight = lax.fori_loop(
                            0,
                            NBPS,
                            bp_body,
                            jnp.zeros((), dtype=curr.dtype),
                        )
                        m2_val = curr[2, i, l]
                        m1_val = curr[1, i, l]
                        m0_val = curr[0, i, l]
                        return (
                            sum_m1_val + weight * m2_val,
                            sum_m0_val + weight * (m1_val + m0_val),
                        )

                    return lax.cond(cond, accumulate, lambda _: (sum_m1_val, sum_m0_val), operand=None)

                init_acc = (
                    jnp.zeros((), dtype=curr.dtype),
                    jnp.zeros((), dtype=curr.dtype),
                )
                sum_m1, sum_m0 = lax.fori_loop(0, table_len, sum_body, init_acc)

                new_m1 = left_m1 + sum_m1
                new_m0 = left_m0 + sum_m0

                updated = curr.at[2, h, l].set(new_m2)
                updated = updated.at[1, h, l].set(new_m1)
                updated = updated.at[0, h, l].set(new_m0)
                return updated

            return lax.cond(valid, update_bar_M, lambda _: current_bar_M, operand=current_bar_M)

        upper = max(table_len - d, 0)
        bar_M = lax.fori_loop(0, upper, body, bar_M)
        return bar_M


    # def fill_bar_OMM(
    #     h: int,
    #     bar_P: Array,
    #     padded_p_seq: Array,
    #     n: int,
    # ) -> Array:
    #     """Accumulate general internal-loop contributions into bar_OMM."""

    #     raise NotImplementedError

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
                cond_j = (l < j) & (j < seq_len)

                def valid_branch(_):
                    ml_val = ML[1, l + 1, j - 1]

                    def accumulate_bp(bp_idx):
                        bp = bp_bases[bp_idx]
                        bi = int(bp[0])
                        bj = int(bp[1])
                        branch_penalty = em.en_multi_branch(bi, bj)
                        nuc_weight = padded_p_seq[i, bi] * padded_p_seq[j, bj]
                        return bar_P[bp_idx, i, j] * branch_penalty * nuc_weight * ml_val

                    return jnp.sum(vmap(accumulate_bp)(jnp.arange(NBPS)))

                return lax.cond(cond_j, valid_branch, lambda _: 0.0, operand=None)

            j_indices = jnp.arange(seq_len)
            total = jnp.sum(vmap(accumulate_single_j)(j_indices))
            return s_table[1] * total

        i_indices = jnp.arange(seq_len - d)
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
                    unpaired_factor = lax.pow(base_multi_unpaired, jnp.asarray(gap, dtype=bar_P.dtype))

                    def accumulate_bp(bp_idx):
                        bp = bp_bases[bp_idx]
                        bi = int(bp[0])
                        bj = int(bp[1])
                        branch_penalty = em.en_multi_branch(bi, bj)
                        nuc_weight = padded_p_seq[i, bi] * padded_p_seq[j, bj]
                        return bar_P[bp_idx, i, j] * branch_penalty * nuc_weight * unpaired_factor

                    inner_sum = jnp.sum(vmap(accumulate_bp)(jnp.arange(NBPS)))
                    return s_table[1] * inner_sum

                return lax.cond(cond_j, valid_branch, lambda _: 0.0, operand=None)

            j_indices = jnp.arange(seq_len)
            total = jnp.sum(vmap(accumulate_single_j)(j_indices))
            return total

        i_indices = jnp.arange(seq_len - d)
        l_indices = i_indices + d
        updates = vmap(accumulate_single_i)(i_indices)
        bar_Pm1 = bar_Pm1.at[i_indices, l_indices].set(updates)
        return bar_Pm1

    def outside_partition(p_seq: Array, inside: InsideComputation) -> tuple[Array, Array, Array]:
        seq_len = inside.E.shape[0]
        bar_E = jnp.zeros(seq_len, dtype=inside.E.dtype)
        bar_E = bar_E.at[1].set(1.0)  # base case: bar_E[0] = 1 in 1-based indexing
        bar_P = jnp.zeros_like(inside.P)
        bar_M = jnp.zeros_like(inside.ML)
        bar_MB = jnp.zeros_like(inside.MB)
        bar_OMM = jnp.zeros_like(inside.OMM)
        bar_Pm = jnp.zeros((seq_len + 1, seq_len + 1), dtype=inside.P.dtype)
        bar_Pm1 = jnp.zeros((seq_len + 1, seq_len + 1), dtype=inside.P.dtype)

        padded_p_seq = jnp.zeros((seq_len+1, 4), dtype=f64)
        padded_p_seq = padded_p_seq.at[:seq_len].set(p_seq)

        # first off, fill bar_E.
        bar_E = fill_bar_E(bar_E, inside.P, padded_p_seq, em, seq_len)

        # filling the other tables
        def fill_tables_by_step(carry, d):
            bar_OMM, bar_P, bar_M, bar_MB, bar_E, bar_Pm, bar_Pm1 = carry

            bar_P = fill_bar_P(
                d, padded_p_seq, inside.OMM, inside.ML, inside.P, bar_P, bar_Pm, bar_Pm1, bar_E
            )
            bar_Pm = fill_bar_Pm(d, padded_p_seq, inside.ML, bar_P, bar_Pm)
            bar_Pm1 = fill_bar_Pm1(d, padded_p_seq, bar_P, bar_Pm1)
            # bar_OMM = fill_bar_OMM(h, bar_P, padded_p_seq, seq_len)
            # bar_MB = fill_bar_MB(carry, inside, h)
            bar_M = fill_bar_M(
                d, bar_M, bar_P, padded_p_seq, inside.P
            )

            return (bar_OMM, bar_P, bar_M, bar_MB, bar_E, bar_Pm, bar_Pm1), None
        (bar_OMM, bar_P, bar_M, bar_MB, bar_E, bar_Pm, bar_Pm1), _ = scan(fill_tables_by_step,
                                        (bar_OMM, bar_P, bar_M, bar_MB, bar_E, bar_Pm, bar_Pm1),
                                        jnp.arange(seq_len-1, -1, -1))
        return (bar_P, bar_M, bar_E)

    return outside_partition



