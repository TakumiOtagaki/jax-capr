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


    @jit
    def psum_outer_internal_loops(bh, bl, h, l, padded_p_seq, P, OMM): 
        """
        コピーペーストして、引数と関数名だけ変更した。
        コメントをつけているように、個々の関数は結構やばいし、追加すべき変数がありそう。
        mmij は OMM の逆バージョンみたいな感じかも？違うかな
        """
        # bip1 --> bh, bjm1 --> bl
        # ここどうすればいいか全くわからない。bip1, bijm1 としたいけどそもそも i, j をまだ知らない.
            # TODO: おそらく OMM のように事前計算する必要がありそう。
        def get_mmij_term(bip1, bjm1):
            return padded_p_seq[i+1, bip1]*padded_p_seq[j-1, bjm1] * \
                em.en_il_inner_mismatch(bi, bj, bip1, bjm1)
        mmij_terms = vmap(vmap(get_mmij_term, (0, None)), (None, 0))(N4, N4)
        mmij = jnp.sum(mmij_terms)

        sm = 0.0

        # Note: 1x1 and 1xN and Nx1. Not just 1xN.
        @jit
        def get_bp_1n_sm(bp_idx, bip1, bjm1):
            bp_1n_sm = 0.0
            bp = bp_bases[bp_idx]
            bk = bp[0]
            bl = bp[1]

            pr_ij_mm = padded_p_seq[i+1, bip1]*padded_p_seq[j-1, bjm1]
            # 1x1. Don't need safe_P since we pad on both sides.
            bp_1n_sm += P[bp_idx, i+2, j-2]*padded_p_seq[i+2, bk] \
                        * padded_p_seq[j-2, bl]*pr_ij_mm \
                        * em.en_internal(bi, bj, bk, bl, bip1, bjm1, bip1, bjm1, 1, 1) \
                        * s_table[4]

            # FIXME: change to z_offset or kl_offset
            def z_b_fn(z_offset, b):
                z_b_sm = 0.0

                l = j-3-z_offset
                l_cond = (l >= i+3)
                il_en = em.en_internal(
                     bi, bj, bk, bl, bip1, bjm1, bip1, b, 1, j-l-1)
                right_term = P[bp_idx, i+2, l]*padded_p_seq[i+2, bk] \
                             * padded_p_seq[l, bl]*padded_p_seq[l+1, b]*pr_ij_mm*il_en \
                             * s_table[j-l+2]
                z_b_sm += jnp.where(l_cond, right_term, 0.0)

                k = i+3+z_offset
                k_cond = (k < j-2)
                il_en = em.en_internal(
                     bi, bj, bk, bl, bip1, bjm1, b, bjm1, k-i-1, 1)
                left_term = P[bp_idx, k, j-2]*padded_p_seq[k, bk] \
                           * padded_p_seq[j-2, bl]*padded_p_seq[k-1, b]*pr_ij_mm*il_en \
                           * s_table[k-i+2]
                z_b_sm += jnp.where(k_cond, left_term, 0.0)

                return z_b_sm

            get_all_zb_terms = vmap(vmap(z_b_fn, (0, None)), (None, 0))

            # z_offsets = jnp.arange(seq_len+1)
            z_offsets = jnp.arange(two_loop_length)
            bp_1n_sm += jnp.sum(get_all_zb_terms(z_offsets, N4))
            return bp_1n_sm
        get_all_1n_terms = vmap(vmap(vmap(get_bp_1n_sm, (0, None, None)),
                                     (None, 0, None)), (None, None, 0))
        sm += jnp.sum(get_all_1n_terms(jnp.arange(NBPS), N4, N4))


        # 2x2, 3x2, 2x3
        def get_bp_22_23_32_sm(bp_idx, k_offset, l_offset):
            k = i + k_offset + 2
            l = j - l_offset - 2

            bp = bp_bases[bp_idx]
            bk = bp[0]
            bl = bp[1]
            lup = k-i-1
            rup = j-l-1

            cond_lup_rup = ((lup == 2) & (rup == 2)) \
                | ((lup == 2) & (rup == 3)) \
                | ((lup == 3) & (rup == 2))
            cond_idx = (k < j-2) & (l >= k+1)
            cond = cond_lup_rup & cond_idx


            def get_bp_22_23_32_summand(bip1, bjm1, bkm1, blp1):
                return P[bp_idx, k, l]*padded_p_seq[k, bk]*padded_p_seq[l, bl] \
                    * em.en_internal(bi, bj, bk, bl, bip1, bjm1, bkm1, blp1, lup, rup) \
                    * padded_p_seq[k-1, bkm1]*padded_p_seq[l+1, blp1] \
                    * padded_p_seq[i+1, bip1]*padded_p_seq[j-1, bjm1] \
                    * s_table[lup+rup+2]
            get_all_summands = vmap(get_bp_22_23_32_summand, (None, None, None, 0))
            get_all_summands = vmap(get_all_summands, (None, None, 0, None))
            get_all_summands = vmap(get_all_summands, (None, 0, None, None))
            get_all_summands = vmap(get_all_summands, (0, None, None, None))

            all_summands = get_all_summands(N4, N4, N4, N4)
            return jnp.where(cond, jnp.sum(all_summands), 0.0)
        get_all_special_terms = vmap(vmap(vmap(get_bp_22_23_32_sm, (0, None, None)),
                                          (None, 0, None)), (None, None, 0))
        sm += jnp.sum(get_all_special_terms(jnp.arange(NBPS), jnp.arange(3), jnp.arange(3)))


        # general internal loops
        def general_kl_sm(k_offset, l_offset):
            k = k_offset + i + 2
            l = j - l_offset - 2

            lup = k-i-1
            rup = j-l-1

            # idx_cond = (k >= i+2) & (k < j-2) & (l >= k+1) & (l < j-1)
            idx_cond = (k < l)
            is_not_n1 = (lup > 1) & (rup > 1)
            is_22_23_32 = ((lup == 2) & (rup == 2)) \
                          | ((lup == 2) & (rup == 3)) \
                          | ((lup == 3) & (rup == 2))
            cond = idx_cond & is_not_n1 & ~is_22_23_32

            general_term = em.en_internal_init(lup+rup) * em.en_internal_asym(lup, rup) \
                           * OMM[k, l] * mmij * s_table[lup+rup+2]

            return jnp.where(cond, general_term, 0.0)
        get_all_general = vmap(vmap(general_kl_sm, (0, None)), (None, 0))
        # k_offsets, l_offsets = jnp.arange(seq_len+1), jnp.arange(seq_len+1)
        k_offsets, l_offsets = jnp.arange(two_loop_length), jnp.arange(two_loop_length)
        sm += jnp.sum(get_all_general(k_offsets, l_offsets))

        return sm


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
            return jnp.sum(all_i_terms)

        def get_bp_l_sm(bp_idx_hl, l): # ある l に対してそれに対応する summation を計算する。
            # bar_P(h, l) の計算をしている。sum_{i, j} B(f_2) * bar_P(i, j) の部分に該当する。
            h = l - d
            bp = bp_bases[bp_idx_hl]
            bh = int(bp[0]) # bi; i = h - 1
            bl = int(bp[1]) # bj; j = l + 1
            sm = jnp.zeros((), dtype=bar_P.dtype)

            sm += psum_outer_bulges(bh, bl, h, l, padded_p_seq, P)
            sm += psum_outer_internal_loops(bh, bl, h, l, padded_p_seq, P, OMM)

            # stacks
            stack_summands = vmap(get_bp_stack, (0, None, None, None))(jnp.arange(NBPS), l, bh, bl)
            sm += jnp.sum(stack_summands) * s_table[2]

            # Multi-loops
            sm += get_bp_l_multi_sm(l)

            return 


        def get_bp_all_ls(bp_idx):
            ls = jnp.arange(seq_len + 1)
            return vmap(get_bp_l_sm, (None, 0))(bp_idx, ls)
        
        all_bp_js = vmap(get_bp_all_ls)(jnp.arange(NBPS))
        hs, ls = jnp.arange(seq_len + 1 - d), jnp.arange(d, seq_len + 1)
        bar_P = bar_P.at[:, hs, ls].set(all_bp_js)

        return bar_P

    def fill_bar_MB(carry: OutsideCarry, inside: InsideTablesLike, i: int) -> Array:
        """Propagate multibranch helper contributions at position i."""

        raise NotImplementedError


    def fill_bar_M(carry: OutsideCarry, inside: InsideTablesLike, i: int) -> Array:
        """Propagate multibranch DP contributions at position i."""
        # 一旦 forward を貼り付けてます↓
        def nb_j_fn(nb, j):
            nb_j_cond = (j >= i) & (j < seq_len)
            nb_j_sm = ML[nb, i+1, j] * s_table[1]

            idx = jnp.where(nb-1 > 0, nb-1, 0)
            def k_fn(k):
                k_cond = (k >= i) & (k < j+1)
                return jnp.where(k_cond, ML[idx, k+1, j] * MB[i, k], 0.0)
            nb_j_sm += jnp.sum(vmap(k_fn)(jnp.arange(seq_len+1)))

            return jnp.where(nb_j_cond, nb_j_sm, ML[nb, i, j])
        get_nb_j_terms = vmap(vmap(nb_j_fn, (None, 0)), (0, None))

        nb_j_terms = get_nb_j_terms(jnp.arange(3), jnp.arange(seq_len+1))
        ML = ML.at[:, i, :].set(nb_j_terms)
        return ML


    def fill_bar_OMM(
        h: int,
        bar_P: Array,
        padded_p_seq: Array,
        n: int,
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
        def fill_tables_by_step(carry, d):
            bar_OMM, bar_P, bar_M, bar_MB, bar_E, bar_Pm, bar_Pm1 = carry

            bar_P = fill_bar_P(
                d, padded_p_seq, inside.OMM, inside.ML, inside.P, bar_P, bar_Pm, bar_Pm1, bar_E
            )
            # bar_OMM = fill_bar_OMM(h, bar_P, padded_p_seq, seq_len)
            # bar_MB = fill_bar_MB(carry, inside, h)
            bar_M = fill_bar_M(carry, inside, d)

            return (bar_OMM, bar_P, bar_M, bar_MB, bar_E, bar_Pm, bar_Pm1), None
        (bar_OMM, bar_P, bar_M, bar_MB, bar_E, bar_Pm, bar_Pm1), _ = scan(fill_tables_by_step,
                                        (bar_OMM, bar_P, bar_M, bar_MB, bar_E, bar_Pm, bar_Pm1),
                                        jnp.arange(seq_len-1, -1, -1))
        return (bar_P, bar_M, bar_E)

    return outside_partition



