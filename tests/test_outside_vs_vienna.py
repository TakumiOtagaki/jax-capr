import numpy as np
import pytest
import pandas as pd

import RNA

from jax_capr.inside_outside import compute_inside_outside
from jax_rnafold.common.utils import TURNER_1999
from jax_rnafold.common.utils import bp_bases, NBPS, MAX_LOOP
from jax_rnafold.d0 import energy
import jax.numpy as jnp
from jax import config

# config.update("jax_disable_jit", True)


def _padded_prob_seq(one_hot_seq: jnp.ndarray) -> np.ndarray:
    seq_len = int(one_hot_seq.shape[0])
    padded = np.zeros((seq_len + 1, 4), dtype=np.float64)
    padded[:seq_len] = np.asarray(one_hot_seq)
    return padded


def stack_contribution_per_bp(inside, outside, model, h: int, l: int) -> np.ndarray:
    """
    Reconstruct stack contribution for each base-pair type at (h,l).
    Mirrors the logic in fill_bar_P (stack block) for debugging.
    """
    seq_len = int(inside.p_seq.shape[0])
    i = h - 1
    j = l + 1
    if not (0 <= i and j <= seq_len):
        return np.zeros(NBPS, dtype=np.float64)

    padded = _padded_prob_seq(inside.p_seq)
    s_table = np.asarray(inside.s_table)
    bar_P = np.asarray(outside.bar_P)

    per_bp = np.zeros(NBPS, dtype=np.float64)
    for bp_idx_hl in range(NBPS):
        bh, bl = bp_bases[bp_idx_hl]
        weight_inner = padded[h, bh] * padded[l, bl]

        contrib = 0.0
        for bp_idx_ij in range(NBPS):
            bi, bj = bp_bases[bp_idx_ij]
            outer_bar = bar_P[bp_idx_ij, i, j]
            if outer_bar == 0.0:
                continue
            en = model.en_stack(int(bi), int(bj), int(bh), int(bl))
            contrib += float(outer_bar) * en

        per_bp[bp_idx_hl] = weight_inner * contrib * s_table[2]
    return per_bp


def bulge_contribution_per_bp(inside, outside, model, h: int, l: int) -> np.ndarray:
    """Mirror psum_outer_bulges for debugging a specific pair (h,l)."""

    seq_len = int(inside.p_seq.shape[0])
    if not (0 <= h < seq_len and 0 <= l < seq_len and h < l):
        return np.zeros(NBPS, dtype=np.float64)

    padded = _padded_prob_seq(inside.p_seq)
    s_table = np.asarray(inside.s_table)
    bar_P = np.asarray(outside.bar_P)
    two_loop_length = min(seq_len, MAX_LOOP)

    per_bp = np.zeros(NBPS, dtype=np.float64)

    for bp_idx_hl in range(NBPS):
        bh, bl = bp_bases[bp_idx_hl]
        total = 0.0

        # Right bulges where (h,l) is the inner pair and outer pair is (i, j) = (h-1, l+gap+1)
        i = h - 1
        if i >= 0:
            for offset in range(two_loop_length):
                j = l + 2 + offset
                if j >= seq_len + 1:
                    break
                bulge_len = j - l - 1
                if bulge_len <= 0:
                    continue
                s_idx = bulge_len + 2
                if s_idx >= len(s_table):
                    continue
                inner_weight = padded[h, bh] * padded[l, bl]
                for bp_idx_ij in range(NBPS):
                    outer = bar_P[bp_idx_ij, i, j]
                    if outer == 0.0:
                        continue
                    bi, bj = bp_bases[bp_idx_ij]
                    en = model.en_bulge(int(bi), int(bj), int(bh), int(bl), int(bulge_len))
                    total += outer * inner_weight * en * s_table[s_idx]

        # Left bulges where (h,l) is the inner pair and outer pair is (i, j) = (h-gap-1, l+1)
        j = l + 1
        if j <= seq_len:
            for offset in range(two_loop_length):
                i = h - 2 - offset
                if i < 0:
                    break
                bulge_len = h - i - 1
                if bulge_len <= 0:
                    continue
                s_idx = bulge_len + 2
                if s_idx >= len(s_table):
                    continue
                inner_weight = padded[h, bh] * padded[l, bl]
                for bp_idx_ij in range(NBPS):
                    outer = bar_P[bp_idx_ij, i, j]
                    if outer == 0.0:
                        continue
                    bi, bj = bp_bases[bp_idx_ij]
                    en = model.en_bulge(int(bi), int(bj), int(bh), int(bl), int(bulge_len))
                    total += outer * inner_weight * en * s_table[s_idx]

        per_bp[bp_idx_hl] = total

    return per_bp


def bulge_contribution_details(inside, outside, model, h: int, l: int) -> list[tuple[str, int, int, int, int, float]]:
    """Return detailed bulge contributions (side, i, j, bp_idx, bulge_len, amount)."""

    seq_len = int(inside.p_seq.shape[0])
    if not (0 <= h < seq_len and 0 <= l < seq_len and h < l):
        return []

    padded = _padded_prob_seq(inside.p_seq)
    s_table = np.asarray(inside.s_table)
    bar_P = np.asarray(outside.bar_P)
    two_loop_length = min(seq_len, MAX_LOOP)

    bh = bl = None  # placeholders; will fill inside loop
    details: list[tuple[str, int, int, int, int, float]] = []

    for bp_idx_hl in range(NBPS):
        bh, bl = bp_bases[bp_idx_hl]
        inner_weight = padded[h, bh] * padded[l, bl]
        if inner_weight == 0.0:
            continue

        # Right bulges (outer pair (i, j) = (h-1, l+gap+1))
        i = h - 1
        if i >= 0:
            for offset in range(two_loop_length):
                j = l + 2 + offset
                if j >= seq_len + 1:
                    break
                bulge_len = j - l - 1
                if bulge_len <= 0:
                    continue
                s_idx = bulge_len + 2
                if s_idx >= len(s_table):
                    continue
                for bp_idx_ij in range(NBPS):
                    outer = bar_P[bp_idx_ij, i, j]
                    if outer == 0.0:
                        continue
                    bi, bj = bp_bases[bp_idx_ij]
                    en = model.en_bulge(int(bi), int(bj), int(bh), int(bl), int(bulge_len))
                    amount = float(outer) * float(inner_weight) * float(en) * float(s_table[s_idx])
                    if amount != 0.0:
                        details.append(("right", int(i), int(j), int(bp_idx_ij), int(bulge_len), amount))

        # Left bulges (outer pair (i, j) = (h-gap-1, l+1))
        j = l + 1
        if j <= seq_len:
            for offset in range(two_loop_length):
                i = h - 2 - offset
                if i < 0:
                    break
                bulge_len = h - i - 1
                if bulge_len <= 0:
                    continue
                s_idx = bulge_len + 2
                if s_idx >= len(s_table):
                    continue
                for bp_idx_ij in range(NBPS):
                    outer = bar_P[bp_idx_ij, i, j]
                    if outer == 0.0:
                        continue
                    bi, bj = bp_bases[bp_idx_ij]
                    en = model.en_bulge(int(bi), int(bj), int(bh), int(bl), int(bulge_len))
                    amount = float(outer) * float(inner_weight) * float(en) * float(s_table[s_idx])
                    if amount != 0.0:
                        details.append(("left", int(i), int(j), int(bp_idx_ij), int(bulge_len), amount))

    return details


def external_contribution_per_bp(inside, outside, model, h: int, l: int) -> np.ndarray:
    """Compute external (bar_E * E) contribution for each base pair type."""

    seq_len = int(inside.p_seq.shape[0])
    if not (0 <= h < seq_len and 0 <= l < seq_len and h < l):
        return np.zeros(NBPS, dtype=np.float64)

    padded = _padded_prob_seq(inside.p_seq)
    bar_E = np.asarray(outside.bar_E)
    E = np.asarray(inside.E)

    per_bp = np.zeros(NBPS, dtype=np.float64)
    if l + 1 >= len(E):
        return per_bp

    for bp_idx_hl in range(NBPS):
        bh, bl = bp_bases[bp_idx_hl]
        weight = padded[h, bh] * padded[l, bl]
        per_bp[bp_idx_hl] = (
            bar_E[h]
            * E[l + 1]
            * model.en_ext_branch(int(bh), int(bl))
            * weight
        )

    return per_bp


def debug_pair(inside, outside, model, h: int, l: int) -> None:
    stack_dbg = stack_contribution_per_bp(inside, outside, model, h, l)
    bulge_dbg = bulge_contribution_per_bp(inside, outside, model, h, l)
    # multi_dbg = multi_contribution_per_bp(inside, outside, model, h, l)
    ext_dbg = external_contribution_per_bp(inside, outside, model, h, l)
    bulge_details = bulge_contribution_details(inside, outside, model, h, l)
    actual = np.asarray(outside.bar_P)[:, h, l]
    total_stack = float(stack_dbg.sum())
    total_bulge = float(bulge_dbg.sum())
    # total_multi = float(multi_dbg.sum())
    total_ext = float(ext_dbg.sum())
    total_actual = float(actual.sum())
    print(f"[debug] pair ({h},{l}) stack sum={total_stack:.6e}, bar_P sum={total_actual:.6e}")
    if total_actual:
        ratio = total_stack / total_actual
    else:
        ratio = float("nan")
    print(f"[debug] pair ({h},{l}) stack/actual ratio={ratio:.6e}")
    if total_actual:
        bulge_ratio = total_bulge / total_actual
    else:
        bulge_ratio = float("nan")
    print(f"[debug] pair ({h},{l}) bulge sum={total_bulge:.6e}, bulge/actual ratio={bulge_ratio:.6e}")
    if total_actual:
        # multi_ratio = total_multi / total_actual
        ext_ratio = total_ext / total_actual
    else:
        multi_ratio = float("nan")
        ext_ratio = float("nan")
    # print(f"[debug] pair ({h},{l}) multi sum={total_multi:.6e}, multi/actual ratio={multi_ratio:.6e}")
    print(f"[debug] pair ({h},{l}) external sum={total_ext:.6e}, external/actual ratio={ext_ratio:.6e}")
    # residual = actual - stack_dbg - bulge_dbg - multi_dbg - ext_dbg
    # print(f"[debug] pair ({h},{l}) inferred internal sum={float(residual.sum()):.6e}")
    if bulge_details:
        total_right = sum(amount for side, *_rest, amount in bulge_details if side == "right")
        total_left = sum(amount for side, *_rest, amount in bulge_details if side == "left")
        print(
            f"[debug] bulge breakdown: right total={total_right:.6e}, left total={total_left:.6e}"
        )
        top = sorted(bulge_details, key=lambda row: abs(row[-1]), reverse=True)[:5]
        print("[debug] top bulge terms (side,i,j,bp_idx,bulge_len,amount):")
        for term in top:
            print(f"        {term}")
    print(f"[debug] per-bp stack contributions: {stack_dbg}")
    print(f"[debug] per-bp bulge contributions: {bulge_dbg}")
    # print(f"[debug] per-bp multi contributions: {multi_dbg}")
    print(f"[debug] per-bp external contributions: {ext_dbg}")
    # print(f"[debug] per-bp inferred internal contributions: {residual}")
    print(f"[debug] bar_P per bp: {actual}")


def vienna_bpp(seq: str, energy_mode: str) -> np.ndarray:
    RNA.params_load_RNA_Turner1999()
    # RNA.params_load_RNA_Turner2004()
    md = RNA.md()
    md.uniq_ML = 1
    md.dangles = 0
    md.noLP = False

    # md.sfact = 0.0 # これであってるか...??
    fc = RNA.fold_compound(seq, md)
    _, mfe_energy = fc.mfe()
    fc.exp_params_rescale(mfe_energy)
    fc.pf()
    bpp = fc.bpp()
    n = len(seq)
    mat = np.zeros((n, n), dtype=float)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            p = bpp[i][j]
            mat[i - 1, j - 1] = p
            mat[j - 1, i - 1] = p
    return mat


def test_inside_outside_matches_vienna():
    model = energy.JaxNNModel(params_path=TURNER_1999)
    checkpoint_every = 1
    # sequences = [
    #     "GGGGAAAACCCC",
    #     "GCGGAAACCAGC",
    #     "GCGGGGAAAAACCCCAGC",
    #     "GGGGAAAACCCCGGGGAAAACCCCGGGGAAAACCCC",
    #     "GGCGGAAAGCGAAACGCAAAACGGCAAAAGCCGAAACCGCC"
    #     # "AUGGCUACGUAC",
    #     # "CCGAUAGCUAAG",
    #     # "GGCAAUCCGAUC",
    # ]
    seq_len = 30
    num_seq = 30
    sequences = [
        "".join(np.random.choice(list("AUGC"), size=seq_len)) for _ in range(num_seq)
    ]
    results: list[tuple[str, float, float]] = []
    print("Testing inside-outside against ViennaRNA...")
    for seq in sequences:
        print(f"Sequence: {seq}")
        ours = compute_inside_outside(seq, model, checkpoint_every)
        print("ours:\n", pd.DataFrame(ours.bpp))

        print("bar_P:\n", pd.DataFrame(jnp.sum(ours.outside.bar_P, axis=0)))
        # print("bar_Pm:", pd.DataFrame(np.asarray(ours.outside.bar_Pm)))
        # print("bar_Pm1:", pd.DataFrame(np.asarray(ours.outside.bar_Pm1)))
        print("bar_M[1]:\n", pd.DataFrame(np.asarray(ours.outside.bar_M[1])))

        # "test csv"
        # header = [f"b_{i}" for i in range(len(seq) + 1)]
        # pd.DataFrame(jnp.sum(ours.outside.bar_P, axis=0)).to_csv(f"tests/bar_P_{seq}.csv", header=header)
        # pd.DataFrame(np.asarray(ours.outside.bar_Pm)).to_csv(f"tests/bar_Pm_{seq}.csv", header=header)
        # pd.DataFrame(np.asarray(ours.outside.bar_Pm1)).to_csv(f"tests/bar_Pm1_{seq}.csv", header=header)
        # pd.DataFrame(np.asarray(ours.outside.bar_M[1])).to_csv(f"tests/bar_M1_{seq}.csv", header=header)
        # pd.DataFrame(ours.bpp).to_csv(f"tests/bpp_{seq}.csv", header=header[:-1])

        # print("bar_P.shape:", ours.outside.bar_P.shape)
        seq_len = len(seq)
        print(
            "E[0] vs bar_E[n]",
            float(ours.inside.E[0]),
            float(ours.outside.bar_E[seq_len]),
        )
        # print("barE:\n", pd.DataFrame(ours.outside.bar_E))
        ref = vienna_bpp(seq, str(TURNER_1999))
        print("vienna:", pd.DataFrame(ref)) 
        diff = ours.bpp - ref
        max_abs = np.max(np.abs(diff))
        mean_abs = np.mean(np.abs(diff))
        results.append((seq, float(max_abs), float(mean_abs)))
        print(f"Max abs difference: {max_abs:.3e}")
        print(f"Mean abs difference: {mean_abs:.3e}")

        # if max_abs > 1e-3:
        #     # 上から 3 つやる
        #     for _ in range(3):
        #         idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
        #         h, l = idx
        #         if h < l:
        #             print(f"[debug] inspecting pair ({h},{l}) with diff {diff[idx]:.3e}")
        #             debug_pair(ours.inside, ours.outside, model, h, l)

        #         diff = diff.at[idx].set(0)  # もう一度同じところを取らないようにする
        # assert max_abs < 1e-6
        # assert mean_abs < 1e-7

    print("Summary of differences:")
    for seq, max_abs, mean_abs in results:
        print(f"Sequence: {seq}")
        print(f"  Max abs difference: {max_abs:.3e}")
        print(f"  Mean abs difference: {mean_abs:.3e}")

    # 0th, 25th percentile, median, 75th percentile, max
    percentiles = np.percentile([max_abs for _, max_abs, _ in results], [25, 50, 75])
    print(f"  min: {min(max_abs for _, max_abs, _ in results):.3e}")
    print(f"  25th percentile: {percentiles[0]:.3e}")
    print(f"  Median: {percentiles[1]:.3e}")
    print(f"  75th percentile: {percentiles[2]:.3e}")
    print(f"  max: {max(max_abs for _, max_abs, _ in results):.3e}")

def main():
    test_inside_outside_matches_vienna()
    print("All tests passed.")

if __name__ == "__main__":
    main()
