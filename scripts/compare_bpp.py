#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

import RNA

from jax_capr.inside_outside import compute_inside_outside
from jax_rnafold.common.utils import TURNER_1999
from jax_rnafold.d0 import energy


def random_sequences(length: int, count: int, seed: int) -> List[str]:
    rng = np.random.default_rng(seed)
    alphabet = np.array(list("ACGU"))
    return ["".join(rng.choice(alphabet, size=length)) for _ in range(count)]


def vienna_bpp(seq: str) -> np.ndarray:
    fc = RNA.fold_compound(seq)
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


@dataclass
class ComparisonResult:
    sequence: str
    max_abs: float
    mean_abs: float
    rmse: float
    ours: np.ndarray
    vienna: np.ndarray
    diff: np.ndarray


def compare_sequences(
    seqs: Iterable[str], model: energy.Model
) -> List[ComparisonResult]:
    results: List[ComparisonResult] = []
    for seq in seqs:
        ours = compute_inside_outside(seq, model)
        ref = vienna_bpp(seq)
        diff = ours.bpp - ref
        max_abs = float(np.max(np.abs(diff)))
        mean_abs = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff**2)))
        results.append(
            ComparisonResult(
                sequence=seq,
                max_abs=max_abs,
                mean_abs=mean_abs,
                rmse=rmse,
                ours=ours.bpp,
                vienna=ref,
                diff=diff,
            )
        )
    return results


def print_matrix(label: str, matrix: np.ndarray, seq: str) -> None:
    n = len(seq)
    index = [f"{i+1}" for i in range(n)]
    df = pd.DataFrame(matrix, index=index, columns=index)
    pd.set_option("display.width", None)
    pd.set_option("display.max_rows", n)
    pd.set_option("display.max_columns", n)
    print(f"{label}:")
    print(df.round(4))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare JAX-based inside/outside BPPs with ViennaRNA."
    )
    parser.add_argument("--length", type=int, default=20, help="Sequence length.")
    parser.add_argument(
        "--count", type=int, default=5, help="Number of random sequences."
    )
    parser.add_argument("--seed", type=int, default=17, help="Random seed.")
    args = parser.parse_args()

    seqs = random_sequences(args.length, args.count, args.seed)
    model = energy.StandardNNModel(params_path=TURNER_1999)

    print(f"Comparing {len(seqs)} sequences (length={args.length})")
    for res in compare_sequences(seqs, model):
        print(
            f"Sequence: {res.sequence}\n"
            f"  max|Δ|={res.max_abs:.3e}, mean|Δ|={res.mean_abs:.3e}, RMSE={res.rmse:.3e}"
        )
        print_matrix("Our BPP", res.ours, res.sequence)
        print_matrix("ViennaRNA BPP", res.vienna, res.sequence)
        print_matrix("Difference (ours - ViennaRNA)", res.diff, res.sequence)
        print("-" * 60)


if __name__ == "__main__":
    main()
