import numpy as np
import pytest

import RNA

from jax_capr.inside_outside import compute_inside_outside
from jax_rnafold.common.utils import TURNER_1999
from jax_rnafold.d0 import energy


def vienna_bpp(seq: str, energy_mode: str) -> np.ndarray:
    RNA.params_load_RNA_Turner1999()
    fc = RNA.fold_compound(seq, RNA.md())
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
    model = energy.StandardNNModel(params_path=TURNER_1999)
    sequences = [
        "AUGGCUACGUAC",
        "CCGAUAGCUAAG",
        "GGCAAUCCGAUC",
    ]

    for seq in sequences:
        ours = compute_inside_outside(seq, model)
        ref = vienna_bpp(seq)
        diff = ours.bpp - ref
        max_abs = np.max(np.abs(diff))
        mean_abs = np.mean(np.abs(diff))
        assert max_abs < 1e-6
        assert mean_abs < 1e-7
