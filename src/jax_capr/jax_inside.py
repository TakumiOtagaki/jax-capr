"""Utilities for constructing inside DP tables via jax-rnafold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import jax.numpy as jnp

from jax_rnafold.common import utils as rna_utils
from jax_rnafold.common.utils import MAX_LOOP
from jax_rnafold.d0 import ss as ss_module


Array = jnp.ndarray


@dataclass(slots=True)
class InsideComputation:
    """Raw outputs produced by the jax-rnafold inside routine."""

    partition: Array
    E: Array
    P: Array
    ML: Array
    MB: Array
    OMM: Array
    p_seq: Array
    s_table: Array
    scale: float


def prepare_probability_sequence(sequence: str | Array) -> Array:
    """Convert a nucleotide sequence or probabilistic encoding to `[n,4]` array."""
    if isinstance(sequence, str):
        seq = sequence.strip().upper()
        if not seq:
            raise ValueError("Sequence string must not be empty.")
        one_hot = rna_utils.seq_to_one_hot(seq)
        return jnp.array(one_hot, dtype=jnp.float64)

    arr = jnp.asarray(sequence, dtype=jnp.float64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(
            "Probabilistic sequence must have shape [length, 4]; "
            f"received {tuple(arr.shape)}."
        )
    return arr


def default_scale(seq_len: int) -> float:
    """Return the recommended scaling factor for a given sequence length."""
    return -1.5 * float(seq_len)


def compute_inside(sequence: str | Array,
                   model,
                   *,
                   max_loop: int | None = None,
                   scale: float | None = None,
                   checkpoint_every: int | None = 10) -> InsideComputation:
    """Run jax-rnafold's inside recursion and collect intermediate tables."""
    p_seq = prepare_probability_sequence(sequence)
    seq_len = int(p_seq.shape[0])

    if max_loop is None:
        max_loop = MAX_LOOP
    if scale is None:
        scale = default_scale(seq_len)

    ss_fn = ss_module.get_ss_partition_fn(
        model,
        seq_len,
        max_loop=max_loop,
        scale=scale,
        checkpoint_every=checkpoint_every,
    )

    partition, tables = ss_fn(p_seq)
    OMM, P, ML, MB, E = tables

    # The inside routine internally constructs s_table using the same scale.
    s_table = jnp.array([jnp.exp(i * scale / seq_len) for i in range(seq_len + 5)],
                        dtype=jnp.float64)

    return InsideComputation(
        partition=partition,
        E=E,
        P=P,
        ML=ML,
        MB=MB,
        OMM=OMM,
        p_seq=p_seq,
        s_table=s_table,
        scale=scale,
    )
