"""Public API scaffolding for JAX-based inside/outside computations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import jax_inside
from . import jax_outside

import jax.numpy as jnp


Array = jnp.ndarray


@dataclass(slots=True)
class InsideTables:
    """Inside DP tables and metadata returned by the forward recursion."""

    partition: float
    E: Array
    P: Array
    ML: Array
    MB: Array
    OMM: Array
    p_seq: Array
    s_table: Array
    scale: float


@dataclass(slots=True)
class OutsideTables:
    """Outside DP tables produced by the backward recursion."""

    bar_E: Array
    bar_P: Array
    bar_M: Array
    bar_MB: Array
    bar_OMM: Array


@dataclass(slots=True)
class InsideOutsideResult:
    """Aggregated result bundling partition, BPP, and optional loop profiles."""

    partition: float
    bpp: Array
    loop_profile: Optional[Array]
    inside: Optional[InsideTables] = None
    outside: Optional[OutsideTables] = None


def compute_inside_tables(
    sequence: str | Array,
    model,
    *,
    max_loop: int | None = None,
    scale: float | None = None,
    checkpoint_every: int | None = 10,
) -> InsideTables:
    """Run the inside DP and return tables needed by the outside recursion."""
    inside = jax_inside.compute_inside(
        sequence,
        model,
        max_loop=max_loop,
        scale=scale,
        checkpoint_every=checkpoint_every,
    )

    partition = float(inside.partition)
    return InsideTables(
        partition=partition,
        E=inside.E,
        P=inside.P,
        ML=inside.ML,
        MB=inside.MB,
        OMM=inside.OMM,
        p_seq=inside.p_seq,
        s_table=inside.s_table,
        scale=inside.scale,
    )


def compute_outside_tables(sequence: str | Array, model) -> OutsideTables:
    """Run the outside DP using inside tables as prerequisites."""
    inside = compute_inside_tables(sequence, model)
    outside = jax_outside.compute_outside(inside, model)
    return OutsideTables(
        bar_E=outside.bar_E,
        bar_P=outside.bar_P,
        bar_M=outside.bar_M,
        bar_MB=outside.bar_MB,
        bar_OMM=outside.bar_OMM,
    )


def assemble_bpp_matrix(inside: InsideTables, outside: OutsideTables) -> Array:
    """Compute the base-pair probability matrix."""
    if inside.partition == 0.0:
        raise ValueError("Partition function is zero; cannot assemble BPP matrix.")

    weighted = jnp.sum(inside.P * outside.bar_P, axis=0) / inside.partition
    seq_len = inside.p_seq.shape[0]
    bpp = jnp.array(weighted[:seq_len, :seq_len])

    bpp = jnp.triu(bpp, k=1)
    bpp = bpp + bpp.T
    bpp = bpp.at[jnp.arange(seq_len), jnp.arange(seq_len)].set(0.0)
    return bpp


def compute_inside_outside(sequence: str | Array, model) -> InsideOutsideResult:
    """Compute partition function and base-pair probabilities."""
    inside = compute_inside_tables(sequence, model)
    outside_raw = jax_outside.compute_outside(inside, model)
    outside = OutsideTables(
        bar_E=outside_raw.bar_E,
        bar_P=outside_raw.bar_P,
        bar_M=outside_raw.bar_M,
        bar_MB=outside_raw.bar_MB,
        bar_OMM=outside_raw.bar_OMM,
    )
    bpp = assemble_bpp_matrix(inside, outside)

    return InsideOutsideResult(
        partition=inside.partition,
        bpp=bpp,
        loop_profile=None,
        inside=inside,
        outside=outside,
    )


def compute_bpp_matrix(sequence: str | Array, model) -> Array:
    """Convenience helper returning only the base-pair probability matrix."""
    return compute_inside_outside(sequence, model).bpp


__all__ = [
    "InsideTables",
    "OutsideTables",
    "InsideOutsideResult",
    "compute_inside_tables",
    "compute_outside_tables",
    "compute_inside_outside",
    "compute_bpp_matrix",
]
