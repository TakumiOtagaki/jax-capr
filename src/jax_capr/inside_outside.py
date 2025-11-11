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
    bar_Pm: Array
    bar_Pm1: Array


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


def compute_outside_tables(
    sequence: str | Array,
    model,
    *,
    max_loop: int | None = None,
    scale: float | None = None,
    checkpoint_every: int | None = 10,
) -> OutsideTables:
    """Run the outside DP using inside tables as prerequisites."""
    inside = compute_inside_tables(
        sequence,
        model,
        max_loop=max_loop,
        scale=scale,
        checkpoint_every=checkpoint_every,
    )
    outside = jax_outside.compute_outside(
        inside,
        model,
        max_loop=max_loop,
        checkpoint_every=checkpoint_every,
    )
    return OutsideTables(
        bar_E=outside.bar_E,
        bar_P=outside.bar_P,
        bar_M=outside.bar_M,
        bar_Pm=outside.bar_Pm,
        bar_Pm1=outside.bar_Pm1,
    )


def assemble_bpp_matrix(inside: InsideTables, outside: OutsideTables) -> Array:
    """Compute the base-pair probability matrix."""
    if inside.partition == 0.0:
        raise ValueError("Partition function is zero; cannot assemble BPP matrix.")

    seq_len = inside.p_seq.shape[0]
    # Only retain real sequence positions before combining inside/outside terms.
    inside_pairs = inside.P[:, :seq_len, :seq_len]
    outside_pairs = outside.bar_P[:, :seq_len, :seq_len]

    # bpp[i, j] = sum_bp inside.P[bp, i, j] * outside.bar_P[bp, i, j].
    bpp = jnp.einsum("kij,kij->ij", inside_pairs, outside_pairs)

    bpp = jnp.triu(bpp, k=1)
    bpp = bpp + bpp.T
    bpp = bpp.at[jnp.arange(seq_len), jnp.arange(seq_len)].set(0.0)
    return bpp


def compute_inside_outside(
    sequence: str | Array,
    model,
    *,
    max_loop: int | None = None,
    scale: float | None = None,
    checkpoint_every: int | None = 10,
) -> InsideOutsideResult:
    """Compute partition function and base-pair probabilities."""
    inside = compute_inside_tables(
        sequence,
        model,
        max_loop=max_loop,
        scale=scale,
        checkpoint_every=checkpoint_every,
    )
    outside_raw = jax_outside.compute_outside(
        inside,
        model,
        max_loop=max_loop,
        checkpoint_every=checkpoint_every,
    )
    outside = OutsideTables(
        bar_E=outside_raw.bar_E,
        bar_P=outside_raw.bar_P,
        bar_M=outside_raw.bar_M,
        bar_Pm=outside_raw.bar_Pm,
        bar_Pm1=outside_raw.bar_Pm1,
    )
    bpp = assemble_bpp_matrix(inside, outside)

    return InsideOutsideResult(
        partition=inside.partition,
        bpp=bpp,
        loop_profile=None,
        inside=inside,
        outside=outside,
    )


def compute_bpp_matrix(
    sequence: str | Array,
    model,
    *,
    max_loop: int | None = None,
    scale: float | None = None,
    checkpoint_every: int | None = 10,
) -> Array:
    """Convenience helper returning only the base-pair probability matrix."""
    return compute_inside_outside(
        sequence,
        model,
        max_loop=max_loop,
        scale=scale,
        checkpoint_every=checkpoint_every,
    ).bpp


__all__ = [
    "InsideTables",
    "OutsideTables",
    "InsideOutsideResult",
    "compute_inside_tables",
    "compute_outside_tables",
    "compute_inside_outside",
    "compute_bpp_matrix",
]
