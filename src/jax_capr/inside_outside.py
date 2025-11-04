"""Public API scaffolding for JAX-based inside/outside computations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import jax_inside

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
    raise NotImplementedError("Outside tables computation is not implemented yet.")


def compute_inside_outside(sequence: str | Array, model) -> InsideOutsideResult:
    """Compute partition function and base-pair probabilities."""
    raise NotImplementedError("Inside/outside computation is not implemented yet.")


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
