# Outside Algorithm Notes (Turner 1999/2004)

## Reference Highlights

- **McCaskill (1990)** `references/McCaskill 1990 - The equilibrium partition function and base pair binding probabilities for RNA secondary structure.pdf`  
  Introduces the classical inside/outside factorisation for RNA partition functions. Defines the `Q(i,j)`/`Q^b(i,j)` recursion (Eq. 12–17) and the outside marginalisation used to obtain base-pair probabilities `p(i,j) = Q^b(i,j) \cdot Q^{b,\text{out}}(i,j) / Q(1,n)`. Establishes the template for Turner-style energy models and justifies symmetry factors for multibranch and exterior loops.

- **Mathews et al. (2004)** `references/Mathews 2004 - Using an RNA secondary structure partition function to determine confidence in base pairs predicted by free energy minimization.pdf`  
  Updates the Turner 1999/2004 parameter set and discusses numerical stabilisation via multiplicative scaling (Section “Partition function calculations”). Highlights how ViennaRNA rescales by `exp(-(βΔG_37)/RT)^L` to prevent under-/overflow and explains that outside recursions reuse the same scaling factors.

- **Fukunaga & Hamada (2014)** `references/Fukunaga et al. 2014 - CapR - revealing structural specificities of RNA-binding protein target recognition using CLIP-seq data.pdf`  
  Describes CapR’s structural profile computation built on Rfold (Turner1999-only) and clarifies the need for explicit outside tables to assemble loop occupancy probabilities (Section 2.2).

- **Matthies & Krueger (Differentiable Partition Function)** `references/Matthies and Krueger - DIFFERENTIABLE PARTITION FUNCTION CALCULATION FOR RNA.pdf`  
  Basis for `jax-rnafold`; presents the JAX-compatible McCaskill formulation, differentiability concerns, and the scaling trick `exp(scale * |segment|)` used instead of log-domain summation (Section 3).

- **ViennaRNA documentation and update notes** `references/btaf203.pdf`  
  Summarises ViennaRNA’s implementation of Turner1999/2004 energies, especially the `pf_scale` heuristic and separate recursions for exterior/multibranch/internal contributions, which we confirmed in code.

## ViennaRNA Inside / Outside States

The canonical implementation lives in `submodules/ViennaRNA/src/ViennaRNA/partfunc/partfunc.c` and `.../probabilities/equilibrium_probs.c`.

| Inside state | Location | Meaning |
| --- | --- | --- |
| `q[i,j]` | `partfunc.c:332-386` | Exterior partition for interval `[i,j]` with free ends (McCaskill `Q(i,j)`). |
| `qb[i,j]` | `partfunc.c:342-383` | Paired segment `[i,j]` forced to pair (`Q^b(i,j)`), assembled from hairpin/internal/multibranch contributions via `decompose_pair`. |
| `qm[i,j]` | `partfunc.c:349-358` | Multibranch aggregator analogous to Vienna’s `QM`. |
| `qm1[i,j]` | `partfunc.c:349-361` | Single-branch helper (`QM1`) for multiloop bifurcations. |
| `qm2[i,j]` | `partfunc.c:347-352` | Optional fast-path for multi-loop closing in circular mode. |
| `q1k[k]`, `qln[k]` | `partfunc.c:372-382` | Linear projections `Q(1,k)` and `Q(k,n)` for post-processing / outside recursions. |
| Scaling arrays | `partfunc.c:336-340`, `probabilities/equilibrium_probs.c:285-320` | Precomputed `scale[L]` ensures inside/outside numerical stability. |

Outside quantities are not stored explicitly; instead, `pf_create_bppm` in `probabilities/equilibrium_probs.c:269-640` recomputes outside contributions on-the-fly, writing base-pair probabilities into `probs[i,j]`. Helper routines such as `compute_bpp_internal` and `compute_bpp_multibranch` realise the classical outside recursion: they multiply the outside weight (a temporary `pr` accumulator) with the matching inside `qb`, `qm`, `q` states and the energy factors that were used in `fill_arrays`. This mirrors the formulae in McCaskill 1990 and Mathews 2004.

## jax-rnafold Inside Tables (Turner1999, dangles 0/2, non-circular)

All recursions are in `submodules/jax-rnafold/src/jax_rnafold/d0/ss.py`.

| Tensor | Location | Meaning |
| --- | --- | --- |
| `E[i]` | `ss.py:56-70` | Exterior partition prefix ending at position `i`. Final partition is `E[0]`. Uses rescaling vector `s_table`. |
| `P[bp_idx, i, j]` | `ss.py:375-411` | Paired interval `[i,j]` with explicit base-pair type index (`NBPS` dimension). Aggregates hairpin, bulge, internal, stacking, and multibranch terms. |
| `OMM[i, j]` | `ss.py:73-97` | “Outer mismatch” accumulator for internal loops; stores Boltzmann weight of the mismatches flanking pair `(i,j)` so that general internal loops can reuse pre-summed contributions (see `ss.py:216-371`). |
| `MB[i, j]` | `ss.py:100-120` | Multibranch helper analogous to Vienna’s `QM1`: collects branch contributions where `(i,j)` is the entering base pair. |
| `ML[nb, i, j]` | `ss.py:413-429` | Multibranch DP layered by `nb∈{0,1,2}`: `nb=0` is the running partition of remaining segment, `nb=1` corresponds to one required branch, `nb=2` matches the “fully formed” core that can close with `P`. `ML[2, i+1, j-1]` is injected into `P` for multiloop closures (`ss.py:399`). |
| Scaling | `s_table` in `ss.py:45-48` | Rescaling term `exp(scale * |segment| / seq_len)` preventing overflow without log-sum-exp. |

These tables are sufficient for the inside (forward) pass but jax-rnafold currently lacks the matching outside tensors, which prevents structural profile and gradient propagation analogous to ViennaRNA’s base-pair probabilities.

## Candidate Outside Tables for jax-rnafold

To mirror ViennaRNA/McCaskill, we propose the following outside tensors (`β` denotes outside weights):

| Outside tensor | Role |
| --- | --- |
| `β_E[i]` | Outside weight for exterior prefix `E[i]`. Initialise `β_E[0] = 1`. |
| `β_P[bp_idx, i, j]` | Outside weight for paired state `P`. Produces base-pair probabilities via `p(i,j) = padded_p_seq[i,bi]·padded_p_seq[j,bj]·P[bp_idx,i,j]·β_P[bp_idx,i,j] / Z`. |
| `β_OMM[i, j]` | Outside accumulator for `OMM`, needed by general internal loop residuals. |
| `β_MB[i, j]` | Outside weight for multibranch helper `MB`. |
| `β_ML[nb, i, j]` | Outside weights for each `ML` layer (`nb ∈ {0,1,2}`). |

All tensors inherit the same scaling factors as their inside counterparts (the `s_table` multipliers cancel between forward/backward passes).

## Outside Recurrences (Draft)

Let `Z = E[0]`. Derivatives are taken with respect to the inside quantities; multiplication by nucleotide priors `padded_p_seq` and Boltzmann factors is implied where present in the forward recurrence.

### Exterior loop (`fill_external`, `ss.py:56-70`)

Forward:  
`E[i] = s₁ · E[i+1] + Σ_{j>i} Σ_{bp} E[j+1] · P[bp,i,j] · B_ext(bp,i,j)`  
with `B_ext = padded_p_seq[i,bi]·padded_p_seq[j,bj]·em.en_ext_branch(bi,bj)·s₂`.

Outside updates:

```
β_E[i+1]     += β_E[i] · s₁
β_E[j+1]     += β_E[i] · P[bp,i,j] · B_ext(bp,i,j)
β_P[bp,i,j]  += β_E[i] · E[j+1]      · B_ext(bp,i,j)
```

Initial condition: `β_E[0] = 1`, all other `β_* = 0`.

### Outer mismatch precomputation (`fill_outer_mismatch`, `ss.py:73-97`)

Forward:  
`OMM[i,j] += Σ_{bp,bim1,bjp1} P[bp,i,j] · B_omm(bp,i,j,bim1,bjp1)`  
with `B_omm = em.en_il_outer_mismatch · padded_p_seq[i-1,bim1] · padded_p_seq[j+1,bjp1] · padded_p_seq[i,bi] · padded_p_seq[j,bj]`.

Outside:

```
β_P[bp,i,j]  += β_OMM[i,j] · B_omm(bp,i,j,bim1,bjp1)  (summed over mismatches)
```

No propagation into `β_OMM` here; it is consumed later by internal-loop general terms.

### Multibranch entry (`fill_multibranch`, `ss.py:100-120`)

Forward:  
`MB[i,j] += Σ_{bp} P[bp,i,j] · B_mb(bp,i,j)` where `B_mb = em.en_multi_branch · padded_p_seq[i,bi] · padded_p_seq[j,bj] · s₂`.

Outside:

```
β_P[bp,i,j]  += β_MB[i,j] · B_mb(bp,i,j)
```

### Multibranch dynamic (`fill_multi`, `ss.py:413-429`)

Forward (per `nb`):
`ML[nb,i,j] = s₁ · ML[nb,i+1,j] + Σ_{k=i}^{j} ML[idx, k+1, j] · MB[i,k]`
with `idx = max(nb-1, 0)`.

Outside:

```
β_ML[nb,i+1,j]       += β_ML[nb,i,j] · s₁
β_ML[idx,k+1,j]      += β_ML[nb,i,j] · MB[i,k]
β_MB[i,k]            += β_ML[nb,i,j] · ML[idx,k+1,j]
```

Boundary handling: `ML[0,*,*]` is initialised to 1 (`ss.py:444`), so only `β_ML[0]` needs a seed when it feeds into other transitions.

### Paired state (`fill_paired`, `ss.py:375-411`)

Forward decomposition:

```
P[bp,i,j] = H(bp,i,j)                      (hairpin)
          + Σ bulge terms using P[bp, i+1, l] or P[bp, k, j-1]
          + Σ internal-loop terms using P[bp, …] and OMM[…]
          + Σ stack terms: Σ_{bp'} P[bp', i+1, j-1] · B_stack
          + em.en_multi_closing · ML[2, i+1, j-1]
```

Outside contributions (schematic):

- **Hairpin**: terminal, no propagation.
- **Bulges** (`psum_bulges`, `ss.py:216-302`):

```
β_P[bp,i+1,l]  += β_P[bp,i,j] · B_bulge_right
β_P[bp,k,j-1]  += β_P[bp,i,j] · B_bulge_left
```

where `B_bulge_*` match the forward factors in `psum_bulges`.

- **Internal loops** (`ss.py:216-371`):

  - Special 1×1/1×n/2×2/2×3/3×2 cases propagate to the corresponding `P[bp,k,l]` the same way bulges do, with weights `B_int_special` drawn from the forward code.
  - General loops (`general_kl_sm`) propagate to `β_OMM[k,l]`:

```
β_OMM[k,l]    += β_P[bp,i,j] · em.en_internal_init(lup+rup)
                 · em.en_internal_asym(lup,rup) · mmij · s_table[lup+rup+2]
```

- **Stacking** (`get_bp_stack`):

```
β_P[bp', i+1, j-1] += β_P[bp,i,j] · B_stack(bp,bp',i,j)
```

with `B_stack = padded_p_seq[i+1,bk]·padded_p_seq[j-1,bl]·em.en_stack`.

- **Multibranch closure**:

```
β_ML[2, i+1, j-1]   += β_P[bp,i,j] · em.en_multi_closing(bi,bj)
```

All additive factors carry the same scaling multiplier (`s_table`) as in the forward sums; because `β_P` multiplies an already scaled `P`, the rescaling cancels out in the final probabilities.

### General Notes

- Boundary conditions (e.g. `i-1` or `j+1` indexing) follow the same masking as the forward pass (`cond` guards in the JAX code).
- Once the backward sweep finishes, loop-profile probabilities can be computed analogously to CapR: multiply the relevant inside weight with its outside counterpart, divide by `Z`, and normalise per nucleotide.

## Next Steps

1. Formalise the bulge/internal-loop propagation factors by extracting the exact `psum_bulges` and `psum_internal_loops` expressions.
2. Prototype the outside sweep in pure Python/JAX to validate against ViennaRNA base-pair probabilities (Turner1999, dangles {0,2}).
3. Extend the probability assembly to loop-level occupancies (`hairpin`, `bulge`, `internal`, `multi`, `exterior`) mirroring CapR, using the newly defined outside tensors.
