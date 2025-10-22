# jax-capr

## Environment setup

```bash
uv sync
uv pip install -e ./submodules/jax-rnafold
uv pip install ViennaRNA
```

The ViennaRNA Python API is required for the comparison script and tests:

```bash
uv pip install ViennaRNA
```

## Usage

Compare base-pair probabilities produced by the JAX inside/outside DP with ViennaRNA:

```bash
uv run python scripts/compare_bpp.py --length 25 --count 3 --seed 123
```

Run the regression test suite:

```bash
uv run pytest
```

# Modification on Jax-RNAfold
jax-rnafold が出力する dp table を活用したいので、 jax-rnafold のコードを一部修正する必要がある。
具体的には `submodules/jax-rnafold/src/jax_rnafold/d0/ss.py` の `get_ss_partition_fn` で出力される `ss_partition_fn` が,
分配関数 \xi(0) だけしか返さないようになっているところを修正して dp table を返すようにする。

```py
    def ss_partition(p_seq):

        # Pad appropriately
        padded_p_seq = jnp.zeros((seq_len+1, 4), dtype=f64)
        padded_p_seq = padded_p_seq.at[:seq_len].set(p_seq)

        E = jnp.zeros((seq_len+1), dtype=f64)
        P = jnp.zeros((NBPS, seq_len+1, seq_len+1), dtype=f64)
        ML = jnp.zeros((3, seq_len+1, seq_len+1), dtype=f64)
        MB = jnp.zeros((seq_len+1, seq_len+1), dtype=f64)
        OMM = jnp.zeros((seq_len+1, seq_len+1), dtype=f64)
        E = E.at[seq_len].set(1)
        ML = ML.at[0, :, :].set(1)

        @jit
        def fill_table(carry, i):
            OMM, P, ML, MB, E = carry

            P = fill_paired(i, padded_p_seq, OMM, ML, P)
            OMM = fill_outer_mismatch(i, OMM, P, padded_p_seq)
            MB = fill_multibranch(i, MB, P, padded_p_seq)
            ML = fill_multi(i, padded_p_seq, ML, MB)
            E = fill_external(i, E, P, padded_p_seq)

            return (OMM, P, ML, MB, E), None

        (OMM, P, ML, MB, E), _ = scan(fill_table,
                                      (OMM, P, ML, MB, E),
                                      jnp.arange(seq_len-1, -1, -1))

        # return E[0]
        return E[0], (OMM, P, ML, MB, E) # ここだけ変更
``` 


