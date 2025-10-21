# jax-capr

## Environment setup

```bash
uv sync
uv pip install -e ./submodules/jax-rnafold
uv pip install ViennaRNA pandas
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
