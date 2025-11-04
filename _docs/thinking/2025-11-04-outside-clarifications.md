# 2025-11-04 Outside Clarifications

- `MB` は multibranch の 1 枝テーブル（ViennaRNA の `QM1` 相当）で、`P · en_multi_branch · s_table` を再利用するための補助状態。outside では `bar_MB` を保持し、`bar_ML` から `bar_MB` を経由して `bar_P` へ寄与を戻す。
- `OMM` は一般内部ループ用の outer mismatch テーブル。forward で `P` に加算される係数を outside で `bar_OMM` に積み上げ、最終的に `bar_P` へ戻す必要があるため、`bar_OMM` を明示的に保持する。
- 補助テーブル（`MB`, `OMM`）にも outside テーブルを持たせる方針で Stage 1 を進める。forward の構造と一致させておくことで、逆伝播の流れが明瞭になり、後続フェーズ（loop profile, AD）でも扱いやすい。

## 2025-11-05 進捗メモ
- `src/jax_capr/jax_outside.py` に outside 再帰の全項目（外部、マルチ、スタック、bulge/internal、`bar_OMM` 伝播）を実装。現状は Python の多重 `for` ループ主体で、CPU 実行は動作するが GPU 並列化には不向き。
- ViennaRNA 比較テストでは `energy.JaxNNModel` へ切り替えることで JIT 利用時の `TracerIntegerConversionError` を解消し、`uv run tests/test_outside_vs_vienna.py` が通るようになった。
- TODO: `jax_outside.py` を `submodules/jax-rnafold/src/jax_rnafold/d0/ss.py` と同様に `vmap`/`jnp.einsum`/`lax.scan` へ書き換え、Python ループを排除して GPU でも高速に動作させる。既存のロジックを保ちながら段階的にベクトル化する方針で進める。
- ViennaRNA 側は `RNA.md().dangles = 0` を設定し Turner1999 と比較する。デフォルト（dangles=2）のままだと差分が大きくなる。
