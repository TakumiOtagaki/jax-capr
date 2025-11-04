# 2025-11-04 Outside Clarifications

- `MB` は multibranch の 1 枝テーブル（ViennaRNA の `QM1` 相当）で、`P · en_multi_branch · s_table` を再利用するための補助状態。outside では `bar_MB` を保持し、`bar_ML` から `bar_MB` を経由して `bar_P` へ寄与を戻す。
- `OMM` は一般内部ループ用の outer mismatch テーブル。forward で `P` に加算される係数を outside で `bar_OMM` に積み上げ、最終的に `bar_P` へ戻す必要があるため、`bar_OMM` を明示的に保持する。
- 補助テーブル（`MB`, `OMM`）にも outside テーブルを持たせる方針で Stage 1 を進める。forward の構造と一致させておくことで、逆伝播の流れが明瞭になり、後続フェーズ（loop profile, AD）でも扱いやすい。

## 2025-11-05 進捗メモ
- `src/jax_capr/jax_outside.py` に outside 再帰の全項目（外部、マルチ、スタック、bulge/internal、`bar_OMM` 伝播）を実装。現状は Python の多重 `for` ループ主体で、CPU 実行は動作するが GPU 並列化には不向き。
- ViennaRNA 比較テストでは `energy.JaxNNModel` へ切り替えることで JIT 利用時の `TracerIntegerConversionError` を解消し、`uv run tests/test_outside_vs_vienna.py` が通るようになった。
- TODO: `jax_outside.py` を `submodules/jax-rnafold/src/jax_rnafold/d0/ss.py` と同様に `vmap`/`jnp.einsum`/`lax.scan` へ書き換え、Python ループを排除して GPU でも高速に動作させる。既存のロジックを保ちながら段階的にベクトル化する方針で進める。
- ViennaRNA 側は `RNA.md().dangles = 0` を設定し Turner1999 と比較する。デフォルト（dangles=2）のままだと差分が大きくなる。

## 2025-11-05 設計メモ
- inside 側の実装（`submodules/jax-rnafold/src/jax_rnafold/d0/ss.py`）と擬似コードを照合して、外部 DP も同じ shape とスケーリング（`s_table`）を厳守する。`p_seq` は forward と同じくパディング済み配列を共有し、`bp_bases`・エネルギー項は事前に `jnp.array` 化して使い回す。
- 外部テーブル更新はすべて `lax.scan`／`vmap`／`jnp.where` で表現する。`bar_E` は外部ループを `scan` で降順処理し、base condition（`bar_E[0] = 1`）から `bar_P` と `bar_E` へベクトル化した一括更新で伝播する。
- `bar_MB` と `bar_M` は forward の `fill_multibranch`／`fill_multi` と鏡写しになるよう、`i` 降順の `scan` と `j`・`nb` の `vmap` で構成する。`ML[idx, k+1, j]` と `MB[i, k]` の掛け合わせで `bar_M` ⇄ `bar_MB` を連結する。
- スタック／マルチ閉じ／バルジ／内部ループの寄与は `BP` 次元を `vmap` し、ループ長の条件は `jnp.arange` とマスクで吸収する。特別ループ（2×2, 2×3, 3×2）と一般内部ループは `MAX_LOOP` を上限に一括演算し、`bar_OMM` を経由して最終的に `bar_P` へ戻す。
- 全行程で `s_table` の添字と指数が forward と一致するかを都度確認し、実装後は `uv run python tests/test_outside_vs_vienna.py` を用いて `max_abs < 1e-6`, `mean_abs < 1e-7` を達成するまで微調整する。
