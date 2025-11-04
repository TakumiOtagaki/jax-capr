# 2025-11-04 Outside Clarifications

- `MB` は multibranch の 1 枝テーブル（ViennaRNA の `QM1` 相当）で、`P · en_multi_branch · s_table` を再利用するための補助状態。outside では `bar_MB` を保持し、`bar_ML` から `bar_MB` を経由して `bar_P` へ寄与を戻す。
- `OMM` は一般内部ループ用の outer mismatch テーブル。forward で `P` に加算される係数を outside で `bar_OMM` に積み上げ、最終的に `bar_P` へ戻す必要があるため、`bar_OMM` を明示的に保持する。
- 補助テーブル（`MB`, `OMM`）にも outside テーブルを持たせる方針で Stage 1 を進める。forward の構造と一致させておくことで、逆伝播の流れが明瞭になり、後続フェーズ（loop profile, AD）でも扱いやすい。
