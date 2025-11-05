# 2025-11-05 Outside Refactor Plan

## 問題認識
- `src/jax_capr/jax_outside.py` の `fill_bar_P` / `psum_outer_internal_loops` が forward (`psum_internal_loops`) と整合しておらず、以下のミスがある。
  - outside にもかかわらず inside テーブル `P` を参照している箇所がある（`bar_P` を使うべき箇所）。
  - `s_table`, `em`, `two_loop_length` を `psum_outer_internal_loops` に渡しておらず呼び出し側で型崩れ。
  - 一般内部ループの寄与を `bar_OMM` に蓄積した後、`fill_bar_OMM` で `bar_P` へ戻す処理が未実装。
  - multibranch 補助テーブル `bar_MB` も未実装のままで、`fill_multi` の逆写像が存在しない。

## 方針
1. forward の `fill_paired` → `psum_internal_loops` をそのまま鏡写しにし、outside で必要な係数を抽出する。  
2. `psum_outer_internal_loops` は `(delta_bar_P, delta_bar_OMM)` のタプルを返し、呼び出し側で `bar_OMM` に加算する。  
3. `fill_bar_OMM` を実装し、forward の `fill_outer_mismatch` と同じ係数 `B_omm` を用いて `bar_P` を更新する。  
4. `fill_multibranch`/`fill_multi` に対応する outside (`bar_MB`, `bar_M`) を整理し、`bar_MB` を経由した `bar_P` 更新を入れる。  
5. `lax.scan` と `vmap` で記述し直し、Python ループを排除。`s_table` の添字は forward と一致させる。

## 検証計画
- ViennaRNA（`pf_create_bppm`）との base-pair probability 比較テストを再利用し、`P*bar_P/Z` が十分一致するかを確認。
- 一般内部ループだけが発生する短鎖ケース、1×1/1×N/2×2 の各特殊ケースで個別に検証。
- `bar_OMM` と `bar_MB` を使わずに実装した旧版と差分比較し、欠落していたループ寄与が正しく足し戻されるかチェック。

## 次のアクション
1. `psum_outer_internal_loops` と `fill_bar_OMM` の仕様書きを先にまとめ、テストケースを決める。
2. outside 全体 (`bar_E`, `bar_P`, `bar_M`, `bar_MB`, `bar_OMM`) の更新順序を整理し、`scan` のキャリーを再設計する。
3. 実装後、既存の `tests/test_outside_vs_vienna.py` を復旧させて比較用の CLI スクリプト (`scripts/compare_bpp.py`) も更新する。

