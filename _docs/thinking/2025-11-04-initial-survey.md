# 2025-11-04 初期観察メモ

## プロジェクト目標メモ
- CapR（Rfold ベース）の構造プロファイル計算を JAX ベースに置き換える目的で、jax-rnafold の分配関数計算を流用しつつ outside DP を実装する。
- jax-rnafold 側は forward（inside）DP が提供されており、`get_ss_partition_fn` で `E, P, ML, MB, OMM` テーブルを入手できる状態。
- outside DP を JAX 化し、最終的には p(i, loop) と p(i, j) を CapR・ViennaRNA と比較可能な精度で返すことがゴール。

## 現状把握
- `src/jax_capr` 内の実装は未着手に近く、`jax_inside.py` は import コメントのみ、`jax_outside.py` / `simple_outside.py` も空。
- パッケージの公開 API として期待されている `inside_outside.py`（`InsideOutsideResult` や `compute_inside_outside`）が存在せず、`__init__.py` とテスト・スクリプトの import が即座に壊れる。
- `tests/test_outside_vs_vienna.py` と `scripts/compare_bpp.py` は欠落している API を前提にしており、現状ではテストを実行できない。
- `pseudocodes/` には outside アルゴリズムの擬似コード（スケーリング有無・ベクトル化案）が置かれているが、本体コードには組み込まれていない。
- `notes/dp_mapping_turner1999.md` で CapR と jax-rnafold の DP 対応表が整理され、outside 実装に必要なテーブル・漸化式が列挙されている。
- `notes/outside_algorithm_turner.md` と `notes/openai_gpt5pro.jax-rnafold_report.md` に outside 再帰と OMM テーブルの扱いに関する詳細な整理があり、実装の下敷きになりそう。
- `notes/jax-capr_labnote_1104.pdf` は「本質的な情報」があると記されており、今後の要件定義で必読。

## 気付き・懸念
- outside DP では jax-rnafold のスケーリング（`s_table`）や `OMM` 逆伝播が複雑で、擬似コードの補完と notes の再確認が不可欠。
- CapR との比較を想定するなら、Turner1999 パラメータおよび最大スパン制限 `_maximal_span` の扱いを要件に含めるか事前に決めておきたい。
- API 設計（どの関数が何を返すか、返すテーブルをどう表現するか）を要件化しないと、テストやドキュメント整備が曖昧になりそう。
- 既存テストは ViennaRNA との BPP 差分のみをチェックしており、loop プロファイルの受け入れ条件が未定義。要件定義段階で補正が必要。

## 次のステップ案
1. 要件定義のアジェンダを整理し、期待成果・必須機能・受け入れ基準・テスト方針を順に詰める（features ドキュメント化も検討）。
2. 擬似コードと notes、`jax-capr_labnote_1104.pdf` を精読し、outside DP のスケーリング・境界条件・計算量を要件に落とし込む。
3. 欠落 API の最小構成（データクラス・テーブル構造・公開メソッド）を決定し、将来の実装タスク化に備える。
