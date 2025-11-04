# 2025-11-04 BPP Outside Stage 1 要件（ドラフト）

## 目的とスコープ
- CapR の構造プロファイル計算を JAX ベースへ移植する第一段階として、jax-rnafold の inside テーブルを利用した outside DP を実装し、**塩基対確率 (BPP)** の再現を目標とする。
- loop profile（hairpin/bulge/internal/multi/exterior）は Stage 2 以降で扱う。Stage 1 では必要な outside テーブルを確保したうえで、API レベルでは `None` で占位する。
- CapR が導入していた `_maximal_span` 制約は適用しない。計算量は GPU 並列化と JAX のベクトル化で吸収する前提。

## 期待する成果物
- 入力 RNA 配列（または確率的配列）に対し、jax-rnafold の inside/outside を用いて BPP 行列を返す関数群。
- 将来の loop profile 計算で再利用できる inside/outside テーブルを外部 API から取得可能にする。
- ViennaRNA (Turner1999) の BPP と高精度で一致すること（テスト基準は既存 `tests/test_outside_vs_vienna.py` の閾値に従う）。

## 必要な inside テーブル
- `get_ss_partition_fn`（submodules/jax-rnafold/src/jax_rnafold/d0/ss.py）から取得できる以下を最低限保持する:
  - `E[i]`: exterior prefix DP。`E[0]` が分配関数 `Z`。
  - `P[bp, i, j]`: 塩基対付き区間。
  - `ML[nb, i, j]`: multibranch 補助状態 (`nb ∈ {0,1,2}`)。
  - `MB[i, j]`: multibranch の 1 枝テーブル。
  - `OMM[i, j]`: 一般内部ループ用 outer mismatch 集約。
- スケーリングテーブル `s_table` や再利用する energy パラメータ（`en_*`）の参照が必要。

## outside 再帰の要件（BPP 用）
- `bar_P[bp, i, j]`（paired outside）を計算する。forward で `P` を更新している各項に対し、研究ノートと擬似コードで示された **二次構造遷移ベースの書き下し** をそのまま反転させる。
  - **Hairpin**: outside 伝播なし（終端でのみ寄与）。
  - **Bulge/Internal 特殊ケース**（1×n, 2×2, 2×3, 3×2 等）:
    - forward で参照した `P[bp, i+1, l]` や `P[bp, k, j-1]` に、対応する Boltzmann 因子・塩基確率・`s_table` を掛けて `bar_P` に加算。
  - **一般内部ループ**（`OMM` を含む `B(f_2)` 項）:
    - `OMM` は forward 同様に保持し、outside では `bar_OMM` を用意する。
    - forward で `P[bp, i, j] += coeff · OMM[h, l]` と更新される一般内部ループ（`k < l`, `lup = k-i-1 > 1`, `rup = j-l-1 > 1`, かつ {2×2, 2×3, 3×2} 以外）に対し、outside では  
      `coeff = mmij(i,j) · en_internal_init(lup+rup) · en_internal_asym(lup, rup) · s_table[lup+rup+2]`  
      を用いて `bar_OMM[h, l] += bar_P[bp, i, j] · coeff` を加算する。
    - `bar_OMM` から `bar_P` へ戻す際は、`fill_outer_mismatch` の定義通りに  
      `bar_P[bp, h, l] += bar_OMM[h, l] · p_seq[h-1, a] · p_seq[l+1, b] · p_seq[h, bi] · p_seq[l, bj] · en_il_outer_mismatch(bi, bj, a, b)`  
      を `(a, b)` 全組み合わせで合算する。
  - **Stacking**:
    - forward で参照した `P[bp', i+1, j-1]` に対し、`en_stack` と塩基確率を掛けて outside を加算。
  - **Multibranch 関連**:
    - `bar_P` から `bar_ML[2, i+1, j-1]` へ閉鎖ペナルティ `en_multi_closing` を掛けて伝播。
    - `fill_multibranch` で `MB[i, k] = Σ_{bp} P[bp, i, k] · en_multi_branch · p_seq · s_table[2]` が構築されるため、outside では `bar_MB[i, k]` を保持し、対応する係数で `bar_P` へ戻す。
    - `fill_multi` の更新式  
      `ML[nb, i, j] = ML[nb, i+1, j] · s_table[1] + Σ_{k} ML[idx, k+1, j] · MB[i, k]`  
      を反転し、`bar_ML` → `bar_MB` → `bar_P` の順に逆伝播させる（`idx = max(nb-1, 0)` の分岐と境界条件を保持）。
- exterior ループ:
  - `E[i] = E[i+1]*s_table[1] + Σ_j Σ_bp (...)` に対して outside を実装し、`bar_E[0] = 1` として `bar_P` へ寄与を伝搬する。
- `bar_M` など multibranch outside も Stage 1 で計算しておく（将来ループ確率で必要になる）。

## 数値スケーリング要件
- inside と同じ `s_table` を outside にも適用し、forward と逆方向のスケールがキャンセルするようにする。
- `bar_xi`（=`bar_E`）の初期値は `bar_E[0] = 1`（pseudocode 由来の `xi` は 1-indexed を想定）。分配関数は jax-rnafold 実装に合わせて `Z = E[0]` を利用する。
- JAX 実装では underflow/overflow 防止のため、各再帰の加算前にブロードキャスト済みの `s_table` を必ず掛ける。
- 一般内部ループでは `s_table[lup+rup+2]`、スタック・マルチ分岐ではそれぞれ `s_table[2]`、外部・マルチ枝の未対合移動では `s_table[1]` を用いるなど、forward と同じ添字を outside でも踏襲する。

## 実装方針
- 反復方向は inside と同様に右から左 (`i` を減少) の `lax.scan` を使う。長さ方向のループ (`d`) には追加の `lax.fori_loop` も検討。
- vectorization が前提：`vmap` / `einsum` / `lax.dot_general` を活用し、Python レベルの 3 重ループを避ける。
- メモリを抑えるため `checkpoint_scan` を outside でも利用する方向で設計し、必要な中間値だけ保持する。
- 非 ASCII コメントは避け、必要な場合のみ短い説明コメントを残す。

- `OutsideTables`（dataclass）: `bar_P`, `bar_M`, `bar_MB`, `bar_E`, `bar_OMM` を保持。後続フェーズで loop profile 用に項目を追加できるよう拡張余地を残す（値は Stage 1 時点で `None` 設定可）。
- `InsideOutsideResult`: 
  - `partition: float`
  - `bpp: Array`
  - `loop_profile: Optional[LoopProfile]`（Stage 1 は `None` 固定）
  - `inside: Optional[InsideTables]`, `outside: Optional[OutsideTables]` を保持する場合は別コンストラクタや専用関数で提供し、スイッチ引数は導入しない。
- 公開関数候補（全て引数・戻り値を固定し、フラグを避ける）:
  1. `compute_inside_tables(seq: str | Array, model) -> InsideTables`
  2. `compute_outside_tables(seq: str | Array, model) -> OutsideTables`（必要に応じて inside を内部で再利用）
  3. `compute_bpp_matrix(seq: str | Array, model) -> InsideOutsideResult`
  - inside/outside テーブルを併せて欲しい場合は `compute_inside_outside(seq, model) -> (InsideTables, OutsideTables, InsideOutsideResult)` 等、戻り値で明示的に返す案を検討。

## テスト & 検証
- 既存テスト `tests/test_outside_vs_vienna.py` を再生可能にする。`ViennaRNA` Turner1999 と比較して最大誤差 `1e-6`、平均誤差 `1e-7` 以下を目標。
- `scripts/compare_bpp.py` で複数配列の比較出力をサポートする。
- Partition 関数 `E[0]` の再現性（ViennaRNA との比率差が小さいか）を確認する追加テストを検討。
- 逆伝播の整合性チェック（`jax.grad` による inside/outside 逆伝播 sanity）を任意で実施できるように設計。

## 開発環境メモ
- 依存管理は `uv` で実施：`uv sync`, `uv pip install -e ./submodules/jax-rnafold`, `uv pip install ViennaRNA` が README に記載済み。
- サブモジュール:
  - `submodules/jax-rnafold`: inside 実装および energy モデル。
  - `submodules/CapR`, `submodules/LinearCapR`, `submodules/ViennaRNA`: 将来比較用。
- Python バージョンは `pyproject.toml` にて `==3.10.4` を指定。`venv/.python-version` も同値。
- GPU での実行を想定し、JAX の X64 モード（`jax_enable_x64=True`）は既定で有効化済み。

## 未確定事項・フォローアップ
- 一般内部ループ（`B(f_2)`）のスケーリング係数や指数（例: `lup+rup+2`）を jax-rnafold の forward 実装と突き合わせて確定する。
- multibranch outside の逆伝播順序・境界条件を擬似コードと一致させるため、`notes/jax-capr_labnote_1104.pdf` の該当節と照合する。
- `LoopProfile` のデータ設計（hairpin 等の保持形式）を Stage 2 要件として別途策定する。

## 実装チェックポイント
1. `compute_inside_tables` で `get_ss_partition_fn` を呼び出し、`E`, `P`, `ML`, `MB`, `OMM` を `InsideTables` へ詰める。partition は `E[0]` を利用。
2. `compute_outside_tables` で擬似コード通りの outside 再帰を実装し、`bar_E`, `bar_P`, `bar_M`, `bar_MB`, `bar_OMM` を揃えたうえで返す。
3. `compute_inside_outside` で inside/outside テーブルを組み合わせ、`bpp = P_total * bar_P / Z` を作成（`P_total` は塩基対種類を合算）し、必要に応じてテーブルを `InsideOutsideResult` へ格納。
4. `compute_bpp_matrix` を `compute_inside_outside` 上のヘルパとして実装し、既存テスト・比較スクリプトが呼び出せる状態にする。
5. `tests/test_outside_vs_vienna.py` を修復し、ViennaRNA との誤差基準（最大 `1e-6`, 平均 `1e-7`）を満たすことを確認。必要なら数値安定化の検証ログを追加する。
    - 1×1 / 1×n / n×1 / 2×2 / 2×3 / 3×2 の特殊形では forward の係数（例: `s_table[4]`, `s_table[j-l+2]`, `s_table[k-i+2]`, `s_table[lup+rup+2]`）をそのまま掛けたうえで、対象 `P` へ outside を伝播する。
