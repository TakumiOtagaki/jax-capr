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
- `bar_P[bp, i, j]`（paired outside）を計算する。forward で `P` を更新している各項に対し逆向きの寄与を実装する。
  - **Hairpin**: outside 伝播なし（終端でのみ寄与）。
  - **Bulge/Internal 特殊ケース**（1×n, 2×2, 2×3, 3×2 等）:
    - forward で参照した `P[bp, i+1, l]` や `P[bp, k, j-1]` に、対応する Boltzmann 因子・塩基確率・`s_table` を掛けて `bar_P` に加算。
  - **一般内部ループ**（`OMM` 経由）:
    - forward で `P[bp, i, j] += coeff * OMM[k, l]` を行っているので、outside では `bar_OMM[k, l] += bar_P[bp, i, j] * coeff`。
    - ここで `coeff = mmij * en_internal_init(lup+rup) * en_internal_asym(lup, rup) * s_table[lup+rup+2]`。
  - **OMM から P への逆伝播**:
    - `bar_P[bp, i, j] += bar_OMM[i, j] * padded_p_seq[i-1, a] * padded_p_seq[j+1, b] * padded_p_seq[i, bi] * padded_p_seq[j, bj] * en_il_outer_mismatch(...)` を `(a, b)` で総和。
  - **Stacking**:
    - forward で参照した `P[bp', i+1, j-1]` に対し、`en_stack` と塩基確率を掛けて outside を加算。
  - **Multibranch 関連**:
    - `bar_P` から `bar_ML[2, i+1, j-1]` へ閉鎖ペナルティ `en_multi_closing` を掛けて伝播。
    - `ML` テーブル更新式の逆向き再帰を実装し、`bar_ML` → `bar_MB` → `bar_P` へと波及させる。
- exterior ループ:
  - `E[i] = E[i+1]*s_table[1] + Σ_j Σ_bp (...)` に対して outside を実装し、`bar_E[0] = 1` として `bar_P` へ寄与を伝搬する。
- `bar_M` など multibranch outside も Stage 1 で計算しておく（将来ループ確率で必要になる）。

## 数値スケーリング要件
- inside と同じ `s_table` を outside にも適用し、forward と逆方向のスケールがキャンセルするようにする。
- `bar_xi`（=`bar_E`）の初期値は `bar_E[0] = 1`（pseudocode 由来の `xi` は 1-indexed を想定）。`Z = E[0]` と `Z = xi[1]` の対応関係を実装前に確認する。
- JAX 実装では underflow/overflow 防止のため、各再帰の加算前にブロードキャスト済みの `s_table` を必ず掛ける。

## 実装方針
- 反復方向は inside と同様に右から左 (`i` を減少) の `lax.scan` を使う。長さ方向のループ (`d`) には追加の `lax.fori_loop` も検討。
- vectorization が前提：`vmap` / `einsum` / `lax.dot_general` を活用し、Python レベルの 3 重ループを避ける。
- メモリを抑えるため `checkpoint_scan` を outside でも利用する方向で設計し、必要な中間値だけ保持する。
- 非 ASCII コメントは避け、必要な場合のみ短い説明コメントを残す。

## API 設計メモ
- `InsideTables`（dataclass）: inside テーブル (`P`, `ML`, `MB`, `OMM`, `E`, `partition`) を外部へ公開。
- `OutsideTables`（仮称）: outside 値 (`bar_P`, `bar_M`, `bar_MB`, `bar_E`, `bar_OMM`) を保持。Stage 1 では loop profile 用の項目も確保するが `None` で初期化可能。
- `InsideOutsideResult`: 
  - `partition: float`
  - `bpp: Array`
  - `loop_profile: Optional[LoopProfile]`（Stage 1 は `None`）
  - 必要なら inside/outside テーブル参照をオプションで持たせる。
- 公開関数:
  1. `compute_inside_tables(seq: str | Array, model) -> InsideTables`
  2. `compute_inside_outside(seq, model, *, return_tables: bool = False) -> InsideOutsideResult`
  - スイッチ的な引数は避ける指針があるため、テーブルを取得したい場合は別関数やメソッドで返す案を検討（要調整）。

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
- `xi` / `E` のインデックス整合（`Z = E[0]` vs `xi[1]`）を `notes/jax-capr_labnote_1104.pdf` で確認し、実装前に明文化する。
- outside の一般内部ループで用いる `s_table` 係数の正確な指数（`lup+rup+2`）をソースから再度検証する。
- `LoopProfile` のデータ設計（hairpin 等の保持形式）を Stage 2 要件として別途策定する。
