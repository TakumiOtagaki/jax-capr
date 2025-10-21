# CapR と jax-rnafold (Turner1999) の DP 対応まとめ

本ノートは `CapR` と `jax-rnafold` (d0 実装, Turner1999 パラメータ想定) の DP テーブルと漸化式を整理し、`p(i, loop)` プロファイルを jax ベースで再実装するための要件をまとめたもの。

## 1. リポジトリ構成
- `submodules/CapR` : Rfold ベースの CapR 本体。log 空間で内外側再帰を実装し、各ループ確率を出力。
- `submodules/LinearCapR` : CapR のビーム探索版。構造は CapR とほぼ同型だが近似あり。
- `submodules/jax-rnafold` : JAX による一般化 McCaskill。`src/jax_rnafold/d0/ss.py` が厳密 DP。

## 2. DP テーブル対応表

| CapR (log 空間) | 役割 | jax-rnafold d0 (`ss.py`) | 備考 |
| --- | --- | --- | --- |
| `_Alpha_outer[i]` (`CapR.cpp:181-192`) | 区間 [1,i] の外部 (外側) 貢献 | `E[i]` (`d0/ss.py:56-69`) | jax は線形空間。`E[i]` は prefix の外部 PF。 |
| `_Alpha_stem[i][d]` (`CapR.cpp:69-94`) | 塩基対 (i+1, j) を含む区間の内部 PF | `P[bp_idx, i, j]` (`d0/ss.py:375-411`) | jax は塩基対種類ごとに保持。CapR は BP 種類を添字に使わずに logsum。 |
| `_Alpha_stemend` (`CapR.cpp:148-176`) | 塩基対 (i, j+1) の末端状態 (hairpin/multi/内部ループ) | `fill_paired` 内の hairpin/bulge/internal/multi 足し込み (`d0/ss.py:385-400`) | jax は都度合算し `P` に書き込むので独立テーブル不要。 |
| `_Alpha_multi` / `_Alpha_multi1` / `_Alpha_multi2` / `_Alpha_multibif` (`CapR.cpp:95-147`) | マルチループ分割用の補助状態 (Vienna の `QM`, `QM1` 相当) | `ML[nb, i, j]` (`d0/ss.py:413-429`) と `MB[i, j]` (`d0/ss.py:100-120`) | `ML[2]` が閉鎖状態、`ML[1]` が 1 本分岐保持、`ML[0]` が遷移バッファ。`MB` が CapR の `_Alpha_multibif` に該当。 |
| `_Beta_*` 群 (`CapR.cpp:426-555`) | 外側再帰 (outside) | 現状未実装 | jax-rnafold d0 には outside テーブルが無い。 |
| 外部ループ dangle (`CalcDangleEnergy`) | 外側/内側共通の 5'/3' dangle エネルギー | `en_ext_branch`, `en_multi_branch`, dangle は energy モデル内 (`d0/energy.py`) | jax ではエネルギーが Boltzmann 重みに変換済み。 |

**主な差分**
- CapR は log ドメイン (`log Z`) で `logsumexp` を明示的に取る。jax-rnafold は Boltzmann 重みをそのまま積和。
- CapR は固定長 `_maximal_span` 制限あり、jax-rnafold d0 は `max_loop` のみ制限 (塩基対距離制限は無し)。
- jax-rnafold は塩基ペア種類ごとの 3 次元テンソル (`NBPS`×`N`×`N`) を保持し、Turner パラメータは `read_vienna_params` からフェッチ。
- 内部ループの一般項で CapR は `LoopEnergy` に `mismatchI`/`ninio` を含める (`CapR.cpp:562-595`)。jax-rnafold は `OMM` テーブル (`d0/ss.py:73-99`) により外側ミスマッチ重みを再利用。

## 3. 漸化式対応

### 3.1 塩基対付き区間 (`Qb` 相当)
- **CapR**: `_Alpha_stem[i][d]` に対し hairpin, internal, multiloop, stack を logsum (`CapR.cpp:69-147`)。外側は `_Beta_stem` で補完 (`CapR.cpp:518-549`)。
- **jax**: `fill_paired` が `P[bp_idx, i, j]` を更新 (`d0/ss.py:386-402`)。hairpin (`psum_hairpin`), bulge/internal (`psum_bulges`, `psum_internal_loops`), multi (`ML[2]`) を加算。

### 3.2 マルチループ
- **CapR**: `_Alpha_multi2` が追加のステム or 未対塩基を吸収、`_Alpha_multibif` が二分木分割 (`CapR.cpp:95-147`)。外側は `_Beta_multi*` 連鎖 (`CapR.cpp:450-515`)。
- **jax**: `MB[i,j]` が 1 枝 (`QM1`)、`ML` で枝数を畳み込み (`d0/ss.py:100-120`, `413-429`)。`ML[2]` がステム閉鎖へ接続 (`d0/ss.py:399`)。

### 3.3 外部ループ
- **CapR**: `_Alpha_outer` / `_Beta_outer` が左側スキャン (`CapR.cpp:181-214`, `426-438`)。
- **jax**: `fill_external` で `E[i] = E[i+1]` (未対塩基) + 枝 (`d0/ss.py:56-70`)。outside は未実装。

### 3.4 内部ループエネルギー
- **CapR**: 1x1 / 1x2 / 2x1 / 2x2 / 一般を `LoopEnergy` で分類 (`CapR.cpp:562-595`)。
- **jax**: 同等の分岐を `psum_internal_loops` で扱い、1×N と 2×2/2×3/3×2 特殊形は明示、一般形は `OMM` を介して外部ミスマッチを別テーブルから取得 (`d0/ss.py:216-371`)。

## 4. Turner1999 パラメータの扱い
- CapR: `energy_par.h` に Turner1999 テーブルを静的配列として内包。Kelvin 換算で `-ΔG / kT` を取ることで log 空間に直接足し込み。
- jax-rnafold: `read_vienna_params.py` が `rna_turner1999.par` をパースし、`boltz_jnp` で Boltzmann 因子化。`d0/energy.Model` サブクラスが `en_*` 系メソッドで重みを返す (`d0/energy.py` トップレベル `Model` 定義)。
- ここまでで、両者とも Turner1999 を利用する前提条件は揃っている (jax 側はモデル生成時に `TURNER_1999` を指定する必要あり)。

## 5. jax-rnafold で `p(i, loop)` を得るための要件

1. **outside DP の導入**: `P`, `ML`, `MB`, `E`, `OMM` に対する逆方向再帰が未実装。CapR の `_Beta_*` と同様のテンソルを JAX で構築する必要がある。<br> → 具体的には `fill_paired` の寄与元を解析して、対応する外側寄与を逆伝播する関数群 (`fill_external_outside`, `fill_multi_outside`, `fill_paired_outside` など) を追加する。
2. **数値スケーリング整合**: jax 側は rescaling (`s_table` 由来) を使用。outside 実装も同じスケーリングを踏襲し、最終的に `prob = inside * outside / Z` を計算できるようにする。
3. **確率計算モジュール**: CapR はクラス内で直接ループ確率を算出 (`CalcHairpinProbability`, `CalcMultiProbability` など)。jax 側でも outside テーブルを得た後、以下の項目を計算するモジュールが必要。
   - Hairpin: `P(bi=i, bj=j)` が hairpin となる寄与を `beta_stemend` 相当 (外側) と `hairpin` 貢献 (内側) で合成。
   - Bulge/Internal: 内側 `P` と外側 `beta` を用いて各未対塩基位置に確率を割り当てる (CapR の `CalcBulgeAndInternalProbability` の JAX 化)。
   - Multibranch: `ML` / `MB` と対応する outside テーブルから `p(i, multi)` を取得。
   - Exterior: `E` と outside 外部状態 (`beta_E`) から計算。
4. **ループ帰納式の JAX 化**: CapR の `CalcBulgeAndInternalProbability` ではループを総当たりしている。jax では `vmap` と `where` を使い、行列演算として外側結合を表現する設計方針を決める。
5. **API 設計**: ユーザーが `jax_rnafold` の `partition_fn` を呼び出したときにオプションで構造プロファイルを取得できるようにするか、別メソッド (`get_structure_profile`) として切り出すかを決定する。
6. **Turner1999 テスト**: CapR の出力との比較を行うシナリオを準備。`test.fa` 等を使い、hairpin / bulge / multi 各確率が一致するかを検証するユニットテストが必要。

## 6. 今後のタスク指針
1. `jax_rnafold/d0/ss.py` に outside DP を追加し、`P`, `ML`, `MB`, `E`, `OMM` に対応する外側テンソルを返す JIT 互換関数を構築する。
2. outside を用いた `p(i, loop)` 計算の JAX 実装を作成し、CapR の log 空間計算との整合性を確認する。
3. CapR (Turner1999) と jax-rnafold (Turner1999) の出力比較テストを追加。`np.allclose` ではなく相対誤差ベースで評価。
4. 線形化 (LinearCapR) との比較は後続。まずは厳密版で一致を目指す。

## 7. 未解決事項・質問候補
- outside DP を効率的に実装するにあたり、`OMM` のような補助テーブルをどう逆方向に扱うか (外部ミスマッチの勾配相当)。
- jax-rnafold で Turner1999 パラメータを使う際の標準 API 呼び出し手順 (エネルギーモデル生成部の確認が必要)。
- 最大スパン制限 (`_maximal_span`) を取り入れる必要があるか。CapR との比較を厳密にするなら導入が望ましい。

