# 2025-11-04 Outside ベクトル化設計メモ

## 目的
- `src/jax_capr/jax_outside.py` を段階的にベクトル化し、ViennaRNA との BPP 誤差を解消する準備を整える。
- 既存 forward 実装（`submodules/jax-rnafold/src/jax_rnafold/d0/ss.py`）から逆伝播時のテンプレートを抽出し、outside の各テーブル更新を整理する。

## forward 実装の整理
- `lax.scan` で `i` を右から左へ走査し、`(OMM, P, ML, MB, E)` を更新する骨格。
- 各更新関数は `vmap` や `jnp.where` を用いて NBPS・インデックス範囲を一括処理する。
  - `fill_external`: `(j, bp)` の 2 次元 `vmap` で外部枝寄与を合算。`E[i+1]` 未対合遷移と NBPS 全種の枝ペナルティを同列に加算。
  - `fill_outer_mismatch`: `(j, bp, bim1, bjp1)` を 4 次元 `vmap` 化し、`OMM[i, j]` に外側ミスマッチの期待値を蓄積。
  - `fill_multibranch`: `MB[i, j] = Σ_bp P·en_multi_branch·s_table[2]` を `vmap` で構築。内部で NBPS の加重和のみを持つ。
  - `fill_multi`: `(nb, j)` の `vmap` による多枝 DP。`idx = max(nb-1, 0)` を `jnp.where` で処理し、`ML[idx, k+1, j]` と `MB[i, k]` の畳み込みを一括加算。
  - `fill_paired`: ヘアピン/バルジ/内部ループの各サブケースを `vmap` で列挙。`two_loop_length = min(seq_len, max_loop)` により `(k, l)` 走査範囲を限定し、スタックおよび multibranch closing も同ステップで処理。
- スケーリング係数は forward と outside を通じて `s_table[1]`（未対合）、`s_table[2]`（枝/スタック）、`s_table[4]`（1x1 内部）などが対応付け済み。

## outside ベクトル化の分割方針
1. **外部ループ (`bar_E`)**  
   - `bar_E[0] = 1` を起点に `fill_external` の逆写像を構築。`einsum` もしくは `dot_general` で `E[j+1]` と NBPS をまとめて計算し、`bar_P[:, i, j]` へ一括伝播。
   - 未対合遷移 `bar_E[i] → bar_E[i+1]` も `jnp.where` で配列化。

2. **multibranch (`bar_MB`, `bar_M`)**  
   - `MB` の構築と同形のテンソル式を用い、`bar_MB[i, k]` から `bar_P[:, i, k]` へ逆伝播する `vmap` を用意。
   - `fill_multi` の逆方向は `(nb, j)` の `vmap` を保ちつつ、`ML[idx, k+1, j]` と `MB[i, k]` へ勾配を配分。境界条件 (`nb=0` 時の idx 固定) を `jnp.where` で表現。

3. **スタック / multibranch closing**  
   - `stack_weights` を `jnp.einsum` で処理し、`bar_P[:, i, j]` → `bar_P[:, i+1, j-1]` にまとめて伝搬。
   - `en_multi_closing` の寄与は `bar_P` の NBPS 和を `ML[2, i+1, j-1]` に集約するテンソル演算へ変換。

4. **バルジ・内部ループ特殊形**  
   - forward の `psum_bulges`・`psum_internal_loops` で使われている `two_loop_length` と `vmap` の構造を再利用。`padded_p_seq` の切り出しを配列化し、`bar_P` のブロードキャスト乗算で更新。

5. **一般内部ループ (`bar_OMM`)**  
   - `bar_P` → `bar_OMM` は `lup/rup` のマスクを用いて `mmij · en_internal_* · s_table` を掛けた加算。`OMM` の `(k, l)` インデックスを生成する `lax.dynamic_slice` もしくは `vmap` ベースのオフセット行列で対応。
   - `bar_OMM` → `bar_P` は forward の `fill_outer_mismatch` と同一形の `vmap` を適用。

6. **構造化**  
   - outside 処理を関数分割し、`lax.scan` で `i` を降順に処理する骨格へ統合。計算途中で必要なテンソルを保持しつつ checkpointing を検討。

## bar_P 伝播詳細（実装順序候補）
| 伝播元                   | forward 参照                         | outside で必要な演算                                                                                                                                            | 補助情報                                             |
| ------------------------ | ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| `bar_E[i]`               | `fill_external` (`E[i] += Σ_{j,bp}`) | `einsum` で `bar_E[i] * s_table[2] * E[j+1] * p_i * p_j * en_ext_branch` を算出し `bar_P[:, i, j]` に加算。並行して `bar_E[j+1]` に `P[:,i,j]` 源の勾配を集約。 | `i` 固定、`j` と NBPS を同時展開。                   |
| `bar_MB[i, k]`           | `fill_multibranch`                   | `bar_P[:, i, k] += bar_MB[i, k] * s_table[2] * p_i * p_k * en_multi_branch`                                                                                     | `vmap` 2 段 (`k`, `bp`)。                            |
| `bar_M[2, i+1, j-1]`     | `fill_paired` 内 multibranch closing | `bar_P[:, i, j]` の和を `en_multi_closing` で加重し `bar_M[2, i+1, j-1]` へ転送。逆向きは `bar_P[:, i, j] += bar_M[2, i+1, j-1] * en_multi_closing`。           | NBPS 方向の内積。                                    |
| `bar_P[:, i, j]` (stack) | `fill_paired` の stack               | `bar_P[:, i+1, j-1] += s_table[2] * stack_weights @ bar_P[:, i, j]`。ブロック対角積で処理。                                                                     | `stack_weights[bp_out, bp_in]`。                     |
| bulge (右/左)            | `psum_bulges`                        | `bar_P[:, i+1, l]` と `bar_P[:, k, j-1]` へ `two_loop_length` 分だけ同時加算。`l`/`k` の候補を `tril` マスクで制御。                                            | `en_bulge`、`s_table[bulge_len+2]`。                 |
| 1×1 内部                 | `psum_internal_loops` 1×1            | `bar_P[:, i+2, j-2] += ...` を `bip1`・`bjm1` の `vmap` で列挙し一括演算。                                                                                      | `en_internal(..., 1, 1)`、`s_table[4]`。             |
| 1×n / n×1                | `psum_internal_loops` 1×n 系         | `z_offset` 走査を `two_loop_length` 限定の `vmap` で行い、`bar_P[:, i+2, l]` や `bar_P[:, k, j-2]` に加算。                                                     | `en_internal`、`s_table[bulge_len+3]`。              |
| 特殊 2×2 / 2×3 / 3×2     | `psum_internal_loops` special        | `k_offset`, `l_offset` に対する `vmap` で NBPS と４つのミスマッチ組合せを一括加算。                                                                             | `en_internal(..., lup, rup)`、`s_table[lup+rup+2]`。 |
| 一般内部 (`bar_OMM`)     | `psum_internal_loops` general        | `bar_OMM[k, l] += bar_P[:, i, j] * coeff`、`coeff = mmij · en_internal_init · en_internal_asym · s_table[...]` を `lup/rup` のブールマスクで制御。              | `mmij` は `p_seq` 2 点の内積。                       |

### 処理順序（1 `i` 固定時）
1. `bar_E` → `bar_P` → `bar_E`（`j+1` への逆伝播）。
2. `bar_MB` → `bar_P`。
3. `bar_M` → `bar_P`（multibranch closing）。
4. `bar_P` の自己項（stack / bulge / 内部ループ / `bar_OMM`）をまとめて処理。
   - `mmij` や `padded_p_seq` の切り出しはバッチ生成し、複数のケースで再利用。
5. `bar_OMM` → `bar_P` は `scan` の後段（`i` 固定）で同時に評価し、副産物として `bar_P` を更新。

この順序で `bar_P` を更新すると、forward と対になる依存関係を保ちつつ、各サブケースをテンソル演算に写像できる。

## 実装ステップ案
1. `compute_outside` を関数分割し、`initialize_tables` 後に `lax.scan` へ移行する土台を構築。
2. `fill_bar_external`（仮称）を実装し、`bar_E`→`bar_P`／`bar_E` 更新だけをベクトル化してテスト。
3. `fill_bar_multibranch`（`bar_MB` / `bar_M` 関連）を追加し、multibranch までの伝播を確認。
4. `fill_bar_paired_core` を実装し、スタックと multibranch closing を処理。
5. バルジ・内部ループ特殊ケースを `vmap` ベースに移植し、`bar_P` の自己項を完成させる。
6. 一般内部ループ (`bar_OMM`) のベクトル化と `bar_OMM`→`bar_P` を仕上げる。
7. `tests/test_outside_vs_vienna.py` を段階的に実行し、誤差減衰を確認。必要に応じて途中段階用の一時テストを追加して差分検証を行う。

## forward / outside 係数対応表（2025-11-04）
forward の `ss.py` で適用している係数・スケールを outside へ鏡写しにするためのメモ。`p_i = padded_p_seq[i, bi]`, `p_j = padded_p_seq[j, bj]`。

| forward 寄与             | 係数（forward）                                             | outside で掛ける係数                      | 備考                                  |
| ------------------------ | ----------------------------------------------------------- | ----------------------------------------- | ------------------------------------- |
| 外部ループ枝             | `s_table[2]·en_ext_branch·p_i·p_j`                          | 同係数で `bar_E[i]` を乗算                | `E[j+1]` への伝播も同一係数           |
| 外部未対合               | `s_table[1]`                                                | 同係数で `bar_E[i]` を乗算                | `i+1` 境界チェック必須                |
| multibranch entry (`MB`) | `s_table[2]·en_multi_branch·p_i·p_k`                        | `bar_MB[i,k]` を掛けて `bar_P[:,i,k]` へ  | `k` は `[i, n]`                       |
| multibranch 閉鎖         | `en_multi_closing(bi,bj)`                                   | NBPS 内積で `bar_M[2,i+1,j-1]`            | `bar_P` の NBPS 和が必要              |
| multibranch 未対合       | `s_table[1]`                                                | `bar_M[nb, i, j]` の係数                  | `idx = max(nb-1, 0)` への配分も同係数 |
| スタック                 | `s_table[2]·en_stack·p_{i+1}·p_{j-1}`                       | `stack_weights @ bar_P[:,i,j]` に同係数   | `einsum` or `dot_general`             |
| バルジ右/左              | `s_table[bulge_len+2]·en_bulge` と外側塩基                  | `bar_P[:,i,j]` を掛けて反対側へ           | offset マスクが必要                   |
| 1×1 内部                 | `s_table[4]·en_internal(...,1,1)`                           | `bip1,bjm1` ごとに乗算                    | `pr_ip1·pr_jm1` をベクトル化          |
| 1×n / n×1                | `s_table[bulge_len+3]·en_internal`                          | `bulge_len` マスク込み                    | `z_offset` で範囲制限                 |
| 特殊 2×2/3×2/2×3         | `s_table[lup+rup+2]·en_internal`                            | `(bip1,bjm1,bkm1,blp1)` 全列挙            | 4 方向 `vmap`                         |
| 一般内部                 | `s_table[lup+rup+2]·en_internal_init·en_internal_asym·mmij` | `bar_P[:,i,j]` を掛けて `bar_OMM[k,l]` へ | `(lup,rup)` マスク                    |
| outer mismatch           | `en_il_outer_mismatch·p_{i-1}·p_{j+1}·p_i·p_j`              | `bar_OMM[i,j]` を掛けて `bar_P[:,i,j]` へ | `i=0` / `j=n-1` の境界注意            |

## `compute_outside` kernel 分割案
擬似コードにならい、テーブルを「対角線ごとに埋める (`fill_*`)」関数へ分割する方針。

1. `fill_bar_xi`: `xi` 再帰の逆伝播。`bar_E`（= `bar_xi`）のみを更新し、塩基対寄与は補助バッファ `bar_Pm`, `bar_Pm1` へ積算する。
2. `fill_bar_MB`: `MB` に対応する補助テーブルを処理し、`bar_Pm`/`bar_Pm1` を介して `bar_P` へ伝播するための係数を作る。
3. `fill_bar_M`: multibranch DP (`ML`) の逆伝播。`fill_bar_MB` と同様に `bar_M`・`bar_MB` を更新。
4. `fill_bar_Ps`: スタック、multibranch closing、バルジ、内部ループをまとめた中核処理。ここで `bar_Pm`, `bar_Pm1` と `bar_P` を組み合わせ、最終的に `bar_P` を更新する。
5. `fill_bar_OMM_from_P`: `bar_P` から `bar_OMM` を生成。
6. `fill_bar_P_from_OMM`: `bar_OMM` から `bar_P` へ戻す。

それぞれの `fill_*` はスパン（`d = j - i`）や位置 `i` を入力に取り、対象テーブルの該当対角線のみを書き換える。

## `lax.scan` 骨格草案（2025-11-04）
PyTree として扱いやすい `OutsideCarry` dataclass を用意し、`checkpoint_scan` にも適合させる。

```python
@dataclass
class OutsideCarry:
   bar_E: Array
   bar_P: Array
   bar_M: Array
   bar_MB: Array
   bar_OMM: Array
   bar_Pm: Array
   bar_Pm1: Array

def outside_scan_body(carry: OutsideCarry, i: int):
   bar_E, bar_P, bar_M, bar_MB, bar_OMM, bar_Pm, bar_Pm1 = carry
   bar_E, bar_Pm, bar_Pm1 = fill_bar_xi(bar_E, bar_Pm, bar_Pm1, i,
                               inside.E, P, padded_p_seq,
                               ext_branch_weights, s_table)
   bar_Pm = fill_bar_MB(bar_Pm, bar_MB, MB, i,
                        padded_p_seq, multi_branch_weights, s_table)
   bar_M, bar_MB = fill_bar_M(bar_M, bar_MB, ML, MB, i, s_table)
   bar_P, bar_Pm, bar_Pm1, bar_M, bar_OMM = fill_bar_Ps(
       bar_P, bar_Pm, bar_Pm1, bar_M, bar_OMM, P, padded_p_seq,
       stack_weights, multi_closing_weights, model, s_table,
       two_loop_length, i)
   return OutsideCarry(bar_E, bar_P, bar_M, bar_MB, bar_OMM, bar_Pm, bar_Pm1), None

indices = jnp.arange(seq_len - 1, -1, -1)
scan = checkpoint_scan if checkpoint_every else lax.scan
carry_final, _ = scan(outside_scan_body, carry0, indices)
bar_P = fill_bar_P_from_OMM(
    carry_final.bar_OMM,
    carry_final.bar_P,
    padded_p_seq,
    model,
)
bar_P = bar_P + carry_final.bar_Pm_contrib  # exterior由来の加算を span 更新前に反映するイメージ
```

各 kernel の戻り値は PyTree を不変更新する設計にし、`scan` 内では `at[...]` を用いた永続化更新のみ行う。`outside_paired_core` の二重ループは offset 配列を事前生成し、`jnp.where` マスクで合法範囲を抽出する方針。

## 次のステップ
- 上記分割に沿って `bar_P` 更新ロジックから実装に着手する。特に `bar_P` は他テーブルへの入力が多いため、最初に vectorized kernel を組み立てる。
- 実装前に `bar_P` の計算単位ごとに入出力テンソル形状と `s_table` 添字を一覧化し、回帰テスト前にセルフチェックできる表を作成する予定。 

## 実装進捗ログ
- 2025-11-04: `fill_bar_xi` の原型を整理し、外部ループ outside 伝播で `bar_E` のみを更新する構造へ変更。塩基対への寄与は `bar_Pm` / `bar_Pm1` に蓄積し、当面は span 降順フェーズで消費する暫定実装とした。
