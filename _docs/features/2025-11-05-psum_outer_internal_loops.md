# psum_outer_internal_loops 要件定義（2025-11-05）

## 目的
- `src/jax_capr/jax_outside.py` における `psum_outer_internal_loops` の仕様を確立し、paired 外側テーブル `bar_P` および一般内部ループ補助テーブル `bar_OMM` の更新を正しく行うための指針を共有する。
- 既存 inside 実装（`submodules/jax-rnafold/src/jax_rnafold/d0/ss.py`）で定義された `psum_internal_loops` の逆写像を、JAX 互換・ベクトル化可能な形で実現する。

## スコープ
- 対象: 内部ループ項（1×1、1×N/N×1、2×2/2×3/3×2、一般内部ループ）の外側再帰。
- 入力: `(bh, bl, h, l, padded_p_seq, bar_P, bar_OMM, s_table, em, two_loop_length)` を想定。`P`（inside の paired テーブル）には依存しない。
- 出力: `bar_P` への加算分と `bar_OMM` への加算分を返却する純粋関数。呼び出し側で `bar_P.at[...]`, `bar_OMM.at[...]` を更新する。
- 非対象: multibranch／外部ループ outside、`fill_bar_OMM` の実装、スカラー `bar_E` 伝播。

## 機能要件
- **1×1 ループ**  
  - インデックス `(i, j) = (h-2, l+2)` の組み合わせだけを許可し、`mmij`（末端ミスマッチ）と `em.en_internal(..., 1, 1)`、`s_table[4]` を掛けた係数で `bar_P[..., h, l] += bar_P[..., i, j] * coeff` を行う。
  - `mmij` は `padded_p_seq[h-1] × padded_p_seq[l+1]` の全組合せで再計算する。
- **1×N / N×1 ループ**  
  - `two_loop_length = min(seq_len, max_loop)` を利用し、左右それぞれの `z_offset` を列挙して合法範囲を `jnp.where` でマスクする。
  - 右腕・左腕の長さに応じて `em.en_internal` と `s_table[j-l+2]` / `s_table[k-i+2]` 等を適用し、同じ `bp_idx` の `bar_P[k, l]` へ寄与を加算する。
- **2×2 / 2×3 / 3×2 ループ**  
  - `lup` / `rup` が `{(2,2), (2,3), (3,2)}` のみ True となるマスクを作成し、四重 `vmap` で末端塩基の組合せを列挙する。
  - 合法条件 `(k < j-2) ∧ (l ≥ k+1)` を満たした場合のみ `bar_P[..., h, l] += ...` を行う。
- **一般内部ループ**  
  - `lup > 1 ∧ rup > 1` かつ前述の特殊形でないケースのみを対象とする。
  - `em.en_internal_init(lup+rup) * em.en_internal_asym(lup, rup) * mmij * s_table[lup+rup+2]` を係数として `bar_OMM[k, l]` に加算する。
  - `bar_P` への寄与は行わず、後続の `fill_bar_OMM` で `bar_P` へ戻す設計に従う。
- **スケーリング整合性**  
  - forward 実装と同じ `s_table` 添字を使い、最終的な `P * bar_P / Z` がスケールに依存しない性質を保持する。
- **ベクトル化と境界処理**  
  - `jax.vmap` / `lax.cond` / `jnp.where` で全分岐を表現し、Python ループ・命令ベースの `if` 文を含めない。
  - 無効なインデックスは `jnp.zeros((), dtype=bar_P.dtype)` を返す条件式で抑制する。

## 非機能要件
- **JAX/JIT 互換**: `jit` でラップ可能な純関数とし、副作用を持たせない。
- **再利用性**: `fill_bar_P` から呼び出しやすいよう、`bar_P`/`bar_OMM` を入力で受け取りタプル `(delta_bar_P, delta_bar_OMM)` を返す形を推奨。
- **検証方針**: ViennaRNA との BPP 差分テストに加え、1×1・1×2・一般内部ループのみが発生する短鎖ケースで forward/外側係数の一致を確認するユニットテストを別途準備する。

## 参照資料
- `submodules/jax-rnafold/src/jax_rnafold/d0/ss.py` の `psum_internal_loops` 実装（L216-L371）。
- `notes/outside_algorithm_turner.md` §Paired state backward（L126-L175）。
- `notes/jax-capr_labnote_1104.pdf` p.4-6（内部ループ outside の導出とスケーリング）。
- `_docs/thinking/2025-11-04-outside-vectorization-plan.md` §outside ベクトル化方針。
