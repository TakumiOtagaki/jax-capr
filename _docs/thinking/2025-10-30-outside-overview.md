# 最近のメモ – outside 実装方針整理

- CapR 本体は ViennaRNA 1.8.5 の d=2（両端ダングルあり）パラメータを前提にしている。`energy_par.h` で 5'/3' ダングルと TermAU を組み込んでおり、Rfold/T99 モデルと整合する。
- jax-rnafold では d0 が最小構成で、エネルギー API の引数も単純なので outside DP の最初の実装ターゲットにすべきと判断。d2 版は内部状態の次元が増えるため後回し。
- d0 forward の内部ループは `OMM[i,j]` に外側ミスマッチと塩基対重みを合算し、一般内部ループ項の重みを `OMM[k,l] · mmij · en_internal_*` で注入している。このため outside でも `β_P → β_OMM` の逆伝播を忘れずに入れる必要がある。

## OMM テーブルの整理
- **定義**: `fill_outer_mismatch()` で
  ```
  OMM[i,j] = Σ_{bp,a,b} P[bp,i,j]
                         · p_{i-1,a} · p_{j+1,b}
                         · p_{i,bi}  · p_{j,bj}
                         · en_il_outer_mismatch(bi, bj, a, b)
  ```
  を蓄積する。`(i,j)` を内側ペアに固定し、外側ミスマッチの Boltzmann 因子だけをまとめた前計算。
- **用途**: 一般内部ループの項では
  ```
  mmij · en_internal_init(lup+rup) · en_internal_asym(lup,rup)
       · OMM[k,l] · s_table[…]
  ```
  を `P[bp,i,j]` に加算する。特例（1×n、2×2/2×3/3×2）は個別処理。
- **意義**: 外側ミスマッチの列挙を事前に分離することで、一般内部ループの forward/outside ともに O(n³) で回せるようにしている。JAX 実装では tensordot ベースの計算が成立する。

## ViennaRNA の調査メモ
- `src/ViennaRNA/probabilities/equilibrium_probs.c` では内部ループの outside 処理を都度 `expintern[...]` や `expmismatchI[...]` を掛け合わせて実行しており、OMM 相当の補助テーブルは見当たらない。
- Vienna は C 実装でループを直接回しても十分高速と判断し、外側ミスマッチを別テーブルに切り出す最適化は行っていない模様。

## Outside 変数（OMM / P）の勾配的な伝播
Forward での依存をそのまま計算グラフと見なすと、一般内部ループに関して次の 2 段の伝播が必要になる。

### 1. `β_P → β_OMM`
Forward では
```
P[bp, i, j] += mmij(bi,bj,i,j) · en_internal_init(lup+rup)
               · en_internal_asym(lup,rup) · s_table[lup+rup+2]
               · OMM[k, l]
```
が加算される（`k = i + lup + 1`, `l = j - rup - 1` を満たす一般内部ループのみ）。  
従って outside では
```
β_OMM[k, l] += β_P[bp, i, j]
               · mmij(bi,bj,i,j)
               · en_internal_init(lup+rup)
               · en_internal_asym(lup,rup)
               · s_table[lup+rup+2]
```
を積み上げる。`lup > 1`, `rup > 1`, かつ 2×2/2×3/3×2 などの特例を除外する条件は forward と同じ。

### 2. `β_OMM → β_P`
OMM 自体は
```
OMM[i, j] = Σ_{bp,a,b} P[bp, i, j]
                        · p_{i-1,a} · p_{j+1,b}
                        · p_{i,bi}  · p_{j,bj}
                        · en_il_outer_mismatch(bi, bj, a, b)
```
なので、各 `P[bp, i, j]` には
```
β_P[bp, i, j] += β_OMM[i, j]
                 · p_{i-1,a} · p_{j+1,b}
                 · p_{i,bi}  · p_{j,bj}
                 · en_il_outer_mismatch(bi, bj, a, b)
```
を全ての `(a,b)` について加算する。自動微分で言えば、OMM は `P[bp, i, j]` に掛けられた係数の勾配をそのまま伝える役割になる。
