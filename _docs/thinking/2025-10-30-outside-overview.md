# 最近のメモ – outside 実装方針整理

- CapR 本体は ViennaRNA 1.8.5 の d=2（両端ダングルあり）パラメータを前提にしている。`energy_par.h` で 5'/3' ダングルと TermAU を組み込んでおり、Rfold/T99 モデルと整合する。
- jax-rnafold では d0 が最小構成で、エネルギー API の引数も単純なので outside DP の最初の実装ターゲットにすべきと判断。d2 版は内部状態の次元が増えるため後回し。
- d0 forward の内部ループは `OMM[i,j]` に外側ミスマッチと塩基対重みを合算し、一般内部ループ項の重みを `OMM[k,l] · mmij · en_internal_*` で注入している。このため outside でも `β_P → β_OMM` の逆伝播を忘れずに入れる必要がある。
