# デバッグメモ（inside/outside・JAX）

1) 現在の実装・検証状況

実装中：
 - `src/jax_capr/jax_outside.py`（outside）
 - `submodules/jax-rnafold/src/jax_rnafold/d0/ss.py`（inside）。

研究ノート
 - notes/research_note_1112.pdf

検証：ViennaRNA (`submodules/ViennaRNA/src/ViennaRNA`) の base pairing probability（bpp）と比較中。


 - scaling 周りの既知事項
     - scaling 周りの既知のエラーは解消済み.
     - scaling は信じて先へ進める.

 - 入力表現
     - 配列は one-hot。padded_p_seq 由来のエラーは一旦無視してよい。

 - 短鎖での精度：長さ ≈12 の配列で 1e-16 程度の高精度（scale=0 でも scale=-1.0 でも）。


2) 観測された挙動（再現ケース）

2-1. 長い配列（例1）

配列：GGAUAGUACGAAUUUAGACUCUCACUUACCGCAGUAAGUUACCCUCGUCU

scale=-2.0 と -4.0 で bpp の 最大・平均誤差が完全一致：


Max abs diff: 1.402e-03

Mean abs diff: 6.887e-06

解釈：スケール不変の誤差 → 単純な項の抜け漏れではなく、稀な条件でのみ発生するロジックミス、特に Internal Loop / Bulge 近辺の可能性。



2-2. 長い配列（例2：大誤差）

配列：

AAUUUUCCCAGCAGUCCCCACUAUAGCUACCCAUACGGUACCAGGGGCAAACGUGAAAUUGCCCCGCGGGAGUAC

bpp の大きい不一致（multibranch から伸びる stem の中央付近）：



Max abs diff: 8.856e-01, Mean: 2.181e-03

(46,61): ours 1.138e-01 vs Vienna 9.994e-01

(45,62): ours 1.138e-01 vs Vienna 9.994e-01

特徴：multi closing の塩基対 (11,65) と multi branch の開始 (43,64) の間に unpaired が 0。

その区間に A を 1 個挿入（…UCCCC[A]CUA…）すると誤差が大幅縮小：



Max abs diff: 6.697e-02, Mean: 1.629e-04

(46,61): ours 9.328e-01 vs Vienna 9.997e-01

(45,62): ours 9.327e-01 vs Vienna 9.997e-01

解釈：「マルチクローズ対とブランチ開始が隣接（unpaired=0）」の境界条件で outside/inside のどこかが崩れている可能性が高い
 * しかし、かなり細かく multiloop (inside/outside)のコードを読んだが、今のところバグは見つからない...他のところにある可能性も出てきた。



3) これまでの仮説と潰し込み結果

最初の疑い：psum_outer_internal_loops (L183) と psum_outer_bulges (L90) で

 - s_table の index が稀に 0（s_table[0]=1.0）になる off-by-one

 - inside（ss.py）との転置ミス
     - → 詳細比較の結果、該当ミスは未確認。outside_1105.md の数式や s_table の適用も表面上は整合。

 - 次の疑い（有力）：エネルギー関数の引数順序ミス
     - これも詳細な調査の結果、そういったミスはないことがわかった。



4) いま疑うべき箇所（優先度順）

Multibranch の「unpaired=0」境界の取り扱い（本当かなぁ...）

outside/inside の再帰境界・分割条件（length や i<k<l<j の制約）・寄与の合成順が一致しているか。
最小未対合長の制約（multi の branch 分解時）が 0 を許す経路で 二重カウント/過剰除外がないか。
en_internal 系の引数並び（mismatch を含む）

outside と inside の厳密な転置対応をコードレベルで照合。
psum_outer_internal_loops / psum_outer_bulges の小ループ端（lup,rup が 1,2）

ij_cond（L223–L225）を含む条件分岐と inside 側（ss.py L311 付近）の真の転置になっているか。
端点近傍で s_table index が0 に落ちないことの再確認（生成側・使用側の両方）。


5) 参考メモ（信頼してよいもの / そうでないもの）



信頼してよい：

- outside_1112.md の Multiloop 項の式と fill_bar_P 実装は一致している。

- scale 依存の大域的な不具合ではなさそう（スケール不変の誤差あり）。

要再点検：

- Multibranch 隣接境界（unpaired=0）
    - これは最近の調査で、この周辺にはやっぱりエラーがなさそうに思えている...
- 他にそもそもどういうエラーがあり得るかわからなくなっていて、漠然としているのが現状...

_____


付録：数値ログ（そのまま再掲）

例1：GGAUAGUACGAAUUUAGACUCUCACUUACCGCAGUAAGUUACCCUCGUCU

Max 1.402e-03, Mean 6.887e-06（scale=-2.0 と -4.0で一致）

例2（問題大）：

AAUUUUCCCAGCAGUCCCCACUAUAGCUACCCAUACGGUACCAGGGGCAAACGUGAAAUUGCCCCGCGGGAGUAC



Max 8.856e-01, Mean 2.181e-03

(46,61): ours 1.138e-01 vs Vienna 9.994e-01

(45,62): ours 1.138e-01 vs Vienna 9.994e-01

unpaired=0（(11,65) と (43,64) の間）

例2’（A を挿入）：

AAUUUUCCCAGCAGUCCCC[A]CUAUAGCUACCCAUACGGUACCAGGGGCAAACGUGAAAUUGCCCCAGCGGGAGUAC



Max 6.697e-02, Mean 1.629e-04

(46,61): ours 9.328e-01 vs 9.997e-01

(45,62): ours 9.327e-01 vs 9.997e-01


