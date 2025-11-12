# デバッグメモ（inside/outside・JAX）




1) 現在の実装・検証状況



実装中：
 - `src/jax_capr/jax_outside.py`（outside）
 - `submodules/jax-rnafold/src/jax_rnafold/d0/ss.py`（inside）。

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



Outside:



em.en_bulge(bi, bj, bh, bl, ...)（L110）

em.en_internal(bi, bj, bh, bl, ...)（L247）

Inside:



em.en_bulge(bi, bj, bk, bl, ...)（L283）

em.en_internal(...)（L337）

**bi,bj（外側ペア）と bh,bl（内側ペア）、および mismatch（bip1,bjm1,bhm1,blp1）**の対応が、関数ごとに完全一致しているかを再点検する価値あり。

en_bulge は見た目OKだが、en_internal は引数が多く、取り違えが起きやすい。







4) いま疑うべき箇所（優先度順）



Multibranch の「unpaired=0」境界の取り扱い



outside/inside の再帰境界・分割条件（length や i<k<l<j の制約）・寄与の合成順が一致しているか。

最小未対合長の制約（multi の branch 分解時）が 0 を許す経路で 二重カウント/過剰除外がないか。

en_internal 系の引数並び（mismatch を含む）



outside と inside の厳密な転置対応をコードレベルで照合。

psum_outer_internal_loops / psum_outer_bulges の小ループ端（lup,rup が 1,2）



ij_cond（L223–L225）を含む条件分岐と inside 側（ss.py L311 付近）の真の転置になっているか。

端点近傍で s_table index が0 に落ちないことの再確認（生成側・使用側の両方）。







5) 競プロ流チェックリスト（上から潰す）



再現テスト：



例2の配列で (i,j)=(11,65) と (43,64) の「unpaired=0」ケースを明示フラグ化し、当該経路のみの寄与をトレース出力（term-by-term）。

scale ∈ {0,-1,-2,-4} で 差分が不変かを各項で確認。

境界条件アサート：



multibranch 分解で unpaired=0 を許す/許さない箇所を inside/outside で同一化。

i+1==k や l+1==j の隣接ケースで、不可能状態を数えていないかを assert。

en_internal 引数対応表の固定：



outer=(bi,bj,bip1,bjm1)、inner=(bh,bl,bhm1,blp1) の マッピング表をコメントで明示し、inside/outside/エネルギー関数の3者で整合を取る。

一時的に assert not jnp.isnan(E) だけでなく、引数タプルのハッシュをログして両側一致を確認。

小ループ端の off-by-one：



lup,rup ∈ {1,2} のみを対象に ブルートフォース一致テスト（inside vs outside の項別サム）。

s_table[idx] の idx 最小値を集計し、0 が出現しないことをカウントで検証。

単項隔離テスト：



bulge/internal を 他の項から切り離し、その項だけで inside/outside の bpp 寄与が一致するかを確認（multiloop を一時無効化→次に有効化の A/B）。

12-mer スモークテストの強化：



unpaired=0 を強制できる 設計配列で最小反例生成（プロパティベース：branch を詰める操作で誤差が発火するか）。







6) 参考メモ（信頼してよいもの / そうでないもの）



信頼してよい：



outside_1105.md の Multiloop 項の式と fill_bar_P 実装は一致している（バグ可能性低）。

scale 依存の大域的な不具合ではなさそう（スケール不変の誤差あり）。

要再点検：



Multibranch 隣接境界（unpaired=0）

en_internal の引数順（特に mismatch の位置合わせ）







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


________


Based on your debug memo and the provided code, you are almost certainly correct to focus on the **multibranch "unpaired=0" boundary condition**.

The fact that inserting a single `A` into the problematic sequence (Example 2) drastically reduces the error is a massive clue. It confirms the logic for handling a "branch-right-next-to-a-branch" (unpaired=0) is diverging from the logic for "branch-unpaired-branch" (unpaired=1).

The error is likely in the *adjoint* (outside) implementation of this specific boundary case.

### 🎯 Primary Suspect: Adjoint of Multibranch Decomposition

The core of the multibranch "unpaired=0" problem lies in the inside `fill_multi` function (`ss.py`, L 444) and its corresponding adjoint (gradient) terms in the outside pass (`jax_outside.py`).

**Inside Logic (`ss.py`, L 444-453):**

The `ML` table is built on this decomposition:
`ML[nb, i, j] = (Unpaired Term) + (Branch Term)`

1.  **Unpaired Term:** `ML[nb, i+1, j] * F_unpair` (L 447)
2.  **Branch Term:** `sum_k(ML[idx, k+1, j] * MB[i, k])` (L 451-453)

Your problematic sequence (Example 2) fails when it's forced to take the **Branch Term** immediately (unpaired=0). The `A`-inserted sequence takes the **Unpaired Term**, which works. This implies the **Branch Term** or its adjoint is buggy.

**Outside (Adjoint) Logic (`jax_outside.py`):**

The gradient (outside value) must flow back from `bar_M` to `bar_M` and from `bar_M` to `bar_P` (via `MB`).

1.  **`bar_M` -> `bar_M` (Adjoint of Unpaired Term):**
    * `fill_bar_M` (L 490): `sm_M... += bar_M[..., h - 1, l] * multi_unpaired_factor`
    * This seems correct.

2.  **`bar_M` -> `bar_P` (Adjoint of Branch Term via `MB`):**
    * This is the most complex part. The gradient for `MB[i, k]` (which is `P[i, k]`) comes from `bar_M[i, j]`.
    * This is handled in `fill_bar_P` (`jax_outside.py`, L 403-411).
    * `multi_branch = (ML[1, l + 1, j] * bar_M[2, h, j] + ML[0, l + 1, j] * (bar_M[0, h, j] + bar_M[1, h, j])) * ...`
    * This term is the *exact* adjoint of the `sum_k(ML[idx] * MB[i, k])` term. It *looks* correct, but it's the most likely place for an off-by-one or indexing error, especially in how `(h, l)` (the `P` pair) and `(h, j)` (the `bar_M` fragment) are related.

3.  **`bar_M` -> `bar_M` (Adjoint of Branch Term via `ML`):**
    * This is the gradient from `bar_M[i, j]` to `bar_M[k+1, j]`.
    * This is handled in `fill_bar_M` (`jax_outside.py`, L 507) by `get_i_term(i)`.
    * `sm_M1 += bp_sum_i * ml_i_to_M1` (L 527)
    * This translates to: `bar_M[1, h, l] += MB[i, h-1] * bar_M[2, i, l]`
    * This also *looks* like a correct adjoint.

### 🕵️ Recommended Action

Your "競プロ流チェックリスト" (competitive programming checklist) is excellent. The bug is subtle. Given that the adjoint code *looks* syntactically correct, the error may be a subtle logic mismatch in the boundary conditions.

I recommend focusing on **Checklist Item 5: 単項隔離テスト (Single-term isolation test)** and **Item 1: 再現テスト (Reproduction test)**.

Your hypothesis `(ML[1, 65, 64] * MB[43, 64])` being zero is *correct* by the logic of the `ML[2]` (>= 2 branches) table. The fact that Vienna disagrees means Vienna's state decomposition or boundary condition is different.

**The most likely bug is in `fill_bar_P` (L 403).**

The `get_multi_j_term(j)` function computes the gradient for `P[h, l]`. It iterates over all `j` (the end of the multiloop fragment).

* `ML[1, l + 1, j]` is the "rest" of the loop, `ML[k+1, j]`.
* `bar_M[2, h, j]` is the "full" outside fragment, `bar_M[i, j]`.

In the "unpaired=0" case (e.g., `(11, 65)` closes, `(12, k)` is the first branch), the `h` in `bar_M[2, h, j]` might be misaligned with the `h` for the `P[h, l]` pair it's supposed to be updating.

This is a deep-level DP adjoint bug. Your own memo has correctly identified the exact, high-priority area to investigate. The discrepancy is almost certainly in how the `ML` table's "branch" term (`sum_k`) is inverted in the `outside` pass.