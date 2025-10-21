以下は、いただいた文献と実装情報を突き合わせて整理した JAX‑RNAfold（Bioinformatics 41(5):btaf203） の forward（inside）アルゴリズムの要約、対応する outside 再帰式の導出、そして CapR の構造プロファイル p(i,\text{loop}) を計算するための具体的な方法です。数式は McCaskill（1990）の inside–outside の枠組みに、Matthies ら（2024）／JAX‑RNAfold の「確率的配列」拡張を合わせた形で書いています。アルゴリズム上の要点や近似は btaf203 の記述に依拠し、outside の導出は McCaskill／LinearPartition の outside 設計と CapR の SCFG 的説明も踏まえています。 ￼

⸻

1) JAX‑RNAfold の forward（inside）— 何を計算し、何を近似／工夫しているか

1.1 どの分配関数を計算するか（確率的配列）
	•	JAX‑RNAfold は、各塩基位置に独立な 4 項カテゴリ分布 \;x \in [0,1]^{4\times n}\;(\sum_{b}x_{b,i}=1)\; を入力とし、構造と配列の両方にまたがる分配関数
Z_{\mathrm{ss}}(x)=\sum_{s\in\mathcal S}\sum_{q\in\mathcal Q} x_q\; e^{-\beta E_{q,s}}
を McCaskill 型 DP で厳密に計算します（連続入力・自動微分可能）。これは Matthies ら（2024）の定式化に準じます。
※btaf203 は、この「期待分配関数」を GPU 上で勾配計算までスケールさせる実装最適化（次項）を与えます。 ￼

1.2 inside の DP テーブルと再帰（最小集合）

論文本体（NAR 2024）で明示されている最小構成は次です（記号は同論文のもの）。
	•	\mathcal E(i)：外部ループの右側部分列 [i,n) の分配関数。
基本形は
\mathcal E(i) \;=\; \mathcal E(i+1) + \sum_{i<j<n}\sum_{b_i,b_j}\; x_{b_i,i}\,x_{b_j,j}\;\mathcal P(b_i,b_j,i,j)\;\color{#555}{\mathcal E(j+1)}
とするのが自然です（\mathcal P は [i,j] 内部の寄与なので、右側 [j+1,n) を掛け持つ \mathcal E(j+1) が必要）。NAR 本文の式（8）は \mathcal E(j+1) を紙面上省略しているように読めますが、定義の整合性のため実装では \mathcal E(j+1) を掛ける形になります（McCaskill との対応）。  ￼
	•	\mathcal P(b_i,b_j,i,j)：[i,j] で i と j が対合（かつ塩基種が b_i,b_j）に条件付けた分配関数。
\begin{aligned}
\mathcal P(b_i,b_j,i,j) &= \underbrace{\mathcal B(\mathrm{ONE\!-\!LOOP}(b_i,b_j,i,j))}{\text{ヘアピン}} \\
&\quad + \sum{i<k<l<j}\sum_{b_k,b_l}\underbrace{\mathcal P(b_k,b_l,k,l)}{\text{内部対}}\;x{b_k,k}x_{b_l,l}\;\underbrace{\mathcal B(\mathrm{TWO\!-\!LOOP}(b_i,b_j,b_k,b_l,i,j,k,l))}{\text{内部/バルジ}}\\
&\quad + \underbrace{\mathcal M(2,i+1,j-1)\; \mathcal B(M_c)\;\mathcal B(M_p)}{\text{多分岐ループ（閉鎖＋分岐ペナルティ）}}
\end{aligned}
ここで \mathcal B(\cdot) はボルツマン因子、M_c,M_p は多分岐の閉鎖・分岐ペナルティ。
	•	\mathcal M(p,i,j)：[i,j] が多分岐ループ内部で、内部に少なくとも p 本の枝を含む部分の分配関数（閉鎖対は含まない）。
\mathcal M(p,i,j)=\underbrace{\mathcal M(p,i+1,j)\;\mathcal B(M_u)}{\text{未対合の 1 塩基を吸収}} \;+\!\!\sum{i<k<j}\sum_{b_i,b_k}\!\!\!\mathcal P(b_i,b_k,i,k)\;x_{b_i,i}x_{b_k,k}\;\mathcal B(M_p)\;\mathcal M(\max(0,p\!-\!1),k\!+\!1,j)
\mathcal M(0,\cdot,\cdot)=1・p>0 かつ i>j は 0。

Vienna 風の対応づけ：\mathcal E\leftrightarrow Q^{\text{ext}}、\mathcal P\leftrightarrow Q^{\mathrm{b}}、\mathcal M\leftrightarrow Q^{\mathrm{m}}/Q^{\mathrm{m1}}（多分岐内部）と見なすと、McCaskill 系の既存 DP と 1 対 1 に対応が取れます。 ￼

1.3 どんな近似・実装上の工夫が入っているか（btaf203）
	•	再帰式の「因数分解」による巨大な定数因子削減：コアキシャルスタック・末端ミスマッチ・ダングリングエンド・多分岐枝の扱いを組み替え、最大 4^4=256 倍の定数因子削減。内部ループも同規模で最適化。ここでの変更は「近似」ではなく **再帰の再編成（等価変形）**で、出力の厳密性は保たれます。
	•	チェックポイント法：逆伝播で必要な中間値を全部保持せず、部分的に再計算してメモリ O(n^3)（理論上 O(n^{2.5}) まで）に削減。forward 値自体は変わりません。
	•	数値安定化：log 空間では加法の近似が勾配に悪影響を及ぼすため、塩基ごとのスケーリング（Vienna 流）を採用（分配関数の値は変えない）。
	•	計算量：基本は O(n^4)（Lyngsø の最適化や内部ループ長の上限で O(n^3) へ）。GPU 並列で行方向を並列化。

⸻

2) JAX‑RNAfold の inside に対応する outside テーブルと再帰式

以下、Z=\mathcal E(0) を全体分配関数とし、outside を真上流の寄与（「その非終端が右辺で使われるときの前後の重み」）として定義します。初期条件は \overline{\mathcal E}(0)=1、他は 0。McCaskill の outside／「バックスコア」設計に準拠します。 ￼

記法：
inside: \mathcal E,\mathcal P,\mathcal M／ outside: \overline{\mathcal E},\overline{\mathcal P},\overline{\mathcal M}。
x_{b,i} は位置 i が塩基 b である確率。 \mathcal B(\cdot) はボルツマン因子。

2.1 外部ループ \overline{\mathcal E}

inside の（自然な）再帰 \mathcal E(i)=\mathcal E(i+1)+\sum_{i<j}\sum_{b_i,b_j} x_{b_i,i}x_{b_j,j}\,\mathcal P(b_i,b_j,i,j)\,\mathcal E(j+1) から：
	•	スキップ（未対合）：
\overline{\mathcal E}(i\!+\!1)\;{+}{=}\;\overline{\mathcal E}(i)
	•	対合を張る項（\mathcal P 側へ）：
\overline{\mathcal P}(b_i,b_j,i,j)\;{+}{=}\;\overline{\mathcal E}(i)\;x_{b_i,i}x_{b_j,j}\;\mathcal E(j\!+\!1)
	•	右側の外部部分（\mathcal E(j+1) 側へ）：
\overline{\mathcal E}(j\!+\!1)\;{+}{=}\;\overline{\mathcal E}(i)\;\sum_{b_i,b_j} x_{b_i,i}x_{b_j,j}\;\mathcal P(b_i,b_j,i,j)

2.2 ペア状態 \overline{\mathcal P}

inside の \mathcal P 再帰から 3 つの行き先：
	1.	ヘアピン（\mathrm{ONE\!-\!LOOP}）：末端（子なし）なので outside 伝播は無し（\overline{\mathcal P} 自身がヘアピン確率の重みになる）。
	2.	内部/バルジ（二重ループ）：内側の \mathcal P(b_k,b_l,k,l) へ
\overline{\mathcal P}(b_k,b_l,k,l)\;{+}{=}\;\sum_{\substack{i<k<l<j\\ b_i,b_j}}\overline{\mathcal P}(b_i,b_j,i,j)\;x_{b_k,k}x_{b_l,l}\;\mathcal B(\mathrm{TWO\!-\!LOOP}(\cdots))
	3.	多分岐の内部：\mathcal M(2,i+1,j-1) へ
\overline{\mathcal M}(2,i\!+\!1,j\!-\!1)\;{+}{=}\;\overline{\mathcal P}(b_i,b_j,i,j)\;\mathcal B(M_c)\,\mathcal B(M_p)

2.3 多分岐内部 \overline{\mathcal M}

inside の \mathcal M 再帰
\mathcal M(p,i,j)=\underbrace{\mathcal M(p,i+1,j)\mathcal B(M_u)}{\text{unpaired}}+\sum{i<k<j}\sum_{b_i,b_k}\mathcal P(b_i,b_k,i,k)\,x_{b_i,i}x_{b_k,k}\,\mathcal B(M_p)\,\mathcal M(\max(0,p-1),k+1,j)
から：
	•	未対合の 1 塩基消費（左へ）：
\overline{\mathcal M}(p,i\!+\!1,j)\;{+}{=}\;\overline{\mathcal M}(p,i,j)\;\mathcal B(M_u)
	•	枝を 1 本張る分岐：左右の子へ分配
\overline{\mathcal P}(b_i,b_k,i,k)\;{+}{=}\;\overline{\mathcal M}(p,i,j)\;x_{b_i,i}x_{b_k,k}\;\mathcal B(M_p)\,\mathcal M(\max(0,p\!-\!1),k\!+\!1,j)
\overline{\mathcal M}(\max(0,p\!-\!1),k\!+\!1,j)\;{+}{=}\;\overline{\mathcal M}(p,i,j)\;\mathcal B(M_p)\!\!\sum_{b_i,b_k}\!x_{b_i,i}x_{b_k,k}\,\mathcal P(b_i,b_k,i,k)

以上の outside は、LinearPartition の inside/outside 擬似コード（SKIP/POP での伝播）とも整合します（ただし LinearPartition は近似ビーム探索を導入）。

2.4 確率の取り出し
	•	塩基対確率：
p(i,j) \;=\; \frac{1}{Z}\sum_{b_i,b_j} x_{b_i,i}x_{b_j,j}\; \mathcal P(b_i,b_j,i,j)\;\overline{\mathcal P}(b_i,b_j,i,j)
（McCaskill の Q^{\mathrm b}\,Q^{\text{out}}/Q に一致。） ￼
	•	スケーリングを入れていても、比としての確率は不変（スケール因子が分子分母で相殺）。

注：コアキシャルスタックや末端ミスマッチ，ダングリングなど btaf203 が再編成した項は、上の「ヘアピン」「二重ループ」「多分岐分岐」それぞれの \mathcal B(\cdot) に吸収されます。外側への伝播は 「その項が現れる右辺に現れる子」に対して鎖状に連鎖させればよく、outside の構造は保たれます。

⸻

3) CapR 相当の構造プロファイル p(i,\text{loop}) は inside/outside から計算できるか？

結論：JAX‑RNAfold の \mathcal E,\mathcal P,\mathcal M（とその outside）だけで 可能 です。CapR は SCFG 流の状態（Outer/Stem/StemEnd/Multi/…）で構造文脈を注釈し inside–outside で p(i,\delta) を出しますが、JAX‑RNAfold の DP でも ヘアピン／二重ループ／多分岐の 3 区分と外部を明示しているため、「どのループ事象に属するか」を事後に集計すれば同等量が得られます。以下に各文脈の寄与の「重み」と集計式を与えます。

CapR の定義：\delta\in\{\text{B(バルジ)},\text{E(外部)},\text{H(ヘアピン)},\text{I(内部)},\text{M(多分岐)},\text{S(ステム)}\} で \sum_\delta p(i,\delta)=1。 ￼

3.1 ステム（paired）S

p(i,S)=\sum_{j} p(i,j)
（2.4 の p(i,j) を合計。） ￼

3.2 外部 E（未対合）

p(i,E)\;=\;\frac{1}{Z}\;\overline{\mathcal E}(i)\;\mathcal E(i+1)
（\mathcal E(i)\to\mathcal E(i+1) の skip 事象の重みを集計。外部未対合にエネルギーは通常付かない。） ￼

3.3 ヘアピン H（未対合）

w_{\mathrm{hp}}(k,l)\;=\;\sum_{b_k,b_l} \overline{\mathcal P}(b_k,b_l,k,l)\;x_{b_k,k}x_{b_l,l}\;\mathcal B(\mathrm{ONE\!-\!LOOP}(b_k,b_l,k,l))
p(i,H)\;=\;\frac{1}{Z}\sum_{k<i<l} w_{\mathrm{hp}}(k,l)
（対 (k,l) がヘアピンを閉じる確率重みを、その内部位置 i ごとに集計。） ￼

3.4 バルジ B／内部 I（未対合）

二重ループ（\mathrm{TWO\!-\!LOOP}）は外側 (k,l) と内側 (p,q)（k<p<q<l）で規定。
w_{\mathrm{2loop}}(k,l,p,q)\!=\!\!\!\sum_{b_k,b_l,b_p,b_q}\!\!\overline{\mathcal P}(b_k,b_l,k,l)\;x_{b_p,p}x_{b_q,q}\;\mathcal B(\mathrm{TWO\!-\!LOOP}(b_k,b_l,b_p,b_q;k,l,p,q))\;\mathcal P(b_p,b_q,p,q)
	•	左腕の未対合区間 U_L=(k\!+\!1\,..\,p\!-\!1)、右腕 U_R=(q\!+\!1\,..\,l\!-\!1)。
|U_L|=0 xor |U_R|=0 なら B（バルジ）、両方 >0 なら I（内部）。
よって
p(i,B)=\frac{1}{Z}\!\!\!\sum_{\substack{k<p<q<l\\ i\in U_L\cup U_R\\ \text{one side }=0}}\!\! w_{\mathrm{2loop}}(k,l,p,q),\quad
p(i,I)=\frac{1}{Z}\!\!\!\sum_{\substack{k<p<q<l\\ i\in U_L\cup U_R\\ \text{both}>0}}\!\! w_{\mathrm{2loop}}(k,l,p,q)
（CapR の図式説明と同じ発想：外側の outside と内側の inside の積を全候補で総和。）

3.5 多分岐 M（未対合）

p(i,M)\;=\;\frac{1}{Z}\sum_{p\ge 0}\sum_{j>i}\underbrace{\overline{\mathcal M}(p,i,j)\;\mathcal B(M_u)\;\mathcal M(p,i+1,j)}_{\mathcal M(p,i,j)\text{ の skip（未対合）に伴う重み}}
（\mathcal M の 未対合吸収 項で i が未対合として現れたときの重みを集計。）
CapR で「未構造 U=E+M」を使うときは p(i,E)+p(i,M) を用います。 ￼

以上の集計で \sum_\delta p(i,\delta)=1 が数値的に検証できます（わずかな差は丸め／スケーリング由来）。CapR の「最大スパン W」のようなヒューリスティクスを入れる場合は、該当する総和の添字に制限を課してください。 ￼

3.6 既存 DP だけで足りるか？
	•	**足ります。**JAX‑RNAfold は inside で ヘアピン／二重ループ／多分岐を明示し、outside を導入すれば 各ループ事象の周辺確率が取れます。
	•	注意点：
	•	btaf203 の「コアキシャル・ダングル等の再編成」は \mathcal B(\cdot) へ吸収でき、文脈（E/H/B/I/M/S）自体は不変。
	•	実装上は \mathcal E(j{+}1) を伴う形で inside/outside を組むこと（上 2.1 参照）。
	•	多分岐の枝数を p で数える設計でも、上の p(i,M) は p で総和すればよい（CapR の Multi/Multi1/… に相当）。 ￼
	•	もし不足があるとすれば：
	1.	擬似結びや 離散的な「特殊ケース」エネルギー（特殊ヘアピン等）を除外していると、その分だけループ重みが欠落します。この場合は \mathcal B(\cdot) に該当項を追加し、outside の伝播先（通常は \mathcal P 内部の髙次項）を 1 本増やします。
	2.	コアキシャル専用状態を別に持つ実装にする場合、その状態を子に持つ親の outside 伝播式を 1 行追加。
いずれも outside の書き方は一意で、右辺に新しい子が増えれば、その子へ \overline{\text{親}}\times(\text{右辺の他子の inside と係数}) を足す、というルールです。 ￼

⸻

4) 実装ノート（JAX で書く際の要点）
	•	順序：inside（i を降順・区間長を昇順）→ outside（依存グラフの逆位相順；\overline{\mathcal E}(0){=}1 から前方へ）。
	•	数値安定化：btaf203 同様に 塩基ごとのスケーリングを inside と outside の両方に適用（確率は比なので打ち消し合う）。
	•	複雑度：素朴に書くと O(n^4)。内部ループ長や分岐本数のカットで O(n^3)（McCaskill／CapR と同様）。
	•	検証：
	•	\sum_j p(i,j)+p(i,E)+p(i,H)+p(i,B)+p(i,I)+p(i,M)\approx 1。
	•	\sum_{i<j} p(i,j)= 既存実装のペア確率総和と一致。
	•	CapR の U=E+M も同様に再現できるはずです。 ￼

⸻

5) 参考：どこを何で参照したか
	•	**JAX‑RNAfold の inside 形式（\mathcal E,\mathcal P,\mathcal M）**は NAR 論文の式（8–10）をベースに整理（上では \mathcal E(j{+}1) を明示）。
	•	btaf203 の最適化点（コアキシャル等の扱いの組替えによる定数因子 256 削減、チェックポイント、スケーリング選択）は本文 2.1 に準拠。  ￼
	•	outside の導出原理は McCaskill の restricted partition と outside（確率の取り出し）に準拠、LinearPartition の outside 擬似コードとも整合。
	•	CapR の p(i,\delta) の定義と SCFG 的内外再帰は原著に準拠し、本回答では JAX‑RNAfold の状態に落とし込み。

⸻

付録：式の「使い方」まとめ（実装チェックリスト）
	1.	inside：\mathcal E,\mathcal P,\mathcal M を上記で埋める（\mathcal E(j+1) を掛ける形で）。
	2.	outside：\overline{\mathcal E}(0)=1 から伝播（2.1–2.3 の 3 族）。
	3.	ペア確率：p(i,j) を 2.4 で取得。
	4.	文脈確率：
	•	p(i,S)=\sum_j p(i,j)
	•	p(i,E)=(\overline{\mathcal E}(i)\,\mathcal E(i+1))/Z
	•	p(i,H)=\frac{1}{Z}\sum_{k<i<l} \sum_{b_k,b_l}\overline{\mathcal P}\cdot x\cdot \mathcal B(\mathrm{ONE\!-\!LOOP})
	•	p(i,B),p(i,I)：二重ループの w_{\mathrm{2loop}}(k,l,p,q) を区間分類で配分
	•	p(i,M)=\frac{1}{Z}\sum_{p,j}\overline{\mathcal M}(p,i,j)\,\mathcal B(M_u)\,\mathcal M(p,i+1,j)
（必要なら U=E+M も計算。） ￼

⸻

補足的コメント
	•	btaf203 の「より良い再帰」は厳密性を保ったまま定数因子を削る設計です（outside の形はそのまま保てる）。
	•	NAR（2024）の \mathcal E の式表示は \mathcal E(j+1) が省かれて見えるので、コードでは必ず掛ける形で実装してください（McCaskill と一致）。  ￼

⸻

参考文献（本文で逐次参照）
	•	JAX‑RNAfold（btaf203）：アルゴリズム最適化と実装の要点（係数 256 削減、チェックポイント、スケール）。
	•	Matthies et al., NAR 2024：\mathcal E,\mathcal P,\mathcal M の inside 定義（確率的配列上の厳密 DP）。
	•	McCaskill 1990：outside の原理と Q^{\mathrm b}Q^{\text{out}}/Q による p(i,j)。
	•	LinearPartition 2020：inside/outside の実装視点（SKIP/POP）と outside 擬似コード。
	•	CapR 2014：構造プロファイル p(i,\delta) の定義と SCFG inside–outside。

⸻

必要であれば、上記の outside を JAX のベクトル化（vmap）に載せるための添字順やメモリ配置（[i,j] 3 重和の効率化）まで書き下します。実装途中で詰まった箇所（特に \mathcal E の右側連結、二重ループの 4 重和、\mathcal M の p 取り扱い）を具体的に挙げていただければ、そこをコードレベルの擬似コードで補います。