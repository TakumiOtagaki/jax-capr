## Multiloop unpaired=0 pathology (2025-02-??)

### 現状
- inside 実装 (`submodules/jax-rnafold/src/jax_rnafold/d0/ss.py`) の `ML` テーブルは  
  `ML[nb, i, j] = s_table[1] * em.en_multi_unpaired() * ML[nb, i+1, j] + Σ ML[idx, k+1, j] * MB[i, k]`  
  の形になっており、**常に1塩基「未対合」を消費してから**再帰に入る。
- `md.uniq_ML = 1` 設定でも ViennaRNA の `qm/qm1` 分解に相当する「unpaired=0 の枝分かれ」が存在せず、マルチ閉じペア直後にブランチが始まる経路が **inside 側で生成されない**。

### ViennaRNA との違い
- ViennaRNA (`partfunc.c`) では `qm1` テーブルで「ブランチ直後」の部分構造を別途保持し、`qm` では `expMLbase * qm[i+1,j]` に加えて `Σ qb[i,k] * qm1[k+1, j]` のように **未対合0の分岐**を明示的に扱う。
- 現行 JAX 実装では `ML[0,:,:], ML[1,:,:]` のベースケースを 1 に設定しているが、その後の更新で未対合0の遷移が出現しないため、  
  `(multi closing pair) -- (branch start)` が隣接するケースで inside がゼロを返し、outside も同経路を一切受け取れない。

### 観測されたズレ
- シーケンス `AAUUUUCCCAGCAGUCCCCACUAUAGCUACCCAUACGGUACCAGGGGCAAACGUGAAAUUGCCCCGCGGGAGUAC` の  
  `(i,j) = (11,65)` から `(43,64)` にかけて未対合0のマルチ分岐が存在すると、Vienna の bpp は ~0.999 だが JAX 実装は ~0.11。
- 同区間に 1 塩基挿入して未対合を 1 にすると誤差が大幅に縮小するため、境界条件に依存した欠落が確定。

### 現在の問題整理
1. **内側 DP 構造の欠落**  
   - `ML` が Vienna の `qm/qm1` に相当する2系統を一つに畳み込んでいるが、遷移式が未対合を必須にしているため表現力が不足している。
   - その結果、multiloop closing pair と最初の branch pair が隣接する構造は partition 関数に寄与しない。

2. **Outside 側への伝播**  
   - inside がゼロを返すため outside も同ルートを復元できず、bpp への寄与が完全に欠落。scale を変更してもズレが不変。

### TODO
- `ML` を Vienna の `qm/qm1` に合わせて再設計し、unpaired=0 を許す分解を追加する (例: 追加テーブル or index を増やす)。
- inside/outside のベースケースとスケーリング (`s_table[2]` など) を再確認し、Vienna の `expMLclosing`, `expMLbase` との対応を文書化。
