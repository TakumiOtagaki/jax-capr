2025-11-06

- ViennaRNA 側の設定を確認。`submodules/jax-rnafold/src/jax_rnafold/common/vienna_rna.py:26-39` で `md.uniq_ML = 1` および `fc.exp_params_rescale(mfe_energy)` を呼んでから `pf()` を実行している。JAX 実装もユニークな multiloop 分解前提なので、比較スクリプトでも `uniq_ML = 1` と rescale の呼び出しが必要。
- `JaxNNModel` は Turner1999 パラメータ読み込み時でも d0（ダングルなし）のボルツマン重みを返す。Vienna 側も `dangles = 0`、`RNA.params_load(str(TURNER_1999))` を明示して整合させる。
- `src/jax_capr/jax_outside.py:403-422` の outside 再帰で hairpin 項が抜けている。forward の `psum_hairpin`（`submodules/jax-rnafold/src/jax_rnafold/d0/ss.py:210-213`）に対応する寄与を追加しないと、hairpin で閉じるペアの `bar_P` が不足して BPP の差が残る。特殊ヘアピン補正も forward と同じ組み合わせで扱う必要がある。
- hairpin ペア `P[bp_idx_hl, h, l]` は forward 側で直接 `psum_hairpin(...)*s_table[u+1]` を掛けて加算されており、outside では「外側にどう繋がるか（外部ループ / stack / bulge / internal / multiloop）」の列挙だけで十分。hairpin 専用の逆再帰は生じない点を意識してデバッグする。

優先調査の洗い出し（2025-11-06）
1. **hairpin 差分**: `GCGGAAACCAGC` の最大差が 2.5e-2 に残る。`jax_outside.py` の各項（外部・stack・bulge・internal・multibranch）を Vienna 参照と突き合わせ、特に bulge/internal の係数や `padded_p_seq` の掛け方を確認。
2. **内部ループの指数因子**: `psum_outer_internal_loops` で `s_table` のインデックスや `lup/rup` 条件が inside 実装と一致しているか再点検。指数の取り違えがあると hairpin 周辺でも誤差が跳ねやすい。
3. **bar_Pm/bar_Pm1 フロー**: multiloop の外側逆伝播 (`fill_bar_Pm`, `fill_bar_Pm1`, `get_bp_h_multi_sm`) が forward の `fill_multibranch` / `fill_multi` とシンメトリックになっているか、`en_multi_unpaired` の累乗や `s_table` の段数を再確認。

追加メモ（2025-11-06, multiloop 調査）
- `get_bp_h_multi_sm` の `bar_Pm` 項で `s_table[h-i]` を掛ける位置を修正。`s(h-i)` を外に出すのではなく、`s_table[1]*ML + s_table[h-i]*pow(en_multi_unpaired, …)` という形にしないと係数が過大になる。
- `fill_bar_Pm1` では `s_table[1]` を余分に掛けていた。擬似コードの定義は `s(j-l)` のみなので、外側で `s_table[1]` を掛けないようにする。
