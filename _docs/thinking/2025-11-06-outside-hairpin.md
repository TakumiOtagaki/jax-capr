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
- `fill_bar_Pm1` の `j` は forward 同様 `j < seq_len + 1` まで走査し、さらに `gap = j - l - 1` が 1 以上のときに限定する。これで `Pm` と `Pm1` の役割がきれいに分離する。
- multiloop 未対塩基のボルツマン係数は energy モデル側で `em.en_multi_unpaired()` を gap 回べき乗する設計。outside ではこれに `s_table[h-i]` や `s_table[j-l]` を掛けるだけに留め、`s_table[1]` を底に含めない。

2025-11-06 jit 無効化トライ
- `tests/test_outside_vs_vienna.py` の `debug_pair` ブロックを復活させて実行しようとしたが、ローカル環境に `jax` がインストールされていないため `ModuleNotFoundError` で停止。REPL からの簡易確認 (`python3 - <<'PY' ...`) でも同様。手元で再現検証するには jax 付きの仮想環境が必要。
- 今回は CSV に吐き出した `bar_P`, `bar_Pm`, `bar_Pm1`, `bar_M1` などを用いた静的解析に専念し、JIT を切ったトレース実行は見送り。

2025-11-07 multiloop divergence 調査メモ
- `/tmp/outside-debug.log` で `GGCGGAAAGCGAAACGCAAAACGGCAAAAGCCGAAACCGCC` の `debug_pair` を再確認。最大差分ペア `(22,31)` と `(23,30)` の multi 項がそれぞれ `5.57×10^8` / `5.29×10^8` まで伸び、`bar_P` との差分の約 12% を占めている。スタック比率は ~0.98 で、マルチ寄与が主因。
- `tests/bar_Pm1_*.csv` の列走査で `bar_Pm1[10,11] ≈ 8.09×10^14`, `bar_Pm1[4,31] ≈ 1.78×10^8` など桁外れの値を確認。`bar_Pm` も `bar_Pm[4,7] ≈ 2.70×10^11` と同様に膨張しており、長距離ペアが指数スケールと多重に掛かっている。
- `fill_bar_Pm`（`src/jax_capr/jax_outside.py:524-571`）は forward の `fill_multibranch`（`submodules/jax-rnafold/src/jax_rnafold/d0/ss.py:102-120`）と異なり `s_table[2]` ではなく `s_table[1]` を掛けている。forward では枝導入時に `s_table[2]` を付与しているため、outside の係数が 1 段抜けている可能性がある。
- `fill_bar_Pm1`（`src/jax_capr/jax_outside.py:573-625`）は `unpaired_factor = s_table[j-l] * en_multi_unpaired()^{gap}` を用い、`get_bp_h_multi_sm`（同:374-405）内でさらに `s_table[h-i] * en_multi_unpaired()^{h-i-1}` を乗せている。この結果 `s_table[1]` の累乗が左右から重複し、gap が 15–20 の領域で `~e^{(gap+1)*scale/len}` まで膨らむ。左側の `ML[1, i+1, h-1]` が 0 でも第 2 項 `bar_Pm[i,l] * s_table[h-i] * …` が非ゼロのまま残るため、左枝の有無に関わらず外側重みが流れ込む点も forward との不整合として要確認。
- 11/06 に試した multiloop スケール調整では、`fill_bar_Pm1` の gap 条件緩和と `s_table` の付け直しを行ったが、`gap_pow` 経由で `s_table[1]` が二重に掛かる副作用を誘発し、`bar_Pm1` がさらに悪化したため `git restore` で巻き戻し済み。今回の検証で再発が確認できたので、修正方針をまとめてから再度実装する。

2025-11-09 multiloop scaling rollback
- `fill_bar_Pm1` から `s_table[1]` を外し、`gap >= 1` の条件を forward に合わせるパッチを試したが、`get_bp_h_multi_sm` 側で `gap_pow` 相当の計算に既にスケールが含まれていたため、左右双方で `s_table[1]^gap` が掛かり直しとなり `bar_Pm1` が 10^1〜10^2 倍に発散。
- 同時に `fill_bar_Pm` にも `s_table[h-i]` をまとめて掛ける変更を入れた結果、`ml_val` が 0 のケースでも右枝の未対塩基項が残存し、`bar_Pm` が gap 長に比例して負方向へ崩れる挙動を確認。
- 上記副作用により ViennaRNA との差分が 10^3 台まで跳ね上がったため、`git restore` で 11/05 時点の outside 実装へ巻き戻し。今後は `ML` テーブルのスケール構造を整理した上で、`bar_Pm`/`bar_Pm1` の係数を段階的に見直す。
