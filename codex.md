ここの workspace の運用ルールです.

# _docs/ の運用ルール
| ディレクトリ名  | 目的                                     | 記録内容                                                                         |
| --------------- | ---------------------------------------- | -------------------------------------------------------------------------------- |
| _docs/thinking/ | 設計判断や思考過程を外部化               | 迷った点、却下した案、判断理由などを短文で残す。codex が設計意図を理解できる形に |
| _docs/features/ | 新機能の追加・改修の目的と背景を記録     | 実装目的、画面構成、データ構造、リスク、完了条件などを簡潔にまとめる             |
| _docs/deleted/  | 削除・廃止した機能やファイルの履歴を残す | 削除理由、影響範囲、代替手段、再発防止策を記録する                               |


# この repo の概要
ここの repo では CapR という Rfold ベースで書かれた software を、
jax-rnafold という McCaskill ベースで書かれた partition function 計算のための software によって書き直す project を行います。
基本的に jax rnafold によって計算される partition function を利用して capr で計算する p(i, hairpin) のような各塩基が各構造をとる確率を計算します。
現在 jax rnafold では確率的文脈自由文法の forward (inside) algorithm のみが実装されており、outside algorithm は未実装です。
 - 現在管理者は outside algorithm の書き下しを行なっています。（2025-10-30 現在）

未来に行う 計算結果の比較のために CapR (Rfold の turner1999しか対応していない) と LinCapR (linearization されているが turner1999 と turner2004 両者に対応) を install しました（git add submodule みたいなやつ）。

## 具体的な実装のイメージ
 -  jax-rnafold の inside algorithm を実装している部分（submodules/jax-rnafold/src/jax_rnafold/d0/ss.py の line 432-462 あたり）を丸ごと借りてきて、それを用いて inside 変数を計算する
    -  `import jax-rnafold` 的なノリのことをする。
 -  その上で notes/ 以下に記載されている outside table の埋め方（具体的な recursion）を jax で実装する
 * notes や README.md を読みながら、慎重に実装を進めて行う必要あり
 - すでにある src/jax_capr はクオリティが極めて低い粗悪なコードですから、一から作るつもりで、コーディングを進めてください。
結構量の多い仕事ですが、step by step で少しずつ進めていけば大丈夫です。よろしくお願いいたします！