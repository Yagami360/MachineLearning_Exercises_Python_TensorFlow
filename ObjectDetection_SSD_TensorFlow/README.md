## TensorFlow を用いた SSD [Single Shot muitibox Detector] の実装と簡単な応用

> 実装中...

ニューラルネットワークによる一般物体検出アルゴリズムの１つである、SSD [Single Shot muitibox Detector] を TensorFlow で実装。

この `README.md` ファイルには、各コードの実行結果、概要、SSD の背景理論の説明を記載しています。
分かりやすいように `main.py` ファイル毎に１つの完結した実行コードにしています。

尚、SSD [Single Shot muitibox Detector] に関しての、背景理論は以下のサイトに記載してあります。

- [星の本棚 : ニューラルネットワーク / ディープラーニング](http://yagami12.hatenablog.com/entry/2017/09/17/111935#ID_11-4)


### 項目 [Contents]

1. [使用するライブラリ](#ID_1)
1. [使用するデータセット](#ID_2)
1. [コード説明＆実行結果](#ID_3)
    1. [コード説明＆実行結果 : `main1.py`](#ID_3-1)
    1. [コード説明＆実行結果 : `main2.py`](#ID_3-2)
    1. [](#)
1. [背景理論](#ID_4)
    1. [背景理論１](#ID_4-1)
    1. [](#)
1. [参考サイト](#ID_5)


<a id="ID_1"></a>

### 使用するライブラリ

- TensorFlow ライブラリ
    - xxx

- その他ライブラリ
    - xxx


<a id="ID_2"></a>

### 使用するデータセット

- Open Images Dataset V4
    - https://storage.googleapis.com/openimages/web/download.html


<a id="ID_3"></a>

## コード説明＆実行結果

<a id="ID_3-1"></a>

## コード説明＆実行結果１ : `main1.py`
> 実装中...

- xxx データセットを使用
- 特徴行列 `X_features` は、特徴数 x 個 × サンプル数 x 個 :<br> `X_features = `
- 教師データ `y_labels` は、サンプル数 x 個 : <br >`y_labels = `
- トレーニングデータ xx% 、テストデータ xx% の割合で分割 : <br>`sklearn.cross_validation.train_test_split( test_size = , random_state =  )`
- 正規化処理を実施して検証する。<br> 

<br>

---

<a id="ID_3-2"></a>

## コード説明＆実行結果２ : `main2.py`
> 実装中...

<br>

---

<a id="ID_4"></a>

## 背景理論

<a id="ID_4-1"></a>

## 背景理論１

