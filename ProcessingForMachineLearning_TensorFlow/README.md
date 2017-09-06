# TensorFlow ライブラリの機械学習処理フローの練習コード集

TensorFlow における基本的な機械学習処理（特にニューラルネットワーク）の練習用コード集。</br>
この README.md ファイルには、各コードの実行結果、概要、機械学習の背景理論の説明を記載しています。（予定）</br>
分かりやすいように `main.py` ファイル毎に１つの完結したコードにしています。（予定）


## 項目 [Contents]

1. [使用するライブラリ](#ID_1)
1. [使用するデータセット](#ID_2)
1. [コードの実行結果](#ID_3)
    1. [ニューラルネットにおける活性化関数の実装](#ID_3-1)
    1. [損失関数の実装](#ID_3-2)
    1. [誤差逆伝播法の実装](#ID3-3)
    1. [](#)
1. [背景理論](#ID_4)
    1. [ニューラルネットワークの概要](#ID_4-1)
    1. [活性化関数](#ID_4-2)
    1. [単純パーセプトロン](#ID_4-3)
    1. [パーセプトロンによる論理演算](#ID_4-4) 
    1. [最急降下法による学習](#ID_4-5)
    1. [誤差逆伝播法](#ID_4-6)
    1. [多層パーセプトロン](#ID_4-7)
    1. [](#)


<a id="ID_1"></a>

## 使用するライブラリ

> TensorFlow ライブラリ </br>
>> API 集 </br>
https://www.tensorflow.org/api_docs/python/ </br>


> その他ライブラリ
> scikit-learn ライブラリ </br>


<a id="ID_2"></a>

## 使用するデータセット

> Iris データセット : `datasets.load_iris()`


<a id="ID_3"></a>

## コードの実行結果

<a id="ID_3-1"></a>

## ニューラルネットにおける活性化関数の実装 : `main1.py`
> コード実行中...

- xxx データセットを使用
- 特徴行列 `X_features` は、特徴数 x 個 × サンプル数 x 個 :</br> `X_features = `
- 教師データ `y_labels` は、サンプル数 x 個 : </br >`y_labels = `
- トレーニングデータ xx% 、テストデータ xx% の割合で分割 : </br>`sklearn.cross_validation.train_test_split( test_size = , random_state =  )`
- 正規化処理を実施して検証する。</br> 

</br>

<a id="ID_3-2"></a>

## 損失関数の実装 : `main2.py`
> コード実装中...

</br>

<a id="ID_3-3"></a>

## 誤差逆伝播法（バックプロパゲーション）の実装 : `main3.py`
> コード実装中...

---

<a id="ID_4"></a>

## 背景理論

<a id="ID_4-1"></a>

## ニューラルネットの概要
![twitter_nn1_1_160825](https://user-images.githubusercontent.com/25688193/30112643-09c7ef7a-934d-11e7-91d2-fcc93505baa0.png)
![twitter_nn1_2_160825](https://user-images.githubusercontent.com/25688193/30112644-09c88430-934d-11e7-9450-6d4861190175.png)
![twitter_nn3 -1_160827](https://user-images.githubusercontent.com/25688193/30112645-09c8e42a-934d-11e7-95f9-87e0ca316b2f.png)


<a id="ID_4-2"></a>

## 活性化関数
![twitter_nn2-1_160826](https://user-images.githubusercontent.com/25688193/30112640-09b4803e-934d-11e7-993d-4e35263cda81.png)
![twitter_nn2-2_160826](https://user-images.githubusercontent.com/25688193/30112641-09b5d6d2-934d-11e7-861d-06792890d2f9.png)

<a id="ID_4-3"></a>

## 単純パーセプトロン
![twitter_nn4 -1_160829](https://user-images.githubusercontent.com/25688193/30112642-09b65e90-934d-11e7-9cac-2472c4add901.png)

<a id="ID_4-4"></a>

## パーセプトロンによる論理演算
![twitter_nn6-1_160829](https://user-images.githubusercontent.com/25688193/30112770-703f5f68-934d-11e7-845d-be2240ef4d17.png)
![twitter_nn6-2_160829](https://user-images.githubusercontent.com/25688193/30112772-7042419c-934d-11e7-9330-d8292a108c1c.png)
![twitter_nn8-1 _160902](https://user-images.githubusercontent.com/25688193/30112777-70842ee0-934d-11e7-9486-d3d14be4d6bd.png)
![twitter_nn10-1_160903](https://user-images.githubusercontent.com/25688193/30112972-1a64417a-934e-11e7-96f1-775f232a2767.png)


<a id="ID_4-5"></a>

## 最急降下法による学習
![twitter_nn8-2 _160902](https://user-images.githubusercontent.com/25688193/30112771-7041b13c-934d-11e7-88c7-8692f42b5799.png)
![twitter_nn8-3 _160902](https://user-images.githubusercontent.com/25688193/30112769-703f3cb8-934d-11e7-81f0-f78ef37cb2b2.png)

<a id="ID_4-6"></a>

## 誤差逆伝播法
![twitter_nn9-2_160902](https://user-images.githubusercontent.com/25688193/30112776-70665816-934d-11e7-95d5-fbe5e349b94c.png)
![twitter_nn9-2 _160903](https://user-images.githubusercontent.com/25688193/30112775-70663048-934d-11e7-8280-b27a02dc1e16.png)
![twitter_nn9-3_160903](https://user-images.githubusercontent.com/25688193/30112774-706594d0-934d-11e7-89a7-50814730aafe.png)


<a id="ID_4-7"></a>

## 多層パーセプトロン
![twitter_nn5 -1_160829](https://user-images.githubusercontent.com/25688193/30112646-09d7f8fc-934d-11e7-81fa-4cc74b1e3e39.png)
![twitter_nn5-1_160829](https://user-images.githubusercontent.com/25688193/30112647-09da02d2-934d-11e7-96a1-a8c4592993cc.png)
![twitter_nn9-1_160902](https://user-images.githubusercontent.com/25688193/30112773-7050f1c4-934d-11e7-9343-398900bd8a2d.png)

<a id="ID_4-8"></a>

## 連想記憶ニューラルネットワーク
![twitter_nn11-1_160904](https://user-images.githubusercontent.com/25688193/30112974-1a8ff1b2-934e-11e7-81de-933019772299.png)
![twitter_nn11-2_160904](https://user-images.githubusercontent.com/25688193/30112976-1a965e58-934e-11e7-98a7-f80bdee26b35.png)
![twitter_nn12-1_160904](https://user-images.githubusercontent.com/25688193/30112977-1aa3d1aa-934e-11e7-98fd-626e1a46fc30.png)
![twitter_nn13-1_160904](https://user-images.githubusercontent.com/25688193/30112975-1a9358fc-934e-11e7-871f-dd2b55ff3657.png)
![twitter_nn14-1_160905](https://user-images.githubusercontent.com/25688193/30112978-1aa4cc4a-934e-11e7-9a7b-97c4f9d415b5.png)
![twitter_nn14-2_160905](https://user-images.githubusercontent.com/25688193/30112973-1a8f9122-934e-11e7-8ef0-2b0f82c00645.png)
![twitter_nn15-1_160905](https://user-images.githubusercontent.com/25688193/30112979-1abb3e26-934e-11e7-8e8d-23a72b07fe7c.png)

