# TensorFlow ライブラリの機械学習処理フローの練習コード集

TensorFlow における基本的な機械学習処理（特にニューラルネットワーク）の練習用コード集。</br>
この README.md ファイルには、各コードの実行結果、概要、機械学習の背景理論の説明を記載しています。</br>
分かりやすいように `main.py` ファイル毎に１つの完結したコードにしています。


## 項目 [Contents]

1. [TensorFlow での機械学習処理の全体フロー](#ID_0)
1. [使用するライブラリ](#ID_1)
1. [使用するデータセット](#ID_2)
1. [コードの実行結果](#ID_3)
    1. [ニューラルネットにおける活性化関数の実装 : `main1.py`](#ID_3-1)
    1. [損失関数の実装 : `main2.py`](#ID_3-2)
    1. [誤差逆伝播法の実装 : `main3.py`](#ID3-3)
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


<a id="ID_0"></a>

## TensorFlow での機械学習処理の全体フロー

- データセットを読み込み or 生成する。
- データを変換、正規化する。 </br>
  （例）正規化処理 : `data = tf.nn.batch_norm_with_global_normalization(...)`
- データセットをトレーニングデータ、テストデータ、検証データセットに分割する。
- アルゴリズム（モデル）のパラメータ（ハイパーパラメータ）を設定する。
- 変数とプレースホルダを設定
    - TensorFlow は、損失関数を最小化するための最適化において、</br>
      変数と重みベクトルを変更 or 調整する。
    - この変更や調整を実現するためには、</br>
      "プレースホルダ [placeholder]" を通じてデータを供給（フィード）する必要がある。
    - そして、これらの変数とプレースホルダと型について初期化する必要がある。
- モデルの構造を定義する。</br>
  データを取得 or 設定し、変数とプレースホルダを初期化した後は、</br>
  モデルを定義する必要がある。これは、計算グラフを作成するという方法で行う。 </br>
    （例）線形モデル : `y_predict = tf.add( tf.mul( x_input, weight_matrix), b_matrix )`
- 損失関数を設定する。
- モデルの初期化と学習（トレーニング）
    - ここまでの準備で, 実際に, 計算グラフ（有向グラフ）のオブジェクトを作成し,</br>プレースホルダを通じて, データを計算グラフ（有向グラフ）に供給する。</br>
    - そして、構築した計算グラフは　セッション [Session] により、その処理を実行する。</br>
    （例） 
        ```python
        with tf.Session( graph = graph ) as session:
            ...
            session.run(...)
        ```
- モデルの評価 (Optional)
- ハイパーパラメータのチューニング (Optional)
- デプロイと新しい成果指標の予想 (Optional)

<a id="ID_1"></a>

## 使用するライブラリ

> TensorFlow ライブラリ </br>
>> API 集 </br>
https://www.tensorflow.org/api_docs/python/ </br>

>> ニューラルネットワーク :</br>
https://www.tensorflow.org/versions/r0.12/api_docs/python/nn/</br>
>>> 活性化関数 </br>
https://www.tensorflow.org/versions/r0.12/api_docs/python/nn/activation_functions_
>>>> ReLu 関数 : `tf.nn.relu(...)` </br>
https://www.tensorflow.org/versions/r1.1/api_docs/python/tf/nn/relu</br>
>>>> ReLu6 関数 : `tf.nn.relu6(...)` </br>
https://www.tensorflow.org/versions/r0.12/api_docs/python/nn/activation_functions_#relu6</br>
>>>> ソフトプラス関数 : `tf.nn.softplus(...)`</br>
https://www.tensorflow.org/api_docs/python/tf/nn/softplus </br>
>>>> ELU 関数 : `tf.nn.elu(...)` </br>
https://www.tensorflow.org/api_docs/python/tf/nn/elu </br>
>>>> シグモイド関数 : ``</br>
</br>
>>>> tanh 関数 : `` </br>
</br>

> その他ライブラリ </br>

> scikit-learn ライブラリ </br>


<a id="ID_2"></a>

## 使用するデータセット

- 

<a id="ID_3"></a>

## コードの実行結果

<a id="ID_3-1"></a>

## ニューラルネットにおける活性化関数の実装 : `main1.py`

ニューラルネットワークにおいて、各ニューロンが情報を取得する時、それぞれの入力信号に対し結合状態に応じた重み付け（結合荷重）を行うが、ニューロンの出力は、重み付けされた入力の和であるに対し、非線形関数で変換した値となる。</br>
この関数を活性化関数といい、多くの場合に飽和的性質をもつ関数となる。</br>
これは、生体のニューロンがパルスを発火するときに、入力を増やして出力パルスの時間密度を上げていくと密度が飽和する特性を持つ。この特性を真似たものである。</br>
</br>
ここでは、代表的な幾つかの活性化関数（ReLu 関数、シグモイド関数、tanh 関数等）の図を plot するサンプルコードを実装する。

- 活性化関数は、TensorFlow のニューラルネットワークライブラリ : `tensorflow.nn` でサポートされている。
    - 標準的な活性化関数を使用したい場合は、</br>
      このライブラリでサポートされている組み込み関数を使用すればよい。
    - 独自のカスタムな活性化関数を使用したい場合は、</br>
      TensorFlow の各種オペレーションを応用することで、これを実現できる。
- まず初めに、活性化関数の１つである ReLu [Rectified Liner Unit] 関数の実装 & 描写を行う。</br>
  この活性化関数は、出力値が 0 の範囲が存在するため、出力に過疎生を作るのが特徴である。
    - 具体的には、以下の TensorFlow の組み込み関数を使用する。
    - `tf.nn.Relu( features )` : Relu オペレーション（Opノード）を返す。
        - x < 0 の範囲では値が 0、x > 0 の範囲では x に比例した単調増加値となる関数。 ( = `max(0,x)` )
        - 引数の `features` は Tensor 型。但し、`numpy.ndarray` オブジェクトに限り、直接渡すことが可能。
    - `tf.nn.Relu6( features )` : Relu6 オペレーション（Opノード）を返す。
        - 上記の Relu 関数の上限値を 6 にした関数。 ( = `min( max(0,x), 6 )` )
        - 引数の `features` は Tensor 型。但し、`numpy.ndarray` オブジェクトに限り、直接渡すことが可能。
- ソフトプラス関数は、Relu 関数の x=0 における非連続性を滑らかにした関数であり、微分可能な関数である。
    - ソフトプラス関数の TensorFlow における組み込み関数は、以下のようになる。
    - `tf.nn.softplus( features )` : Softplus オペレーション（Opノード）を返す。
        - 引数の `features` は Tensor 型。但し、`numpy.ndarray` オブジェクトに限り、直接渡すことが可能。
- ELU [Exponetial Liner Unit] 関数は、ソフトプラス関数に似た特徴を持つが、下の漸近線が 0 ではなく -1 となる。
    - ELU 関数の TensorFlow における組み込み関数は、以下のようになる。
    - `tf.nn.elu( features )` : Elu オペレーション（Opノード）を返す。
        - 引数の `features` は Tensor 型。但し、`numpy.ndarray` オブジェクトに限り、直接渡すことが可能。
- 
</br>

> 活性化関数のグラフ１
>> 活性化関数の内、Relu, Relu6, softplus, ELU 関数の図
![processingformachinelearning_tensorflow_1-1](https://user-images.githubusercontent.com/25688193/30203903-ac94e5ec-94be-11e7-867f-fc78b059ef44.png)
[拡大図]
![processingformachinelearning_tensorflow_1-1](https://user-images.githubusercontent.com/25688193/30203883-9ac00c48-94be-11e7-888d-fff494e5d1f7.png)


> 活性化関数のグラフ２
>> 活性化関数の内、sigmoid, tanh, softsign 関数の図


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

