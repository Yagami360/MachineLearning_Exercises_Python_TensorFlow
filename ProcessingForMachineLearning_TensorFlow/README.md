# TensorFlow ライブラリの機械学習処理フローの練習コード集

TensorFlow における基本的な機械学習処理（特にニューラルネットワーク）の練習用コード集。</br>
この README.md ファイルには、各コードの実行結果、概要、機械学習の背景理論の説明を記載しています。</br>
分かりやすいように `main.py` ファイル毎に１つの完結したコードにしています。

> 参考 URL :
>> ニューラルネットワークにおける損失関数について
>>> http://s0sem0y.hatenablog.com/entry/2017/06/19/084210
>>> http://s0sem0y.hatenablog.com/entry/2017/06/20/135402

## 項目 [Contents]

1. [TensorFlow での機械学習処理の全体ワークフロー](#ID_0)
1. [使用するライブラリ](#ID_1)
1. [使用するデータセット](#ID_2)
1. [コードの実行結果](#ID_3)
    1. [ニューラルネットにおける活性化関数の実装 : `main1.py`](#ID_3-1)
    1. [損失関数（評価関数、誤差関数）の実装 : `main2.py`](#ID_3-2)
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


</br>
<a id="ID_0"></a>

## TensorFlow での機械学習処理の全体ワークフロー

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

</br>
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
>>>> シグモイド関数 : `tf.nn.sigmoid(x)` or `tf.sigmoid(x)` </br>
https://www.tensorflow.org/api_docs/python/tf/sigmoid </br>
>>>> tanh 関数 : `tf.nn.tanh(x)` or `tf.tanh(x)` </br>
https://www.tensorflow.org/api_docs/python/tf/tanh </br>

> scikit-learn ライブラリ </br>
>> Iris データセット : `sklearn.datasets.load_iris()`</br>
http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html</br>

> その他ライブラリ </br>


</br>
<a id="ID_2"></a>

## 使用するデータセット

- Iris データセット : `sklearn.datsets.load_iris()`

</br>
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
        - x < 0 の範囲では値域が 0、</br>
          x > 0 の範囲では値域が x に比例した単調増加値となる関数。 ( = `max(0,x)` )
        - 引数の `features` は Tensor 型。但し、`numpy.ndarray` オブジェクトに限り、直接渡すことが可能。
    - `tf.nn.Relu6( features )` : Relu6 オペレーション（Opノード）を返す。
        - 上記の Relu 関数の値域の上限値を 6 にした関数。 ( = `min( max(0,x), 6 )` )
        - 引数の `features` は Tensor 型。但し、`numpy.ndarray` オブジェクトに限り、直接渡すことが可能。
- ソフトプラス関数は、Relu 関数の x=0 における非連続性を滑らかにした関数であり、微分可能な関数である。
    - softplus(x) = log( exp(x) + 1 ) で与えられる関数。
    - ソフトプラス関数の TensorFlow における組み込み関数は、以下のようになる。
    - `tf.nn.softplus( features )` : Softplus オペレーション（Opノード）を返す。
        - 引数の `features` は Tensor 型。但し、`numpy.ndarray` オブジェクトに限り、直接渡すことが可能。
- ELU [Exponetial Liner Unit] 関数は、ソフトプラス関数に似た特徴を持つが、値域の下の漸近線が 0 ではなく -1 となる。
    - elu = exp(x) - 1 ( x < 0 ), elu = x ( x > 0 ) で与えられる関数
    - ELU 関数の TensorFlow における組み込み関数は、以下のようになる。
    - `tf.nn.elu( features )` : Elu オペレーション（Opノード）を返す。
        - 引数の `features` は Tensor 型。但し、`numpy.ndarray` オブジェクトに限り、直接渡すことが可能。
- シグモイド関数は、以下のような式で与えられる関数で、値域が [0, 1] となる。
    - sigmoid(x) = 1 / { 1 + exp(-x) } で与えられる関数
    - シグモイド関数の、TensorFlow における組み込み関数は、以下のようになる。
    - `tf.nn.sigmoid( x )` or `tf.sigmoid( x )`
        - 引数の `x` は Tensor 型。但し、`numpy.ndarray` オブジェクトに限り、直接渡すことが可能。
    - 値域が [0,1] となるが、都合に合わせてオフセットして [-1,1] などにすれば良い。
- ハイパブリックタンジェントは、以下のような式で与えられる関数で、値域が [-1,1] となる。
    - tanh(x) = { exp(x)-exp(-x) }/{ exp(x)+exp(-x) }
    - ハイパブリックタンジェントの TensorFlow における組み込み関数は、以下のようになる。
    - `tf.nn.tanh( x )` or `tf.tanh( x) `
        - 引数の `x` は Tensor 型。但し、`numpy.ndarray` オブジェクトに限り、直接渡すことが可能。
- ソフトサイン関数は、
    - `tf.nn.softsign( features )` : Softsign オペレーション（Opノード）を返す。
        - 引数の `features` は Tensor 型。但し、`numpy.ndarray` オブジェクトに限り、直接渡すことが可能。
</br>

> 活性化関数のグラフ１
>> 活性化関数の内、Relu, Relu6, softplus, ELU 関数の図
![processingformachinelearning_tensorflow_1-1](https://user-images.githubusercontent.com/25688193/30203903-ac94e5ec-94be-11e7-867f-fc78b059ef44.png)
[拡大図]
![processingformachinelearning_tensorflow_1-1](https://user-images.githubusercontent.com/25688193/30203883-9ac00c48-94be-11e7-888d-fff494e5d1f7.png)


> 活性化関数のグラフ２
>> 活性化関数の内、sigmoid, tanh, softsign 関数の図
![processingformachinelearning_tensorflow_1-2](https://user-images.githubusercontent.com/25688193/30211949-e16ce07a-94dd-11e7-9562-6d121aeeb59e.png)
[拡大図]
![processingformachinelearning_tensorflow_1-2](https://user-images.githubusercontent.com/25688193/30211950-e16e1922-94dd-11e7-9320-7b16dd6006f6.png)


</br>
<a id="ID_3-2"></a>

## 損失関数（評価関数、誤差関数）の実装 : `main2.py`

損失関数（評価関数、誤差関数）は、モデルの出力と目的値（真の値、教師データ）との差（いいえ変えれば、誤差、距離）を計測する関数であり、モデルの学習に適用されるものである。<br>

ここで、ニューラルネットワーク、より広義には機械学習は、<br>
大きく分けて以下の２つの問題設定＆解決のための手法に分けることが出来た。<br>
① 回帰問題の為の手法。（単回帰分析、重回帰分析、等）<br>
② （クラスの）分類問題の為の手法（SVM、k-NN、ロジスティクス回帰、等）<br>

従って、損失関数も同様にして、回帰問題の為の損失関数と、分類問題の為の損失関数が存在することになる。<br>
ここでは、それぞれの問題（回帰問題分類問題）の損失関数を分けて TensorFlow で実装する。

### ① 回帰問題の為の損失関数（評価関数、誤差関数）

- 回帰の為の損失関数は、言い換えると、連続値をとる独立変数を予想する損失関数である。
- そのための TensorFlow での実装として、<br>
  一連の予測値（ `x_predicts_tsr` ）と、１つの目的値（ `target_tsr` ）を Tensor として作成する。
    - 一連の予想値 : `x_predicts_tsr = tf.linspace( -1. , 1. , 500 )`
        - -1 ~ +1 の範囲の 500 個のシーケンス Tensor
    - １つの目的値 : `target_tsr = tf.constant( 0. )`
        - 定数 Tensor で値（目的値）を 0 としている。
- そして（TensorFlow を用いて）以下の損失関数に関しての関数値を求め、それらのグラフを描写する。
    - L2 正則化の損失関数（L2ノルムの損失関数、ユークリッド損失関数）は、<br>
      目的値への距離の２乗で表される損失関数であり、<br>
      その TensorFlow による関数値の算出＆グラフ化のための実装は、以下のように書ける。<br>
        - `l2_loss_op = tf.square( target_tsr - x_predicts_tsr )` : L2 正則化の損失関数のオペレーション Square
        - `output_l2_loss = session.run( l2_loss_op )` : L2 正則化の損失関数の値 （グラフの y 軸値の list）
        - 尚、ここでは、目的値が 0 のケースのグラフを描写する。
    - L1 正則化の損失関数（L1ノルムの損失関数、絶対損失関数）は、<br>
      目的値の距離への絶対値で表される損失関数であり、<br>
      その TensorFlow による関数値の算出＆グラフ化のための実装は、以下のように書ける。<br>
        - `l1_loss_op = tf.abs( target_tsr - x_predicts_tsr )` : L1 正則化の損失関数のオペレーション Abs
        - `output_l1_loss = session.run( l1_loss_op )` : L2 正則化の損失関数の値（グラフの y 軸値の list）
        - 尚、ここでは、目的値が 0 のケースのグラフを描写する。
    - 損失関数のグラフ化のためのグラフ化のための x 軸の値のリスト `axis_x_list` は、<br>
      先に定義した シーケンス Tensor を `session.run(...)` することで取得できる。
        - `axis_x_list = session.run( x_predicts_tsr )`

#### ① 回帰の為の、損失関数（L2正則化、L1正則化）のグラフ
![processingformachinelearning_tensorflow_2-1](https://user-images.githubusercontent.com/25688193/30562150-6efa89e8-9cf8-11e7-924f-43a3f3623248.png)

- L2 正則化の損失関数は、目的値への距離の２乗で表されるので、下に凸な２次関数の形状をしており、<br>
  目的値（この場合 0）の近くで急なカーブを描く。<br>
  この特性が、損失関数と扱う際に優れているのが特徴である。
- L1 正則化の損失関数は、目的値への距離の絶対値で表される損失関数である。</br>
  その為、目的値（この場合 0）からのズレが大きくなっても（ズレの大きなに関わらず）、その傾き（勾配）は一定である。<br>
  その為、L1 正則化は L2 正則化よりも、外れ値にうまく対応するケースが多いのが特徴である。<br>
  又、目的値（この場合 0）にて、関数が連続でないために、対応するアルゴリズムがうまく収束しないケースが存在することに注意が必要となる。


<br>

### ② クラスの分類問題の為の損失関数（評価関数、誤差関数）

クラスの分類問題の為の損失関数は、現在の学習結果が与えられたデータに対してどの程度「良い感じなのか」を定量化するために使われる。（誤差逆伝播法時の計算等）<br>
分類問題でのニューラルネットワークの最終結果は、例えば２クラスの分類問題の場合、正解は -1 or 1（又は 0 or 1）の負例と正例となる。従って、損失関数による損失は、連続な値ではなく sign 化したもの 、即ち正解ラベルと、ニューラルネットワークの出力の符号が一致しているならば損失は 0（＝分類が成功）であり、符号が一致していなければ損失は 1 となる。

- クラスの分類問題の為の損失関数の TensorFlow での実装として、<br>
  一連の予測値（ `x_predicts_tsr` ）と、１つの目的値（ `target_tsr` ）を Tensor として作成する。
    - 一連の予想値 : `x_predicts_tsr = tf.linspace( -3. , 5. , 500 )`
        - -3 ~ +5 の範囲の 500 個のシーケンス Tensor
    - １つの目的値 : `target_tsr = tf.constant( 0. )`
        - 定数 Tensor で値（目的値）を 1 としている。（分類問題なので -1 or 1, 又は 0 or 1 ）
    - 目的値のリスト  : `targets_tsr = tf.fill( [500,], 1. )`
        - 値を 1 とする 500 個のリスト（重みベクトルとの演算で使用）
- そして（TensorFlow を用いて）以下の損失関数に関しての関数値を求め、それらのグラフを描写する。
    - ヒンジ損失関数 [hinge loss funtion] は、２つのターゲットクラス（ -1 or 1 ）の間での損失値を計算する関数である。（※ 0 or 1 のクラス分類ではないことに注意）<br>
    その式は、目的値が 1 の場合、y = 1 - 1*x ( x < 1 ) , y = 0 ( x >= 1 ) で与えられる。<br>
    そして TensorFlow による関数値の算出＆グラフ化のための実装は、以下のように書ける。<br>
        - `hinge_loss_op = tf.maximum( 0., 1. - tf.multiply( target_tsr, x_predicts_tsr ) )` : ヒンジ損失関数のオペレーション 
        - `output_hinge_loss = session.run( hinge_loss_op )` : ヒンジ損失関数の値（グラフの y 軸値の list）
        - 尚、ここでは、目的値が 1 のケースのグラフを描写する。
    - クロス・エントロピー損失関数 [cross-entropy loss funtion]（２クラスの場合）は、２つのクラス 0 or 1 を予想する場合に使用される。<br>
    （※ -1 or 1 のクラス分類ではないことに注意）<br>
    この関数は、シャノンの情報理論でいうところの２つの確率分布 P,Q の相互情報量であり、<br>
    H ( P, Q ) = P_1 * log( 1/Q_1 ) + P_2 * log( 1/Q_2 ) = - P_1 * log( Q_1 ) - P_2 * log( Q_2 ) という式で書ける。<br>
    （ここでは、目的値を 1 とするので、P_1 = 1, P_2 = 0 ）<br>
    そして TensorFlow による関数値の算出＆グラフ化のための実装は、以下のように書ける。<br>
        - `cross_entopy_loss_op =`<br>
          `- tf.multiply( target_tsr, tf.log( x_predicts_tsr ) )`<br>
          `- tf.multiply( 1 - target_tsr, tf.log( 1 - x_predicts_tsr ) )` <br>
          クロス・エントロピー関数のオペレーション 
        - `output_cross_entopy_loss = session.run( cross_entopy_loss_op )` <br>
          クロス・エントロピー関数の値（グラフの y 軸値の list）
        - 尚、ここでは、目的値が 1 のケースのグラフを描写する。
    - シグモイド・クロス・エントロピー損失関数は、クロス・エントロピー関数とよく似ているが、<br>
        x 値をシグモイド変換してからクロス・エントロピー関数を計算するという違いがあり、<br>
        loss = z * - log( sigmoid(x) ) + (1 - z) * -log( 1 - sigmoid(x) ) とい式で書ける。<br>
        そして TensorFlow による関数値の算出＆グラフ化のための実装は、以下のように書ける。<br>
        ```python
        # シグモイド・クロス・エントロピー関数のオペレーション
        # x = logits, z = labels. 
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        # tf.nn.softmax_cross_entropy_with_logits(...) : 
        # 推計結果の softmax 値を計算して、cross-entropy を計算。
        sigmoid_cross_entropy_loss_op \
        = tf.nn.softmax_cross_entropy_with_logits( 
              logits = x_expanded_predicts_tsr,   # 最終的な推計値。softmax はする必要ない
              labels = expaned_targets_tsr        # 教師データ
        )

        # シグモイド・クロス・エントロピー損失関数の値 （グラフの y 軸値の list）
        output_sigmoid_cross_entropy_loss = session.run( sigmoid_cross_entropy_loss_op )
        ```
    - 重み付きクロス・エントロピー損失関数は、重み付きのシグモイド・クロスエントロピー損失関数のことである。<br>
    ここでは、正の目的値 1 を 0.5 で重み付けするようにする。<br>
    そして TensorFlow による関数値の算出＆グラフ化のための実装は、以下のように書ける。<br>
    ```python
    # 重み付けクロス・エントロピー損失関数
    # loss = targets * -log(sigmoid(logits)) * pos_weight + (1 - targets) * -log(1 - sigmoid(logits))
    weight_tsr = tf.constant( 0.5 )      # 重み付けの値の定数 Tensor
    weighted_cross_entropy_loss_op = tf.nn.weighted_cross_entropy_with_logits(
                                         logits = x_predicts_tsr, 
                                         targets = targets_tsr,
                                         pos_weight = weight_tsr
                                     )
    
    # 重み付けクロス・エントロピー損失関数の値 （グラフの y 軸値の list）
    output_weighted_cross_entropy_loss = session.run( weighted_cross_entropy_loss_op )
    ```
    - ソフトマックスクロス・エントロピー損失関数 [softmax cross-entrpy loss function]は、正規化されていない出力を操作する。
    - ...

#### ◎ 分類の為の、損失関数のグラフ
![processingformachinelearning_tensorflow_2-2](https://user-images.githubusercontent.com/25688193/30594195-2d46c742-9d88-11e7-8989-585977c7865b.png)

- ヒンジ損失関数は、...
- クロス・エントロピー交差関数（２クラスの場合）は、
- シグモイド・クロスエントロピー損失関数は、
- 重み付きクロス・エントロピー損失関数は、


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

