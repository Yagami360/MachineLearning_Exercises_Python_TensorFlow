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
    1. [誤差逆伝播法の実装 : `main3.py`](#ID_3-3)
    1. [](#)
1. [背景理論](#ID_4)
    1. [ニューラルネットワークの概要](#ID_4-1)
    1. [活性化関数](#ID_4-2)
    1. [学習方法の分類](#ID_4-3)
        1. [教師あり学習 [supervised learning] と教師なし学習 [Unsupervised learning]](#ID_4-3-1)
        1. [バッチ学習 [batch learning] とオンライン学習 [online learning]](#ID_4-3-2)
        1. [強化学習 [reinforcement learning]](#ID_4-3-3)
        1. [転移学習 [transfer learning]](#ID_4-3-4)
    1. [単純パーセプトロン [Simple perceptron]](#ID_4-4)
        1. [単層パーセプトロンのアーキテクチャ [architecture]](#ID_4-4-1)
        1. [誤り訂正学習 [error correction learning rule]（パーセプトロンの学習規則 [perceptron learing rule] ）<br>＜教師あり学習、オンライン学習＞](#ID_4-4-2)
        1. [最急降下法 [gradient descent method] による学習（重みの更新）</br>＜教師あり学習、パッチ学習＞](#ID_4-4-3)
        1. [確率的勾配降下法 [stochastic gradient descent method]](#ID_4-4-4)
     1. [多層パーセプトロン [ MLP : Multilayer perceptron]](#ID_4-5)
        1. [多層パーセプトロンのアーキテクチャ [architecture]](#ID_4-5-1)
        1. [最急降下法 [gradient descent method] による学習（重みの更新）<br>＜教師あり学習、パッチ学習＞](#ID_4-5-2)
        1. [確率的勾配降下法 [stochastic gradient descent method] <br>＜教師あり学習、オンライン学習＞](#ID_4-5-3)
        1. [誤差逆伝播法（バックプロパゲーション）[Backpropagation]<br>＜教師あり学習、バッチ学習 or オンライン学習＞](#ID_4-5-4)
    1. [パーセプトロンによる論理演算](#ID_4-7) 


<br>
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

<br>
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
    - ソフトマックスクロス・エントロピー損失関数 [softmax cross-entrpy loss function] は、正規化されていない出力を操作する。
        - この関数は、ソフトマックス関数を用いて、出力を確率分布に変換して、真の確率分布から損失関数を計算する。
        ```python
        # ソフトマックスクロス・エントロピー損失関数 [softmax cross-entrpy loss function] 
        # L = -actual * (log(softmax(pred))) - (1-actual)(log(1-softmax(pred)))
        unscaled_logits = tf.constant( [[1., -3., 10.]] )   # 正規化されていない予測値
        target_dist = tf.constant( [[0.1, 0.02, 0.88]] )    # 目的値の確率分布
        softmax_entropy_op = \
        tf.nn.softmax_cross_entropy_with_logits(
            logits = unscaled_logits,   # 最終的な推計値。softmax はする必要ない
            labels = target_dist        # 教師データ（目的値の確率分布）
        )

        # ソフトマックスクロス・エントロピー損失関数の値 （グラフの y 軸値の list）
        output_softmax_entropy_loss = session.run( softmax_entropy_op )

        print( "output_softmax_entropy_loss : ", output_softmax_entropy_loss )
        ```
        ```python
        [出力]
        output_softmax_entropy_loss :  [ 1.16012561] 
        ↑ 分類クラス{-1, 1} の内、クラス 1 に分類される値となっている。
        ```
    - 疎なソフトマックスクロス・エントロピー損失関数 [sparse softmax cross-entrpy loss function] は、ソフトマックスクロス・エントロピー損失関数とよく似ているが、目的値が確率分布ではなく、分類クラス {-1 or 1} が真となるクラスのインデックス（ここでは、インデックス 0, 1 が存在）である。
        - ここでは、教師データ : `labels` として、（この関数の一般的な使われ方である 1 の値が 1 つ含まれている以外は、すべて 0 の疎なベクトルではなく）分類が真の値（正解クラス）であるクラスのインデックスを渡す。
        ```python
        # 疎なソフトマックスクロス・エントロピー損失関数 [sparse softmax cross-entrpy loss function]
        # 正規化されていない予測値
        unscaled_logits = tf.constant( [[1., -3., 10.]] )
        # 分類クラス {-1 or 1} が真となるクラスのインデックス
        sparse_target_dist = tf.constant( [2] )

        sparse_entropy_op = \
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=unscaled_logits,     # 最終的な推計値。softmax はする必要ない
            labels=sparse_target_dist   # 教師データ（ここでは、クラスラベルを渡す）
        )

        # 疎なソフトマックスクロス・エントロピー損失関数の値 （グラフの y 軸値の list）
        output_sparse_entropy_loss = session.run( sparse_entropy_op )
        print( "output_sparse_entropy_loss : ", output_sparse_entropy_loss )
        ```
        ```python
        [出力]
        output_sparse_entropy_loss :  [ 0.00012564]
        ```

#### ◎ 分類の為の、損失関数のグラフ
![processingformachinelearning_tensorflow_2-2](https://user-images.githubusercontent.com/25688193/30594195-2d46c742-9d88-11e7-8989-585977c7865b.png)

- ヒンジ損失関数は、...
- クロス・エントロピー交差関数（２クラスの場合）は、
- シグモイド・クロスエントロピー損失関数は、
- 重み付きクロス・エントロピー損失関数は、


<a id="ID_3-3"></a>

## 誤差逆伝播法（バックプロパゲーション）の実装 : `main3.py`
誤差逆伝播法（バックプロパゲーション）は、多層パーセプトロンの学習等、ニューラルネットワークによる学習（重みの更新）で広く使われる学習手法である。<br>
ここでは、TensorFlow で誤差逆伝播法を簡単な例（回帰問題、分類問題）で実装する。<br>
尚、ここでの誤差逆伝播法の実装例は、ニューラルネットワークに対して適用したものではないが、一般的には、誤差逆伝播法は多層パーセプトロン等のニューラルネットワークの学習として使われる手法である。<br>

誤差逆伝播法は、誤差関数（コスト関数、損失関数）に対しての、最急降下法で最適なパラメータ（ニューラルネットワークの場合、重みベクトル）を求め、又の誤差項を順方向（ニューラルネットワークの場合、入力層→隠れ層→出力層）とは逆方向に逆伝播で伝播させる手法であるが、<br>
この最急降下法を TensorFlow で実装する場合は、TensorFlow の組み込み関数 `tf.tf.train.GradientDescentOptimizer( learning_rate )` を使用すれば良い。

### ① 回帰問題での誤差逆伝播法
単純な回帰モデルとして、以下のようなモデルを実装し、最急降下法での最適なパラメータ逐次計算過程、及びそのときの誤差関数の値の過程をグラフ化する

- 平均値 1, 標準偏差 0,1 の正規分布 N(1, 0.1) に従う乱数を 100 個生成する。
    - `x_rnorms = numpy.random.normal( 1.0, 0.1, 100 )`
- そして、この乱数値に対し、Variable（モデルのパラメータで、最適値は 10 になる）を乗算する演算を行う。
    - `a_var = tf.Variable( tf.random_normal( shape = [1] ) )` : 乗算する Variable
    - `x_rnorms_holder = tf.placeholder( shape = [1], dtype = tf.float32 )`<br>
    オペレーター mul_op での x_rnorm にデータを供給する placeholder
    - `mul_op = tf.multiply( x_rnorms_holder, a_var )` : 計算グラフに乗算追加 
- 教師データ（目的値）として、 10 の値の list を設定する。
    - `y_targets = numpy.repeat( 10., 100 )` : 教師データ（目的値）の list
    - `y_targets_holder = tf.placeholder( shape = [1], dtype = tf.float32 )`<br>
    L2 損失関数（オペレーター）loss = y_targets - x_rnorms = 10 - N(1.0, 0.1) での `y_targets` に教師データを供給する placeholder
- 損失関数として、この教師データと正規分布からの乱数に Variable を乗算した結果の出力の差からなる L2 ノルムの損失関数を設定する。（変数 a が 10 のとき誤差関数の値が 0 になるようなモデルにする。）
    - `l2_loss_op = tf.square( mul_op - y_targets_holder )` : 目的値と乗算の出力の差
- 最急降下法によるパラメータの最適化を行う。（学習プロセスは、損失関数の最小化）
    - `optGD_op = tf.train.GradientDescentOptimizer( learning_rate = 0.01 )`
    - `train_step = optGD_op.minimize( l2_loss_op )`
- 各エポックに対し、オンライン学習を行い、パラメータを最適化していく。
    ```python
    a_var_list = []
    loss_list = []
    # for ループでオンライン学習
    for i in range( num_train ):
        # RNorm のイテレータ : ランダムサンプリング
        it = numpy.random.choice( num_train )

        x_rnorm = [ x_rnorms[ it ] ]    # shape を [1] にするため [...] で囲む
        y_target = [ y_targets[ it ] ]  # ↑

        session.run( 
            train_step,                     # 学習プロセス（オペレーター）
            feed_dict = { x_rnorms_holder: x_rnorm, y_targets_holder: y_target } 
        )

        a_var_list.append( session.run( a_var ) )
        loss_list.append( 
            session.run( l2_loss_op, feed_dict = { x_rnorms_holder: x_rnorm, y_targets_holder: y_target } )
        )
    ```

以下、最急降下法での最適なパラメータの値の逐次計算過程、及びそのときの誤差関数の値の過程のグラフ
> ![processingformachinelearning_tensorflow_3-1](https://user-images.githubusercontent.com/25688193/31498382-bdca09f2-af9c-11e7-8688-d7cc707f2d8c.png)
>> エポック数（学習回数）が増えるにつれ、パラメータ a → 10 （最適値）に近づいていく様子と、又その過程で誤差関数の値が小さくなっていく（０に近づいていく）様子が見て取れる。


### ② 分類問題での誤差逆伝播法
> コード実装中...

<br>

---

<a id="ID_4"></a>

## 背景理論

<a id="ID_4-1"></a>

## ニューラルネットの概要
![twitter_nn1_1_160825](https://user-images.githubusercontent.com/25688193/30112643-09c7ef7a-934d-11e7-91d2-fcc93505baa0.png)
![twitter_nn1_2_160825](https://user-images.githubusercontent.com/25688193/30112644-09c88430-934d-11e7-9450-6d4861190175.png)

### ニューラルネットワークの主動作
![twitter_nn3 -1_160827](https://user-images.githubusercontent.com/25688193/30112645-09c8e42a-934d-11e7-95f9-87e0ca316b2f.png)

ニューラルネットワーク、より広義には機械学習は、</br>
大きく分けて以下の３つの問題設定＆解決のための手法に分けることが出来る。</br>
① 回帰問題の為の手法。（単回帰分析、重回帰分析、等）</br>
② （クラスの）分類問題の為の手法（SVM、k-NN、ロジスティクス回帰、等）</br>
③ クラスタリング問題の為の手法（k-means法、等）


<a id="ID_4-2"></a>

## 活性化関数
![twitter_nn2-1_160826](https://user-images.githubusercontent.com/25688193/30112640-09b4803e-934d-11e7-993d-4e35263cda81.png)
![twitter_nn2-2_160826](https://user-images.githubusercontent.com/25688193/30112641-09b5d6d2-934d-11e7-861d-06792890d2f9.png)

<a id="ID_4-3"></a>


### 学習方法の分類

<a id="ID_4-3-1"></a>

#### 教師あり学習 [supervised learning] と教師なし学習 [Unsupervised learning]
![image](https://user-images.githubusercontent.com/25688193/30948617-1cb9a46a-a44c-11e7-824b-1f0f23f6780a.png)

<a id="ID_4-3-2"></a>

#### バッチ学習 [batch processing] とオンライン学習 [online learning]
![image](https://user-images.githubusercontent.com/25688193/30580233-c7f83474-9d56-11e7-8a0f-38a54892e3d0.png)

<a id="ID_4-3-3"></a>

#### 強化学習 [reinforcement learning]
![image](https://user-images.githubusercontent.com/25688193/30580261-dd196eea-9d56-11e7-8ae6-6f2df8557307.png)

<a id="ID_4-3-4"></a>

#### 転移学習 [transfer learning]
![image](https://user-images.githubusercontent.com/25688193/30949112-85641f60-a44f-11e7-9430-a0a2fd068e1e.png)


<a id="ID_4-4"></a>

## 単純パーセプトロン
![twitter_nn4 -1_160829](https://user-images.githubusercontent.com/25688193/30112642-09b65e90-934d-11e7-9cac-2472c4add901.png)

<a id="ID_4-4-1"></a>

#### 誤り訂正学習 [error correction learning rule]（パーセプトロンの学習規則 [perceptron learing rule] ）</br>＜教師あり学習、オンライン学習＞
![image](https://user-images.githubusercontent.com/25688193/30771972-171532fc-a08e-11e7-86ab-663fd81fbb75.png)
![image](https://user-images.githubusercontent.com/25688193/30772185-7c0aca0c-a091-11e7-8a22-f258792b99df.png)
![image](https://user-images.githubusercontent.com/25688193/30772194-922be5fa-a091-11e7-8f35-26f52b029e14.png)

<a id="ID_4-4-2"></a>

#### 最急降下法 [gradient descent method] による学習（重みの更新）</br>＜教師あり学習、パッチ学習＞
![image](https://user-images.githubusercontent.com/25688193/30624595-3a3797da-9df9-11e7-95eb-5edb913e080f.png)
![image](https://user-images.githubusercontent.com/25688193/30772096-ec426f7a-a08f-11e7-8fa6-47ce74a29bb9.png)
![image](https://user-images.githubusercontent.com/25688193/30772213-fbeaeaa4-a091-11e7-8838-e8ceccc4b96e.png)
![image](https://user-images.githubusercontent.com/25688193/30772274-78479b3c-a093-11e7-8f6b-6b7ed6c29751.png)

<a id="ID_4-4-3"></a>

#### 確率的勾配降下法 [stochastic gradient descent method]
![image](https://user-images.githubusercontent.com/25688193/30772388-ac53aa3c-a094-11e7-80f2-28703a2931b8.png)
![image](https://user-images.githubusercontent.com/25688193/30772400-d949d8e0-a094-11e7-8d31-87ebc9e8913e.png)

<a id="ID_4-5"></a>

### 多層パーセプトロン [ MLP : Multilayer perceptron]

<a id="ID_4-5-1"></a>

#### 多層パーセプトロンのアーキテクチャ [architecture]
![image](https://user-images.githubusercontent.com/25688193/30770644-c6575a60-a070-11e7-9a4b-c31a0743abf7.png)
![image](https://user-images.githubusercontent.com/25688193/30770558-ed0b3fe8-a06e-11e7-99b9-15278ee6f60e.png)
![image](https://user-images.githubusercontent.com/25688193/30760907-32b178f8-a017-11e7-8605-b087b92c9442.png)
![image](https://user-images.githubusercontent.com/25688193/30770651-e0155c40-a070-11e7-94b4-9fa49980ff91.png)
![image](https://user-images.githubusercontent.com/25688193/30761470-541ad50a-a019-11e7-8ece-b0cf55e14cee.png)
> 【参考 URL】softmax関数について
>> https://mathtrain.jp/softmax<br>
>> http://s0sem0y.hatenablog.com/entry/2016/11/30/012350<br>

![image](https://user-images.githubusercontent.com/25688193/30770538-6591cad2-a06e-11e7-9440-290d3957af7e.png)
![image](https://user-images.githubusercontent.com/25688193/30770761-e01c8a26-a073-11e7-9e49-fc70a23bd63d.png)

![image](https://user-images.githubusercontent.com/25688193/30748067-111c05b4-9fea-11e7-8841-f6e9029ea2b4.png)

<a id="ID_4-5-2"></a>

#### 最急降下法 [gradient descent mesod] による学習（重みの更新）<br>＜教師あり学習、パッチ学習＞
![image](https://user-images.githubusercontent.com/25688193/30624595-3a3797da-9df9-11e7-95eb-5edb913e080f.png)
![image](https://user-images.githubusercontent.com/25688193/30772455-74ac9e16-a096-11e7-99b4-69618fdd8ab8.png)
![image](https://user-images.githubusercontent.com/25688193/30778507-db5903a8-a112-11e7-8a5e-65e356aa2a3c.png)
![image](https://user-images.githubusercontent.com/25688193/30778884-6f95d782-a11b-11e7-8e2d-885da200a2bf.png)
![image](https://user-images.githubusercontent.com/25688193/30778895-b24e28c2-a11b-11e7-8b5a-6a4129206fd1.png)
![image](https://user-images.githubusercontent.com/25688193/30778967-6d01b3ae-a11d-11e7-9ea7-f86b5a6dfeae.png)
![image](https://user-images.githubusercontent.com/25688193/30772701-111084a2-a09c-11e7-939e-d3f5a2198157.png)

<a id="ID_4-5-3"></a>

#### 確率的勾配降下法 [stochastic gradient descent method] <br>＜教師あり学習、オンライン学習＞
![image](https://user-images.githubusercontent.com/25688193/30773009-98407c24-a0a2-11e7-8e94-2bad0b818786.png)
![image](https://user-images.githubusercontent.com/25688193/30773013-a883396e-a0a2-11e7-867e-ad3e9e34188b.png)

<a id="ID_4-5-4"></a>

#### 誤差逆伝播法（バックプロパゲーション）[Backpropagation] <br>＜教師あり学習、バッチ学習 or オンライン学習＞
![image](https://user-images.githubusercontent.com/25688193/30778562-c4fc9074-a113-11e7-9df5-3af84b3e26fb.png)
![image](https://user-images.githubusercontent.com/25688193/30778693-392d659c-a117-11e7-9a2c-8658144bc5f2.png)
![image](https://user-images.githubusercontent.com/25688193/30778686-14bd91be-a117-11e7-8a16-e1651534fc32.png)
![image](https://user-images.githubusercontent.com/25688193/30779065-4543fc84-a120-11e7-82af-8028fa8e05ef.png)
![image](https://user-images.githubusercontent.com/25688193/30779458-65f39640-a12c-11e7-848a-fb9cd82e2248.png)

![image](https://user-images.githubusercontent.com/25688193/30780761-9f2678bc-a14d-11e7-8dfb-7e3d5e8591e9.png)
![image](https://user-images.githubusercontent.com/25688193/30846403-832289f4-a2d2-11e7-9dc7-2842bba5abf9.png)
![image](https://user-images.githubusercontent.com/25688193/30850059-4522b9aa-a2df-11e7-87b2-77b4b689dfd4.png)


<a id="ID_4-6"></a>

## パーセプトロンによる論理演算
![twitter_nn6-1_160829](https://user-images.githubusercontent.com/25688193/30112770-703f5f68-934d-11e7-845d-be2240ef4d17.png)
![twitter_nn6-2_160829](https://user-images.githubusercontent.com/25688193/30112772-7042419c-934d-11e7-9330-d8292a108c1c.png)
![twitter_nn8-1 _160902](https://user-images.githubusercontent.com/25688193/30112777-70842ee0-934d-11e7-9486-d3d14be4d6bd.png)
![twitter_nn10-1_160903](https://user-images.githubusercontent.com/25688193/30112972-1a64417a-934e-11e7-96f1-775f232a2767.png)

