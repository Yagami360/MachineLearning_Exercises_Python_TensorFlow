# TensorFlow ライブラリの機械学習処理フローの練習コード集

TensorFlow における基本的な機械学習処理（特にニューラルネットワークに関わる処理）の練習用コード集。</br>
この README.md ファイルには、各コードの実行結果、概要、機械学習の背景理論の説明を記載しています。</br>
分かりやすいように `main.py` ファイル毎に１つの完結したコードにしています。

## 項目 [Contents]

1. [使用するライブラリ](#ID_1)
1. [使用するデータセット](#ID_2)
1. [コードの実行結果](#ID_3)
    1. [TensorFlow での機械学習処理の全体ワークフロー : `main_templete.py`](#ID_3-0)
    1. [ニューラルネットにおける活性化関数の実装 : `main1.py`](#ID_3-1)
    1. [損失関数（評価関数、誤差関数）の実装 : `main2.py`](#ID_3-2)
        1. [① 回帰問題の為の損失関数（評価関数、誤差関数）](#ID_3-2-1)
        1. [② クラスの分類問題の為の損失関数（評価関数、誤差関数）](#ID_3-2-2)
    1. [誤差逆伝播法の実装 : `main3.py`](#ID_3-3)
        1. [① 回帰問題での誤差逆伝播法](#ID_3-3-1)
        1. [② 分類問題での誤差逆伝播法](#ID_3-3-2)
    1. [バッチ学習（ミニバッチ学習）とオンライン学習（確率的トレーニング）の実装 : `main4.py`](#ID_3-4)
    1. [モデルの評価 : `main5.py`](#ID_3-5)
        1. [ ① 回帰モデルの評価](#ID_3-5-1)
        1. [ ② 分類モデルの評価](#ID_3-5-2)
1. [背景理論](#ID_4)
    1. [ニューラルネットワークの概要](#ID_4-1)
    1. [活性化関数](#ID_4-2)
        1. [sigmoid, tanh, softsign](#ID_4-2-1)
        1. [Relu, Relu6, softplus, ELU](#ID_4-2-2)
            1. [ReLu 関数による勾配消失問題 [vanishing gradient problem] への対応と softmax 関数](#ID_4-2-3-1)
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
    1. [パーセプトロンの収束定理](#ID_4-8)
    1. [【補足】ロジスティクス回帰によるパラメータ推定](#ID_4-9)
    1. [【補足】最尤度法によるロジスティクス回帰モデルのパラメータ推定](#ID_4-10)

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

> その他ライブラリ </br>


<br>
<a id="ID_2"></a>

## 使用するデータセット


<br>
<a id="ID_3"></a>

## コードの実行結果

<a id="ID_3-0"></a>

## TensorFlow での機械学習処理の全体ワークフロー : `main_templete.py`

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
    （例）線形モデル : `y_predict_op = tf.add( tf.mul( x_input, weight_matrix), b_matrix )`
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


<br>
<a id="ID_3-2"></a>

## 損失関数（評価関数、誤差関数）の実装 : `main2.py`

> 参考 URL :
>> ニューラルネットワークにおける損失関数について
>>> http://s0sem0y.hatenablog.com/entry/2017/06/19/084210
>>> http://s0sem0y.hatenablog.com/entry/2017/06/20/135402

損失関数（評価関数、誤差関数）は、モデルの出力と目的値（真の値、教師データ）との差（いいえ変えれば、誤差、距離）を計測する関数であり、モデルの学習に適用されるものである。<br>

ここで、ニューラルネットワーク、より広義には機械学習は、<br>
大きく分けて以下の２つの問題設定＆解決のための手法に分けることが出来た。<br>
① 回帰問題の為の手法。（単回帰分析、重回帰分析、等）<br>
② （クラスの）分類問題の為の手法（SVM、k-NN、ロジスティクス回帰、等）<br>

従って、損失関数も同様にして、回帰問題の為の損失関数と、分類問題の為の損失関数が存在することになる。<br>
ここでは、それぞれの問題（回帰問題分類問題）の損失関数を分けて TensorFlow で実装する。

<a id="ID_3-2-1"></a>

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

<a id="ID_3-2-2"></a>

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
    そして TensorFlow による関数値の算出＆グラフ化のための実装は、以下のように書ける。
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

<!-- - ヒンジ損失関数は、... -->
<!-- - クロス・エントロピー交差関数（２クラスの場合）は、-->
<!-- - シグモイド・クロスエントロピー損失関数は、-->
<!-- - 重み付きクロス・エントロピー損失関数は、 -->

<br>
<a id="ID_3-3"></a>

## 誤差逆伝播法（バックプロパゲーション）の実装 : `main3.py`
誤差逆伝播法（バックプロパゲーション）は、多層パーセプトロンの学習等、ニューラルネットワークによる学習（重みの更新）で広く使われる学習手法である。<br>
ここでは、TensorFlow で誤差逆伝播法を簡単な例（回帰問題、分類問題）で実装する。<br>
尚、ここでの誤差逆伝播法の実装例は、ニューラルネットワークに対して適用したものではないが、一般的には、誤差逆伝播法は多層パーセプトロン等のニューラルネットワークの学習として使われる手法である。<br>

誤差逆伝播法は、誤差関数（コスト関数、損失関数）に対しての、最急降下法で最適なパラメータ（ニューラルネットワークの場合、重みベクトル）を求め、又の誤差項を順方向（ニューラルネットワークの場合、入力層→隠れ層→出力層）とは逆方向に逆伝播で伝播させる手法であるが、<br>
これを TensorFlow で実現するには、計算グラフに誤差を逆伝播させることで、変数の値を更新し、損失関数を最小化する必要がある。<br>
これは、TensorFlow に定義されている最適化アルゴリズム : Optimizer を設定するという方法で実現できる。より詳細には、`tf.XXXOptimizer(...)` と名付けられている最適化アルゴリズムを設定すると、TensorFlow が計算グラフのすべての計算過程でバックプロパゲーションの誤差項を洗い出して計算する。そして、データを供給して誤差関数を最小化すると、それに従って、TensorFlow が計算グラフの（予め定義しておいた） Variable の値を適切に変更する。<br>
尚、誤差逆伝播法では、最急降下法（勾配降下法）を用ちいるが、この最急降下法を TensorFlow で実装する場合は、最適化アルゴリズム (Optimizer) : `tf.tf.train.GradientDescentOptimizer( learning_rate )` を使用すれば良い。

<a id="ID_3-3-1"></a>

### ① 回帰問題での誤差逆伝播法
単純な回帰モデルとして、以下のようなモデルを実装し、最急降下法での最適なパラメータ逐次計算過程、及びそのときの誤差関数の値の過程をグラフ化する。

- ますは、このモデルでのデータを生成する。
    - 平均値 1, 標準偏差 0,1 の正規分布 N(1, 0.1) に従う乱数を 100 個生成する。
        - `x_rnorms = numpy.random.normal( 1.0, 0.1, 100 )`
    - 教師データ（目的値）として、 10 の値の list を設定する。
        - `y_targets = numpy.repeat( 10., 100 )` : 教師データ（目的値）の list
- オペレーターにデータ供給用の各種 placeholder を設定する。
    - `x_rnorms_holder = tf.placeholder( shape = [1], dtype = tf.float32 )`<br>
    オペレーター mul_op での x_rnorm にデータを供給する placeholder
    - `y_targets_holder = tf.placeholder( shape = [1], dtype = tf.float32 )`<br>
    L2 損失関数（オペレーター）loss = y_targets - x_rnorms = 10 - N(1.0, 0.1) での `y_targets` に教師データを供給する placeholder
- そして、この乱数値 `x_rnorms` に対し、Variable（モデルのパラメータで、最適値は 10 になる）を乗算する演算を行う。<br>
最適化アルゴリズムの実行過程で、この Variable の値が TensorFlow によって、適切に変更されていることを確認するのが、このサンプルコードでの目的の１つである。
    - `a_var = tf.Variable( tf.random_normal( shape = [1] ) )` : 乗算する Variable
    - `mul_op = tf.multiply( x_rnorms_holder, a_var )` : 計算グラフに乗算追加 
- Varibale を初期化する。
    - `init_op = tf.global_variables_initializer()`
    - `session.run( init_op )`
- 損失関数として、この教師データと正規分布からの乱数に Variable を乗算した結果の出力の差からなる L2 ノルムの損失関数を設定する。（変数 a が 10 のとき誤差関数の値が 0 になるようなモデルにする。）<br>
最適化アルゴリズムの実行過程で、この誤差関数の値が TensorFlow によって、適切に最小化されていることを確認するのが、このサンプルコードでの目的の１つである。
    - `l2_loss_op = tf.square( mul_op - y_targets_holder )` : 目的値と乗算の出力の差
- 最適化アルゴリズム : Optimizer として、最急降下法（勾配降下法）を設定し、パラメータの最適化を行う。（学習プロセスは、損失関数の最小化）
    - `optGD_op = tf.train.GradientDescentOptimizer( learning_rate = 0.01 )` :<br>
     最適化アルゴリズム : Optimizer として、最急降下法（勾配降下法）を設定（学習率 : 0.01）
    - `train_step = optGD_op.minimize( l2_loss_op )` : トレーニングは、誤差関数の最小化
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

以下、最急降下（学習率 : 0.01）法での最適なパラメータの値の逐次計算過程、及びそのときの誤差関数の値の過程のグラフ
> ![processingformachinelearning_tensorflow_3-1](https://user-images.githubusercontent.com/25688193/31498382-bdca09f2-af9c-11e7-8688-d7cc707f2d8c.png)
>> エポック数（学習回数）が増えるにつれ、パラメータ a → 10 （最適値）に近づいていく様子と、又その過程で誤差関数の値が小さくなっていく（０に近づいていく）様子が見て取れる。


<a id="ID_3-3-2"></a>

### ② 分類問題での誤差逆伝播法
単純な分類モデルとして、以下のようなモデルを実装し、最急降下法での最適なパラメータ逐次計算過程、及びそのときの誤差関数の値の過程をグラフ化する。（※分類問題であるが分類結果は調べない。）

- ますは、このモデルでのデータを生成する。
    - 2 つの正規分布 N(-1, 1) , N(3, 1) の正規分布に従う乱数をそれぞれ 50 個、合計 100 個生成する。
        - `x_rnorms = numpy.concatenate( ( numpy.random.normal(-1,1,50), numpy.random.normal(3,1,50) ) )`
            - `numpy.concatenate(...)` : ２個以上の配列を軸指定して結合する。軸指定オプションの axis はデフォルトが 0
    - 生成した 2 つの正規分布 N(-1, 1) , N(3, 1) のそれぞれのクラスラベルを {0,1} とする。
        - `y_targets = numpy.concatenate( ( numpy.repeat(0,50), numpy.repeat(1,50) ) )` 
- オペレーターにデータを供給するための各種 placeholder を設定する。
    - `x_rnorms_holder = tf.placeholder( shape = [1], dtype = tf.float32 )`<br>
    `x_rnorms` にデータを供給する placeholder
    - `y_targets_holder = tf.placeholder( shape = [1], dtype = tf.float32 )`<br>
    `y_targets` にデータを供給する placeholder
- このモデルのパラメータとなる Variable として、シグモイド関数の平行移動分 sigmoid( x + a_var ) となる変数 `a_var` を設定する。<br>
この平行移動したシグモイド関数が、クラスラベル {0,1} を識別するモデルとなる。つまり、ロジスティクス回帰による２クラス識別モデル（※ロジスティクス回帰は、回帰という名前であるが、実際には分類問題のための手法である。）<br>
そして、最適化アルゴリズムの実行過程で、この Variable の値が TensorFlow によって、適切に変更されていることを確認するのが、このサンプルコードでの目的の１つである。
    - `a_var = tf.Variable( tf.random_normal(mean=10, shape=[1]) )`<br>
    値を 10 で初期化しているが、これは、最適値である -1 に学習の過程で収束する様子を明示することを意図しての値である。
- シグモイド関数の平行移動演算（オペレーター）を計算グラフに追加する。<br>
この際、加算演算 `tf.add(...)` に加えてシグモイド関数 `tf.sigmoid(...)` をラップする必要があるように思えるが、後に定義する損失関数が、シグモイド・クロス・エントロピー関数 `tf.nn.sigmoid_cross_entropy_with_logit(...)` であるため、このシグモイド関数の演算が自動的に行われるために、ここでは必要ないことに注意。
    - `add_op = tf.add( x_rnorms_holder, a_var )`
- 後に定義する損失関数が、シグモイド・クロス・エントロピー関数 `tf.nn.sigmoid_cross_entropy_with_logit(...)` が、追加の次元（バッチ数が関連付けられているバッチデータ）を必要とするため、`tf.expand_dims(...)` を用いて、オペレーター、placeholder に次元を追加する。
    - `add_expaned_op = tf.expand_dims(add_op, 0)` : shape=(1,) → shape=(1,1)
    - `y_expaned_targets_holder = tf.expand_dims(y_targets_holder, 0)` : shape=(1,) → shape=(1,1)
- Varibale を初期化する。
    - `init_op = tf.global_variables_initializer()`
    - `session.run( init_op )`
- 損失関数として、正規分布乱数 `x_rnorms` をシグモイド関数で変換し、クロス・エントロピーをとる、シグモイド・クロス・エントロピー関数 : `tf.nn.sigmoid_cross_entropy_with_logit(...)` を使用する。<br>
この関数は、引数 `logits`, `labels` が特定の次元であることを要求するため、先の次元を拡張した `add_expaned_op`, `y_expaned_targets_holder` で引数を指定する。<br>
最適化アルゴリズムの実行過程で、この誤差関数の値が TensorFlow によって、適切に最小化されていることを確認するのが、このサンプルコードでの目的の１つである。
    - `loss_op = tf.nn.sigmoid_cross_entropy_with_logits( logits = add_expaned_op, labels = y_expaned_targets_holder)`
        - `logits = add_expaned_op` : 最終的な推計値。sigmoid 変換する必要ない
        - `labels = y_expaned_targets_holder` : 教師データ
- 最適化アルゴリズム : Optimizer として、最急降下法（勾配降下法）を設定し、パラメータの最適化を行う。（学習プロセスは、損失関数の最小化）
    - `optGD_op = tf.train.GradientDescentOptimizer( learning_rate = 0.05 )` :<br>
     最適化アルゴリズム : Optimizer として、最急降下法（勾配降下法）を設定（学習率 : 0.05）
    - `train_step = optGD_op.minimize( loss_op )` : トレーニングは、誤差関数の最小化
- 各エポックに対し、オンライン学習を行い、パラメータを最適化していく。
    ```python
    a_var_list = []
    loss_list = []
    # for ループでオンライン学習
    for i in range( 1500 ):
        # RNorm のイテレータ : ランダムサンプリング
        it = numpy.random.choice( num_train )

        x_rnorm = [ x_rnorms[ it ] ]    # shape を [1] にするため [...] で囲む
        y_target = [ y_targets[ it ] ]  # ↑

        session.run( 
            train_step,                     # 学習プロセス（オペレーター）
            feed_dict = { x_rnorms_holder: x_rnorm, y_targets_holder: y_target } 
        )

        a_var_list.append( session.run( a_var ) )
        
        loss = session.run( loss_op, feed_dict = { x_rnorms_holder: x_rnorm, y_targets_holder: y_target } )
        loss_reshaped = loss[0, 0]  # shape = (1,1) [[ xxx ]] → xxx に変換
        loss_list.append( loss_reshaped )
    ```

<br>

最急降下法（学習率 : 0.05）での最適なパラメータの値の逐次計算過程、及びそのときの誤差関数（シグモイド・クロス・エントロピー関数）の値の過程のグラフ
> ![processingformachinelearning_tensorflow_3-2](https://user-images.githubusercontent.com/25688193/31508764-111ec31e-afba-11e7-9d41-07a5b05a9726.png)
>> エポック数（学習回数）が増えるにつれ、パラメータ a → -1 （最適値）に近づいていく様子と、又その過程で誤差関数の値が小さくなっていく（０に近づいていく）様子が見て取れる。

<br>


<a id="ID_3-4"></a>

## バッチ学習（ミニバッチ学習）とオンライン学習（確率的トレーニング）の実装 : `main4.py`
TensorFlow は誤差逆伝播法（バックプロパゲーション）に従い、設定したモデルの変数の更新を行なっていくが、この誤差逆伝播法で行われる、学習アルゴリズム（最急降下法（勾配降下法）等）は、
- バッチ学習 : 全てのトレーニングデータセットを一度にまとめて学習する手法。
- オンライン学習 : １つのトレーニングデータ度に学習を行う）による手法。
    - 特に、最急降下法（勾配降下法）の場合は、確率的勾配降下法という。
- ミニバッチ学習 : バッチ学習とオンライン学習の折込案で、ある程度個のトレーニングデータセットを一度にまとめて学習する手法。

が存在する。<br>
先の `main3.py` で示したコードは、このオンライン学習である確率的最急降下法での実装例である。<br>
このトレーニングの種類での違いによるメリット・デメリットをまとめると、以下の表のようになる。

|トレーニングの種類|メリット|デメリット|
|---|---|---|
|オンライン学習<br>確率的トレーニング|最適化過程で、局所的最適解（ローカルミニマム）を脱出できる可能性がある。<br>又、トレーニングデータの変化に素早く対応できる。<br>但し、１つトレーニングサンプル毎に学習を行うので、エポック度にトレーニングデータをシャッフルすることが重要となる。|一般的に、収束に必要なイテレーション回数（トレーニング回数）が多くなる。|
|バッチ学習<br>ミニバッチ学習|最適解を素早く特定することが出来る。|計算リソースを多く消費する。<br>又、オンライン学習に比べて、最適化過程で、局所的最適解（ローカルミニマム）を脱出できる可能性が低い。|

ここでは、比較的簡単なサンプルコードで、バッチ学習（正確には、ミニバッチ学習）とオンライン学習（確率的トレーニング）の学習過程の違いを確認する。

- 以下、ミニバッチ学習での処理を記載。
- ますは、このモデルでのデータを生成する。
    - 正規分布 N(1, 0.1) に従う乱数の list `x_rnorms` を 100 個生成する。
        - `x_rnorms = numpy.random.normal(1, 0.1, 100)`
    - 教師データ（目的値）の list `y_targets` として、 10 の値の list を設定する。
        - `y_targets = numpy.repeat(10.0, 100)` 
- 次に、ミニバッチ学習でのバッチサイズを指定する。<br>
  これは、計算グラフに対し、一度に供給するトレーニングデータの数となる.
    - `batch_size = 20`
- オペレーターにデータを供給するための各種 placeholder を設定する。<br>
  ここでの placeholder の形状 shape は、先のコードとは異なり、1 次元目を `None`, 2 次元目をバッチサイズとする。( `shape = [None, 1]` )<br>
  1 次元目を明示的に `20` としても良いが、`None` とすることでより汎用的になる。
    - `x_rnorms_holder = tf.placeholder( shape = [None, 1], dtype = tf.float32 )`<br>
    `x_rnorms` にデータを供給する placeholder<br>
    後のバッチ学習では, shape = [None, 1] → shape = [20, 1] と解釈される。<br>
    オンライン学習では、shape = [None, 1] → shape = [1, 1] と解釈される。
    - `y_targets_holder = tf.placeholder( shape = [None, 1], dtype = tf.float32 )`<br>
    `y_targets` にデータを供給する placeholder<br>
    後のバッチ学習では, shape = [None, 1] → shape = [20, 1] と解釈される。<br>
    オンライン学習では、shape = [None, 1] → shape = [1, 1] と解釈される。
- このモデルのパラメータとなる Variable として、そして、この正規分布乱数値の list である `x_rnorms` に対し、行列での乗算演算を行うための変数 `A_var` を設定する。<br>
最適化アルゴリズムの実行過程で、この Variable の値が TensorFlow によって、適切に変更されていることを確認するのが、このサンプルコードでの目的の１つである。
    - `A_var = tf.tf.Variable( tf.random_normal(shape=[1,1]) )` <br>
    placeholder の shape の変更に合わせて、shape = [1,1] としている。
- モデルの構造（計算グラフ）を定義する。
    - 先に記述したように、正規分布乱数値の list である `x_rnorms` と変数 `A_var` の行列での乗算演算を設定する。
        - `matmul_op = tf.matmul( x_rnorms_holder, A_var )` : 行列での乗算演算
- 損失関数を設定する。
    - 損失関数として、この教師データと正規分布からの乱数に Variable を乗算した結果の出力の差からなる L2 ノルムの損失関数を定義するが、<br>
    バッチのデータ点ごとに、すべての L2 ノルムの損失の平均を求める必要があるので、L2 ノルムの損失関数を `reduce_mean(...)` という関数（平均を算出する関数）でラップする。
        - `loss_op = tf.reduce_mean( tf.square( matmul_op - y_targets_holder ) )`
- 最適化アルゴリズム : Optimizer として、最急降下法（勾配降下法）を設定し、パラメータの最適化を行う。（学習プロセスは、損失関数の最小化）
    - `GD_opt = tf.train.GradientDescentOptimizer( learning_rate = 0.02 )` : 学習率 = 0.02
    - `train_step = GD_opt.minimize( loss_op )` : トレーニングは、誤差関数の最小化
- Varibale を初期化する。
    - `init_op = tf.global_variables_initializer()`
    - `session.run( init_op )`
- 各エポックに対し、ミニバッチ学習を行い、パラメータを最適化していく。
    ```python
    A_var_list_batch = []
    loss_list_batch = []

    # for ループで各エポックに対し、ミニバッチ学習を行い、パラメータを最適化していく。
    for i in range( 100 ):
        # RNorm のイテレータ : ランダムサンプリング
        it = numpy.random.choice( 100, size = batch_size )  # ミニバッチ処理

        x_rnorm = numpy.transpose( [ x_rnorms[ it ] ] )    # shape を [1] にするため [...] で囲む
        y_target = numpy.transpose( [ y_targets[ it ] ] )  # ↑

        session.run( 
            train_step,                     # 学習プロセス（オペレーター）
            feed_dict = { x_rnorms_holder: x_rnorm, y_targets_holder: y_target } 
        )

        A_batch = session.run( A_var )
        loss_batch = session.run( loss_op, feed_dict = { x_rnorms_holder: x_rnorm, y_targets_holder: y_target } )

        A_var_list_batch.append( A_batch )
        loss_list_batch.append( loss_batch )
    ```
<br>

- オンライン学習（確率的トレーニング）に関しても、同様の処理を実施していく。（詳細略、コード `main4.py` 参照）
- そして、各エポックに対し、オンライン学習（確率的トレーニング）を行い、パラメータを最適化していく。
    ```python
    A_var_list_online = []
    loss_list_online = []

    # for ループで各エポックに対し、オンライン学習（確率的トレーニング）を行い、パラメータを最適化していく。
    for i in range( 100 ):
        # RNorm のイテレータ : ランダムサンプリング
        it = numpy.random.choice( 100 )  # online 処理

        x_rnorm = [ x_rnorms[ it ] ]    # shape を [1] にするため [...] で囲む
        y_target = [ y_targets[ it ] ]  # ↑

        session.run( 
            train_step,                     # 学習プロセス（オペレーター）
            feed_dict = { x_rnorms_holder: x_rnorm, y_targets_holder: y_target } 
        )

        A_online = session.run( A_var )
        loss_online = session.run( loss_op, feed_dict = { x_rnorms_holder: x_rnorm, y_targets_holder: y_target } )

        A_var_list_online.append( A_online )
        loss_list_online.append( loss_online )
    ```

<br>

ミニバッチ学習、オンライン学習（確率的トレーニング）での勾配降下法（学習率 : 0.02）での最適なパラメータの値の逐次計算過程、及びそのときの誤差関数の値の過程のグラフ。
> ![processingformachinelearning_tensorflow_4-1](https://user-images.githubusercontent.com/25688193/31527653-b7c25dd0-b009-11e7-907e-c4ba599cffb6.png)
>> バッチ学習による学習過程のほうが滑らかで、オンライン学習（確率的トレーニング）による学習過程のほうが不規則であることが見て取れる。この不規則な動きが、局所的最適解（ローカルミニマム）を脱出する可能性を高めるが、逆に収束速度を遅くする。<br>
>> 又、エポック数（学習回数）が増えるにつれ、パラメータ A_var → 10 （最適値）に近づいていく様子と、又その過程で誤差関数の値が小さくなっていく（０に近づいていく）様子が見て取れる。

<br>

<a id="ID_3-5"></a>

### TensorFlow でのモデルの評価 : `main5.py`
ここでは、比較的簡単な回帰問題、分類問題、それぞれに対して、TensorFlow を用いてのモデルの評価手法のサンプルコードを実装する。

<a id="ID_3-5-1"></a>

#### ① 回帰モデルの評価
回帰問題は、与えられたデータから数値を予想する問題であり、分類問題のように、その評価指数として正解率というものは存在しない。<br>
回帰問題における、モデルの評価指数としては、予想値と目的値の間の距離、又はそれに準じるものに関しての集約したものが指標となり得る（平均２乗誤差 : MES 等）。<br>
多くの場合、誤差関数は、この指標に準じるものになり得るので、誤差関数の値を調べることが、そのまま回帰モデルの性能評価に繋がる。

ここでは、先の `main4.py` で実装した回帰モデルと同じモデルで、モデルの評価指数として、L2 損失関数から求められる平均２乗誤差（MES）を求め、モデルの評価を行う。

- 以下、先の `main4.py` とは異なる箇所（特に、モデルの評価に関わる部分）を中心的に説明する。
- 正規分布 N(1, 0.1) から生成した 100 個乱数の list `x_rnorms` を、トレーニングデータとテストデータに分割する。<br>
データの分割割合は、トレーニングデータ : 80%, テストデータ : 20 % とする。
    ```python
    # トレーニングデータのインデックス範囲
    train_indices = numpy.random.choice(                # ランダムサンプリング
                        len( x_rnorms ),                # 100
                        round( len(x_rnorms)*0.8 ),     # 80% : 100*0.8 = 80
                        replace = False                 # True:重複あり、False:重複なし
                    )

    # テストデータのインデックス範囲
    test_indices = numpy.array( 
                       list( set( range(len(x_rnorms)) ) - set( train_indices ) )   #  set 型（集合型） {...} の list を ndarry 化
                   )

    # トレーニングデータ、テストデータに分割
    x_train = x_rnorms[ train_indices ]
    x_test = x_rnorms[ test_indices ]
    y_train = y_targets[ train_indices ]
    y_test = y_targets[ test_indices ]
    ```
    - 尚、このデータの分割は scikit-learn の `sklearn.model_selection.train_test_split(...)` を用いて、より簡単に行うことも可能である。但し、この場合は、`numpy.reshape(..)` 等で TensorFlow 用にデータの次元 : shape との整合性をとる必要がある。
- データをトレーニングデータとテストデータに分割したので、学習は、トレーニングデータで行うようにする。
    - 具体的には、先の `main4.py` での該当箇所が、`100` → `len(x_train)`, `x_rnorms` → `x_train`, `y_targets` → `y_train` に置き換わる。 
    ``` python
    A_var_list_batch = []
    loss_list_batch = []

    # for ループで各エポックに対し、ミニバッチ学習を行い、パラメータを最適化していく。
    for i in range( 100 ):
        # RNorm のイテレータ : ランダムサンプリング
        it = numpy.random.choice( len(x_train), size = batch_size )  # ミニバッチ処理

        x_rnorm = numpy.transpose( [ x_train[ it ] ] )    # shape を [1] にするため [...] で囲む
        y_target = numpy.transpose( [ y_train[ it ] ] )  # ↑

        session.run( 
            train_step,                     # 学習プロセス（オペレーター）
            feed_dict = { x_rnorms_holder: x_rnorm, y_targets_holder: y_target } 
        )

        A_batch = session.run( A_var )
        loss_batch = session.run( loss_op, feed_dict = { x_rnorms_holder: x_rnorm, y_targets_holder: y_target } )

        A_var_list_batch.append( A_batch )
        loss_list_batch.append( loss_batch )
    ```
- この回帰モデルの評価として、トレーニングデータとテストデータの L2 損失関数から、平均２乗誤差（MSE）を算出する。
    - 尚、この MSE 値は、トレーニングデータでのトレーニング終了後の、ランダムサンプリングされた 1 つのデータに対する値となる。
    ```python
    # MSE の算出
    loss_train = session.run( 
                    loss_op, 
                    feed_dict = { 
                        x_rnorms_holder: numpy.transpose( [x_train] ), 
                        y_targets_holder: numpy.transpose( [y_train] )
                    } 
                 )

    loss_test = session.run( 
                    loss_op, 
                    feed_dict = { 
                        x_rnorms_holder: numpy.transpose( [x_test] ), 
                        y_targets_holder: numpy.transpose( [y_test] )
                    } 
                 )

    mse_train = numpy.round( loss_train, 2 )    # ２乗
    mse_test = numpy.round( loss_test, 2 )

    print( "MSE (train data)", mse_train )
    print( "MSE (test data)", mse_test )
    ```
    ```python
    [出力]
    MSE (train data) :  0.93
    MSE (test data) :  1.29
    ```

<br>

<a id="ID_3-5-2"></a>

#### ② 分類モデルの評価
次に、先の `main3.py` で実装した分類モデルと同じモデル（２つの正規分布 N(-1, 1)、N(2, 1) の分類問題）で、分類モデルの評価の TensorFlow での実装を行う。<br>
分類モデルの評価指数は様々あるが、ここでは最も代表的な分類の正解率の TensorFlow での実装を行う。

- 以下、先の `main3.py` の分類モデルの実装箇所とは異なる箇所（特に、モデルの評価に関わる部分）を中心的に説明する。
- 正規分布 N(-1, 1) から生成した 50 個の乱数と、別の正規分布 N(2, 1) から生成した 50 個の乱数、合計 100 個の list `x_rnorms` を、トレーニングデータとテストデータに分割する。<br>
データの分割割合は、トレーニングデータ : 80%, テストデータ : 20 % とする。
- データをトレーニングデータとテストデータに分割したので、学習は、トレーニングデータで行うようにする。
    - 具体的には、先の `main3.py` での該当箇所が、`100` → `len(x_train)`, `x_rnorms` → `x_train`, `y_targets` → `y_train` に置き換わる。 
- このコードでの目的である分類モデルの評価指数として、TensorFlow を用いて正解率を算出する。
    - そのためにはまず、モデルと同じ演算を実施し、予測値を算出する。
        ``` python
        y_pred_op = tf.squeeze( 
                        tf.round(
                            tf.nn.sigmoid( tf.add(x_rnorms_holder, a_var) )
                        ) 
                    )
        ```
    - 続いて、`tf.equal(...)` を用いて、予測値のオペレーター `y_pred_op` と目的値を与える placeholder `y_targets_holder` が等価であるか否かを確認する。
        - `correct_pred_op = tf.equal( y_pred_op, y_targets_holder )`
    - これにより、bool 型の Tensor `Tensor("Equal:0", dtype=bool)` が `correct_pred_op` に代入されるので、この値を `float32` 型に `tf.cast(...)` で明示的にキャストした後、平均値を `tf.reduce_mean(...)` で求める。これが、正解率となる。（但し、この段階では Session を run していないので、値は代入されていない）
        - `accuracy_op = tf.reduce_mean( tf.cast(correct_pred_op, tf.float32) )`
    - トレーニングデータと、テストデータを feed_dict として、Session を run して、それぞれの正解率を求める。
        ```python
        accuracy_train = session.run( 
            accuracy_op, 
            feed_dict = {
                x_rnorms_holder: [x_train],     # shape を合わせるため,[...] で囲む
                y_targets_holder: [y_train]
            }
        )

        accuracy_test = session.run( 
            accuracy_op, 
            feed_dict = {
                x_rnorms_holder: [x_test], 
                y_targets_holder: [y_test]
            }
        )

        print( "Accuracy (train data) : ", accuracy_train )
        print( "Accuracy (test data) : ", accuracy_test )
        ```
        ```python
        [出力]
        Accuracy (train data) :  0.9375
        Accuracy (test data) :  0.95
        ```



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

<a id="ID_4-2-1"></a>

#### sigmoid, tanh, softsign
活性化関数の内、sigmoid, tanh, softsign 関数の図
> ![processingformachinelearning_tensorflow_1-2](https://user-images.githubusercontent.com/25688193/30211949-e16ce07a-94dd-11e7-9562-6d121aeeb59e.png)

<a id="ID_4-2-2"></a>

#### Relu, Relu6, softplus, ELU
活性化関数の内、Relu, Relu6, softplus, ELU 関数の図
> ![processingformachinelearning_tensorflow_1-1](https://user-images.githubusercontent.com/25688193/30203903-ac94e5ec-94be-11e7-867f-fc78b059ef44.png)

ReLu関数（ランプ関数）は、x=0 にて非連続で微分不可能な関数であるが、その他の領域では微分可能なので、ニューラルネットワークにおいては、微分可能な活性化関数として取り扱われることが多い。<br>
そして、この ReLu は、勾配が一定なので、ディープネットワークにおける学習アルゴリズムにおいて発生する、勾配損失問題 [vanishing gradient problem] に対応することが出来るのが最大の利点である。（後述）

<a id="ID_4-2-2-1"></a>

##### ReLu 関数による勾配消失問題 [vanishing gradient problem] への対応と softmax 関数
勾配消失問題 [vanishing gradient problem] とは、ニューラルネットワークの層が深くなるにつれて、誤差逆伝播法等の学習の際に損失関数の勾配（傾き）が 0 に近くなり、入力層に近い層で入出力誤差が消失してしまい、結果として、うまく学習（重みの更新）ができなくなるような問題である。<br>

この問題に対応するために開発されたのが、ReLU [rectified linear unit] や MaxOut という活性化関数である。<br>
これらの活性化関数では、勾配（傾き）が一定なので、誤差消失問題を起こさない。従って、深い層のネットワークでも学習が可能となり、現在多くのニューラルネットワークで採用されている。<br>

但し、これらの活性化関数を通して出力される値は、先に示したグラフのように負の値が出てきたりと、そのままでは扱いづらい欠点が存在する。

従って、softmax 関数を通じて出力を確率に変換するようにする。
この softmax 関数の式は以下のように与えられる。

```math
y_i=\dfrac{e^{x_i}}{e^{x_1}+e^{x_2}+\cdots +e^{x_n}}
```

![image](https://user-images.githubusercontent.com/25688193/30590115-37a895ae-9d78-11e7-9012-50cc868b6321.png)

> 参考サイト : [画像処理とか機械学習とか / Softmaxって何をしてるの？](http://hiro2o2.hatenablog.jp/entry/2016/07/21/013805)

##### 【Memo】softmax 関数と統計力学での分配関数の繋がり
ニューラルネットワークの softmax 関数の形は、<br>
統計力学で言う所のカノニカルアンサンブルでの sub system の微視的状態を与える確率の式<br>

![image](https://user-images.githubusercontent.com/25688193/31034610-bfe29f12-a59f-11e7-8d90-6541e8fa216c.png)

$$ P_n = \dfrac{ e^{\frac{E_n}{k_B T}} }{ \sum_{i=1}^{n} e^{ \frac{E_i}{k_B \times T } } } $$

の形に対応している。<br>

この確率の式の分母を統計力学では分配関数<br>

![image](https://user-images.githubusercontent.com/25688193/31034696-21d2f636-a5a0-11e7-9f6d-81de5b7f9f39.png)

$$ Z = \sum_{i=1}^{n} e^{ \frac{-E_i}{k_B \times T} } $$

といい重要な意味を持つが、これは、エントロピー最大化に繋がる話しであり、<br>

Helmholtz の自由エネルギーは、この分配関数 Z を用いて、<br>

![image](https://user-images.githubusercontent.com/25688193/31034742-51e4a0ae-a5a0-11e7-8d87-704124ad5467.png)

$$ F = - k_B \times T \times log_e{Z} $$

で表現できるが、これを使えば、カノニカルアンサンブルのエントロピー S が<br>

![image](https://user-images.githubusercontent.com/25688193/31034868-dba484e4-a5a0-11e7-85fe-ba7d5e011a04.png)

$$ S = - k_B \times \sum_{i=1}^{n} P_i \times \log_e{P_i} $$<br>

と書ける。これはまさに、情報理論でいうとこのシャノンの情報量に対応している。

<br>

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

<a id="ID_4-7"></a>

### パーセプトロンの収束定理
パーセプトロンの学習は、** 線形分離可能な問題であれば、有限回の学習の繰り返しにより収束する ** ことが証明されている。<br>
このことをパーセプトロンの収束定理と呼ぶ。

---

<a id="ID_4-8"></a>

### 【補足】ロジスティクス回帰によるパラメータ推定

![twitter_ 18-1_161130](https://user-images.githubusercontent.com/25688193/29994398-b3cb8b5e-9009-11e7-9ca3-947c8ede9407.png)
![twitter_ 18-2_161130](https://user-images.githubusercontent.com/25688193/29994397-b3ca7f84-9009-11e7-8e86-9677931b681e.png)
![twitter_ 18-3_161130](https://user-images.githubusercontent.com/25688193/29994396-b3c9dcd2-9009-11e7-8db0-c342aac2725c.png)
![twitter_ 18-4_161130](https://user-images.githubusercontent.com/25688193/29994399-b3cb73f8-9009-11e7-8f86-52d112491644.png)
![twitter_ 18-5_161201](https://user-images.githubusercontent.com/25688193/29994401-b3ceb5d6-9009-11e7-97b6-9470f10d0235.png)

<a id="ID_4-9"></a>

### 【補足】最尤度法によるロジスティクス回帰モデルのパラメータ推定 [MLE]
![twitter_ 18-6_161201](https://user-images.githubusercontent.com/25688193/29994400-b3cdbcf8-9009-11e7-9dba-fdaf84d592f8.png)
![twitter_ 18-6 _170204](https://user-images.githubusercontent.com/25688193/29994403-b3ed4870-9009-11e7-8432-0468dfc2b841.png)
![twitter_ 18-7_161201](https://user-images.githubusercontent.com/25688193/29994405-b3ee6e94-9009-11e7-840d-50d2a5c10aba.png)
![twitter_ 18-7 _170204](https://user-images.githubusercontent.com/25688193/29994406-b3efd13a-9009-11e7-817d-6f0d5373f178.png)