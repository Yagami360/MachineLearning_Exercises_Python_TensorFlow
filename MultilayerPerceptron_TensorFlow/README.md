# TensorFlow での多層パーセプトロン [MLP : Multilayer perceptron] の実装

TensorFlow での多層パーセプトロンの練習用実装コード集。<br>

TensorFlow での多層パーセプトロンの処理をクラス（任意の層にディープラーニング化可能な柔軟なクラス）でラッピングし、scikit-learn ライブラリの classifier, estimator とインターフェイスを共通化することで、scikit-learn ライブラリとの互換性のある自作クラス `MultilayerPerceptron` を使用。<br>
これにより、scikit-learn ライブラリを使用した自作クラス `MLPlot` 等が再利用可能になる。

この README.md ファイルには、各コードの実行結果、概要、ニューラルネットワーク（パーセプトロン）の背景理論の説明を記載しています。<br>
分かりやすいように `main.py` ファイル毎に１つの完結した実行コードにしています。

## 項目 [Contents]

1. [使用するライブラリ](#ID_1)
1. [使用するデータセット](#ID_2)
1. [コードの説明＆実行結果](#ID_3)
    1. [ニューラルネットワークのフレームワークのコードの説明](#ID_3-1)
    1. [多層パーセプトロンによる２クラスの識別 : `main1.py`](#ID_3-2)
    1. [多層パーセプトロンによる多クラスの識別 : `main2.py`](#ID_3-3)
    1. [多層パーセプトロンによる MIST データの識別 : `main3.py`](#ID_3-3)
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

- 半月状データ : ２クラスの識別 
- 円状データ : ２クラスの識別
- Iris データ : ３クラスの識別
- MIST : 多クラスの識別＆パターン認識

<br>
<a id="ID_3"></a>

## コードの説明＆実行結果

<a id="ID_3-1"></a>

## ニューラルネットワークのフレームワークのコードの説明

<!--
- `NeuralNetworkBase`
    - TensorFlow でのニューラルネットワークの処理のラッピングした基底クラス
    - このクラスを継承するクラスの共通メソッド（インターフェイス）として、以下のメソッドを抽象メソッドで定義している。このクラスを継承したクラスは、これらのメソッドの具体的な実装（オーバーライド）を行わなければならない。
        - `models()` : モデルの定義を行い、最終的なモデルの出力のオペレーターを設定する。
        - `loss()` : 損失関数（誤差関数、コスト関数）の定義を行う。
        - `optimizer()` : モデルの最適化アルゴリズムの設定を行う。
-->
### `MultilayerPerceptron` クラス : `MultilayerPerceptron.py`
- TensorFlow での多層パーセプトロンの処理をクラス（任意の層数に DNN 化可能な柔軟なクラス）でラッピングし、scikit-learn ライブラリの classifier, estimator とインターフェイスを共通化することで、scikit-learn ライブラリとの互換性のある自作クラス。<br>
これにより、scikit-learn ライブラリを使用した自作クラス `MLPlot` 等が再利用可能になる。
- 具体的には、scikit-learn ライブラリの classifier, estimator と同じく、fitting 処理するメソッドとして、`fit( X_train, y_train )` 及び、fitting 処理後の推定を行う `predict( X_test )` メソッド（インターフェイス）等を実装している。
    - `fit( X_train, y_train )` : 指定されたトレーニングデータで、モデルの fitting 処理を行う。
        - `X_train` : numpy.ndarray ( shape = [n_samples, n_features] )<br>トレーニングデータ（特徴行列）
        - `y_train` : numpy.ndarray ( shape = [n_samples] ) <br> トレーニングデータ用のクラスラベル（教師データ）のリスト
    - `predict( X_test )` : fitting 処理したモデルで、推定を行い、予想クラスラベル値を返す。
        - `X_test` : numpy.ndarry ( shape = [n_samples, n_features] )<br>予想したい特徴行列
        - `predict_proba( X_test )` : fitting 処理したモデルで、推定を行い、クラスの所属確率の予想値を返す。
- ニューラルネットワークモデルに共通する処理を行うメソッド
    - `model()` : モデルの定義（計算グラフの構築）を行い、最終的なモデルの出力のオペレーターを設定する。
    - `loss()` : 損失関数の定義を行う。
    - `optimizer() :` モデルの最適化アルゴリズムの設定を行う。
- 多層パーセプトロン固有の処理を行うメソッド
    - `init_weight_variable( input_shape )` : 重みの初期化を行う。重みは TensorFlow の Variable で定義することで、学習過程（最適化アルゴリズム Optimizer の session.run(...)）で自動的に TensorFlow により、変更される値となる。
        - `input_shape` : [int,int] <br>重みの Variable を初期化するための Tensor の形状
    - `init_bias_variable( input_shape )` : バイアス項 b の初期化を行う。バイアス項は TensorFlow の Variable で定義することで、学習過程（最適化アルゴリズム Optimizer の session.run(...)）で自動的に TensorFlow により、変更される値となる。
        - `input_shape` : [int,int] <br>バイアス項の Variable を初期化するための Tensor の形状
- その他、評価のためのメソッド
    - `accuracy( X_test, y_test )` : 引数で指定されたデータから正解率を算出する。
<!--
    - ニューラルネットワークの自作基底クラス `NeuralNetworkBase` を継承し、各ニューラルネットワーク用の抽象メソッドに対し、この多層パーセプトロンに固有な具体的な実装（オーバーライド）を行なっている<br>
-->

#### 使用例
```python
from MLPlot import MLPlot
from MLPreProcess import MLPreProcess
from MultilayerPerceptron import MultilayerPerceptron

def main():
    ...

    # データセットを読み込み or 生成
    X_features, y_labels = MLPreProcess.generateMoonsDataSet( input_n_samples = 300, input_noize = 0.3 )

    # データセットをトレーニングデータ、テストデータ、検証データセットに分割
    X_train, X_test, y_train, y_test \
    = MLPreProcess.dataTrainTestSplit( X_input = X_features, y_input = y_labels, ratio_test = 0.2, input_random_state = 1 )

    # 多層パーセプトロンクラスのオブジェクト生成
    # 入力層 : 1 ユニット、
    # 隠れ層 1 : 3 ユニット、
    # 隠れ層 2 : 3 ユニット、
    # 出力層：1 ユニット
    # 学習率 : 0.05
    # エポック数 : 500
    # ミニバッチサイズ : 20
    mlp = MultilayerPerceptron(
               session = tf.Session(),
               n_inputLayer = len(X_features[0]), 
               n_hiddenLayers = [3,3],
               n_outputLayer = 1,
               learning_rate = 0.05,
               epochs = 500,
               batch_size = 20
           )

    # モデルの構造を定義する。
    mlp.models()

    # 損失関数を設定する。
    mlp.loss()

    # モデルの初期化と学習（トレーニング）
    mlp1.fit( X_train, y_train )

    # 識別境界を plot
    MLPlot.drawDiscriminantRegions( X_features, y_labels, classifier = mlp )
    ...

```

<br>
<a id="ID_3-1"></a>

## 多層パーセプトロンによる２クラス識別 : `main1.py`

<a id="ID_3-1-1"></a>

### 半月状データ `sklearn.datasets.make_moons(...)` での識別

- scikit-learn ライブラリの `sklearn.datasets.make_moons(...)` で生成した半月状データにて、２クラスの識別を行なった。<br>データ数は 300 個で、ノイズ値を 0.3 に設定<br>
    - `X_features, y_labels = MLPreProcess.generateMoonsDataSet( input_n_samples = 300, input_noize = 0.3 )`
- データセットをトレーニングデータ、テストデータに 8:2 の割合で分割。
    - `X_train, X_test, y_train, y_test = MLPreProcess.dataTrainTestSplit( X_input = X_features, y_input = y_labels, ratio_test = 0.2, input_random_state = 1 )`
- `MultilayerPerceptron.models()` メソッドにて、多層パーセプトロンのフィードフォワード処理をモデル化。
    - 入力層、及び隠れ層からの出力に対する活性化関数は、シグモイド関数で実装
        - `h_out_op = tf.nn.sigmoid( h_in_op )`
    - 出力層からの出力に対する活性化関数は、シグモイド関数で実装
        - `self._y_out_op = tf.nn.sigmoid( y_in_op )`
- `MultilayerPerceptron.models()` メソッドにて、このモデルの損失関数を設定。<br>このモデルの損失関数は、クロス・エントロピー関数
    ```python
    def loss( self ):
        self._loss_op = -tf.reduce_sum( 
                            self._t_holder * tf.log(self._y_out_op) +
                             (1-self._t_holder)*tf.log(1-self._y_out_op)
                    )

        return self._loss_op
    ```
- 最適化アルゴリズムは、最急降下法（勾配降下法）。学習率は、0.05
    ```python
    def optimizer( self ):
        """
        モデルの最適化アルゴリズムの設定を行う。

        [Output]
            optimizer の train_step
        """
        optimizer = tf.train.GradientDescentOptimizer( learning_rate = self._learning_rate )
        train_step = optimizer.minimize( self._loss_op )

        return train_step
    ```
- エポック数は 500 回で、ミニバッチ学習のサイズは 20 で学習

<br>

#### トレーニング回数（エポック）に対する、損失関数（クロス・エントロピー）の値のグラフ
> ![multilayerperceptron_1-1](https://user-images.githubusercontent.com/25688193/31829121-543ade9c-b5f7-11e7-8929-b98928876989.png)
> １つ目の図が、入力層：１ノード、隠れ層：３ノード、出力層：１ノードの多層パーセプトロンモデル (1-3-1) での損失関数のグラフ。<br>
> ２つ目の図が、入力層：１ノード、隠れ層１：３ノード、隠れ層２：３ノード、出力層：１ノードの多層パーセプトロンモデル (1-3-3-1) での損失関数のグラフ。<br>


#### 識別結果＆境界のグラフ
![multilayerperceptron_1-2](https://user-images.githubusercontent.com/25688193/31829551-8e0df7a2-b5f8-11e7-910e-d56db2bab919.png)
> １つ目の図が、入力層：１ノード、隠れ層：３ノード、出力層：１ノードの多層パーセプトロンモデル (1-3-1) での識別結果＆境界のグラフ。<br>
> ２つ目の図が、入力層：１ノード、隠れ層１：３ノード、隠れ層２：３ノード、出力層：１ノードの多層パーセプトロンモデル (1-3-3-1) での識別結果＆境界のグラフ。<br>

|NN model|accuracy [test data]|
|---|---|
|Multiplelayer Perceptron<br>(1-3-1)|0.883|
|Multiplelayer Perceptron<br>(1-3-3-1)|0.817|


<br>
<a id="ID_3-1-2"></a>

### 円状データ `sklearn.datasets.make_circles(...)` での識別

- scikit-learn ライブラリの `sklearn.datasets.make_circles(...)` で生成した円状データにて、２クラスの識別を行なった。<br>データ数は 300 個で、ノイズ値を 0.1 に設定<br>
    - `X_features, y_labels = MLPreProcess.generateCirclesDataSet( input_n_samples = 300, input_noize = 0.1 )`
- その他の条件は、先の半月状データのときと同様。

#### トレーニング回数（エポック）に対する、損失関数（クロス・エントロピー）の値のグラフ
![multilayerperceptron_1-3](https://user-images.githubusercontent.com/25688193/31832404-daad6094-b601-11e7-92d5-765de4b614ef.png)
> １つ目の図が、入力層：１ノード、隠れ層：３ノード、出力層：１ノードの多層パーセプトロンモデル (1-3-1) での損失関数のグラフ。<br>
> ２つ目の図が、入力層：１ノード、隠れ層１：３ノード、隠れ層２：３ノード、出力層：１ノードの多層パーセプトロンモデル (1-3-3-1) での損失関数のグラフ。<br>
> 1-3-1 の MLP モデルでは、損失関数が早い段階で収束おり、overfitting 状態であることが見て取れる。一方、1-3-3-1 の MLP モデルでは、損失関数が収束しておらず、underfitting 状態であることが見て取れる。

#### 識別結果＆境界のグラフ
![multilayerperceptron_1-4](https://user-images.githubusercontent.com/25688193/31832405-dad313a2-b601-11e7-9c1e-f09d2b960d91.png)
> １つ目の図が、入力層：１ノード、隠れ層：３ノード、出力層：１ノードの多層パーセプトロンモデル (1-3-1) での識別結果＆境界のグラフ。<br>
> ２つ目の図が、入力層：１ノード、隠れ層１：３ノード、隠れ層２：３ノード、出力層：１ノードの多層パーセプトロンモデル (1-3-3-1) での識別結果＆境界のグラフ。<br>
> 先の、損失関数の結果を反映して、1-3-1 の MLP モデルでは、（学習が収束しているので）うまく識別できている。一方、1-3-3-1 の MLP モデルでは、（まだ学習が収束していないので）うまく識別できていない。

|NN model|accuracy [test data]|
|---|---|
|Multiplelayer Perceptron<br>(1-3-1)|1.000|
|Multiplelayer Perceptron<br>(1-3-3-1)|0.783|

<br>

<a id="ID_3-2"></a>

## 多層パーセプトロンによる多クラスの識別 : `main2.py`
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