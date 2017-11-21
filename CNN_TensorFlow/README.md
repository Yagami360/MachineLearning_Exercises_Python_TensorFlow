# TensorFlow で畳み込みニューラルネットワーク [CNN : Convolutional Neural Network] の実装

TensorFlow での CNN の練習用実装コード集。<br>

TensorFlow での CNN の処理をクラス（任意の層に DNN 化可能な柔軟なクラス）でラッピングし、scikit-learn ライブラリの classifier, estimator とインターフェイスを共通化することで、scikit-learn ライブラリとの互換性のあるようにした自作クラス `ConvolutionalNN` を使用する。<br>


この README.md ファイルには、各コードの実行結果、概要、CNN の背景理論の説明を記載しています。<br>
分かりやすいように `main.py` ファイル毎に１つの完結した実行コードにしています。

参考サイト :
- [Tensorflow での MINIST チュートリアル（公式）](https://www.tensorflow.org/get_started/mnist/beginners)
- [Tensorflow での CIFAR-10 チュートリアル（公式）](https://www.tensorflow.org/tutorials/deep_cnn)
- [TensorFlowはじめました / TensorFlowでデータの読み込み ― 画像を分類するCIFAR-10の基礎](http://www.buildinsider.net/small/booktensorflow/0201)
- Queue（画像パイプライン）を用いた処理
    - http://ykicisk.hatenablog.com/entry/2016/12/18/184840<br>

## 項目 [Contents]

1. [使用するライブラリ](#ID_1)
1. [使用するデータセット](#ID_2)
1. [ニューラルネットワークのフレームワークのコードの説明](#ID_3-0)
1. [コード説明＆実行結果](#ID_3)
    1. [CNN による MNIST データの識別 : `main1.py`](#ID_3-1)
    1. [CNN による CIFAR-10 データの識別 : `main2.py`](#ID_3-2)
    1. [Queue（画像パイプライン）を用いた CNN による CIFAR-10 データの識別 : `main3.py`](#ID3-3)
    1. [学習済み CNN モデルの再学習処理（転移学習）](#ID_3-4)
        1. [学習済み CNN モデルの保存、読み込み : `main4_1.py`](#ID_3-4-1)
        1. [GoogLeNet（インセプション） : `main4_2.py`](#ID_3-4-2)
1. [背景理論](#ID_4)
    1. [CNN の概要](#ID_4-1)
    1. [畳み込み [convolution] 処理について](#ID_4-2)
        1. [畳み込みの数学的な一般的な定義](#ID_4-2-1)
        1. [畳み込みニューラルネットワークにおける畳み込み](#ID_4-2-2)
        1. [畳み込みニューラルネットワークにおける畳み込み処理の具体的な例（画像データとそのフィルタ処理）](#ID_4-2-3)
        1. [より一般化した畳み込み層のアーキテクチャの元での定式化](#ID_4-2-4)
        1. [受容野の観点から見た、畳み込み層](#ID_4-2-5)
    1. [プーリング [pooling] 処理について](#ID_4-3)
        1. [平均プーリング [average pooling]](#ID_4-3-1)
        1. [最大プーリング [max pooling]](#ID_4-3-2)
        1. [Lp プーリング [Lp pooling]](#ID_4-3-3)


<a id="ID_1"></a>

## 使用するライブラリ

> TensorFlow ライブラリ </br>
>> 参考サイト<br>
>> https://qiita.com/tadOne/items/b484ce9f973a9f80036e<br>

>> `tf.nn.conv2d(...)` : ２次元の画像の畳み込み処理のオペレーター<br>
>> https://www.tensorflow.org/api_docs/python/tf/nn/conv2d<br>

>> `tf.nn.max_pool(...)` : マックスプーリング処理のオペレーター<br>
>> https://www.tensorflow.org/api_docs/python/tf/nn/max_pool<br>

>> `tf.nn.sparse_softmax_cross_entropy_with_logits(...)` : 疎なソフトマックス・クロス・エントロピー関数のオペレーター<br>
>> https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits<br>

>> `tf.train.MomentumOptimizer(...)` : モーメンタムアルゴリズムの Optimizer<br>
>> https://www.tensorflow.org/api_docs/python/tf/train/MomentumOptimizer<br>

>> ファイル＆画像処理関連
>>> 参考サイト : <br>
http://tensorflow.classcat.com/2016/02/13/tensorflow-how-tos-reading-data/<br>
https://qiita.com/antimon2/items/c7d2285d34728557e81d<br>
>>> `tf.FixedLengthRecordReader(...)` : 固定の長さのバイトを読み取るレコードリーダー<br>
>>> https://www.tensorflow.org/api_docs/python/tf/FixedLengthRecordReader<br>
>>> `tf.train.string_input_producer(...)` : ファイル名（のキュー）を渡すことで、ファイルの内容（の一部（を表す tensor））が得られる<br>
>>> https://www.tensorflow.org/api_docs/python/tf/train/string_input_producer<br>
>>> `tf.decode_raw(...)` : 文字列から uint8 の Tensor に変換する。<br>
>>> https://www.tensorflow.org/api_docs/python/tf/decode_raw<br>
>>> `tf.image.resize_image_with_crop_or_pad(...)` : 指定した値で画像を切り取る<br>
>>> https://www.tensorflow.org/api_docs/python/tf/image/resize_image_with_crop_or_pad<br>
>>> `tf.image.random_flip_left_right(...)` : 画像の左右をランダムに反転<br>
>>> https://www.tensorflow.org/api_docs/python/tf/image/random_flip_left_right<br>
>>> `tf.image.per_image_standardization(...)` : 画像を正規化<br>
>>> https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization<br>

>> キュー（画像パイプライン）関連<br>
>>>

> Numpy ライブラリ
>> `numpy.argmax(...)` : 指定した配列の中で最大要素を含むインデックスを返す関数<br>
>> https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.argmax.html



<a id="ID_2"></a>

## 使用するデータセット
- [MNIST データセット](https://github.com/Yagami360/MachineLearning_Exercises_Python_TensorFlow/blob/master/dataset.md#mnist手書き数字文字画像データ)
    - 多クラスの識別＆パターン認識処理である `main1.py` で使用
- [CIFAR-10 データセット](https://github.com/Yagami360/MachineLearning_Exercises_Python_TensorFlow/blob/master/dataset.md#cifar-10-データセット)
    - 多クラスの識別＆パターン認識処理である `main2.py` で使用


<a id="ID_3-0"></a>

## ニューラルネットワークのフレームワークのコードの説明
> sphinx or API Blueprint で HTML 形式の API 仕様書作成予定...

- `NeuralNetworkBase` クラス
    - scikit-learn ライブラリの推定器 estimator の基本クラス `BaseEstimator`, `ClassifierMixin` を継承している
    - ニューラルネットワークの基本的なフレームワークを想定した仮想メソッドからなる抽象クラス。<br>
    実際のニューラルネットワークを表すクラスの実装は、このクラスを継承し、オーバーライドするを想定している。
        - `model()` : モデルの定義を行い、最終的なモデルの出力のオペレーターを設定する。
        - `loss( nnLoss )` : 損失関数（誤差関数、コスト関数）の定義を行う。
        - `optimizer( nnOptimize )` : モデルの最適化アルゴリズムの設定を行う。
        - `fit( X_test, y_test )` : 指定されたトレーニングデータで、モデルの fitting 処理を行う。
        - `predict( X_test, y_test )` : fitting 処理したモデルで、推定を行い、予想値を返す。
        - `predict_prob( X_test, y_test )` : fitting 処理したモデルで、推定を行い、クラスの所属確率の予想値を返す。
        - `accuracy( X_test )` : 指定したデータでの正解率を算出する。
        - `accuracy_labels( X_test )` : 指定したデータでのラベル毎の正解率を算出する。

- `ConvolutionalNN` クラス
    - `NeuralNetworkBase` クラスの子クラス。
    - 畳み込みニューラルネットワーク [CNN : Convolutional Neural Network] を表すクラス。<br>
    TensorFlow での CNN の処理をクラス（任意の層に DNN 化可能な柔軟なクラス）でラッピングし、scikit-learn ライブラリの classifier, estimator とインターフェイスを共通化することで、scikit-learn ライブラリとの互換性のある自作クラス

- `NNActivation` クラス : ニューラルネットワークの活性化関数を表す親クラス。<br>
    ポリモーフィズムを実現するための親クラス
    - `Sigmoid` クラス : `NNActivation` の子クラス。シグモイド関数の活性化関数を表す
    - `ReLu` クラス : `NNActivation` の子クラス。Relu 関数の活性化関数を表す
    - `Softmax` クラス : `NNActivation` の子クラス。softmax 関数の活性化関数を表す
    
- `NNLoss` クラス : ニューラルネットワークにおける損失関数を表す親クラス。<br>
    ポリモーフィズムを実現するための親クラス
    - `L1Norm` クラス : `NNLoss` クラスの子クラス。損失関数である L1ノルムを表すクラス。
    - `L2Norm` クラス : `NNLoss` クラスの子クラス。損失関数である L2ノルムを表すクラス。
    - `BinaryCrossEntropy` クラス : `NNLoss` クラスの子クラス。２値のクロス・エントロピーの損失関数
    - `CrossEntropy` クラス : `NNLoss` クラスの子クラス。クロス・エントロピーの損失関数
    - `SoftmaxCrossEntropy` クラス : `NNLoss` クラスの子クラス。ソフトマックス・クロス・エントロピーの損失関数
    - `SparseSoftmaxCrossEntropy` クラス : `NNLoss` クラスの子クラス。疎なソフトマックス・クロス・エントロピーの損失関数
    
- `NNOptimizer` クラス : ニューラルネットワークモデルの最適化アルゴリズム Optimizer を表す親クラス<br>
    ポリモーフィズムを実現するための親クラス
    - `GradientDecent` クラス : `NNOptimizer` クラスの子クラス。勾配降下法を表すクラス。
    - `Momentum` クラス : `NNOptimizer` クラスの子クラス。モメンタム アルゴリズムを表すクラス
    - `NesterovMomentum` クラス : `NNOptimizer` クラスの子クラス。Nesterov モメンタム アルゴリズムを表すクラス
    - `Adagrad` クラス : `NNOptimizer` クラスの子クラス。Adagrad アルゴリズムを表すクラス
    - `Adadelta` クラス : `NNOptimizer` クラスの子クラス。Adadelta アルゴリズムを表すクラス

<br>
<a id="ID_3"></a>

## コードの説明＆実行結果

<a id="ID_3-1"></a>

### CNN による MNIST データの識別 : `main1.py`
![image](https://user-images.githubusercontent.com/25688193/33008526-bf1593e2-ce16-11e7-9cf4-8930e1347cf6.png)

#### コードの説明
- MNIST データセットを使用。
    - データは shape = [n_sample, image_width=28, image_height=28] の形状に reshape
    ```python
    def main():
        ...
        X_train, y_train = MLPreProcess.load_mnist( mnist_path, "train" )
        X_test, y_test = MLPreProcess.load_mnist( mnist_path, "t10k" )

        X_train = numpy.array( [numpy.reshape(x, (28,28)) for x in X_train] )
        X_test = numpy.array( [numpy.reshape(x, (28,28)) for x in X_test] )
    ```
- エポック数は 500、ミニバッチサイズは 100 で学習。
- その他、CNN 処理に必要なパラメータは、
以下のように `ConvolutionalNN` クラスのオブジェクト生成時に設定する。
    ```python
    def main():
        ...
        # CNN クラスのオブジェクト生成
        cnn1 = ConvolutionalNN(
                   session = tf.Session( config = tf.ConfigProto(log_device_placement=True) ),
                   epochs = 500,
                   batch_size = 100,
                   eval_step = 1,
                   image_height = 28,                   # 28 pixel
                   image_width = 28,                    # 28 pixel
                   n_channels = 1,                      # グレースケール
                   n_ConvLayer_featuresMap = [25, 50],  # conv1 : 25 枚, conv2 : 50 枚
                   n_ConvLayer_kernels = [4, 4],        # conv1 : 4*4, conv2 : 4*4
                   n_strides = 1,
                   n_pool_wndsize = 2,
                   n_pool_strides = 2,
                   n_fullyLayers = 100,
                   n_labels = 10
               )

        cnn2 = ConvolutionalNN(
                   session = tf.Session( config = tf.ConfigProto(log_device_placement=True) ),
                   epochs = 500,
                   batch_size = 100,
                   eval_step = 1,
                   image_height = 28,                   # 28 pixel
                   image_width = 28,                    # 28 pixel
                   n_channels = 1,                      # グレースケール
                   n_ConvLayer_featuresMap = [25, 50],  # conv1 : 25 枚, conv2 : 50 枚
                   n_ConvLayer_kernels = [4, 4],        # conv1 : 4*4, conv2 : 4*4
                   n_strides = 1,
                   n_pool_wndsize = 2,
                   n_pool_strides = 2,
                   n_fullyLayers = 100,
                   n_labels = 10
               )
    ```
- モデルの構造は、`ConvolutionalNN.model()` メソッドで定義し、<br>
  ｛畳み込み層１ → プーリング層１ → 畳み込み層２ → プーリング層２ → 全結合層１ → 全結合層２｝
   で構成。
    - 畳み込み層１ : `tf.nn.conv2d(...)`
        - 画像の高さ : `_image_height = 28` 
        - 画像の幅 : `_image_width = 28`
        - チャンネル数 : `_n_channels = 1`
        - カーネル（フィルタ行列）: `_n_ConvLayer_kernels[0] = 4` → 4*4
        - 特徴マップ数 : `_n_ConvLayer_featuresMap[0] = 25`
        - ストライド幅 : `_n_strides = 1` → 1*1
        - ゼロパディング : `padding = "SAME"`
    - プーリング層１
        - マックスプーリング : `tf.nn.max_pool(...)`
        - ウィンドウサイズ : `_n_pool_wndsize = 2` → 2*2
        - ストライド幅 : `_n_pool_strides = 2` → 2*2
    - 畳み込み層２ : `tf.nn.conv2d(...)`
        - カーネル（フィルタ行列）: `_n_ConvLayer_kernels[1] = 4` → 4*4
        - 特徴マップ数（入力側）: `_n_ConvLayer_featuresMap[0] = 25`
        - 特徴マップ数（出力側）: `_n_ConvLayer_featuresMap[1] = 50`
        - ストライド幅 : `_n_strides = 1` → 1*1
        - ゼロパディング : `padding = "SAME"`
    - プーリング層２
        - マックスプーリング : `tf.nn.max_pool(...)`
        - ウィンドウサイズ : `_n_pool_wndsize = 2` → 2*2
        - ストライド幅 : `_n_pool_strides = 2` → 2*2
    - 全結合層１（入力側）
        - `n_fullyLayers = 100` → `n_labels = 10`
    - 全結合層２（出力側）
        - `n_labels = 10` → 最終出力
    ```python
    class ConvolutionalNN( NeuralNetworkBase ):
    ...
    def model( self ):
        """
        モデルの定義（計算グラフの構築）を行い、
        最終的なモデルの出力のオペレーターを設定する。

        [Output]
            self._y_out_op : Operator
                モデルの出力のオペレーター
        """
        # 計算グラフの構築
        #----------------------------------------------------------------------
        # １つ目の畳み込み層 ~ 活性化関数 ~ プーリング層 ~
        #----------------------------------------------------------------------
        # 重みの Variable の list に、１つ目の畳み込み層の重み（カーネル）を追加
        # この重みは、畳み込み処理の画像データに対するフィルタ処理（特徴マップ生成）に使うカーネルを表す Tensor のことである。
        self._weights.append( 
            self.init_weight_variable( 
                input_shape = [ 
                    self._n_ConvLayer_kernels[0], self._n_ConvLayer_kernels[0], 
                    self._n_channels, 
                    self._n_ConvLayer_featuresMap[0] 
                ]
            ) 
        )
        
        # バイアス項の Variable の list に、畳み込み層のバイアス項を追加
        self._biases.append( 
            self.init_bias_variable( input_shape = [ self._n_ConvLayer_featuresMap[0] ] ) 
        )

        # 畳み込み層のオペレーター
        conv_op1 = tf.nn.conv2d(
                       input = self._X_holder,
                       filter = self._weights[0],   # 畳込み処理で input で指定した Tensor との積和に使用する filter 行列（カーネル）
                       strides = [ 1, self._n_strides, self._n_strides, 1 ], # strides[0] = strides[3] = 1. とする必要がある
                       padding = "SAME"     # ゼロパディングを利用する場合はSAMEを指定
                   )

        # 畳み込み層からの出力（活性化関数）オペレーター
        # バイアス項を加算したものを活性化関数に通す
        conv_out_op1 = Relu().activate( tf.nn.bias_add( conv_op1, self._biases[0] ) )
                
        # プーリング層のオペレーター
        pool_op1 = tf.nn.max_pool(
                       value = conv_out_op1,
                       ksize = [ 1, self._n_pool_wndsize, self._n_pool_wndsize, 1 ],    # プーリングする範囲（ウィンドウ）のサイズ
                       strides = [ 1, self._n_pool_strides, self._n_pool_strides, 1 ],  # ストライドサイズ strides[0] = strides[3] = 1. とする必要がある
                       padding = "SAME"                                                 # ゼロパディングを利用する場合はSAMEを指定
                   )


        #----------------------------------------------------------------------
        # ２つ目以降の畳み込み層 ~ 活性化関数 ~ プーリング層 ~
        #----------------------------------------------------------------------
        # 畳み込み層のカーネル
        self._weights.append( 
            self.init_weight_variable( 
                input_shape = [ 
                    self._n_ConvLayer_kernels[1], self._n_ConvLayer_kernels[1], 
                    self._n_ConvLayer_featuresMap[0], self._n_ConvLayer_featuresMap[1] 
                ]
            ) 
        )

        self._biases.append( 
            self.init_bias_variable( input_shape = [ self._n_ConvLayer_featuresMap[1] ] ) 
        )

        # 畳み込み層のオペレーター
        conv_op2 = tf.nn.conv2d(
                       input = pool_op1,
                       filter = self._weights[1],   # 畳込み処理で input で指定した Tensor との積和に使用する filter 行列 (Tensor)
                       strides = [ 1, self._n_strides, self._n_strides, 1 ], # strides[0] = strides[3] = 1. とする必要がある
                       padding = "SAME"     # ゼロパディングを利用する場合はSAMEを指定
                   )

        conv_out_op2 = Relu().activate( tf.nn.bias_add( conv_op2, self._biases[1] ) )
        
        # プーリング層のオペレーター
        pool_op2 = tf.nn.max_pool(
                       value = conv_out_op2,
                       ksize = [ 1, self._n_pool_wndsize, self._n_pool_wndsize, 1 ],    # プーリングする範囲（ウィンドウ）のサイズ
                       strides = [ 1, self._n_pool_strides, self._n_pool_strides, 1 ],  # ストライドサイズ strides[0] = strides[3] = 1. とする必要がある
                       padding = "SAME"                                                 # ゼロパディングを利用する場合はSAMEを指定
                   )

        #----------------------------------------------------------------------
        # ~ 全結合層 ~ 出力層
        #----------------------------------------------------------------------
        # 全結合層の入力側
        # 重み & バイアス項の Variable の list に、全結合層の入力側に対応する値を追加
        fullyLayers_width = self._image_width // (2*2)    # ? (2 * 2 : pooling 処理の範囲)
        fullyLayers_height = self._image_height // (2*2)  # ?
        fullyLayers_input_size = fullyLayers_width * fullyLayers_height * self._n_ConvLayer_featuresMap[-1] # ?
        print( "fullyLayers_input_size : ", fullyLayers_input_size )

        self._weights.append( 
            self.init_weight_variable( 
                input_shape = [ fullyLayers_input_size, self._n_fullyLayers ] 
            )
        )
        self._biases.append( self.init_bias_variable( input_shape = [ self._n_fullyLayers ] ) )

        # 全結合層への入力
        # 1 * N のユニットに対応するように reshape
        pool_op_shape = pool_op2.get_shape().as_list()      # ? [batch_size, 7, 7, _n_ConvLayer_features[-1] ]
        print( "pool_op2.get_shape().as_list() :\n", pool_op_shape )
        fullyLayers_shape = pool_op_shape[1] * pool_op_shape[2] * pool_op_shape[3]
        flatted_input = tf.reshape( pool_op2, [ -1, fullyLayers_shape ] )    # 1 * N に平坦化 (reshape) された値
        #flatted_input = numpy.reshape( pool_op2, (None, fullyLayers_shape) )
        print( "flatted_input :", flatted_input )

        # 全結合層の入力側へのオペレーター
        fullyLayers_in_op = Relu().activate( tf.add( tf.matmul( flatted_input, self._weights[-1] ), self._biases[-1] ) )


        # 全結合層の出力側
        # 重み & バイアス項のの Variable の list に、全結合層の出力側に対応する値を追加
        self._weights.append( 
            self.init_weight_variable( 
                input_shape = [ self._n_fullyLayers, self._n_labels ] 
            )
        )
        self._biases.append( self.init_bias_variable( input_shape = [ self._n_labels ] ) )
        
        # 全結合層の出力側へのオペレーター
        fullyLayers_out_op = tf.add( tf.matmul( fullyLayers_in_op, self._weights[-1] ), self._biases[-1] )
        self._y_out_op = fullyLayers_out_op

        return self._y_out_op
    ```
- 損失関数は、`ConvolutionalNN.loss()` メソッドで行い、ソフトマックス・クロス・エントロピー関数を使用
    ```python
    cnn1.loss( SoftmaxCrossEntropy() )
    cnn2.loss( SoftmaxCrossEntropy() )
    ```
- モデルの最適化アルゴリズムは、`ConvolutionalNN.optimizer()` メソッドで行い、モメンタムを使用
    - 学習率 learning_rate は、0.0001 と 0.0005 の２つのモデルで異なる値で検証
    ```python
    cnn1.optimizer( Momentum( learning_rate = 0.0001, momentum = 0.9 ) )
    cnn2.optimizer( Momentum( learning_rate = 0.0005, momentum = 0.9 ) )
    ```

#### コードの実行結果

##### 損失関数のグラフ
![cnn_1-1-1 _softmax-cross-entropy](https://user-images.githubusercontent.com/25688193/32997301-65714272-cdd1-11e7-8b42-ce042bc5042f.png)
> 損失関数として、ソフトマックス・クロス・エントロピー関数を使用した場合の、損失関数のグラフ。<br>
> 赤線が学習率 0.0001 の CNN モデル（最適化アルゴリズムとして、モーメンタムアルゴリズム使用）。
> 青線が学習率 0.0005 の CNN モデル（最適化アルゴリズムとして、モーメンタムアルゴリズム使用）。
> 学習率が 0.0001 の場合、エポック数 500 で損失関数が収束しきれていないことが分かる。

##### 学習済みモデルでの正解率の値

- 学習済みモデルでのテストデータでの正解率 : 学習率=0.0001 の場合

|ラベル|Acuraccy [test data]|サンプル数|
|---|---|---|
|全ラベルでの平均|0.837|10,000 個|
|0|0.924|980（※全サンプル数でない）|
|1|1.000|1135（※全サンプル数でない）|
|2|1.000|1032（※全サンプル数でない）|
|3|0.892|1010（※全サンプル数でない）|
|4|0.941|982（※全サンプル数でない）|
|5|1.000|892（※全サンプル数でない）|
|6|1.000|958（※全サンプル数でない）|
|7|0.985|1028（※全サンプル数でない）|
|8|0.781|974（※全サンプル数でない）|
|9|1.000|1009（※全サンプル数でない）|

→ 8 の正解率が低い傾向がある。

- 学習済みモデルでのテストデータでの正解率 : 学習率=0.0005 の場合

|ラベル|Acuraccy [test data]|サンプル数|
|---|---|---|
|全ラベルでの平均|0.958|10,000 個|
|0|1.000|980（※全サンプル数でない）|
|1|1.000|1135（※全サンプル数でない）|
|2|1.000|1032（※全サンプル数でない）|
|3|0.997|1010（※全サンプル数でない）|
|4|0.987|982（※全サンプル数でない）|
|5|1.000|892（※全サンプル数でない）|
|6|0.973|958（※全サンプル数でない）|
|7|0.914|1028（※全サンプル数でない）|
|8|1.000|974（※全サンプル数でない）|
|9|1.000|1009（※全サンプル数でない）|


##### 識別に正解した画像
![cnn_1-2-1 _softmax-cross-entropy](https://user-images.githubusercontent.com/25688193/32997303-78819aba-cdd1-11e7-8be5-fc032cd0a3e2.png)
> 学習率 0.0005 の CNN モデルにおいて、<br>
> 識別に正解したテストデータの画像の内、前方から 40 個のサンプル。<br>
> 各画像のタイトルの Actual は実際のラベル値、Pred は予測したラベル値を示す。

##### 識別に失敗した画像
![cnn_1-3-2 _softmax-cross-entropy_leraningrate 0 00001](https://user-images.githubusercontent.com/25688193/32997994-668ecd9c-cdda-11e7-9e2e-4d8c8ac6c083.png)
> 学習率 0.0001 の CNN モデルにおいて、<br>
> 識別に失敗したテストデータの画像の内、前方から 40 個のサンプル。<br>
> 各画像のタイトルの Actual は実際のラベル値、Pred は予測したラベル値を示す。

<br>

![cnn_1-3-1 _softmax-cross-entropy](https://user-images.githubusercontent.com/25688193/32997541-c857168e-cdd4-11e7-81b1-198774d15fc3.png)
> 学習率 0.0005 の CNN モデルにおいて、<br>
> 識別に失敗したテストデータの画像の内、前方から 40 個のサンプル。<br>

<br>

<a id="ID_3-2"></a>

### CNN による CIFAR-10 データの識別 : `main2.py`
![image](https://user-images.githubusercontent.com/25688193/33008486-99f0d40a-ce16-11e7-95a1-2adc8abfff39.png)

#### コードの説明

- バイナリー形式の CIFAR-10 データセット CIFAR-10 binary version (suitable for C programs) を使用
    - ファイルフォーマットは [dataset.md](https://github.com/Yagami360/MachineLearning_Exercises_Python_TensorFlow/blob/master/dataset.md#cifar-10-データセット) 参照
    - 処理負荷＆メモリサイズ軽減のため、５つのトレーニング用データ（合計：50000 サンプル）の内、
    １つのファイル（10000 サンプル）のみトレーニングデータとして使用する。
    - このフォーマットに基づき、<br>
    データの読み込みを `MLPreProcess` クラスの static 関数 `load_cifar10_train(...)` and `load_cifar10_test(...)` で行う。
    ```python
    class MLPreProcess(object):
    ...
    @staticmethod
    def load_cifar10_train( path, fileName = "data_batch_1.bin" ):
        """
        検証データ用の CIFAR-10 データの１つのトレーニング用ファイルを読み込む。
        バイナリ形式 : CIFAR-10 binary version (suitable for C programs)

        [Input]
            path : str
                CIFAR-10 データセットが格納されているフォルダへのパス
            fileName :str
                CIFAR-10 データセットの１つのトレーニング用ファイル名
        """
        file = os.path.join( path, fileName )
        
        # 内部データサイズの設定 
        image_height = 32   # CIFAR-10 画像の高さ (pixel)
        image_width = 32    #
        n_channels = 3      # RGB の 3 チャンネル

        image_bytes = image_height * image_width * n_channels
        labels_byte = 1
        record_bytes = image_bytes + labels_byte

        images = numpy.empty( shape = [10000, image_width, image_height, n_channels ] )
        labels = numpy.empty( shape = [10000] )

        # バイナリーモードでファイルオープン
        byte_stream = open( file, mode="rb" )

        # 全レコード長に関しての loop
        for record in range(10000):
            # seek(...) : 起点：record_bytes * record, オフセット：0
            byte_stream.seek( record_bytes * record , 0 )

            # バッファに割り当て
            label_buffer = numpy.frombuffer( byte_stream.read(labels_byte), dtype=numpy.uint8 )
            image_buffer = numpy.frombuffer( byte_stream.read(image_bytes), dtype=numpy.int8 )

            # [n_channel, image_height, image_width] = [3,32,32] に reshape
            image_buffer = numpy.reshape( image_buffer, [n_channels, image_width, image_height ] )
            
            # imshow(), fit()で読める ([1]height, [2]width, [0] channel) の順番に変更するために
            # numpy の transpose() を使って次元を入れ替え
            image_buffer = numpy.transpose( image_buffer, [1, 2, 0] )

            # float
            image_buffer = image_buffer.astype( numpy.float32 )
            image_buffer = image_buffer / 255

            # 各レコードの画像データを格納していく
            images[record] = image_buffer
            labels[record] = label_buffer

        # 
        byte_stream.close()

        return images, labels
    ```
    ```python
    class MLPreProcess(object):
    ...
        @staticmethod
    def load_cifar10_test( path ):
        """
        検証データ用の CIFAR-10 データのテスト用ファイルを読み込む。
        バイナリ形式 : CIFAR-10 binary version (suitable for C programs)

        [Input]
            path : str
                CIFAR-10 データセットが格納されているフォルダへのパス
        """
        file = os.path.join( path, "test_batch.bin" )
        
        # 内部データサイズの設定 
        image_height = 32   # CIFAR-10 画像の高さ (pixel)
        image_width = 32    #
        n_channels = 3      # RGB の 3 チャンネル

        image_bytes = image_height * image_width * n_channels
        labels_byte = 1
        record_bytes = image_bytes + labels_byte

        images = numpy.empty( shape = [10000, image_width, image_height, n_channels ] )
        labels = numpy.empty( shape = [10000] )

        # バイナリーモードでファイルオープン
        byte_stream = open( file, mode="rb" )

        # 全レコード長に関しての loop
        for record in range(10000):
            # seek(...) : 起点：record_bytes * record, オフセット：0
            byte_stream.seek( record_bytes * record , 0 )

            # バッファに割り当て
            label_buffer = numpy.frombuffer( byte_stream.read(labels_byte), dtype=numpy.uint8 )
            image_buffer = numpy.frombuffer( byte_stream.read(image_bytes), dtype=numpy.int8 )

            # [n_channel, image_height, image_width] = [3,32,32] に reshape
            image_buffer = numpy.reshape( image_buffer, [n_channels, image_width, image_height ] )
            
            # imshow(), fit()で読める ([1]height, [2]width, [0] channel) の順番に変更するために
            # numpy の transpose() を使って次元を入れ替え
            image_buffer = numpy.transpose( image_buffer, [1, 2, 0] )
            
            # float
            image_buffer = image_buffer.astype( numpy.float32 )
            image_buffer = image_buffer / 255

            # 各レコードの画像データを格納していく
            images[record] = image_buffer
            labels[record] = label_buffer
 
        byte_stream.close()

        return images, labels
    ```
    ```python
    X_train, y_train = MLPreProcess.load_cifar10_train( cifar10_path )
    X_test, y_test = MLPreProcess.load_cifar10_test( cifar10_path )
    ```
- モデルの構造は、`ConvolutionalNN.model()` メソッドで定義し、<br>
  ｛畳み込み層１ → プーリング層１ → 畳み込み層２ → プーリング層２ → 全結合層１ → 全結合層２｝
   で構成する。
    - 畳み込み層１ : `tf.nn.conv2d(...)`
        - 画像の高さ : `_image_height = 32` 
        - 画像の幅 : `_image_width = 32`
        - チャンネル数 : `_n_channels = 3`
        - カーネル（フィルタ行列）: `_n_ConvLayer_kernels[0] = 5` → 5*5
        - 特徴マップ数 : `_n_ConvLayer_featuresMap[0] = 64`
        - ストライド幅 : `_n_strides = 1` → 1*1
        - ゼロパディング : `padding = "SAME"`
    - プーリング層１
        - マックスプーリング : `tf.nn.max_pool(...)`
        - ウィンドウサイズ : `_n_pool_wndsize = 3` → 3*3
        - ストライド幅 : `_n_pool_strides = 2` → 2*2
    - 畳み込み層２ : `tf.nn.conv2d(...)`
        - カーネル（フィルタ行列）: `_n_ConvLayer_kernels[1] = 5` → 5*5
        - 特徴マップ数（入力側）: `_n_ConvLayer_featuresMap[0] = 64`
        - 特徴マップ数（出力側）: `_n_ConvLayer_featuresMap[1] = 64`
        - ストライド幅 : `_n_strides = 1` → 1*1
        - ゼロパディング : `padding = "SAME"`
    - プーリング層２
        - マックスプーリング : `tf.nn.max_pool(...)`
        - ウィンドウサイズ : `_n_pool_wndsize = 3` → 3*3
        - ストライド幅 : `_n_pool_strides = 2` → 2*2
    - 全結合層１（入力側）
        - `n_fullyLayers = 384` → `n_labels = 10`
    - 全結合層２（出力側）
        - `n_labels = 10` → 最終出力
- 損失関数は、`ConvolutionalNN.loss()` メソッドで行い、ソフトマックス・クロス・エントロピー関数を使用
    ```python
    cnn1.loss( SoftmaxCrossEntropy() )
    cnn2.loss( SoftmaxCrossEntropy() )
    ```
- モデルの最適化アルゴリズムは、`ConvolutionalNN.optimizer()` メソッドで行い、<br>
　最急降下法 `GradientDecent(...)` or モメンタム `Momentum(...)` を使用する。
    - 学習率 `learning_rate` は、の２つのモデルで異なる値、及び固定値 or 減衰する値の組み合わせで検証
<!--
    - 尚、幾何学的に減衰する学習率は `tf.train.exponential_decay(...)` を使用し、エポック数の 1/100 回数度に 10% を学習率を減衰させる。
        - 式で書くと、rate * ( 1.0 - rate )^(n_generation/n_gen_to_wait)<br>
        rate = 0.1, (n_generation/n_gen_to_wait) = 0.01
-->
        ```python
        # 最急降下法：固定値の学習率
        cnn1.optimizer( GradientDecent( learning_rate = learning_rate1 ) )
        cnn2.optimizer( GradientDecent( learning_rate = learning_rate2 ) )
        ```
        ```python
        # モメンタム：減衰する学習率
        cnn1.optimizer( Momentum( learning_rate = learning_rate1, momentum = 0.9 ) )
        cnn2.optimizer( Momentum( learning_rate = learning_rate2, momentum = 0.9 ) )
        ```

#### コードの実行結果

#### 損失関数のグラフ
損失関数として、ソフトマックス・クロス・エントロピー関数を使用した場合の、損失関数のグラフ。<br>

- 学習率 : 0.0001（固定値） と 0.0005（固定値）：最急降下法
    - 全結合層：100 ノード
    - エポック数：500、バッチサイズ：100
    ![cnn_2-3-1 _gradentdecent](https://user-images.githubusercontent.com/25688193/33002539-2d29d70e-cdf8-11e7-888f-b587f693a715.png)
    - 全結合層：384 ノード
    - エポック数：500、バッチサイズ：100
    ![cnn_2-3-4 _gradebtdecent](https://user-images.githubusercontent.com/25688193/33010288-52f77994-ce1d-11e7-9191-07a3e06aa7cc.png)

- 学習率 : 0.001（固定値） と 0.005（固定値）：最急降下法
    - 全結合層：384 ノード<br>
    - エポック数：500、バッチサイズ：100
    ![cnn_2-3-6 _gradebtdecent](https://user-images.githubusercontent.com/25688193/33021995-fe2b5ba4-ce46-11e7-81af-78ebc4448ce0.png)

- 学習率 : 0.01（固定値） と 0.05（固定値）：最急降下法
    - 全結合層：384 ノード<br>
    - エポック数：500、バッチサイズ：100
![cnn_2-3-7 _gradebtdecent](https://user-images.githubusercontent.com/25688193/33040654-13d324f4-ce7f-11e7-8b0b-4876c3a7f3c8.png)

> 損失関数のグラフより、
> - 値が 0 付近に収束しきれていおらず、2.3 付近に収束している。
> - ノイズが大きい。ミニバッチサイズが小さいため？
> - 収束値に近づくにつれ、ノイズが大きい。過学習が発生している？減衰する学習率を使用する必要あり？

- 学習率 : 0.001（減衰値） と 0.005（減衰値）：モメンタム
    - 全結合層：384 ノード<br>
    - エポック数：1000、バッチサイズ：128<br>
![cnn_2-3-1 _momentum](https://user-images.githubusercontent.com/25688193/33065428-27ec833c-ceec-11e7-832b-fa8b55ce0157.png)

- 学習率 0.005（減衰値）：モメンタム
    - 全結合層：384 ノード<br>
    - エポック数：20000、バッチサイズ：128<br>
> 処理中...

<br>

#### 学習済みモデルでの正解率の値

- 学習済みモデルでのテストデータでの正解率
    - 学習率=0.005（減衰値） 、モメンタム
    - エポック数：1000, ミニバッチサイズ：128
> 処理中...

|ラベル|Acuraccy [test data]|サンプル数|
|---|---|---|
|全ラベルでの平均|0.250|10,000 個|
<!--
|0||（※全サンプル数でない）|
|1||（※全サンプル数でない）|
|2||（※全サンプル数でない）|
|3||（※全サンプル数でない）|
|4||（※全サンプル数でない）|
|5||（※全サンプル数でない）|
|6||（※全サンプル数でない）|
|7||（※全サンプル数でない）|
|8||（※全サンプル数でない）|
|9||（※全サンプル数でない）|
-->

- 学習率 0.005（減衰値）：モメンタム
    - 全結合層：384 ノード<br>
    - エポック数：20000、バッチサイズ：128<br>
> 処理中...

#### 識別に正解した画像
識別に正解したテストデータの画像の内、前方から 40 個のサンプル。<br>
各画像のタイトルの Actual は実際のラベル値、Pred は予測したラベル値を示す。

- 学習率 0.005 （減衰値）：モメンタム
    - エポック数：1500、バッチサイズ：128<br>
![cnn_2-6-1 _momentum](https://user-images.githubusercontent.com/25688193/33066693-64b3c9a8-ceef-11e7-99f5-8668e2cad862.png)

- 学習率 0.005（減衰値）：モメンタム
    - 全結合層：384 ノード<br>
    - エポック数：20000、バッチサイズ：128<br>
> 処理中...

<br>

#### 識別に失敗した画像
識別に失敗したテストデータの画像の内、前方から 40 個のサンプル。<br>
各画像のタイトルの Actual は実際のラベル値、Pred は予測したラベル値を示す。
    
- 学習率 0.005 （減衰値）：モメンタム
    - エポック数：1000、バッチサイズ：128<br>
![cnn_2-7-1 _momentum](https://user-images.githubusercontent.com/25688193/33066747-82b01484-ceef-11e7-8903-1ef5fcb8313b.png)

- 学習率 0.005（減衰値）：モメンタム
    - 全結合層：384 ノード<br>
    - エポック数：20000、バッチサイズ：128<br>
> 処理中...


<br>
<a id="ID_3-3"></a>

### Queue （画像パイプライン）を使用した CNN による CIFAR-10 データの識別 : `main3.py`
> コード実装中...

- バイナリー形式の CIFAR-10 データセットを使用
    - xxx
- **画像は、ランダムに加工した上でトレーニングデータとして利用する**
    - 加工は、画像の一部の切り出し、左右の反転、明るさの変更からなる。
    - 画像の分類精度を向上させるには、画像の枚数が必要となるが、画像を加工することで画像を水増しすることが出来るため、このような処理を行う。
- 画像パイプライン（キュー）を用いた処理は以下のようになる。
    - パイプラインを使用する場合は、session の `run(...)` 時に Placeholder を feed_dict する必要はない。
    - パイプラインの初期化は、`tf.train.start_queue_runners(...)` で行う。
    - https://www.tensorflow.org/api_docs/python/tf/train/start_queue_runners
    - http://tensorflow.classcat.com/2016/02/13/tensorflow-how-tos-reading-data/
- xxx
    
<br>

<a id="ID_3-4"></a>

### 学習済み CNN モデルの再学習処理（転移学習） 

<a id="ID_3-4-1"></a>

#### 学習済み CNN モデルの保存、読み込み : `main4_1.py`
> コード実装中...

- `tf.train.Saver` を使用した、モデルの保存＆読み出し
    - https://www.tensorflow.org/api_docs/python/tf/train/Saver
    - https://qiita.com/yukiB/items/a7a92af4b27e0c4e6eb2
    - http://testpy.hatenablog.com/entry/2017/02/02/000000
    - http://arakan-pgm-ai.hatenablog.com/entry/2017/05/17/194414
- xxx

<br>
<a id="ID_3-4-2"></a>

#### GoogLeNet（インセプション）: `main4_2.py`
> コード実装中...

- http://tensorflow.classcat.com/2016/10/21/tensorflow-googlenet-inception/
- チュートリアルのレポジトリを git clone する。
- コマンドプロンプトで指定の python プログラムを実行し ($ python3 ...)、
展開した画像を`TFRecords` オブジェクトに変換する。
- Bazel をインストールし、トレーニングを行う。
- xxx


<br>

---

<a id="ID_4"></a>

## 背景理論
![image](https://user-images.githubusercontent.com/25688193/30858595-4e038b96-a2fb-11e7-9ac2-4e7131148034.png)
![image](https://user-images.githubusercontent.com/25688193/30904563-47b0fd48-a3ad-11e7-8d6c-c1f3c2751131.png)

<a id="ID_4-2"></a>

### 畳み込み [convolution] 処理について

<a id="ID_4-2-1"></a>

#### 畳み込みの数学的な一般的な定義
![image](https://user-images.githubusercontent.com/25688193/30863721-af4cee86-a30c-11e7-8d6d-b47244badc03.png)

<a id="ID_4-2-2"></a>

#### 畳み込みニューラルネットワークにおける畳み込み
![image](https://user-images.githubusercontent.com/25688193/30867484-0d67583a-a317-11e7-9740-d2449e794990.png)

<a id="ID_4-2-3"></a>

#### 畳み込みニューラルネットワークにおける畳み込み処理の具体的な例（画像データとそのフィルタ処理）
![image](https://user-images.githubusercontent.com/25688193/30872260-6c4409fe-a324-11e7-8758-9a9625a5283d.png)
![image](https://user-images.githubusercontent.com/25688193/30872283-77425900-a324-11e7-9cfc-4f7346cbada9.png)
![image](https://user-images.githubusercontent.com/25688193/30872618-adff2058-a325-11e7-94c5-7620941d8a43.png)
![image](https://user-images.githubusercontent.com/25688193/30874529-9e6564d0-a32b-11e7-904e-a08960e693f3.png)
![image](https://user-images.githubusercontent.com/25688193/30874745-3e52abce-a32c-11e7-9492-71b7f4f072e5.png)
![image](https://user-images.githubusercontent.com/25688193/30874981-f4e58672-a32c-11e7-952e-658c105c4782.png)
![image](https://user-images.githubusercontent.com/25688193/30874489-6f731b90-a32b-11e7-94ad-0025899d76e4.png)

> 参考サイト
>> [定番のConvolutional Neural Networkをゼロから理解する#畳み込みとは](https://deepage.net/deep_learning/2016/11/07/convolutional_neural_network.html#畳み込みとは)


<a id="ID_4-2-4"></a>

#### より一般化した畳み込み層のアーキテクチャの元での定式化
![image](https://user-images.githubusercontent.com/25688193/30882264-5eba369a-a343-11e7-84e3-57b5c66c28e7.png)
![image](https://user-images.githubusercontent.com/25688193/30882273-6c7c3e9a-a343-11e7-8225-893c3bde3700.png)
![image](https://user-images.githubusercontent.com/25688193/30882308-7f8b6a06-a343-11e7-9f50-0288bbfd944b.png)
![image](https://user-images.githubusercontent.com/25688193/30926162-3e669cf6-a3ef-11e7-8732-086483b4a2ec.png)
![image](https://user-images.githubusercontent.com/25688193/30884989-9c766018-a34c-11e7-8cf2-adfd0cc891a1.png)

<a id="ID_4-2-5"></a>

#### 受容野の観点から見た、畳み込み層
![image](https://user-images.githubusercontent.com/25688193/30904710-b736ff00-a3ad-11e7-9a4c-f73f76f71cc3.png)
![image](https://user-images.githubusercontent.com/25688193/30926213-5d706af0-a3ef-11e7-84c9-0216233e73ee.png)
![image](https://user-images.githubusercontent.com/25688193/30926318-abde4d10-a3ef-11e7-900a-8d9eb2842995.png)



<a id="ID_4-3"></a>

### プーリング [pooling] 処理について
![image](https://user-images.githubusercontent.com/25688193/30928885-c94bc0b4-a3f7-11e7-9b83-a86dd44abc95.png)
![image](https://user-images.githubusercontent.com/25688193/30928920-d8cf1b94-a3f7-11e7-86b7-3ab149639139.png)
![image](https://user-images.githubusercontent.com/25688193/30947089-aa6e4b62-a442-11e7-94c5-39b4a52f59e1.png)

<a id="ID_4-3-1"></a>

#### 平均プーリング [average pooling]
![image](https://user-images.githubusercontent.com/25688193/30947132-dfbf6eb8-a442-11e7-9b23-d6eeadc5e951.png)

<a id="ID_4-3-2"></a>

#### 最大プーリング [max pooling]
![image](https://user-images.githubusercontent.com/25688193/30947702-286b95c6-a446-11e7-92a2-6a4cd87dd706.png)

<a id="ID_4-3-3"></a>

#### Lp プーリング [Lp pooling]
![image](https://user-images.githubusercontent.com/25688193/30948182-27d90abe-a449-11e7-869d-4d14fbe22904.png)

<br>

---

### デバッグ Memo

[17/11/19]

- InvalidArgumentError (see above for traceback): logits and labels must be same size: logits_size=[100,10] labels_size=[1,100]
	 [[Node: SoftmaxCrossEntropyWithLogits = SoftmaxCrossEntropyWithLogits[T=DT_FLOAT, _device="/job:localhost/replica:0/task:0/cpu:0"](Reshape_2, Reshape_3)]]

- 

```
ValueError: Rank mismatch: Rank of labels (received 2) should equal rank of logits minus 1 (received 2).

    def loss( self, t_holder, y_out_op ):
        self._loss_op = tf.reduce_mean(
                            tf.nn.sparse_softmax_cross_entropy_with_logits(
                                logits = y_out_op,
                                labels = t_holder
                            )
                        )
```