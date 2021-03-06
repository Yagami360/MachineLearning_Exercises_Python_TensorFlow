## TensorFlow での DCGAN [Deep Convolutional GAN] の実装

TensorFlow での DCGAN [Deep Convolutional GAN] の練習用実装コード集。

この README.md ファイルには、各コードの実行結果、概要、DCGAN の背景理論の説明を記載しています。<br>
分かりやすいように `main.py` ファイル毎に１つの完結した実行コードにしています。

### 項目 [Contents]

1. [使用するライブラリ](#ID_1)
1. [使用するデータセット](#ID_2)
1. [コード実行結果＆内容説明](#ID_3)
    1. [DCGAN による MNIST ライクな手書き数字画像データの自動生成 : `main1.py`](#ID_3-1)
        1. [コードの実行結果](#ID_3-1-1)
        1. [コードの内容説明](#ID_3-1-2)
    1. xxx : `main2.py`
        1. コードの実行結果
        1. コードの内容説明
1. [背景理論](#ID_4)
    1. [生成モデル [generative model]](#ID_10)
        1. [GAN [Generative Adversarial Network]（敵対的ネットワーク）](#ID_10-1)
            1. [DCGAN [Deep Convolutional GAN]](#ID_10-1-1)
        1. VAE [Variational Autoencoder]

---

<a id="ID_1"></a>

### 使用するライブラリ

- TensorFlow ライブラリ
    - `tf.get_varible(...)`, `tf.variable_scope(...)` : 名前空間と変数のスコープ（重み共有で使用）
        - https://qiita.com/TomokIshii/items/ffe999b3e1a506c396c8
        - https://deepage.net/tensorflow/2017/06/02/tensorflow-variable.html
    - `tf.nn.moments(...)` : 平均と分散を計算（batch normalization で使用）
        - https://www.tensorflow.org/api_docs/python/tf/nn/moments
    - `tf.nn.batch_normalization(...)` : batch normalization
        - https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization
        - https://tyfkda.github.io/blog/2016/09/14/batch-norm-mnist.html
    - `tf.nn.conv2d(...)` : ２次元の画像の畳み込み処理のオペレーター
        - https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
    - `tf.nn.conv2d_transpose(...)` : 逆畳み込み層 deconvolution layers
        - xxx
    - `tf.trainable_variables(...)` : trainable フラグを付けた変数
        - https://qiita.com/TomokIshii/items/84ee55a1c2d335dcab6f
    - `tf.control_dependencies(...)` : sess.run で実行する際のトレーニングステップの依存関係（順序）を定義
        - `tf.no_op(...)` : 何もしない Operator を返す。（トレーニングステップの依存関係を定義するのに使用）
        - xxx
- その他ライブラリ
    - `pickle` :


<a id="ID_2"></a>

### 使用するデータセット
- [MNIST データセット](https://github.com/Yagami360/MachineLearning_Exercises_Python_TensorFlow/blob/master/dataset.md#mnist手書き数字文字画像データ)
    - `main1.py` で使用
- [CIFAR-10 データセット](https://github.com/Yagami360/MachineLearning_Exercises_Python_TensorFlow/blob/master/dataset.md#cifar-10-データセット)

---

<a id="ID_3"></a>

## コード実行結果＆内容説明

<a id="ID_3-1"></a>

### DCGAN による MNIST ライクな手書き数字画像データの自動生成 : `main1.py`
DCGAN モデルに対し MNIST データセットで学習し、MNIST ライクな手書き数字画像を自動生成する。

<a id="ID_3-1-1"></a>

#### コードの実行結果

#### 損失関数のグラフ

|パラメータ名|値（実行条件１）|
|---|---|
|最適化アルゴリズム|Adam|
|学習率<br>`learning_rate`|0.0001|
|学習率の減衰項１<br>`beta1`|0.5|
|学習率の減衰項２<br>`beta2`|0.999|
|ミニバッチサイズ<br>`batch_size`|32|
|batch normalization の有無|Generator に対し batch normalization を適用（出力層は適用外）<br>Descriminator に batch normalization を適用（入力層は適用外）|
|出力層への結合方法|fully connected layer を使用|

- epoch 数 : 2000
![gan_dcgan_1-1_epoch2000](https://user-images.githubusercontent.com/25688193/36040940-5dd1f10a-0e0a-11e8-8d0b-a326009364f6.png)
- epoch 数 : 5000
![gan_dcgan_1-1_epoch5000](https://user-images.githubusercontent.com/25688193/36057482-1be44818-0e52-11e8-8a98-8f09a06fd3b6.png)
- epoch 数 : 20000
![gan_dcgan_1-1 _epoch20000](https://user-images.githubusercontent.com/25688193/36054381-657c04b6-0e39-11e8-943a-9a72293cf1a5.png)

> 損失関数として、疎なソフトマックス・クロス・エントロピー関数を使用した場合の、損失関数のグラフ。
赤線が Generator の損失関数の値。青線が Descriminator の損失関数の値。黒線が、Generator と Descriminator の損失関数の総和。<br>

> 損失関数のグラフより、Generator の損失関数の値が 0 に収束できておらず、又上下に不安定となっていることが分かる。これは GAN に見られる特性である。<br>
つまり、GAN はゲーム理論的に言えば、Generator と Descriminator の２プレイヤーのゼロサムゲームとみなすことが出来るので、相互に干渉しあって動作が不安定になっていると考えられる。<br>

> 又、epoch 数 7500 程度で一旦 Generator の損失関数値が減少した後、その後は、損失関数値が増加傾向になっていることも見て取れる。

<br>

#### Generator から出力された自動生成画像（学習時の途中経過込み）

|パラメータ名|値（実行条件１）|
|---|---|
|最適化アルゴリズム|Adam|
|学習率<br>`learning_rate`|0.0001|
|学習率の減衰項１<br>`beta1`|0.5|
|学習率の減衰項２<br>`beta2`|0.999|
|ミニバッチサイズ<br>`batch_size`|32|
|batch normalization の有無|Generator に対し batch normalization を適用（出力層は適用外）<br>Descriminator に batch normalization を適用（入力層は適用外）|
|出力層への結合方法|fully connected layer を使用|

- 入力ノイズデータ : 32 × 64 pixel<br>
![temp_output_image0](https://user-images.githubusercontent.com/25688193/36032312-ec40f13a-0df0-11e8-8819-68dc1bba41ca.jpg)

- epoch 数 : 50 ~ 5000
![dcgan_fitting_vstack_epoch5000](https://user-images.githubusercontent.com/25688193/36059966-956f0170-0e82-11e8-854c-3cc413e9354b.gif)

- epoch 数 : 50
![temp_output_hstack_image50](https://user-images.githubusercontent.com/25688193/36056472-17722fac-0e48-11e8-92cf-3896928b8504.jpg)
- epoch 数 : 100
![temp_output_hstack_image100](https://user-images.githubusercontent.com/25688193/36056475-19635fac-0e48-11e8-9d0c-9e344270297c.jpg)
- epoch 数 : 200
![temp_output_hstack_image200](https://user-images.githubusercontent.com/25688193/36056477-1c4fe384-0e48-11e8-8794-d86e08cf16f8.jpg)
- epoch 数 : 500
![temp_output_hstack_image500](https://user-images.githubusercontent.com/25688193/36056510-545fef6c-0e48-11e8-9602-4db1ade2cb57.jpg)
- epoch 数 : 2000
![temp_output_hstack_image2000](https://user-images.githubusercontent.com/25688193/36054074-e6d38e5a-0e37-11e8-92aa-eb7945b21d4d.jpg)
- epoch 数 : 5000
![temp_output_hstack_image5000](https://user-images.githubusercontent.com/25688193/36054212-80f8061e-0e38-11e8-88c4-01d2191eef7c.jpg)
- epoch 数 : 7000
![temp_output_hstack_image7000](https://user-images.githubusercontent.com/25688193/36054067-d4661d5a-0e37-11e8-9cb1-b8a8dabe26c5.jpg)
- epoch 数 : 7300
![temp_output_hstack_image7300](https://user-images.githubusercontent.com/25688193/36054254-af260ad6-0e38-11e8-86c8-8b3b9364aa0d.jpg)
- epoch 数 : 7400
![temp_output_hstack_image7400](https://user-images.githubusercontent.com/25688193/36054259-b4cfa5be-0e38-11e8-9feb-215159339a4a.jpg)
- epoch 数 : 7500
![temp_output_hstack_image7500](https://user-images.githubusercontent.com/25688193/36054247-abc5ac84-0e38-11e8-8d66-aeeabf314266.jpg)
- epoch 数 : 10000
![temp_output_hstack_image10000](https://user-images.githubusercontent.com/25688193/36054169-4a22baa8-0e38-11e8-9d3d-52f628182ca3.jpg)

> 縦列が、epoch 数を増加して（学習を進めていった）ときの、Generator から出力された自動生成画像。<br>
> 横列は、Generator に入力した入力ノイズデータの違いによる自動生成画像の違い。<br>

> epoch 数 : 7300 程度までは、順調に epoch数 が増加するにつれ、よりくっきりとした手書き数字の画像が生成出来ているが、<br>
epoch 数 : 7300 程度から突如、手書き数字ライクな画像が生成出来なくなっていることが分かる。
これは、先の損失関数のグラフの、ある epoch 数以上からの Generator の損失関数値の増加傾向と合致している。


<br>

#### 学習済み DCGAN に対し、入力ノイズから画像自動生成

|パラメータ名|値（実行条件１）|
|---|---|
|最適化アルゴリズム|Adam|
|学習率<br>`learning_rate`|0.0001|
|学習率の減衰項１<br>`beta1`|0.5|
|学習率の減衰項２<br>`beta2`|0.999|
|ミニバッチサイズ<br>`batch_size`|32|
|batch normalization の有無|Generator に対し batch normalization を適用（出力層は適用外）<br>Descriminator に batch normalization を適用（入力層は適用外）|
|出力層への結合方法|fully connected layer を使用|

- epoch 数 : 2000<br>
![gan_dcgan_1-2_epoch2000](https://user-images.githubusercontent.com/25688193/36042589-b68d1bda-0e0f-11e8-84ef-ac8cc049c8be.png)
- epoch 数 : 5000<br>
![gan_dcgan_1-2_epoch5000](https://user-images.githubusercontent.com/25688193/36057505-504daa86-0e52-11e8-8727-148df8bb0022.png)

<br>

#### 入力ノイズのパラメータを球の表面上に沿って動かしたときの Generator から出力された自動生成画像

|パラメータ名|値（実行条件１）|
|---|---|
|最適化アルゴリズム|Adam|
|学習率<br>`learning_rate`|0.0001|
|学習率の減衰項１<br>`beta1`|0.5|
|学習率の減衰項２<br>`beta2`|0.999|
|ミニバッチサイズ<br>`batch_size`|32|
|batch normalization の有無|Generator に対し batch normalization を適用（出力層は適用外）<br>Descriminator に batch normalization を適用（入力層は適用外）|
|出力層への結合方法|fully connected layer を使用|

- 入力ノイズデータ : 32 × 64 pixel

- epoch 数 : 500<br>
    - 極座標系 `(theta1, theta2)` : `theta1` 等速、`theta2` 倍速<br>

- epoch 数 : 2000<br>
    - 極座標系 `(theta1, theta2)` : `theta1` 等速、`theta2` 倍速<br>
![dcgan_morphing1_epoch2000-iloveimg-cropped](https://user-images.githubusercontent.com/25688193/36060296-076a8e5a-0e8a-11e8-9194-ab6fcfd72d80.gif)

- epoch 数 : 5000<br>
    - 極座標系 `(theta1, theta2)` : `theta1` 等速、`theta2` 倍速<br>
![dcgan_morphing1_epoch5000](https://user-images.githubusercontent.com/25688193/36060267-816361c4-0e89-11e8-9037-ca945b1aa131.gif)
    - 極座標系 `(theta1, theta2)` : `theta1` 等速、`theta2 = 0`<br>
    - 極座標系 `(theta1, theta2)` : `theta1` 等速、`theta2 = 45π`<br>

---

<a id="ID_3-1-2"></a>

#### コードの内容説明
- DCGAN の Discriminator に入力する学習用データセットとして、MNIST データセットを使用。
    - データは `shape = [n_sample, image_width=28, image_height=28]` の形状に reshape
    - 尚、この処理で取得できる MNIST データの教師データ `y_train`, `y_test` は、
    DCGAN の学習時には使用しない。（教師なしの強化学習のため）
    ```python
    def main():
        ...
        X_train, y_train = MLPreProcess.load_mnist( mnist_path, "train" )
        X_test, y_test = MLPreProcess.load_mnist( mnist_path, "t10k" )

        X_train = np.array( [np.reshape(x, (28,28)) for x in X_train] )
        X_test = np.array( [np.reshape(x, (28,28)) for x in X_test] )
    ```
- DCGAN モデルの各種パラメーターの設定を行う。
    - この設定は、`DeepConvolutionalGAN` クラスのインスタンス作成時の引数にて行う。
    ```python
    [main1.py]
    def main():
        ...
        # DCGAN クラスのオブジェクト生成
        dcgan = DeepConvolutionalGAN(
                    session = tf.Session( config = tf.ConfigProto(log_device_placement=True) ),
                    epochs = epochs,
                    batch_size = batch_size,
                    eval_step = eval_step,
                    image_height = 28,                      # 28 pixel
                    image_width = 28,                       # 28 pixel
                    n_channels = 1,                         # グレースケール
                    n_G_deconv_featuresMap = [128, 64, 1],  # Generator の逆畳み込み層で変換される特徴マップの枚数
                    n_D_conv_featuresMap = [1, 64, 128],    # Descriminator の畳み込み層で変換される特徴マップの枚数
                    n_labels = 2
                )
    ```
- DCGAN のモデルを構築する。
この処理は、`DeepConvolutionalGAN.model()` メソッドで行う。
    ![image](https://user-images.githubusercontent.com/25688193/36060078-037a18f6-0e85-11e8-977e-bc46b49bdc40.png)
    ![image](https://user-images.githubusercontent.com/25688193/36059643-933d4bc6-0e7f-11e8-8cbf-7e52041c8d77.png)
    - まず、Generator に入力する、入力ノイズデータを生成する。
        - このノイズデータは、`tf.random_uniform(...)` を用いて生成した `-1.0f` ~ `1.0f` の間のランダム値をとる Tensor とする。 <br>
        ```python
        [DeepConvolutionalGAN.py]
        def model( self ):
            """
            モデルの定義を行い、
            最終的なモデルの出力のオペレーターを設定する。

            [Output]
                self._y_out_op : Operator
                    モデルの出力のオペレーター
            """
            # 入力データ（ノイズデータ）
            i_depth = self._n_G_deconv_featuresMap[:-1]   # 入力 [Input] 側の layer の特徴マップ数
            z_dim = i_depth[-1]                           # ノイズデータの次数

            input_noize_tsr = tf.random_uniform(
                                  shape = [self._batch_size, z_dim],
                                  minval = -1.0, maxval = 1.0
                              )
        ```
        - 尚、このノイズデータを画像表示したものは、以下の画像のようになる。( 32×64 pixel )<br>
        ![temp_output_image0](https://user-images.githubusercontent.com/25688193/36032312-ec40f13a-0df0-11e8-8819-68dc1bba41ca.jpg)
        - そして、このノイズデータを Generator に入力する。
        ```python
        [DeepConvolutionalGAN.py]
        def model( self ):
            ...
            # Generator : 入力データは, ノイズデータ
            self._G_y_out_op = self.generator( input = input_noize_tsr, reuse = False )
        ```
    - 次に、Generator 側のモデルを構築する。<br>
    この処理は、`DeepConvolutionalGAN.generator(...)` で行う。
        - `DeepConvolutionalGAN.generator(...)` 内部では、まず、引数 `input` で指定された、入力ノイズデータを deconv 層へ入力するために、データの形状を reshape する。
        ```python
        [DeepConvolutionalGAN.py]
        def generator( self, input, reuse = False ):
            """
            GAN の Generator 側のモデルを構築する。

            [Input]
                input : Tensor or placeholder
                    入力ノイズデータの Tensor or 画像データの placeholder
                reuse : bool
                    Variable を共有するか否かのフラグ

            [Output]
                out_G_op : Operator
                    Generator の最終的な出力の Operator
            """
            depths = self._n_G_deconv_featuresMap   # Generator の畳み込み層の特徴マップ数
            f_size = int( self._image_height / 2**(len(depths)-1) )
            i_depth = depths[:-1]                   # 入力 [Input] 側の layer の特徴マップ数
            o_depth = depths[1:]                    # 出力 [Output] 側の layer の特徴マップ数
            z_dim = i_depth[-1]                     # ノイズデータの次数
            
            #---------------------------------------------------------------------
            # 入力データ（ノイズデータ）を deconv 層へ入力するための reshape
            #---------------------------------------------------------------------
            with tf.variable_scope( "Generator", reuse = reuse ):
                # 入力データ → Generator の deconv 層 への重み
                weight0 = self.init_weight_variable( 
                              input_shape = [ z_dim, i_depth[0] * f_size * f_size] 
                          )
            
                # 入力データ → Generator の deconv 層 へのバイアス項
                bias0 = self.init_bias_variable( input_shape = [ i_depth[0] ] )
            
                # weight, bias を list にpush
                if( reuse == False):
                    self._weights.append( weight0 )
                    self._biases.append( bias0 )

                tmp_op = tf.matmul( input, weight0 )
                dc0_op = tf.reshape( tmp_op, [-1, f_size, f_size, i_depth[0]] ) + bias0
        ```
        - 次に、これを batch normalization で正規化し、Relu 出力する。
        ```python
        [DeepConvolutionalGAN.py]
        def generator( self, input, reuse = False ):
            ...
            # batch normarization（ミニバッチごとに平均が0,分散が1）
            # tf.nn.moments(...) : 平均と分散を計算
            # axes = [0, 1, 2] でチャンネル毎の平均と分散を計算
            mean0_op, variance0_op = tf.nn.moments( dc0_op, axes = [0, 1, 2] )
            bn0_op = tf.nn.batch_normalization( dc0_op, mean0_op, variance0_op, None, None, 1e-5 )
            out_G_op = tf.nn.relu( bn0_op )
        ``` 
        - 次に、Generator の特徴マップ数 `self._n_G_deconv_featuresMap` に応じた、
        逆畳み込み層 deconv を構築する。
        ```python
        [DeepConvolutionalGAN.py]
        def generator( self, input, reuse = False ):
            ...
            with tf.variable_scope( "Generator", reuse = reuse ):
                ...
                #---------------------------------------------------------------------
                # DeConvolution layers
                #---------------------------------------------------------------------
                for layer in range( len(self._n_G_deconv_featuresMap)-1 ):
                    with tf.variable_scope( "DeConvLayer_{}".format(layer) ):
                        # layer 番目の畳み込み層の重み（カーネル）
                        # この重みは、畳み込み処理の画像データに対するフィルタ処理（特徴マップ生成）に使うカーネルを表す Tensor のことである。
                        weight = self.init_weight_variable(
                                     input_shape = [ 
                                         5, 5,                              # kernel 行列（フィルタ行列のサイズ） 
                                         o_depth[layer], i_depth[layer]     # tf.nn.conv2d_transpose(...) の filter なので、Output, Input の形状
                                    ]
                                 )
                    
                        # 畳み込み層のバイアス
                        bias = self.init_bias_variable( input_shape = [ o_depth[layer] ] )

                        # weight, bias を list にpush
                        if( reuse == False):
                            self._weights.append( weight )
                            self._biases.append( bias )

                        # deconv
                        dc_op = tf.nn.conv2d_transpose(
                                    value = out_G_op,
                                    filter = weight,            # 畳込み処理で value で指定した Tensor との積和に使用する filter 行列（カーネル）
                                    output_shape = [self._batch_size, f_size*2**(layer+1), f_size*2**(layer+1), o_depth[layer]],    # ?
                                    strides = [1, 2, 2, 1]      # strides[0] = strides[3] = 1. とする必要がある
                                )

                        out_G_op = tf.nn.bias_add( dc_op, bias )

        ```
        - 次に、これをミニバッチサイズに対し、batch normalization で正規化し、Relu 出力する。
        但し、出力層へは batch normalization は適用せず、そのまま線形出力する。
        ```python
        [DeepConvolutionalGAN.py]
        def generator( self, input, reuse = False ):
            ...
            # batch normarization
            # 出力層でない場合 batch normarization を実施
            if( layer < ( len(self._n_G_deconv_featuresMap) - 2 ) ):
                mean_op, variance_op = tf.nn.moments( out_G_op, axes = [0, 1, 2] )
                bn_op = tf.nn.batch_normalization( out_G_op, mean_op, variance_op, None, None, 1e-5 )
                out_G_op = tf.nn.relu( bn_op )
        ```
        - 最後に、シグモイド関数 `tf.nn.sigmoid(...)` で活性化して、Generator の最終的な出力とする。
        ```python
        [DeepConvolutionalGAN.py]
        def generator( self, input, reuse = False ):
            ...
            out_G_op = tf.nn.sigmoid( out_G_op )
        ```
    - 次に、Descriminator 側のモデルを構築する。
    この処理は、`DeepConvolutionalGAN.discriminator(...)` で行う。
        - まず、Generator からの出力 `self._G_y_out_op`、及び、学習用画像データを Descriminator に入力する。
        ```python
        [DeepConvolutionalGAN.py]
        def model( self ):
            ...
            # Descriminator : 入力データは, Generator の出力
            self._D_y_out_op1 = self.discriminator( input = self._G_y_out_op, reuse = False )
        
            # Descriminator : 入力データは, 画像データ
            self._D_y_out_op2 = self.discriminator( input = self._image_holder, reuse = True )
        ```
        - 次に、引数 `input` で指定された、Generator からの出力（フェイク画像データ）、或いは、学習用画像データに対し、Descriminator の特徴マップ数 `self._n_D_deconv_featuresMap` に応じた、畳み込み層 conv を構築する。
        ```python
        [DeepConvolutionalGAN.py]
        def discriminator( self, input, reuse = False ):
            """
            GAN の Discriminator 側のモデルを構築する。

            [Input]
                input : Operator or placeholder
                    Generator の出力の Operator or 画像データの placeholder
                reuse : bool
                    Variable を共有するか否かのフラグ

            [Output]
                self._D_y_out_op : Operator
                    Descriminator の最終的な出力の Operator
            """
            depths = self._n_D_conv_featuresMap     # Descriminator の畳み込み層の特徴マップ数
            i_depth = depths[:-1]                   # 入力 [Input] 側の layer の特徴マップ数
            o_depth = depths[1:]                    # 出力 [Output] 側の layer の特徴マップ数

            with tf.variable_scope( "Descriminator", reuse = reuse ):
                out_D_op = input            # 最初の入力は、Generator の出力

                #----------------------------------------
                # conv layer
                #----------------------------------------
                for layer in range( len(depths) - 1 ):
                    with tf.variable_scope( "ConvLayer_{}".format(layer) ):
                        # layer 番目の畳み込み層の重み（カーネル）
                        # この重みは、畳み込み処理の画像データに対するフィルタ処理（特徴マップ生成）に使うカーネルを表す Tensor のことである。
                        weight = self.init_weight_variable(
                                     input_shape = [ 
                                         5, 5,                              # kernel 行列（フィルタ行列のサイズ） 
                                         i_depth[layer], o_depth[layer]     # tf.nn.conv2d(...) の filter なので、Input, Output の形状
                                     ]
                                 )

                        # 畳み込み層のバイアス項
                        bias = self.init_bias_variable( input_shape = [ o_depth[layer] ] )

                        # weight, bias を list にpush
                        if( reuse == False):
                            self._weights.append( weight )
                            self._biases.append( bias )

                        # conv
                        conv_op = tf.nn.conv2d(
                                      input = out_D_op,         # layer = 0 : Generator の出力 or 入力画像データ, layer = 1~ : 前回の出力
                                      filter = weight,          # 畳込み処理で input で指定した Tensor との積和に使用する filter 行列（カーネル）
                                      strides = [1, 2, 2, 1],   # strides[0] = strides[3] = 1. とする必要がある
                                    padding='SAME'            # ゼロパディングを利用する場合はSAMEを指定
                            )

                        out_D_op = tf.nn.bias_add( conv_op, bias = bias )

        ```
        - 次に、最後の conv 層からの出力を、ミニバッチサイズに対し、batch normalization で正規化し、Leaky ReLu で出力する。
        ```python
        [DeepConvolutionalGAN.py]
        def discriminator( self, input, reuse = False ):
            ...
            with tf.variable_scope( "Descriminator", reuse = reuse ):
                ...
                # batch normalization
                mean_op, variance_op = tf.nn.moments( out_D_op, [0, 1, 2] )
                bn_op = tf.nn.batch_normalization( out_D_op, mean_op, variance_op, None, None, 1e-5 )

                # Leaky ReLu
                out_D_op = tf.maximum( 0.2 * bn_op, bn_op )
        ```
        - conv 層からの出力を平坦化し、全結合層 [fully connected layer] として出力層へ結合し、それらを線形活性したものを Descriminator の最終的な出力とする。
        ```python
        [DeepConvolutionalGAN.py]
        def discriminator( self, input, reuse = False ):
            ...
            #----------------------------------------
            # reshape & fully connected layer
            #----------------------------------------
            with tf.variable_scope( "flatten_fully" ):
                shape = out_D_op.get_shape().as_list()
                dim = shape[1]*shape[2]*shape[3]
                #print( "shape :", shape )
                #print( "dim :", dim )

                # 一列に平坦化 
                out_flatten = tf.reshape( out_D_op, shape = [-1, dim] )

                # 出力ノード
                # flatten layer → outout layer への重み
                weight = self.init_weight_variable( input_shape = [ dim, self._n_labels ] )

                # flatten layer → outout layer へのバイアス項
                bias = self.init_bias_variable( input_shape = [ self._n_labels ] ) 

                # weight, bias を list にpush
                if( reuse == False):
                    self._weights.append( weight )
                    self._biases.append( bias )

                out_D_op = tf.matmul( out_flatten, weight ) + bias
        ```
        - 尚、DCGAN では、fully connected layer ではなく、GAP [global average pooling] のほうが汎化性能が良くなるとされているようだが、ここでは通常の fully connected layer で Descriminator を実装している。
- 損失関数を定義する。
    - 損失関数を、以下の DCGAN での損失関数の更新アルゴリズムに従って、定義する。
    ![image](https://user-images.githubusercontent.com/25688193/36006479-89695612-0d80-11e8-8937-6c4c9d8ef14f.png)
    ![image](https://user-images.githubusercontent.com/25688193/36006524-cbc8eeaa-0d80-11e8-872c-2f5927e121b2.png)
    - まず、Descriminator 側の損失関数を疎なソフトマックスクロスエントロピーで定義する。
    ```python
    [DeepConvolutionalGAN.py]
    def loss( self ):
        """
        損失関数の定義を行う。
        
        [Input]
            
        [Output]
            self._loss_op : Operator
                損失関数を表すオペレーター
        """
        # Descriminator の損失関数
        loss_D_op1 = SparseSoftmaxCrossEntropy().loss(
                         t_holder = tf.zeros( [self._batch_size], dtype=tf.int64 ),      # log{ D(x) } (D(x) = discriminator が 学習用データ x を生成する確率)
                         y_out_op = self._D_y_out_op1                                    # generator が出力する fake data を入力したときの discriminator の出力
                     )
        loss_D_op2 = SparseSoftmaxCrossEntropy().loss( 
                         t_holder = tf.ones( [self._batch_size], dtype = tf.int64 ),     # log{ 1 - D(x) } (D(x) = discriminator が 学習用データ x を生成する確率) 
                         y_out_op = self._D_y_out_op2                                    # generator が出力する fake data を入力したときの discriminator の出力
                     )
        self._D_loss_op =  loss_D_op1 + loss_D_op2
    ```
    - 次に、Generator 側の損失関数を疎なソフトマックスクロスエントロピーで定義する。
    ```python
    [DeepConvolutionalGAN.py]
    def loss( self ):
        ...
        # Generator の損失関数
        self._G_loss_op = SparseSoftmaxCrossEntropy().loss( 
                              t_holder = tf.ones( [self._batch_size], dtype = tf.int64 ),   # log{ 1 - D(x) } (D(x) = discriminator が 学習用データ x を生成する確率)
                              y_out_op = self._D_y_out_op1                                  # generator が出力する fake data を入力したときの discriminator の出力
                          )
    ```
- 最適化アルゴリズム Optimizer として、
Generator, Descriminator 双方とも Adam アルゴリズム を使用する。
    - 学習率 `learning_rate` は、0.001 で検証。減衰項は `beta1 = 0.5`, `beta1 = 0.999`
    ```python
    [main1.py]
    def main():
        ...
        dcgan.optimizer( 
            nnOptimizerG = Adam( learning_rate = learning_rate, beta1 = beta1, beta2 = beta2 ),
            nnOptimizerD = Adam( learning_rate = learning_rate, beta1 = beta1, beta2 = beta2 )
        )
    ```
    ```python
    [DeepConvolutionalGAN.py]
    def optimizer( self, nnOptimizerG, nnOptimizerD ):
        """
        モデルの最適化アルゴリズムの設定を行う。
        [Input]
            nnOptimizerG : NNOptimizer のクラスのオブジェクト
                Generator 側の Optimizer

            nnOptimizerD : NNOptimizer のクラスのオブジェクト
                Descriminator 側の Optimizer

        [Output]
            optimizer の train_step
        """
        # Generator, Discriminator の Variable の抽出
        g_vars = [ var for var in tf.trainable_variables() if var.name.startswith('G') ]
        d_vars = [ var for var in tf.trainable_variables() if var.name.startswith('D') ]

        # Optimizer の設定
        self._G_optimizer = nnOptimizerG._optimizer
        self._D_optimizer = nnOptimizerD._optimizer
        
        # トレーニングステップの設定
        self._G_train_step = self._G_optimizer.minimize( self._G_loss_op, var_list = g_vars )
        self._D_train_step = self._D_optimizer.minimize( self._D_loss_op, var_list = d_vars )

        # tf.control_dependencies(...) : sess.run で実行する際のトレーニングステップの依存関係（順序）を定義
        with tf.control_dependencies( [self._G_train_step, self._D_train_step] ):
            # tf.no_op(...) : 何もしない Operator を返す。（トレーニングの依存関係を定義するのに使用）
            self._train_step = tf.no_op( name = 'train' )
            print( "_train_step", self._train_step )
        
        return self._train_step
    ```
- トレーニング用データ `X_train` に対し、fitting 処理を行う。
    ```python
    [main1.py]
    def main():
        ...
        dcgan.fit( X_train, y_train = None )
    ```
- 尚、このモデルの TensorBorad で描写した計算グラフは以下のようになる。
![graph_large_attrs_key _too_large_attrs limit_attr_size 1024 run 9](https://user-images.githubusercontent.com/25688193/36034906-dba14b88-0df8-11e8-9016-f846c3289401.png)
![graph_large_attrs_key _too_large_attrs limit_attr_size 1024 run 10](https://user-images.githubusercontent.com/25688193/36034907-dbcdf1c4-0df8-11e8-8ae5-86c64f2f649c.png)
![graph_large_attrs_key _too_large_attrs limit_attr_size 1024 run 11](https://user-images.githubusercontent.com/25688193/36034909-dbf6a966-0df8-11e8-916c-eeb0dff6c436.png)

<br>

---

<a id="ID_4"></a>

## 背景理論

<a id="ID_10"></a>

## 生成モデル [generative model]
![image](https://user-images.githubusercontent.com/25688193/35478872-4302b400-042c-11e8-80aa-a187b50eba30.png)

<a id="ID_10-1"></a>

### GAN [Generative Adversarial Network]（敵対的ネットワーク）
- 元論文「Generative Adversarial Nets」
    - arXiv.org : https://arxiv.org/abs/1406.2661
- 参考サイト
    - https://elix-tech.github.io/ja/2017/02/06/gan.html
    - http://mizti.hatenablog.com/entry/2016/12/10/224426
    - http://vaaaaaanquish.hatenablog.com/entry/2017/03/19/220817
    - http://yasuke.hatenablog.com/entry/generative-adversarial-nets

![image](https://user-images.githubusercontent.com/25688193/35478891-ac0a5494-042c-11e8-8781-39c88431fe8f.png)
![image](https://user-images.githubusercontent.com/25688193/35481685-c432d534-046b-11e8-954c-f9b88f5a07fb.png)
![image](https://user-images.githubusercontent.com/25688193/35481115-7b76b87a-0460-11e8-9f3f-293e6afdba22.png)
![image](https://user-images.githubusercontent.com/25688193/35488656-2b95c91c-04cf-11e8-8d06-67ea71c58a72.png)


<a id="ID_10-1-1"></a>

#### DCGAN [Deep Convolutional GAN]
- 元論文「Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks」
    - arXiv.org : https://arxiv.org/abs/1511.06434

![image](https://user-images.githubusercontent.com/25688193/35545399-50f2a4bc-05b2-11e8-853e-11d38971630f.png)
![image](https://user-images.githubusercontent.com/25688193/35545437-72ebb95a-05b2-11e8-9219-e723ee344d54.png)
![image](https://user-images.githubusercontent.com/25688193/36059567-efac113c-0e7d-11e8-8bdd-329fbc70808a.png)
![image](https://user-images.githubusercontent.com/25688193/35549375-93ea4836-05c8-11e8-8279-a8d3d3a659c6.png)
![image](https://user-images.githubusercontent.com/25688193/35545532-cd39d9d2-05b2-11e8-9ab9-a3f4123ab8fd.png)
![image](https://user-images.githubusercontent.com/25688193/35545809-5d14a248-05b4-11e8-854e-caf830ef2972.png)
![image](https://user-images.githubusercontent.com/25688193/35549398-b4a58dce-05c8-11e8-9bd5-883c03aa4564.png)



### デバッグメモ
