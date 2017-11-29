## TensorFlow を用いた CNN-StyleNet / NeuralStyle による画像生成の実装

TensorFlow を用いた CNN-StyleNet / NeuralStyle による画像生成の練習用実装コード。<br>

この README.md ファイルには、各コードの説明＆実行結果、概要の説明を記載しています。<br>
<!--CNN の背景理論 -->

分かりやすいように `main.py` ファイル毎に１つの完結した実行コードにしています。

StyleNet / NeuralStyle（ニューラルスタイル）は、１つ目の画像から「画像スタイル」を学習し、２つ目の画像の「構造（内容）を維持」した上で、１つ目の画像スタイルを２つ目の画像に適用可能な手法である。

これは、一部の CNN に「２種類の中間層」が存在するという特性に基いている。<br>
この２種類の中間層とは、「画像スタイルをエンコード（符号化）するような中間層」と「画像内容をエンコード（符号化）するような中間層」である。<br>
この２つの中間層（それぞれ、スタイル層、内容層と名付ける）に対して、スタイル画像と内容画像でトレーニングし、従来のニューラルネットワークと同様にして、損失関数値をバックプロパゲーションすれば、２つの画像を合成した新たな画像を生成出来る。<br>
そして、この CNN モデルの構築には、事前に学習された CNN モデルを使用する。

- 元論文「A Neural Algorithm of Artistic Style」
    - https://arxiv.org/abs/1508.06576

- 参考サイト
    - https://research.preferred.jp/2015/09/chainer-gogh/
    - https://elix-tech.github.io/ja/2016/08/22/art.html


### 項目 [Contents]

1. [使用するライブラリ](#ID_1)
1. [使用するデータセット](#ID_2)
1. [コードの説明＆実行結果](#ID_3)
    1. [CNN-StyleNet / NeuralStyle（ニューラルスタイル）による画像生成処理 : `main1.py`](#ID_3-1)
<!--
1. [背景理論](#ID_4)
-->

<a id="ID_1"></a>

## 使用するライブラリ

> TensorFlow ライブラリ <br>
>> `tf.reduce_prod(...)` : 積の操作で縮約
>>> https://www.tensorflow.org/api_docs/python/tf/reduce_prod

> Scipy ライブラリ
>> `scipy.misc.imread(...)` : Read an image from a file as an array.<br>
>>> https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.imread.html<br>
>> `scipy.io.loadmat` : MATLAB 用のオブジェクトファイルを読み込む<br>
>>> https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.io.loadmat.html

> その他ライブラリ
>>


<a id="ID_2"></a>

### 使用するデータセット

- 学習済み CNN モデルのデータ : VGG-19
    - [imagenet-vgg-verydee-19.mat]( http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat)
    - https://jp.mathworks.com/help/nnet/ref/vgg19.html
    - https://qiita.com/TokyoMickey/items/a6bbf62571dd79730052<br>
> 合計 19 層からなる CNN モデルの学習済みデータ。MATLAB 形式の mat ファイル。

<a id="ID_3"></a>

## コードの説明＆実行結果

<a id="ID_3-1"></a>

## CNN-StyleNet / NeuralStyle（ニューラルスタイル）による画像生成処理 : `main1.py`

![vgg19_stylenet](https://user-images.githubusercontent.com/25688193/33375981-3bff8b1c-d54f-11e7-8146-36687c224c27.png)

上図で示した 19 層の学習済み CNN モデル（VGG-19）を用いた、StyleNet による内容画像とスタイル画像の自動合成画像処理。<br>
コンテンツ画像層には、VGG-19 の１つの conv → relu 出力の層を割り当て、
スタイル層には、複数の conv → Relu 出力の層を割り当てることで、それぞれ内容とスタイルの（特徴的、抽象的な）特徴量を抽出している。

- まずは、学習済みの CNN モデルのデータ `imagenet-vgg-verydee-19.mat` を読み込み、
    - 学習済み CNN モデルの重み＋バイアス項を含んだ `network_weights` と
    画像を正規化するための正規化行列 `norm_mean_matrix` を取り出す。
    ```python
    [CNNStyleNet.py]
    def load_model_info( self, mat_file_path ):
        """
        学習済みの StyleNet モデル用のデータである mat データから、パラメータを読み込み。
        imagenet-vgg-verydee-19.mat : この mat データは、MATLAB オブジェクトが含まれている
        [Input]
            mat_file_path : str
                imagenet-vgg-verydee-19.mat ファイルの path
        
        [Output]
            matrix_mean :
                画像を正規化するための正規化行列
                [ 123.68   116.779  103.939]
            network_weight :
                学習済み CNN モデルの重み、バイアス項等を含んだ MATLAB データ
        """
        vgg_data = scipy.io.loadmat( mat_file_path )

        normalization_matrix = vgg_data[ "normalization" ][0][0][0]

        # matrix の row , colum に対しての平均値をとった matrix
        matrix_mean = np.mean( normalization_matrix, axis = (0,1) )

        # モデルの重み
        network_weight = vgg_data[ "layers" ][0]

        return ( matrix_mean, network_weight )
    ```
- 内容（コンテンツ）画像層のモデルを構築する。
    - 上図の VGG-19 モデルの `vgg_layers` を構成する layer から、<br>
    `contents_layer = "relu4_2"` に該当する layer を取り出し、種類に応じて、モデルを具体的 `network_content` に構築していく。
    ```python
    [CNNStyleNet.py]
    def model():
        ...
        for ( i, layer ) in enumerate( self._vgg_layers ):
            # layer "convx_x" の先頭の文字列が畳み込み層を表す "c" の場合 
            if ( layer[0] == "c" ):
                # network_weights から weights とバイアス項に対応するデータを抽出
                weights, bias = network_weights[i][0][0][0][0]
                #print( "weights :\n", weights )
                #print( "bias :\n", bias )
                
                # StyleNet モデルに対応するように reshape
                weights = np.transpose( weights, (1,0,2,3) )
                bias = bias.reshape(-1)

                # 畳み込み層を構築
                conv_layer_op = \
                    tf.nn.conv2d(
                        input = image_content_tsr,
                        filter = tf.constant( weights ),                       # 畳込み処理で input で指定した Tensor との積和に使用する filter 行列（カーネル）
                        strides = [ 1, self._n_strides, self._n_strides, 1 ],  # strides[0] = strides[3] = 1. とする必要がある]
                        padding = "SAME"                                       # ゼロパディングを利用する場合は SAME を指定
                    )

                image_content_tsr = tf.nn.bias_add( conv_layer_op, bias )
                
                # リストに追加しておく
                self._weights.append( tf.constant( weights ) )
                self._biases.append( bias )

            # layer "relux_x" の先頭の文字列が Relu を表す "r" の場合 
            elif ( layer[0] == "r" ):
                image_content_tsr = tf.nn.relu( image_content_tsr )

            # layer "pool_x" の先頭の文字列がプーリング層を表す "p" の場合 
            else:
                image_content_tsr = \
                    tf.nn.max_pool(
                        value = image_content_tsr,
                        ksize = [ 1, self._n_pool_wndsize, self._n_pool_wndsize, 1 ],    # プーリングする範囲（ウィンドウ）のサイズ
                        strides = [ 1, self._n_pool_strides, self._n_pool_strides, 1 ],  # ストライドサイズ strides[0] = strides[3] = 1. とする必要がある
                        padding = "SAME"                                                 # ゼロパディングを利用する場合は SAME を指定
                    )
            
            network_content[ layer ] = image_content_tsr
    ```
    - 内容画像の行列を正規化する。
        - 内容画像行列の正規化には、
        先に読み込んだ、画像を正規化するための正規化行列 `norm_mean_matrix` を使用する。
        ```python
        [CNNStyleNet.py]
        def model():
            ...
            # 内容画像の行列を正規化
            content_minus_mean_matrix = self._image_content - norm_mean_matrix
            content_norm_matrix = np.array( [content_minus_mean_matrix] )
        ```
    - 構築した 内容画像層のモデルを `session.run(...)` し、<br>
    学習済み CNN モデルから、画像の内容層の特徴量 `_features_content` を抽出する。
        ```python
        [CNNStyleNet.py]
        def model():
            ...
            self._features_content[ self._content_layer ] =\
                self._session.run( 
                    network_content[ self._content_layer ], 
                    feed_dict = { self._image_content_holder : content_norm_matrix } 
                )
        ```
- スタイル画像層のモデルを構築する。
    - 上図の VGG-19 モデルの `vgg_layers` を構成する layer から、<br>
    `style_layers = { "relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1" }` に該当するlayer を取り出し、種類に応じて、モデルを具体的に構築していく。
    - スタイル画像の行列を正規化する。
        - スタイル画像行列の正規化には、
        先に読み込んだ、画像を正規化するための正規化行列 `norm_mean_matrix` を使用する。
    - 構築した スタイル画像層のモデルを `session.run(...)` し、<br>
    学習済み CNN モデルから、画像の内容層の特徴量 `_features_style` を抽出する。
        ```python
        [CNNStyleNet.py]
        def model():
            ...
            for layer in self._style_layers:
                layer_output =\
                    self._session.run( 
                        network_style[ layer ], 
                        feed_dict = { self._image_style_holder : style_norm_matrix } 
                    )

                layer_output = np.reshape( layer_output, ( -1, layer_output.shape[3] ) )

                # グラム行列 A^T * A
                style_gram_matrix = np.matmul( layer_output.T, layer_output ) / layer_output.size

                # 特徴量のリストに格納
                self._features_style[ layer ] = style_gram_matrix
        ```
- 画像の内容層とスタイル層をノイズ付き合成するための変数＆モデルの構築する。
    - **ここで構築したモデル（Variable） `_image_var` が、StyleNet のトレーニング対象となる。**
    ```python
    [CNNStyleNet.py]
    def model():
        ...
        self._image_var = tf.Variable(
                              tf.random_normal( shape = (1,) + self._image_content.shape ) * 0.256
                          )
    ```
    - この Variable `_image_var` を VGG-19 モデルの入力とし、モデルを構築する。
        - 詳細には、`_vgg_layers` を構成する layer から layer を取り出し、種類に応じて、モデルを具体的に構築していく。
- モデルの損失関数を設定する。
    - 画像の内容（コンテンツ）層の損失関数を設定する。
    ```python
    [CNNStyleNet.py]
    def loss():
        ...
        self._loss_content_op = \
            self._weight_image_content * \
            ( 2 * tf.nn.l2_loss( 
                      self._vgg_network[self._content_layer] - self._features_content[self._content_layer] 
                  ) / self._features_content[ self._content_layer ].size
            )
    ```
    - 画像のスタイル層の損失関数を設定する。
    ```python
    [CNNStyleNet.py]
    def loss():
        ...
        style_losses = []
        for style_layer in self._style_layers:
            # スタイル層の style_layer 番目のモデルの内容を抽出
            layer = self._vgg_network[ style_layer ]

            #
            feats, height, width, channels = [x.value for x in layer.get_shape()]
            size = height * width * channels
            features = tf.reshape( layer, (-1, channels) )

            style_gram_matrix = tf.matmul( tf.transpose(features), features ) / size
            style_expected = self._features_style[ style_layer ]

            style_losses.append(
                2 * tf.nn.l2_loss( style_gram_matrix - style_expected ) / style_expected.size
            )

        self._loss_style_op = 0
        self._loss_style_op += self._weight_image_style  * tf.reduce_sum( style_losses )
    ```
    - 内容層とスタイル層のノイズ付き合成加工の際に、滑らかな結果を得るために、全変動損失関数なるものを設定する。
    ```python
    [CNNStyleNet.py]
    def loss():
        ...
        # tf.reduce_prod(...) : 積の操作で縮約
        total_var_x = self._session.run( 
            tf.reduce_prod( self._image_var[ :, 1:, :, : ].get_shape() )   #  
        )
        total_var_y = self._session.run(
            tf.reduce_prod( self._image_var[ :, :, 1:, : ].get_shape() )
        )

        # ?
        first_term = self._weight_regularization  * 2
        second_term_numerator = tf.nn.l2_loss(
                                    self._image_var[ :, 1:, :, : ] 
                                    - self._image_var[ :, :( (1,) + self._image_content.shape )[1] - 1, :, : ]
                                )
        second_term = second_term_numerator / total_var_y
        third_term = ( 
                         tf.nn.l2_loss( 
                             self._image_var[ :, :, 1:, : ] 
                             - self._image_var[ :, :, :( (1,) + self._image_content.shape )[2] - 1, : ] 
                         ) / total_var_x 
                     )
        self._loss_total_var_op = first_term * ( second_term + third_term )
    ```
    - 最終的な StyleNet モデルの損失関数は、これら損失関数の和とする。
    ```python
    [CNNStyleNet.py]
    def loss():
        ...
         self._loss_op = self._loss_content_op + self._loss_style_op + self._loss_total_var_op
    ```
- ノイズ合成加工モデルに対しての、最適化アルゴリズム（Optimizer）を設定する。
    - ここでは、最適化アルゴリズムとして、
    Adam アルゴリズム`tf.train.AdamOptimizer(...)` を使用する。
    ```python
    [main1.py]
    def main():
        ...
        styleNet1.optimizer( Adam( learning_rate = learning_rate1, beta1 = adam_beta1, beta2 = adam_beta2 ) )
    ```
- 設定したトレーニングステップ `_train_step` に対し、
`_session.run( _train_step )` し、トレーニングを実施していく。
    - このとき、一定のステップ回数 `_eval_step` 度に、逐次生成画像を出力する。
    ```python
    [CNNStyleNet.py]
    def run():
        ....
        if ( (epoch + 1) % self._eval_step == 0 ):
            ...
            # 途中生成画像の保存
            image_eval = self._session.run( self._image_var )
            image_eval = image_eval.reshape( self._image_content.shape ) + self._norm_mean_matrix
        
            output_file = "output_image/temp_output_image{}.jpg".format( epoch + 1 )
            scipy.misc.imsave( output_file, image_eval )
    ```

<br>

### コードの実行結果

- 内容（コンテンツ）画像<br>
![neko-sensei](https://user-images.githubusercontent.com/25688193/33228168-0bc7598e-d1f8-11e7-8a0e-788ea073154a.jpg)

- スタイル画像<br>
![starry_night](https://user-images.githubusercontent.com/25688193/33214900-064aa77c-d171-11e7-9f4a-00220ac4d9a2.jpg)

- 生成画像<br>
![stylenet_50 1000](https://user-images.githubusercontent.com/25688193/33228657-7e4f2558-d203-11e7-8f13-fdbfabfc5699.gif)

|パラメータ|引数名|値|
|---|---|---|
|最適化アルゴリズム|`optimizer`|Adam アルゴリズム|
|学習率 |`learning_rate`|0.500|
||`beta1`|0.900|
||`beta2`|0.990|
|内容画像のウェイト値|`weight_image_content`|200.0|
|スタイル画像のウェイト値|`weight_image_style`|200.0|
|全変動損失のウェイト値|`weight_regularization`|100.0|
|畳み込み処理のストライド値|`n_strides`|1|
|プーリング処理のウィンドウサイズ|`n_pool_wndsize`|2|
|プーリング処理のストライド値|`n_pool_strides`|2|

- エポック数：1 での生成画像<br>
![temp_output_image1](https://user-images.githubusercontent.com/25688193/33231341-3fc4ea8c-d237-11e7-8dcf-ae7e6753c72e.jpg)

- エポック数：2 での生成画像<br>
![temp_output_image2](https://user-images.githubusercontent.com/25688193/33231342-3feb5744-d237-11e7-9d0a-ced02b3777ed.jpg)
- エポック数：3 での生成画像<br>
![temp_output_image3](https://user-iages.githubusercontent.com/25688193/33231343-40134bfa-d237-11e7-8982-2de4a3b3f4d6.jpg)

- エポック数：50 での生成画像<br>
![temp_output_add_mean_image50](https://user-images.githubusercontent.com/25688193/33228156-d6181eea-d1f7-11e7-9aba-9e213f799dcc.jpg)

- エポック数：100 での生成画像<br>
![temp_output_add_mean_image100](https://user-images.githubusercontent.com/25688193/33228160-e3bfeef6-d1f7-11e7-84df-945b3606c019.jpg)

- エポック数：200 での生成画像<br>
![temp_output_add_mean_image200](https://user-images.githubusercontent.com/25688193/33228183-435ceba2-d1f8-11e7-99b5-5b21ed4e26c9.jpg)

- エポック数：500 での生成画像<br>
![temp_output_add_mean_image500](https://user-images.githubusercontent.com/25688193/33228165-f0f04f3a-d1f7-11e7-84e2-5babcaeb6268.jpg)

- エポック数：1000 での生成画像<br>
![temp_output_add_mean_image1000](https://user-images.githubusercontent.com/25688193/33228650-4fd46ca6-d203-11e7-9280-a90279d9d68b.jpg)

<br>

#### 損失関数のグラフ
![figure_1](https://user-images.githubusercontent.com/25688193/33236202-e11001d6-d28e-11e7-8eb6-9e22ea11ee69.png)
> 黒線が、モデル全体の損失関数の値。エポック数：300 付近で収束している。<br>
> 赤線が、コンテンツ画像層のモデルの損失関数の値。エポック数：300 付近で収束している。<br>
> 青線が、スタイル画像層のモデルの損失関数の値。エポック数：300 付近で収束している。<br>
> 緑線が、全変動損失関数の値。エポック数が増えるにつれ、単調増加している。<br>

<br>

---

<!--
<a id="ID_4"></a>

## 背景理論
> 記載中...
-->

---
## デバッグメモ
[]
ValueError: Cannot feed value of shape (1, 326, 458, 3) for Tensor 'Relu_15:0', which has shape '(1, 21, 29, 512)'

----------------------------------
<CNNStyleNet.CNNStyleNet object at 0x0000019B3E10BDD8>

_session :  <tensorflow.python.client.session.Session object at 0x0000019B3E10BE10>
_init_var_op :  None
_loss_op :  None
_optimizer :  None
_train_step :  None
_y_out_op :  None
_epoches :  250
_eval_step :  1
_image_content_path :  D:\Data\MachineLearning_DataSet\CNN-StyleNet\image_content\book_cover.jpg
_image_style_path :  D:\Data\MachineLearning_DataSet\CNN-StyleNet\image_style\starry_night.jpg
_weight_image_content :  5.0
_weight_image_style :  500.0
_weight_regularization :  100
_vgg_layers : 
 ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4']
_content_layer :  relu4_2
_style_layers :  ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
_image_content_holder :  Tensor("Placeholder:0", shape=(1, 326, 458, 3), dtype=float32)
_image_style_holder :  Tensor("Placeholder_1:0", shape=(1, 362, 458, 3), dtype=float32)
_image_content.shape : 
 (326, 458, 3)
_image_style.shape : 
 (362, 458, 3)
_features_content : {}
_features_style : {}
_n_strides :  1
_n_pool_wndsize :  2
_n_pool_strides :  2
_weights : 
 []
_biases : 
 []
----------------------------------
norm_mean_matrix :
 [ 123.68   116.779  103.939]
network_content :
 {'conv1_1': <tf.Tensor 'BiasAdd:0' shape=(1, 326, 458, 64) dtype=float32>, 'relu1_1': <tf.Tensor 'Relu:0' shape=(1, 326, 458, 64) dtype=float32>, 'conv1_2': <tf.Tensor 'BiasAdd_1:0' shape=(1, 326, 458, 64) dtype=float32>, 'relu1_2': <tf.Tensor 'Relu_1:0' shape=(1, 326, 458, 64) dtype=float32>, 'pool1': <tf.Tensor 'MaxPool:0' shape=(1, 163, 229, 64) dtype=float32>, 'conv2_1': <tf.Tensor 'BiasAdd_2:0' shape=(1, 163, 229, 128) dtype=float32>, 'relu2_1': <tf.Tensor 'Relu_2:0' shape=(1, 163, 229, 128) dtype=float32>, 'conv2_2': <tf.Tensor 'BiasAdd_3:0' shape=(1, 163, 229, 128) dtype=float32>, 'relu2_2': <tf.Tensor 'Relu_3:0' shape=(1, 163, 229, 128) dtype=float32>, 'pool2': <tf.Tensor 'MaxPool_1:0' shape=(1, 82, 115, 128) dtype=float32>, 'conv3_1': <tf.Tensor 'BiasAdd_4:0' shape=(1, 82, 115, 256) dtype=float32>, 'relu3_1': <tf.Tensor 'Relu_4:0' shape=(1, 82, 115, 256) dtype=float32>, 'conv3_2': <tf.Tensor 'BiasAdd_5:0' shape=(1, 82, 115, 256) dtype=float32>, 'relu3_2': <tf.Tensor 'Relu_5:0' shape=(1, 82, 115, 256) dtype=float32>, 'conv3_3': <tf.Tensor 'BiasAdd_6:0' shape=(1, 82, 115, 256) dtype=float32>, 'relu3_3': <tf.Tensor 'Relu_6:0' shape=(1, 82, 115, 256) dtype=float32>, 'conv3_4': <tf.Tensor 'BiasAdd_7:0' shape=(1, 82, 115, 256) dtype=float32>, 'relu3_4': <tf.Tensor 'Relu_7:0' shape=(1, 82, 115, 256) dtype=float32>, 'pool3': <tf.Tensor 'MaxPool_2:0' shape=(1, 41, 58, 256) dtype=float32>, 'conv4_1': <tf.Tensor 'BiasAdd_8:0' shape=(1, 41, 58, 512) dtype=float32>, 'relu4_1': <tf.Tensor 'Relu_8:0' shape=(1, 41, 58, 512) dtype=float32>, 'conv4_2': <tf.Tensor 'BiasAdd_9:0' shape=(1, 41, 58, 512) dtype=float32>, 'relu4_2': <tf.Tensor 'Relu_9:0' shape=(1, 41, 58, 512) dtype=float32>, 'conv4_3': <tf.Tensor 'BiasAdd_10:0' shape=(1, 41, 58, 512) dtype=float32>, 'relu4_3': <tf.Tensor 'Relu_10:0' shape=(1, 41, 58, 512) dtype=float32>, 'conv4_4': <tf.Tensor 'BiasAdd_11:0' shape=(1, 41, 58, 512) dtype=float32>, 'relu4_4': <tf.Tensor 'Relu_11:0' shape=(1, 41, 58, 512) dtype=float32>, 'pool4': <tf.Tensor 'MaxPool_3:0' shape=(1, 21, 29, 512) dtype=float32>, 'conv5_1': <tf.Tensor 'BiasAdd_12:0' shape=(1, 21, 29, 512) dtype=float32>, 'relu5_1': <tf.Tensor 'Relu_12:0' shape=(1, 21, 29, 512) dtype=float32>, 'conv5_2': <tf.Tensor 'BiasAdd_13:0' shape=(1, 21, 29, 512) dtype=float32>, 'relu5_2': <tf.Tensor 'Relu_13:0' shape=(1, 21, 29, 512) dtype=float32>, 'conv5_3': <tf.Tensor 'BiasAdd_14:0' shape=(1, 21, 29, 512) dtype=float32>, 'relu5_3': <tf.Tensor 'Relu_14:0' shape=(1, 21, 29, 512) dtype=float32>, 'conv5_4': <tf.Tensor 'BiasAdd_15:0' shape=(1, 21, 29, 512) dtype=float32>, 'relu5_4': <tf.Tensor 'Relu_15:0' shape=(1, 21, 29, 512) dtype=float32>}
_image_content_holder :
 Tensor("Relu_15:0", shape=(1, 21, 29, 512), dtype=float32)
content_minus_mean_matrix :
 (326, 458, 3)
content_norm_matrix.shape :
 (1, 326, 458, 3)
ValueError: Cannot feed value of shape (1, 326, 458, 3) for Tensor 'Relu_15:0', which has shape '(1, 21, 29, 512)'