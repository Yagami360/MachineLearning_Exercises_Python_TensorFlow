## TensorFlow を用いた CNN-StyleNet / NeuralStyle による画像生成の実装

TensorFlow を用いた CNN-StyleNet / NeuralStyle による画像生成の練習用実装コード。<br>

この README.md ファイルには、各コードの実行結果、概要、CNN の背景理論の説明を記載しています。<br>
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
    
### 項目 [Contents]

1. [使用するライブラリ](#ID_1)
1. [使用するデータセット](#ID_2)
1. [コードの説明＆実行結果](#ID_3)
    1. [CNN-StyleNet / NeuralStyle（ニューラルスタイル）による画像生成処理 : `main1.py`](#ID_3-1)
1. [背景理論](#ID_4)


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

- 学習済み CNN モデルのデータ : MATLAB オブジェクトファイル
    - [imagenet-vgg-verydee-19.mat]( http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat)

> 

<a id="ID_3"></a>

## コードの説明＆実行結果

<a id="ID_3-1"></a>

## CNN-StyleNet / NeuralStyle（ニューラルスタイル）による画像生成処理 : `main1.py`

- まずは、学習済みの CNN モデルのデータ `imagenet-vgg-verydee-19.mat` を読み込む。
    - 学習済み CNN モデルの重み＋バイアス項を含んだ `network_weights` と
    画像を正規化するための正規化行列 `norm_mean_matrix` を取り出す。
- 内容画像層のモデルを構築する。
    - `_vgg_layers` を構成する layer から layer を取り出し、
    種類に応じて、モデルを具体的に構築していく。
    - 内容画像の行列を正規化する。
        - 内容画像行列の正規化には、
        先に読み込んだ、画像を正規化するための正規化行列 `norm_mean_matrix` を使用する。
        > CNNStyleNet.py
        ```python
        # 内容画像の行列を正規化
        content_minus_mean_matrix = self._image_content - norm_mean_matrix
        content_norm_matrix = np.array( [content_minus_mean_matrix] )
        ```
    - 構築した 内容画像層のモデルを `session.run(...)` し、<br>
    学習済み CNN モデルから、画像の内容層の特徴量を抽出する。
        - xxx
        ```python
        self._features_content[ self._content_layer ] =\
            self._session.run( 
                network_content[ self._content_layer ], 
                feed_dict = { self._image_content_holder : content_norm_matrix } 
            )
        ```
- スタイル画像層のモデルを構築する。
    - `_vgg_layers` を構成する layer から layer を取り出し、
    種類に応じて、モデルを具体的に構築していく。
    - スタイル画像の行列を正規化する。
        - スタイル画像行列の正規化には、
        先に読み込んだ、画像を正規化するための正規化行列 `norm_mean_matrix` を使用する。
    - 学習済み CNN モデルから、画像のスタイル層の特徴量を抽出する。
        - xxx
- 画像の内容層とスタイル層をノイズ付き合成するためのモデルの構築する。
    - **ここで構築したモデル（Variable）が、StyleNet のトレーニング対象となる。**
    - `_vgg_layers` を構成する layer から layer を取り出し、
    種類に応じて、モデルを具体的に構築していく。
- モデルの損失関数を設定する。
    - 画像の内容層の損失関数を設定する。
    - 画像のスタイル層の損失関数を設定する。
    - 内容層とスタイル層のノイズ付き合成加工の際に、滑らかな結果を得るために、全変動損失関数なるものを設定する。
- ノイズ合成加工モデルに対しての、最適化アルゴリズム（Optimizer）を設定する。
- 設定したトレーニングステップ `_train_step` に対し、
`_session.run( _train_step )` し、トレーニングを実施していく。
    - このとき、一定のステップ回数 `_eval_step` 度に、逐次生成画像を出力する。
<!--
    - ここでは、最適化アルゴリズムとして、
    Adam アルゴリズム`tf.train.AdamOptimizer(...)` を使用する。
-->

<br>

### コードの実行結果

- 内容画像<br>
![book_cover](https://user-images.githubusercontent.com/25688193/33214839-e18e0fdc-d170-11e7-9860-fb31a6dcbf9e.jpg)

- スタイル画像<br>
![starry_night](https://user-images.githubusercontent.com/25688193/33214900-064aa77c-d171-11e7-9f4a-00220ac4d9a2.jpg)

- 生成画像<br>
    - エポック数：50 での生成画像<br>
        - 正規化あり
![temp_output_add_mean_image50](https://user-images.githubusercontent.com/25688193/33220246-0550d0d0-d18a-11e7-8650-e6e32b378b67.jpg)
        - 正規化なし
![temp_output_image50](https://user-images.githubusercontent.com/25688193/33220248-05e8f374-d18a-11e7-9ea0-8afb8956ee7c.jpg)
    - エポック数：100 での生成画像<br>
        - 正規化あり
![temp_output_add_mean_image100](https://user-images.githubusercontent.com/25688193/33220247-05b17692-d18a-11e7-8da3-672da5f8e56a.jpg)
        - 正規化なし
![temp_output_image100](https://user-images.githubusercontent.com/25688193/33220249-062d0636-d18a-11e7-8e67-6868ffddbb9e.jpg)
    - エポック数：200 での生成画像<br>
    > 処理中...
    - 最終生成画像<br>
    > 処理中...

    - エポック数：500 での生成画像<br>
    > 処理中...
    - 最終生成画像<br>
    > 処理中...

    - エポック数：1000 での生成画像<br>
    > 処理中...
    - 最終生成画像<br>
    > 処理中...


<br>

---

<a id="ID_4"></a>

## 背景理論


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