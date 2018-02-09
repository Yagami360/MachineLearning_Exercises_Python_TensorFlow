## TensorFlow での DCGAN [Deep Convolutional GAN] の実装

TensorFlow での DCGAN [Deep Convolutional GAN] の練習用実装コード集。

この README.md ファイルには、各コードの実行結果、概要、DCGAN の背景理論の説明を記載しています。<br>
分かりやすいように `main.py` ファイル毎に１つの完結した実行コードにしています。

### 項目 [Contents]

1. [使用するライブラリ](#ID_1)
1. [使用するデータセット](#ID_2)
1. [コード説明＆実行結果](#ID_3)
    1. [DCGAN による手書き数字画像データ MNIST の自動生成 : `main1.py`](#ID_3-1)
    1. [](#)
1. [背景理論](#ID_4)
    1. [生成モデル [generative model]](#ID_10)
        1. [GAN [Generative Adversarial Network]（敵対的ネットワーク）](#ID_10-1)
            1. [DCGAN [Deep Convolutional GAN]](#ID_10-1-1)
        1. VAE [Variational Autoencoder]


<a id="ID_1"></a>

### 使用するライブラリ

- TensorFlow ライブラリ
    - `tf.get_varible(...)`, `tf.variable_scope(...)` : 名前空間と変数のスコープ
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
    - `tf.train.Saver` : Variable の save/restore
    - `tf.trainable_variables(...)` : trainable フラグを付けた変数
        - https://qiita.com/TomokIshii/items/84ee55a1c2d335dcab6f
    - `tf.control_dependencies(...)` : sess.run で実行する際のトレーニングステップの依存関係（順序）を定義
    - `tf.no_op(...)` : 何もしない Operator を返す。（トレーニングステップの依存関係を定義するのに使用）
        - xxx

- その他ライブラリ
    - `argparse` : コマンドライン引数用ライブラリ
    - `pickle` :


<a id="ID_2"></a>

### 使用するデータセット
- [MNIST データセット](https://github.com/Yagami360/MachineLearning_Exercises_Python_TensorFlow/blob/master/dataset.md#mnist手書き数字文字画像データ)
    - `main1.py` で使用
- [CIFAR-10 データセット](https://github.com/Yagami360/MachineLearning_Exercises_Python_TensorFlow/blob/master/dataset.md#cifar-10-データセット)


<a id="ID_3"></a>

## コードの実行結果

<a id="ID_3-1"></a>

### DCGAN による手書き数字画像データ MNIST の自動生成 : `main1.py`
DCGAN モデルに対し MNIST データセットで学習し、手書き数字画像を自動生成する。

#### コードの説明
- DCGAN の Discriminator に入力する学習用データセットとして、MNIST データセットを使用。
    - データは shape = [n_sample, image_width=28, image_height=28] の形状に reshape
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
- DCGAN のモデルを構築する。
この処理は、`DeepConvolutionalGAN.model()` メソッドで行う。
    - まず、Generator に入力する、入力ノイズデータを生成する。
    - 次に、Descriminator 側のモデルを構築する。
    この処理は、`DeepConvolutionalGAN.generator(...)` で行う。
        - xxx 
    - 最後に、Generator 側の損失関数を定義する。
    この処理は、`DeepConvolutionalGAN.discriminator(...)` で行う。
        - xxx
- 損失関数を定義する。
    - 損失関数を、以下の DCGAN での損失関数の更新アルゴリズムに従って、定義する。
    ![image](https://user-images.githubusercontent.com/25688193/36006479-89695612-0d80-11e8-8937-6c4c9d8ef14f.png)
    ![image](https://user-images.githubusercontent.com/25688193/36006524-cbc8eeaa-0d80-11e8-872c-2f5927e121b2.png)
    - まず、Descriminator 側の損失関数を定義する。
    - 次に、Generator 側の損失関数を定義する。
- xxx

<br>

#### コードの実行結果

#### 損失関数のグラフ

|パラメータ名|値（実行条件１）|
|---|---|
|最適化アルゴリズム|Adam|
|学習率<br>`learning_rate`|0.0001|
|学習率の減衰項１<br>`beta1`|0.5|
|学習率の減衰項２<br>`beta2`|0.999|
|ミニバッチサイズ<br>`batch_size`|32|

![gan_dcgan_1-1_1](https://user-images.githubusercontent.com/25688193/36030883-3aeb8e08-0dec-11e8-8ec4-4966754a12ea.png)
> 損失関数として、疎なソフトマックス・クロス・エントロピー関数を使用した場合の、損失関数のグラフ。
赤線が Generator の損失関数の値。青線が Descriminator の損失関数の値。黒線が、Generator と Descriminator の損失関数の総和。<br>
損失関数のグラフより、Generator の損失関数の値が 0 に収束できておらず、又上下に不安定となっていることが分かる。これは GAN に見られる特性である。<br>
つまり、GAN はゲーム理論的に言えば、Generator と Descriminator の２プレイヤーのゼロサムゲームとみなすことが出来るので、相互に干渉しあって動作が不安定になる。

<br>

#### Generator から出力された自動生成画像（学習時の途中経過込み）
- 入力ノイズデータ : 32 × 64 pixel<br>
![temp_output_image0](https://user-images.githubusercontent.com/25688193/36032312-ec40f13a-0df0-11e8-8819-68dc1bba41ca.jpg)

|パラメータ名|値（実行条件１）|
|---|---|
|最適化アルゴリズム|Adam|
|学習率<br>`learning_rate`|0.0001|
|学習率の減衰項１<br>`beta1`|0.5|
|学習率の減衰項２<br>`beta2`|0.999|
|ミニバッチサイズ<br>`batch_size`|32|

- epoch 数 : 1 ~ 1000（途中経過）, ステップ間隔 : `eval_step = 25`
![temp_output_vhstack_image1000](https://user-images.githubusercontent.com/25688193/36032203-99fddb40-0df0-11e8-989a-cfa1df321dbb.jpg)    
- epoch 数 : 1000（途中経過）
- epoch 数 : 20000（最終結果）

> 縦列が、epoch 数を増加して（学習を進めていった）ときの、Generator から出力された自動生成画像。<br>
> 横列は、Generator に入力した入力ノイズデータの違いによる自動生成画像の違い。

<br>

#### 入力ノイズのパラメータを球の表面上に沿って動かしたときの Generator から出力された自動生成画像

|パラメータ名|値（実行条件１）|
|---|---|
|最適化アルゴリズム|Adam|
|学習率<br>`learning_rate`|0.0001|
|学習率の減衰項１<br>`beta1`|0.5|
|学習率の減衰項２<br>`beta2`|0.999|
|ミニバッチサイズ<br>`batch_size`|32|

- epoch 数 : 1000

![dcgan_morphing_epoch1000](https://user-images.githubusercontent.com/25688193/36030902-4f680e74-0dec-11e8-9c1b-3ec62ca5e089.gif)


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

> 記載中...


<a id="ID_10-1-1"></a>

#### DCGAN [Deep Convolutional GAN]
- 元論文「Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks」
    - arXiv.org : https://arxiv.org/abs/1511.06434

![image](https://user-images.githubusercontent.com/25688193/35545399-50f2a4bc-05b2-11e8-853e-11d38971630f.png)
![image](https://user-images.githubusercontent.com/25688193/35545437-72ebb95a-05b2-11e8-9219-e723ee344d54.png)
![image](https://user-images.githubusercontent.com/25688193/35545467-93e540c2-05b2-11e8-846f-ccd86273a85f.png)
![image](https://user-images.githubusercontent.com/25688193/35549375-93ea4836-05c8-11e8-8279-a8d3d3a659c6.png)
![image](https://user-images.githubusercontent.com/25688193/35545532-cd39d9d2-05b2-11e8-9ab9-a3f4123ab8fd.png)
![image](https://user-images.githubusercontent.com/25688193/35545809-5d14a248-05b4-11e8-854e-caf830ef2972.png)
![image](https://user-images.githubusercontent.com/25688193/35549398-b4a58dce-05c8-11e8-9bd5-883c03aa4564.png)

> 記載中...


### デバッグメモ

[18/02/08]

[main1.py]

> `input_noize_holder` がモデルに関連付けられていない問題
![graph_large_attrs_key _too_large_attrs limit_attr_size 1024 run 3](https://user-images.githubusercontent.com/25688193/35985713-9945101c-0d3a-11e8-8086-8127969c2ec4.png)
![graph_large_attrs_key _too_large_attrs limit_attr_size 1024 run 4](https://user-images.githubusercontent.com/25688193/35985718-9cb06544-0d3a-11e8-85bc-ba08cc993752.png)
![graph_large_attrs_key _too_large_attrs limit_attr_size 1024 run 5](https://user-images.githubusercontent.com/25688193/35985720-9cdd04aa-0d3a-11e8-807f-d5ffe05ef0e4.png)

[main1_1.py]
![graph_large_attrs_key _too_large_attrs limit_attr_size 1024 run](https://user-images.githubusercontent.com/25688193/35968027-07a70a4a-0d06-11e8-8cf2-271db602be33.png)

![graph_large_attrs_key _too_large_attrs limit_attr_size 1024 run 1](https://user-images.githubusercontent.com/25688193/35968028-07d2084e-0d06-11e8-9392-48ff5137a6e6.png)


```python


-----------------------------------------
g_vars : [
    <tf.Variable 'Generator/weight_var:0' shape=(64, 6272) dtype=float32_ref>, 
    <tf.Variable 'Generator/bias_var:0' shape=(128,) dtype=float32_ref>, 
    <tf.Variable 'Generator/DeConvLayer_0/weight_var:0' shape=(5, 5, 64, 128) dtype=float32_ref>, 
    <tf.Variable 'Generator/DeConvLayer_0/bias_var:0' shape=(64,) dtype=float32_ref>, 
    <tf.Variable 'Generator/DeConvLayer_1/weight_var:0' shape=(5, 5, 1, 64) dtype=float32_ref>, 
    <tf.Variable 'Generator/DeConvLayer_1/bias_var:0' shape=(1,) dtype=float32_ref>
    ]

d_vars : [
    <tf.Variable 'Descriminator/ConvLayer_0/weight_var:0' shape=(5, 5, 1, 64) dtype=float32_ref>, 
    <tf.Variable 'Descriminator/ConvLayer_0/bias_var:0' shape=(64,) dtype=float32_ref>, 
    <tf.Variable 'Descriminator/ConvLayer_1/weight_var:0' shape=(5, 5, 64, 128) dtype=float32_ref>, 
    <tf.Variable 'Descriminator/ConvLayer_1/bias_var:0' shape=(128,) dtype=float32_ref>, 
    <tf.Variable 'Descriminator/flatten_fully/weight_var:0' shape=(6272, 2) dtype=float32_ref>, 
    <tf.Variable 'Descriminator/flatten_fully/bias_var:0' shape=(2,) dtype=float32_ref>]


generate_images(...) / out_G_op : Tensor("Generator_1/Sigmoid:0", shape=(32, 28, 28, 1), dtype=float32)

InvalidArgumentError (see above for traceback): Conv2DCustomBackpropInput: input and out_backprop must have the same batch sizeinput batch: 32outbackprop batch: 8 batch_dim: 0
	 [[Node: Generator_1/DeConvLayer_0/conv2d_transpose = Conv2DBackpropInput[T=DT_FLOAT, data_format="NHWC", padding="SAME", strides=[1, 2, 2, 1], use_cudnn_on_gpu=true, _device="/job:localhost/replica:0/task:0/device:CPU:0"](Generator_1/DeConvLayer_0/conv2d_transpose/output_shape, Generator/DeConvLayer_0/weight_var/read, Generator_1/Relu)]]


-----------------------------------------
DeepConvolutionalGAN
DeepConvolutionalGAN(batch_size=None, epochs=None, eval_step=None,
           image_height=None, image_width=None, n_D_conv_featuresMap=None,
           n_G_deconv_featuresMap=None, n_channels=None, n_labels=None,
           session=None)
after building model & loss & optimizer
_session : 
 <tensorflow.python.client.session.Session object at 0x0000023F76E14BE0>
_init_var_op : 
 None
_epoches :  2000
_batch_size :  32
_eval_step :  100
_image_height :  28
_image_width :  28
_n_channels :  1
_n_G_deconv_featuresMap :  [128, 64, 1]
_n_D_conv_featuresMap :  [1, 64, 128]
_n_labels :  2
_image_holder :  Tensor("image_holder:0", shape=(?, 28, 28, 1), dtype=float32)
_dropout_holder : Tensor("dropout_holder:0", dtype=float32)
_G_loss_op : 
 Tensor("Mean_2:0", shape=(), dtype=float32)
_G_optimizer : 
 <tensorflow.python.training.adam.AdamOptimizer object at 0x0000023F77A22CF8>
_G_train_step : 
 name: "Adam"
op: "NoOp"
input: "^Adam/update_Generator/weight_var/ApplyAdam"
input: "^Adam/update_Generator/bias_var/ApplyAdam"
input: "^Adam/update_Generator/DeConvLayer_0/weight_var/ApplyAdam"
input: "^Adam/update_Generator/DeConvLayer_0/bias_var/ApplyAdam"
input: "^Adam/update_Generator/DeConvLayer_1/weight_var/ApplyAdam"
input: "^Adam/update_Generator/DeConvLayer_1/bias_var/ApplyAdam"
input: "^Adam/Assign"
input: "^Adam/Assign_1"

_G_y_out_op : 
 Tensor("Generator/Sigmoid:0", shape=(32, 28, 28, 1), dtype=float32)
_D_loss_op : 
 Tensor("add:0", shape=(), dtype=float32)
_D_optimizer : 
 <tensorflow.python.training.adam.AdamOptimizer object at 0x0000023F77A22D30>
_D_train_step : 
 name: "Adam_1"
op: "NoOp"
input: "^Adam_1/update_Descriminator/ConvLayer_0/weight_var/ApplyAdam"
input: "^Adam_1/update_Descriminator/ConvLayer_0/bias_var/ApplyAdam"
input: "^Adam_1/update_Descriminator/ConvLayer_1/weight_var/ApplyAdam"
input: "^Adam_1/update_Descriminator/ConvLayer_1/bias_var/ApplyAdam"
input: "^Adam_1/update_Descriminator/flatten_fully/weight_var/ApplyAdam"
input: "^Adam_1/update_Descriminator/flatten_fully/bias_var/ApplyAdam"
input: "^Adam_1/Assign"
input: "^Adam_1/Assign_1"

_D_y_out_op1 : 
 Tensor("Descriminator/flatten_fully/add:0", shape=(32, 2), dtype=float32)
_D_y_out_op2 : 
 Tensor("Descriminator_1/flatten_fully/add:0", shape=(?, 2), dtype=float32)
_weights : 
 [<tf.Variable 'Generator/weight_var:0' shape=(64, 6272) dtype=float32_ref>, <tf.Variable 'Generator/DeConvLayer_0/weight_var:0' shape=(5, 5, 64, 128) dtype=float32_ref>, <tf.Variable 'Generator/DeConvLayer_1/weight_var:0' shape=(5, 5, 1, 64) dtype=float32_ref>, <tf.Variable 'Descriminator/ConvLayer_0/weight_var:0' shape=(5, 5, 1, 64) dtype=float32_ref>, <tf.Variable 'Descriminator/ConvLayer_1/weight_var:0' shape=(5, 5, 64, 128) dtype=float32_ref>, <tf.Variable 'Descriminator/flatten_fully/weight_var:0' shape=(6272, 2) dtype=float32_ref>]
_biases : 
 [<tf.Variable 'Generator/bias_var:0' shape=(128,) dtype=float32_ref>, <tf.Variable 'Generator/DeConvLayer_0/bias_var:0' shape=(64,) dtype=float32_ref>, <tf.Variable 'Generator/DeConvLayer_1/bias_var:0' shape=(1,) dtype=float32_ref>, <tf.Variable 'Descriminator/ConvLayer_0/bias_var:0' shape=(64,) dtype=float32_ref>, <tf.Variable 'Descriminator/ConvLayer_1/bias_var:0' shape=(128,) dtype=float32_ref>, <tf.Variable 'Descriminator/flatten_fully/bias_var:0' shape=(2,) dtype=float32_ref>]
-----------------------------------------


```

```python
[main1_1.py]


g_vars :
 [
     <tf.Variable 'generator/reshape/weights:0' shape=(64, 6272) dtype=float32_ref>, 
     <tf.Variable 'generator/reshape/biases:0' shape=(128,) dtype=float32_ref>, 
     <tf.Variable 'generator/conv0/weights:0' shape=(5, 5, 64, 128) dtype=float32_ref>, 
     <tf.Variable 'generator/conv0/biases:0' shape=(64,) dtype=float32_ref>, 
     <tf.Variable 'generator/conv1/weights:0' shape=(5, 5, 1, 64) dtype=float32_ref>, 
     <tf.Variable 'generator/conv1/biases:0' shape=(1,) dtype=float32_ref>]
d_vars :
 [
     <tf.Variable 'descriminator/conv0/weights:0' shape=(5, 5, 1, 64) dtype=float32_ref>, 
     <tf.Variable 'descriminator/conv0/biases:0' shape=(64,) dtype=float32_ref>, 
     <tf.Variable 'descriminator/conv1/weights:0' shape=(5, 5, 64, 128) dtype=float32_ref>, <tf.Variable 'descriminator/conv1/biases:0' shape=(128,) dtype=float32_ref>, 
     <tf.Variable 'descriminator/classify/weights:0' shape=(6272, 2) dtype=float32_ref>, 
     <tf.Variable 'descriminator/classify/biases:0' shape=(2,) dtype=float32_ref>]


outputs_0 : Tensor("Sigmoid:0", shape=(32, 28, 28, 1), dtype=float32)

w_1 : <tf.Variable 'descriminator/conv0/weights:0' shape=(5, 5, 1, 64) dtype=float32_ref>
b_1 : <tf.Variable 'descriminator/conv0/biases:0' shape=(64,) dtype=float32_ref>
outputs_1 : Tensor("descriminator/conv0/Maximum:0", shape=(32, 14, 14, 64), dtype=float32)

w_2 : <tf.Variable 'descriminator/conv1/weights:0' shape=(5, 5, 64, 128) dtype=float32_ref>
b_2 : <tf.Variable 'descriminator/conv1/biases:0' shape=(128,) dtype=float32_ref>
outputs_2 : Tensor("descriminator/conv1/Maximum:0", shape=(32, 7, 7, 128), dtype=float32)

---

logits_from_g : Tensor("descriminator/classify/add:0", shape=(32, 2), dtype=float32)
logits_from_i : Tensor("descriminator_1/classify/add:0", shape=(?, 2), dtype=float32)

train_op
    node_def
        name: "train"op: "NoOp"input: "^Adam"input: "^Adam_1"
name: "train"
op: "NoOp"
input: "^Adam"
input: "^Adam_1"

C:\Users\y0341\Anaconda3\lib\site-packages\matplotlib\cbook\deprecation.py:106: MatplotlibDeprecationWarning: Adding an axes using the same arguments as a previous axes currently reuses the earlier instance.  In a future version, a new instance will always be created and returned.  Meanwhile, this warning can be suppressed, and the future behavior ensured, by passing a unique label to each axes instance.
  warnings.warn(message, mplDeprecation, stacklevel=1)
C:\Users\y0341\Anaconda3\lib\site-packages\matplotlib\animation.py:1218: UserWarning: MovieWriter imagemagick unavailable
  warnings.warn("MovieWriter %s unavailable" % writer)
ValueError: outfile must be *.htm or *.html

```

