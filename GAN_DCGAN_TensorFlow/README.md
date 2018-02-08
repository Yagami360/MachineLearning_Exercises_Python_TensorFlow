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


<a id="#ID_1"></a>

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
    - `tf.control_dependencies(...)` : sess.run で実行する際の Operator の依存関係（順序）を定義
    - `tf.no_op(...)` : 何もしない Operator を返す。（Operator の依存関係を定義するのに使用）
        - xxx

- その他ライブラリ
    - `argparse` : コマンドライン引数用ライブラリ
    - `pickle` :


<a id="#ID_2"></a>

### 使用するデータセット
- [MNIST データセット](https://github.com/Yagami360/MachineLearning_Exercises_Python_TensorFlow/blob/master/dataset.md#mnist手書き数字文字画像データ)
    - `main1.py` で使用
- [CIFAR-10 データセット](https://github.com/Yagami360/MachineLearning_Exercises_Python_TensorFlow/blob/master/dataset.md#cifar-10-データセット)


<a id="#ID_3"></a>

## コードの実行結果

<a id="#ID_3-1"></a>

### DCGAN による手書き数字画像データ MNIST の自動生成 : `main1.py`
DCGAN モデルに対し MNIST データセットで学習し、手書き数字画像を自動生成する。

> 実装中...

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
    - one-hot encode 処理を行う。
    ```python
    def main():
        ...
        session = tf.Session()
        encode_holder = tf.placeholder(tf.int64, [None])
        y_oneHot_enoded_op = tf.one_hot( encode_holder, depth=10, dtype=tf.float32 ) # depth が 出力層のノード数に対応
        session.run( tf.global_variables_initializer() )
        y_train_encoded = session.run( y_oneHot_enoded_op, feed_dict = { encode_holder: y_train } )
        y_test_encoded = session.run( y_oneHot_enoded_op, feed_dict = { encode_holder: y_test } )
        session.close()
    ```
- xxx


<br>

---

<a id="#ID_4"></a>

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
```python

_D_y_out_op_0 Tensor("Generator/Sigmoid:0", shape=(32, 28, 28, 1), dtype=float32)

_weights_1 <tf.Variable 'Descriminator/ConvLayer_0/weight_var:0' shape=(5, 5, 1, 64) dtype=float32_ref>
_biases_1 <tf.Variable 'Descriminator/ConvLayer_0/bias_var:0' shape=(64,) dtype=float32_ref>
_D_y_out_op_1 Tensor("Descriminator/ConvLayer_0/Maximum:0", shape=(32, 14, 14, 64), dtype=float32)

_weights_2 <tf.Variable 'Descriminator/ConvLayer_1/weight_var:0' shape=(5, 5, 64, 128) dtype=float32_ref>
_biases_2 <tf.Variable 'Descriminator/ConvLayer_1/bias_var:0' shape=(128,) dtype=float32_ref>
_D_y_out_op_2 Tensor("Descriminator/ConvLayer_1/Maximum:0", shape=(32, 7, 7, 128), dtype=float32)

---

_t_holder : Tensor("Placeholder_3:0", shape=(?, 2), dtype=int32)
_image_holder : Tensor("Placeholder_2:0", shape=(?, 28, 28, 1), dtype=float32)
_G_y_out_op : Tensor("Generator/Sigmoid:0", shape=(32, 28, 28, 1), dtype=float32)
_D_y_out_op : Tensor("Descriminator/flatten/fully/add:0", shape=(32, 2), dtype=float32)


```

```python
[main1_1.py]

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
```