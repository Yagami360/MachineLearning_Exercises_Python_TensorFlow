# TensorFlow で畳み込みニューラルネットワーク [CNN : Convolutional Neural Network] の実装

TensorFlow での CNN の練習用実装コード集。<br>

TensorFlow での CNN の処理をクラス（任意の層に DNN 化可能な柔軟なクラス）でラッピングし、scikit-learn ライブラリの classifier, estimator とインターフェイスを共通化することで、scikit-learn ライブラリとの互換性のある自作クラス `ConvolutionalNN` を使用。<br>


この README.md ファイルには、各コードの実行結果、概要、CNN の背景理論の説明を記載しています。<br>
分かりやすいように `main.py` ファイル毎に１つの完結した実行コードにしています。

## 項目 [Contents]

1. [使用するライブラリ](#ID_1)
1. [使用するデータセット](#ID_2)
1. [ニューラルネットワークのフレームワークのコードの説明](#ID_3-0)
1. [コード実行結果](#ID_3)
    1. [CNN による MNIST データの識別 : `main1.py`](#ID_3-1)
    1. [CNN による CIFAR-10 データの識別 : `main2.py`](#ID_3-2)
    1. [既存の CNN モデルの再学習処理 : `main3.py`](#ID_3-3)
1. [背景理論](#ID_4)
    1. [CNN の概要](#ID_4-1)

```python
    # MSIT データが格納されているフォルダへのパス
    mist_path = "D:\Data\MachineLearning_DataSet\MIST"

    X_train, y_train = MLPreProcess.load_mist( mist_path, "train" )
    X_test, y_test = MLPreProcess.load_mist( mist_path, "t10k" )

    print( "X_train.shape : ", X_train.shape )
    print( "y_train.shape : ", y_train.shape )
    print( "X_test.shape : ", X_test.shape )
    print( "y_test.shape : ", y_test.shape )
    ...
    session = tf.Session()
    encode_holder = tf.placeholder(tf.int64, [None])
    y_oneHot_enoded_op = tf.one_hot( encode_holder, depth=10, dtype=tf.float32 ) # depth が 出力層のノード数に対応
    session.run( tf.global_variables_initializer() )
    y_train_encoded = session.run( y_oneHot_enoded_op, feed_dict = { encode_holder: y_train } )
    y_test_encoded = session.run( y_oneHot_enoded_op, feed_dict = { encode_holder: y_test } )
    print( "y_train_encoded.shape : ", y_train_encoded.shape )
    print( "y_train_encoded.dtype : ", y_train_encoded.dtype )
    print( "y_test_encoded.shape : ", y_test_encoded.shape )
```
```python
[出力]
X_train.shape :  (60000, 784)
y_train.shape :  (60000,)
X_test.shape :  (10000, 784)
y_test.shape :  (10000,)
y_train_encoded.shape :  (60000, 10)
y_train_encoded.dtype :  float32
y_test_encoded.shape :  (10000, 10)
```
<br>

```python
    # TensorFlow のサポート関数を使用して, MNIST データを読み込み
    mnist = read_data_sets( mist_path )
    print( "mnist :\n", mnist )
    X_train = numpy.array( [numpy.reshape(x, (28,28)) for x in mnist.train.images] )
    X_test = numpy.array( [numpy.reshape(x, (28,28)) for x in mnist.test.images] )
    y_train = mnist.train.labels
    y_test = mnist.test.labels

    print( "X_train.shape : ", X_train.shape )
    print( "y_train.shape : ", y_train.shape )
    print( "X_test.shape : ", X_test.shape )
    print( "y_test.shape : ", y_test.shape )
```
```python
[出力]
mnist :
 Datasets(
 train=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x0000000002BE99E8>, 
 validation=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x0000000002BE9EB8>, 
 test=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x00000000108A5C50>)
X_train.shape :  (55000, 28, 28)
y_train.shape :  (55000,)
X_test.shape :  (10000, 28, 28)
y_test.shape :  (10000,)

fullyLayers_input_size :  78400
pool_op1.get_shape().as_list() :
 [None, 28, 28, 25]
ValueError: Dimensions must be equal, but are 19600 and 78400 for 'MatMul' (op: 'MatMul') with input shapes: [1,19600], [78400,100].
```