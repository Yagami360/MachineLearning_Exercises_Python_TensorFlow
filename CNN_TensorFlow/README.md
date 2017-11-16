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

<Memo>

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

```
InvalidArgumentError (see above for traceback): You must feed a value for placeholder tensor 'Placeholder_2' with dtype int32 and shape [100]
	 [[Node: Placeholder_2 = Placeholder[dtype=DT_INT32, shape=[100], _device="/job:localhost/replica:0/task:0/cpu:0"]()]]
```

```python
X_train.shape :  (60000, 28, 28)
y_train.shape :  (60000,)
X_test.shape :  (10000, 28, 28)
y_test.shape :  (10000,)
X_train : 
 [[[0 0 0 ..., 0 0 0]
  [0 0 0 ..., 0 0 0]
  [0 0 0 ..., 0 0 0]
  ..., 
  [0 0 0 ..., 0 0 0]
  [0 0 0 ..., 0 0 0]
  [0 0 0 ..., 0 0 0]]

 [[0 0 0 ..., 0 0 0]
  [0 0 0 ..., 0 0 0]
  [0 0 0 ..., 0 0 0]
  ..., 
 [[0 0 0 ..., 0 0 0]
  [0 0 0 ..., 0 0 0]
  [0 0 0 ..., 0 0 0]
  ..., 
  [0 0 0 ..., 0 0 0]
  [0 0 0 ..., 0 0 0]
  [0 0 0 ..., 0 0 0]]]

y_train : 
 [5 0 4 ..., 5 6 8]
y_train_encoded.shape :  (60000, 10)
y_train_encoded.dtype :  float32
y_test_encoded.shape :  (10000, 10)
```
