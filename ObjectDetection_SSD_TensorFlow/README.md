## TensorFlow を用いた SSD [Single Shot muitibox Detector] の実装と簡単な応用

> 実装中...

ニューラルネットワークによる一般物体検出アルゴリズムの１つである、SSD [Single Shot muitibox Detector] を TensorFlow で実装。

この `README.md` ファイルには、各コードの実行結果、概要、SSD の背景理論の説明を記載しています。
分かりやすいように `main.py` ファイル毎に１つの完結した実行コードにしています。

尚、SSD [Single Shot muitibox Detector] に関しての、背景理論は以下のサイトに記載してあります。

- [星の本棚 : ニューラルネットワーク / ディープラーニング](http://yagami12.hatenablog.com/entry/2017/09/17/111935#ID_11-4)


### 項目 [Contents]

1. [使用するライブラリ](#ID_1)
1. [使用するデータセット](#ID_2)
1. [コード説明＆実行結果](#ID_3)
    1. [コード説明＆実行結果 : `main2.py`](#ID_3-2)
    1. [](#)
1. [背景理論](#ID_4)
    1. [背景理論１](#ID_4-1)
    1. [](#)
1. [参考サイト](#ID_5)


<a id="ID_1"></a>

### 使用するライブラリ

- TensorFlow ライブラリ
    - xxx

- その他ライブラリ
    - xxx


<a id="ID_2"></a>

### 使用するデータセット
- [Pascal VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/)<br>
    ![image](https://user-images.githubusercontent.com/25688193/40175526-55b9d1fe-5a13-11e8-8829-8c5791383ffb.png)<br>
    - [ 画像ファイル名, N x 24 の 2 次元配列 ]<br>
    - 画像のRGB値は 0.0 ~ 1.0 の範囲での値<br>
    - N は、画像の中にある検出物体の数で、画像によって異なる。<br>
    - 24 というのは、位置とクラス名のデータを合わせたデータを表すベクトルになっていて、<br>
    この内、(xmin, ymin, xmax, ymax) の 4 次元の情報で物体を囲む矩形の位置を表し、残りの 20 次元でクラス名を表す。<br>

- Open Images Dataset V4
    - https://storage.googleapis.com/openimages/web/download.html


<a id="ID_3"></a>

## コード説明＆実行結果

<a id="ID_3-2"></a>

## コード説明＆実行結果２ : `main2.py`
> 実装中...

![image](https://user-images.githubusercontent.com/25688193/39536606-0e4f1416-4e72-11e8-91e2-82b516706bae.png)<br>

<br>

---

<a id="ID_4"></a>

## 背景理論

<a id="ID_4-1"></a>

## 背景理論１

---

## デバッグメモ

[18/05/14]

```python
tensorflow.python.framework.errors_impl.InvalidArgumentError:
 Incompatible shapes: [100] vs. [100,512]
	 [[Node: Loss_CrossEntropy_op/mul = Mul[T=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:CPU:0"](_arg_Placeholder_2_0_1, Loss_CrossEntropy_op/Log)]]

```

```Python
@conv6 layer
Current input size in convolution layer is: [None, 38, 38, 512]
kernel_size : [3, 3, 512, 1024]
strides : [1, 1, 1, 1]
    ===> output size is: [None, 38, 38, 1024]
@conv7 layer
Current input size in convolution layer is: [None, 19, 19, 1024]
kernel_size : [1, 1, 1024, 1024]
strides : [1, 1, 1, 1]
    ===> output size is: [None, 19, 19, 1024]
@conv8_1 layer
Current input size in convolution layer is: [None, 19, 19, 1024]
kernel_size : [1, 1, 1024, 256]
strides : [1, 1, 1, 1]
    ===> output size is: [None, 19, 19, 256]
@conv8_2 layer
Current input size in convolution layer is: [None, 19, 19, 256]
kernel_size : [3, 3, 256, 512]
strides : [1, 2, 2, 1]
    ===> output size is: [None, 10, 10, 512]
@conv9_1 layer
Current input size in convolution layer is: [None, 10, 10, 512]
kernel_size : [1, 1, 512, 128]
strides : [1, 1, 1, 1]
    ===> output size is: [None, 10, 10, 128]
@conv9_2 layer
Current input size in convolution layer is: [None, 10, 10, 128]
kernel_size : [3, 3, 128, 256]
strides : [1, 2, 2, 1]
    ===> output size is: [None, 5, 5, 256]
@conv10_1 layer
Current input size in convolution layer is: [None, 5, 5, 256]
kernel_size : [1, 1, 256, 128]
strides : [1, 1, 1, 1]
    ===> output size is: [None, 5, 5, 128]
@conv10_2 layer
Current input size in convolution layer is: [None, 5, 5, 128]
kernel_size : [3, 3, 128, 256]
strides : [1, 2, 2, 1]
    ===> output size is: [None, 3, 3, 256]
@conv11_1 layer
Current input size in convolution layer is: [None, 3, 3, 256]
kernel_size : [1, 1, 256, 128]
strides : [1, 1, 1, 1]
    ===> output size is: [None, 3, 3, 128]
@conv11_2 layer
Current input size in convolution layer is: [None, 3, 3, 128]
kernel_size : [3, 3, 128, 256]
strides : [1, 3, 3, 1]
    ===> output size is: [None, 1, 1, 256]

concatenated: Tensor("concat:0", shape=(?, 8752, 25), dtype=float32)
confs: [None, 8752, 21]
locs: [None, 8752, 4]

fmap shapes is 
[
    [None, 38, 38, 100], [None, 19, 19, 150], [None, 10, 10, 150], 
    [None, 5, 5, 150], [None, 3, 3, 150], [None, 1, 1, 150]
]

--------

conv6_op : Tensor("conv6/Relu:0", shape=(?, 38, 38, 1024), dtype=float32)
pool6_op : Tensor("pool6/pool6:0", shape=(?, 19, 19, 1024), dtype=float32)
conv7_op : Tensor("conv7/Relu:0", shape=(?, 19, 19, 1024), dtype=float32)
conv8_1_op : Tensor("conv8_1/Relu:0", shape=(?, 19, 19, 256), dtype=float32)
conv8_2_op : Tensor("conv8_2/Relu:0", shape=(?, 10, 10, 512), dtype=float32)
conv9_1_op : Tensor("conv9_1/Relu:0", shape=(?, 10, 10, 128), dtype=float32)
conv9_2_op : Tensor("conv9_2/Relu:0", shape=(?, 5, 5, 256), dtype=float32)
conv10_1_op : Tensor("conv10_1/Relu:0", shape=(?, 5, 5, 128), dtype=float32)
conv10_2_op : Tensor("conv10_2/Relu:0", shape=(?, 3, 3, 256), dtype=float32)
conv11_1_op : Tensor("conv11_1/Relu:0", shape=(?, 3, 3, 128), dtype=float32)
conv11_2_op : Tensor("conv11_2/Relu:0", shape=(?, 1, 1, 256), dtype=float32)

fmaps :
    fmaps[0] : Tensor("fmap1/Relu:0", shape=(?, 38, 38, 100), dtype=float32)
    fmaps[1] : Tensor("fmap2/Relu:0", shape=(?, 19, 19, 150), dtype=float32)
    fmaps[2] : Tensor("fmap3/Relu:0", shape=(?, 10, 10, 150), dtype=float32)
    fmaps[3] : Tensor("fmap4/Relu:0", shape=(?, 5, 5, 150), dtype=float32)
    fmaps[4] : Tensor("fmap5/Relu:0", shape=(?, 3, 3, 150), dtype=float32)
    fmaps[5] : Tensor("fmap6/Relu:0", shape=(?, 1, 1, 150), dtype=float32)

fmap_reshaped[0] : Tensor("Reshape:0", shape=(?, 5776, 25), dtype=float32)
fmap_reshaped[1] : Tensor("Reshape_1:0", shape=(?, 2166, 25), dtype=float32)
fmap_reshaped[2] : Tensor("Reshape_2:0", shape=(?, 600, 25), dtype=float32)
fmap_reshaped[3] : Tensor("Reshape_3:0", shape=(?, 150, 25), dtype=float32)
fmap_reshaped[4] : Tensor("Reshape_4:0", shape=(?, 54, 25), dtype=float32)
fmap_reshaped[5] : Tensor("Reshape_5:0", shape=(?, 6, 25), dtype=float32)
fmap_concatenated : Tensor("concat:0", shape=(?, 8752, 25), dtype=float32)

pred_confidences : Tensor("strided_slice:0", shape=(?, 8752, 21), dtype=float32)
pred_locations : Tensor("strided_slice_1:0", shape=(?, 8752, 4), dtype=float32)

```


```python
[ssd300.py]
def train(...):

images : list
[0] / shape (300, 300, 3) = (image_width, height, n_channels)
[1] / shape (300, 300, 3)
[2] / shape (300, 300, 3)
[3] / shape (300, 300, 3)
[4] / shape (300, 300, 3)
[5] / shape (300, 300, 3)
[6] / shape (300, 300, 3)
[7] / shape (300, 300, 3)
[8] / shape (300, 300, 3)
[9] / shape (300, 300, 3)

actual_data : list
[0] / shape (3, 24) = (画像の中にある物体の数, クラス数 + 長方形位置情報 )
[1] / shape (3, 24)
[2] / shape (2, 24)
[3] / shape (2, 24)
[4] / shape (2, 24)
[5] / shape (2, 24)
[6] / shape (6, 24)
[7] / shape (3, 24)
[8] / shape (1, 24)
[9] / shape (2, 24)

feature_maps : list
[0] / shape (10, 38, 38, 100)
[1] / shape (10, 19, 19, 150)
[2] / shape (10, 10, 10, 150)
[3] / shape (10, 5, 5, 150)
[4] / shape (10, 3, 3, 150)
[5] / shape (10, 1, 1, 150)

pred_confs : list
[0] / shape (10, 8752, 21) = (バッチサイズ, デフォルトボックス数, クラス数)
    [0][0] =
    array(
        [ 0.47469443, -0.31052983, -0.68134677, -0.66496718, -0.35621506,
         -0.56755793, -0.64405829, -0.63288641, -0.63658172, -0.56236774,       
         -0.6255036 ,  0.11520439, -0.66435236, -0.67320728, -0.68250519,       
         -0.63873053,  1.53790379, -0.67591554, -0.70764214, -0.65140432,        
         0.28005606], 
        dtype=float32
    )
[1] / shape (10, 8752, 21)
[2] / shape (10, 8752, 21)
[3] / shape (10, 8752, 21)
[4] / shape (10, 8752, 21)
[5] / shape (10, 8752, 21)
[6] / shape (10, 8752, 21)
[7] / shape (10, 8752, 21)
[8] / shape (10, 8752, 21)
[9] / shape (10, 8752, 21)

pred_locs : list
[0] / shape (10, 8752, 4) = (バッチサイズ, デフォルトボックス数, 長方形位置)
    [0][0] =
    array([ 0.39878446, -0.64267939, -0.29277915, -0.28452805], dtype=float32)
[1] / shape (10, 8752, 4)
...
[9] / shape (10, 8752, 4)


```

```
X_train : list
[0] / shape 
    [0][0] : numapy
    [0][1] : 
[1]
...
[199]

batch_x : list


```


```python
BATCH: 1 / EPOCH: 1, LOSS: 77.99568176269531
BATCH: 2 / EPOCH: 1, LOSS: 72.53934478759766
BATCH: 3 / EPOCH: 1, LOSS: 66.68212127685547
BATCH: 4 / EPOCH: 1, LOSS: 57.453941345214844
BATCH: 5 / EPOCH: 1, LOSS: 65.49876403808594
BATCH: 6 / EPOCH: 1, LOSS: 62.12317657470703
BATCH: 7 / EPOCH: 1, LOSS: 58.043888092041016
BATCH: 8 / EPOCH: 1, LOSS: 53.789306640625
BATCH: 9 / EPOCH: 1, LOSS: 66.81963348388672
BATCH: 10 / EPOCH: 1, LOSS: 51.86540985107422
BATCH: 11 / EPOCH: 1, LOSS: 55.48357391357422
BATCH: 12 / EPOCH: 1, LOSS: 54.44649124145508
BATCH: 13 / EPOCH: 1, LOSS: 49.0512580871582
BATCH: 14 / EPOCH: 1, LOSS: 49.96155548095703
BATCH: 15 / EPOCH: 1, LOSS: 57.35893630981445
BATCH: 16 / EPOCH: 1, LOSS: 58.429710388183594
BATCH: 17 / EPOCH: 1, LOSS: 59.454132080078125
BATCH: 18 / EPOCH: 1, LOSS: 50.92224884033203
BATCH: 19 / EPOCH: 1, LOSS: 52.530296325683594
BATCH: 20 / EPOCH: 1, LOSS: 54.49018096923828
BATCH: 21 / EPOCH: 1, LOSS: 53.80031204223633
BATCH: 22 / EPOCH: 1, LOSS: 55.987430572509766
BATCH: 23 / EPOCH: 1, LOSS: 49.911922454833984
BATCH: 24 / EPOCH: 1, LOSS: 46.470672607421875
BATCH: 25 / EPOCH: 1, LOSS: 44.198753356933594
BATCH: 26 / EPOCH: 1, LOSS: 49.226768493652344
BATCH: 27 / EPOCH: 1, LOSS: 50.17113494873047
BATCH: 28 / EPOCH: 1, LOSS: 46.77484893798828
BATCH: 29 / EPOCH: 1, LOSS: 44.91448974609375
BATCH: 30 / EPOCH: 1, LOSS: 54.14591979980469
BATCH: 31 / EPOCH: 1, LOSS: 50.675567626953125
BATCH: 32 / EPOCH: 1, LOSS: 53.774436950683594
BATCH: 33 / EPOCH: 1, LOSS: 46.288475036621094
BATCH: 34 / EPOCH: 1, LOSS: 45.0712890625
BATCH: 35 / EPOCH: 1, LOSS: 44.72560119628906
BATCH: 36 / EPOCH: 1, LOSS: 47.078311920166016
BATCH: 37 / EPOCH: 1, LOSS: 43.437782287597656
BATCH: 38 / EPOCH: 1, LOSS: 55.951690673828125
BATCH: 39 / EPOCH: 1, LOSS: 45.72810363769531
BATCH: 40 / EPOCH: 1, LOSS: 52.4937744140625
BATCH: 41 / EPOCH: 1, LOSS: 51.1361083984375
BATCH: 42 / EPOCH: 1, LOSS: 53.22508239746094
BATCH: 43 / EPOCH: 1, LOSS: 58.21778869628906
BATCH: 44 / EPOCH: 1, LOSS: 44.99196243286133
BATCH: 45 / EPOCH: 1, LOSS: 47.436241149902344
BATCH: 46 / EPOCH: 1, LOSS: 51.595970153808594
BATCH: 47 / EPOCH: 1, LOSS: 58.18979263305664
BATCH: 48 / EPOCH: 1, LOSS: 53.59778594970703
BATCH: 49 / EPOCH: 1, LOSS: 50.67637252807617
BATCH: 50 / EPOCH: 1, LOSS: 60.64678955078125

*** AVERAGE: 53.7096 ***

========== EPOCH: 1 END ==========
BATCH: 1 / EPOCH: 2, LOSS: 46.43099594116211
BATCH: 2 / EPOCH: 2, LOSS: 55.19695281982422
BATCH: 3 / EPOCH: 2, LOSS: 52.012481689453125
BATCH: 4 / EPOCH: 2, LOSS: 46.90318298339844
BATCH: 5 / EPOCH: 2, LOSS: 46.00563430786133
BATCH: 6 / EPOCH: 2, LOSS: 46.569419860839844
BATCH: 7 / EPOCH: 2, LOSS: 77.05448913574219
BATCH: 8 / EPOCH: 2, LOSS: 45.550270080566406
BATCH: 9 / EPOCH: 2, LOSS: 43.39899444580078
BATCH: 10 / EPOCH: 2, LOSS: 45.24702453613281
BATCH: 11 / EPOCH: 2, LOSS: 45.976402282714844
BATCH: 12 / EPOCH: 2, LOSS: 40.169986724853516
BATCH: 13 / EPOCH: 2, LOSS: 52.714759826660156
BATCH: 14 / EPOCH: 2, LOSS: 41.76013946533203
BATCH: 15 / EPOCH: 2, LOSS: 38.50575256347656
BATCH: 16 / EPOCH: 2, LOSS: 44.95689392089844
BATCH: 17 / EPOCH: 2, LOSS: 46.28858184814453
BATCH: 18 / EPOCH: 2, LOSS: 44.6292724609375
BATCH: 19 / EPOCH: 2, LOSS: 43.51865768432617
BATCH: 20 / EPOCH: 2, LOSS: 41.56088638305664
BATCH: 21 / EPOCH: 2, LOSS: 39.68405532836914
BATCH: 22 / EPOCH: 2, LOSS: 36.9792366027832
BATCH: 23 / EPOCH: 2, LOSS: 49.263885498046875
BATCH: 24 / EPOCH: 2, LOSS: 46.06732177734375
BATCH: 25 / EPOCH: 2, LOSS: 49.49059295654297
BATCH: 26 / EPOCH: 2, LOSS: 46.59154510498047
BATCH: 27 / EPOCH: 2, LOSS: 44.79001235961914
BATCH: 28 / EPOCH: 2, LOSS: 45.509796142578125
BATCH: 29 / EPOCH: 2, LOSS: 44.42538833618164
BATCH: 30 / EPOCH: 2, LOSS: 42.84892272949219
BATCH: 31 / EPOCH: 2, LOSS: 42.58990478515625
BATCH: 32 / EPOCH: 2, LOSS: 47.5404052734375
BATCH: 33 / EPOCH: 2, LOSS: 44.9470100402832
BATCH: 34 / EPOCH: 2, LOSS: 47.209877014160156
BATCH: 35 / EPOCH: 2, LOSS: 43.423709869384766
BATCH: 36 / EPOCH: 2, LOSS: 44.227413177490234
BATCH: 37 / EPOCH: 2, LOSS: 41.00377655029297
BATCH: 38 / EPOCH: 2, LOSS: 50.184600830078125


```

```python
Epoch: 1/20 | minibatch iteration: 1/400 | loss = 42.00503 |
tensorflow.python.framework.errors_impl.InvalidArgumentError: Incompatible shapes: [20,8752,4] vs. [10,8752,4]
	 [[Node: gradients/sub_grad/BroadcastGradientArgs = BroadcastGradientArgs[T=DT_INT32, _device="/job:localhost/replica:0/task:0/device:CPU:0"](gradients/sub_grad/Shape, gradients/sub_grad/Shape_1)]]


```


```python

```
