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
    1. [コード説明＆実行結果 : `main1.py`](#ID_3-1)
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

- VOC 2007
    - http://host.robots.ox.ac.uk/pascal/VOC/

![image](https://user-images.githubusercontent.com/25688193/40175526-55b9d1fe-5a13-11e8-8829-8c5791383ffb.png)

- Open Images Dataset V4
    - https://storage.googleapis.com/openimages/web/download.html


<a id="ID_3"></a>

## コード説明＆実行結果

<a id="ID_3-1"></a>

## コード説明＆実行結果１ : `main1.py`
> 実装中...

- xxx データセットを使用
- 特徴行列 `X_features` は、特徴数 x 個 × サンプル数 x 個 :<br> `X_features = `
- 教師データ `y_labels` は、サンプル数 x 個 : <br >`y_labels = `
- トレーニングデータ xx% 、テストデータ xx% の割合で分割 : <br>`sklearn.cross_validation.train_test_split( test_size = , random_state =  )`
- 正規化処理を実施して検証する。<br> 

<br>

---

<a id="ID_3-2"></a>

## コード説明＆実行結果２ : `main2.py`
> 実装中...

<br>

---

<a id="ID_4"></a>

## 背景理論

<a id="ID_4-1"></a>

## 背景理論１

---

## デバッグ情報

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




