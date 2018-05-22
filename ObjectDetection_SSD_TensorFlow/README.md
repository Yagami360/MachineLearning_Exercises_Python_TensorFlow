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



<a id="ID_2"></a>

### 使用するデータセット
- [Pascal VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/)<br>
    ![image](https://user-images.githubusercontent.com/25688193/40175526-55b9d1fe-5a13-11e8-8829-8c5791383ffb.png)<br>
    - 物体検出用のデータセット。<br>
    - [ 画像ファイル名, N x 24 の 2 次元配列 ]<br>
    - 画像のRGB値は 0.0 ~ 1.0 の範囲での値<br>
    - N は、画像の中にある検出物体の数で、画像によって異なる。<br>
    - 24 というのは、位置とクラス名のデータを合わせたデータを表すベクトルになっていて、<br>
    この内、長方形の左上と右下座標である (xmin, ymin, xmax, ymax) の 4 次元の情報で物体を囲む矩形の位置を表し、<br>
    残りの 20 次元で物体のクラスラベルを表す。<br>


<!--
- Open Images Dataset V4
    - https://storage.googleapis.com/openimages/web/download.html
-->

<a id="ID_3"></a>

## コード説明＆実行結果

<a id="ID_3-2"></a>

## TensorFlow を用いた SSD [Single Shot muitibox Detector] の実装 : `main2.py`
> 実装中...

TensorFlow を用いた SSD [Single Shot muitibox Detector] の実装。<br>
ChainerCV や OpenCV にある実装済み or 学習済み SSD モジュールのような高レベル API 使用せずに、TensorFlow で実装している。<br>

<a id="ID_3-2-1"></a>

### 使用するライブラリ

- TensorFlow ライブラリ
    - xxx

- その他ライブラリ
    - xxx

<br>

<a id="ID_3-2-2"></a>

### コードの内容説明

<a id="ID_3-2-1"></a>

以下、コードの説明。<br>

#### 1. Poscal VOC2007 データセットにある、画像、物体情報の読み込み＆抽出

まず、物体検出用のデータセットである Poscal VOC2007 データセットにある、画像、物体情報の読み込み＆抽出処理を行う。<br>

- これらのデータセットは、pickle 形式 `VOC2007.pkl` で保管されているので、以下の処理で読み込みを行う。<br>
```python
[main2.py]
def main():
    ...
    with open( dataset_path + 'VOC2007.pkl', 'rb' ) as file:
        data = pickle.load( file )
        keys = sorted( data.keys() )
```
- pickle ファイルの中身は、ファイル名 `['000001.jpg', '000002.jpg', '000003.jpg', '000004.jpg', '000006.jpg', '000008.jpg', '000010.jpg', ...]` を key とする辞書型の構造になっている。<br>
- この処理により、`data` には、`[ファイル名 , N × 24 次元の配列]` の情報が格納される。<br>
- この内、`N` は、画像の中にある検出物体の数で、画像によって異なる。<br>
- 又、`24 次元` は、位置とクラス名のデータを合わせたデータを表すベクトルになっていて、<br>
この内、長方形の左上、右下座標 (xmin, ymin, xmax, ymax) の 4 次元の情報で物体を囲む矩形の位置を表し、<br>
残りの 20 次元で、この矩形の所属クラス名を表す。<br>
- なお、20 種類の物体を識別する場合、これら 20 種類のどれにも該当しないというクラスも必要になるので、出力層の出力は21次元だけ必要となる。<br>


#### 2. SSD モデルの各種パラメーターの設定

SSD モデルの各種パラメーターの設定を行う。<br>
    
- この設定は、`SingleShotMultiBoxDetector` クラスのインスタンス作成時の引数にて行う。<br>
```python
[main2.py]
def main():
    ...
    ssd = SingleShotMultiBoxDetector(
              session = tf.Session(),
              epochs = 20,                      # モデルの学習時のエポック数
              batch_size = 10,                  # モデルの学習時の、ミニバッチサイズ
              eval_step = 1,                    #
              save_step = 100,                  #
              image_height = 300,               # 入力画像の高さ（ピクセル数）
              image_width = 300,                #
              n_channels = 3,                   #
              n_classes = 21,                   # 識別クラス数（物体の種類＋１）
              n_boxes = [ 4, 6, 6, 6, 6, 6 ]    # 
      )
```
- `epochs` は、学習時 `ssd.fit(...)` での総エポック数。
- `batch_size` は、学習時 `ssd.fit(...)` でのミニバッチサイズ。
- `image_height` は、入力画像データの高さ（ピクセル単位）<br>
- `image_width` は、入力画像データの幅（ピクセル単位）<br>
- `n_channels` は、入力画像データのチャンネル数（ピクセル単位）<br>
  ⇒ 本コードでは、300 × 300 の画像で実施。
- `n_classes` は、
- `n_boxes` は、

### 3. SSD モデルの構築

以下のアーキテクチャ図に従って、SSD モデルを構築する。<br>
この処理は、`SingleShotMultiBoxDetector` クラスの `model()` メソッドで行う。 <br>

![image](https://user-images.githubusercontent.com/25688193/39536606-0e4f1416-4e72-11e8-91e2-82b516706bae.png)<br>

- SSD モデルの構築では、まず初めにベースネットワークとなる VGG-16 モデルを構築する。<br>
    ```python
    [SingleShotMultiBoxDetector.py]
    class SingleShotMultiBoxDetector( NeuralNetworkBase ):
    ...
    def model():
        #-----------------------------------------------------------------------------
        # ベースネットワーク
        #-----------------------------------------------------------------------------
        self.base_vgg16.model()
        ...
    ```
    - この SSD のベースネットワークとしての VGG16 は、全結合層を畳み込み層に置き換えたモデルであり、<br>
    この処理は `BaseNetworkVGG16.model()` メソッドで行う。<br>
    ```python
    [BaseNetwork.py]
    class BaseNetworkVGG16( BaseNetwork ):
    ...
    def model():
        #-----------------------------------------------------------------------------
        # layer 1
        #-----------------------------------------------------------------------------
        self.conv1_1_op = self.convolution_layer( 
                              input_tsr = self.X_holder, 
                              filter_height = 3, filter_width = 3,
                              n_output_channels = 64,
                              n_strides = 1,
                              name = "conv1_1", 
                              reuse = False
                          )

        self.conv1_2_op = self.convolution_layer( 
                              input_tsr = self.conv1_1_op, 
                              filter_height = 3, filter_width = 3,
                              n_output_channels = 64,
                              n_strides = 1,
                              name = "conv1_2", 
                              reuse = False
                          )

        self.pool1_op = self.pooling_layer( input_tsr = self.conv1_2_op, name = "pool1", reuse = False )

        #-----------------------------------------------------------------------------
        # layer 2
        #-----------------------------------------------------------------------------
        self.conv2_1_op = self.convolution_layer( 
                              input_tsr = self.pool1_op, 
                              filter_height = 3, filter_width = 3,
                              n_output_channels = 128,
                              n_strides = 1,
                              name = "conv2_1",
                              reuse = False
                          )

        self.conv2_2_op = self.convolution_layer( 
                              input_tsr = self.conv2_1_op, 
                              filter_height = 3, filter_width = 3,
                              n_output_channels = 128,
                              n_strides = 1,
                              name = "conv2_2",
                              reuse = False
                          )

        self.pool2_op = self.pooling_layer( input_tsr = self.conv2_2_op, name = "pool2", reuse = False )

        #-----------------------------------------------------------------------------
        # layer 3
        #-----------------------------------------------------------------------------
        self.conv3_1_op = self.convolution_layer( 
                              input_tsr = self.pool2_op, 
                              filter_height = 3, filter_width = 3,
                              n_output_channels = 256,
                              n_strides = 1,
                              name = "conv3_1",
                              reuse = False
                          )

        self.conv3_2_op = self.convolution_layer( 
                              input_tsr = self.conv3_1_op, 
                              filter_height = 3, filter_width = 3,
                              n_output_channels = 256,
                              n_strides = 1,
                              name = "conv3_2",
                              reuse = False
                          )

        self.conv3_3_op = self.convolution_layer( 
                              input_tsr = self.conv3_2_op, 
                              filter_height = 3, filter_width = 3,
                              n_output_channels = 256,
                              n_strides = 1,
                              name = "conv3_3",
                              reuse = False
                          )

        self.pool3_op = self.pooling_layer( input_tsr = self.conv3_3_op, name = "pool3", reuse = False )

        #-----------------------------------------------------------------------------
        # layer 4
        #-----------------------------------------------------------------------------
        self.conv4_1_op = self.convolution_layer( 
                              input_tsr = self.pool3_op, 
                              filter_height = 3, filter_width = 3,
                              n_output_channels = 512,
                              n_strides = 1,
                              name = "conv4_1",
                              reuse = False
                          )

        self.conv4_2_op = self.convolution_layer( 
                              input_tsr = self.conv4_1_op, 
                              filter_height = 3, filter_width = 3,
                              n_output_channels = 512,
                              n_strides = 1,
                              name = "conv4_2",
                              reuse = False
                          )

        #-----------------------------------------------------------------------------
        # model output
        #-----------------------------------------------------------------------------
        self._y_out_op = self.conv4_2_op

        return self._y_out_op
    ```
    ```python
    [BaseNetwork.py]
    class BaseNetworkVGG16( BaseNetwork ):
    ...
    def convolution_layer( 
            self, 
            input_tsr, 
            filter_height, filter_width, 
            n_strides,
            n_output_channels, 
            name = "conv", reuse = False
        ):
        """
        畳み込み層を構築する。
        
        [Input]
            input_tsr : Tensor / Placeholder
                畳み込み層への入力 Tensor
            filter_height : int
                フィルターの高さ（カーネル行列の行数）
            filter_width : int
                フィルターの幅（カーネル行列の列数）
            n_output_channels : int
                畳み込み処理後のデータのチャンネル数
        [Output]
            out_op : Operator
                畳み込み処理後の出力オペレーター
        """
        
        # Variable の名前空間（スコープ定義）
        with tf.variable_scope( name, reuse = reuse ):
            # 入力データ（画像）のチャンネル数取得
            input_shape = input_tsr.get_shape().as_list()
            n_input_channels = input_shape[-1]

            # 畳み込み層の重み（カーネル）を追加
            # この重みは、畳み込み処理の画像データに対するフィルタ処理（特徴マップ生成）に使うカーネルを表す Tensor のことである。
            # kernel_shape : [ [(filterの高さ) , (filterの幅) , (入力チャネル数) , (出力チャネル数) ]
            kernel = self.init_weight_variable( input_shape = [filter_height, filter_width, n_input_channels, n_output_channels] )
            bias = self.init_bias_variable( input_shape = [n_output_channels] )

            # 畳み込み演算
            conv_op = tf.nn.conv2d(
                          input = input_tsr,
                          filter = kernel,
                          strides = [1, n_strides, n_strides, 1],   # strides[0] = strides[3] = 1. とする必要がある
                          padding = "SAME",
                          name = name
                      )
            
            # 活性化関数として Relu で出力
            out_op = tf.nn.relu( tf.add(conv_op,bias) )

        return out_op
    ```
    ```python
    [BaseNetwork.py]
    class BaseNetworkVGG16( BaseNetwork ):
    ...
    def pooling_layer( self, input_tsr, name = "pool", reuse = False ):
        """
        VGG16 のプーリング層を構築する。

        [Input]
            input_tsr : Tensor / Placeholder
                畳み込み層への入力 Tensor
        [Output]
            pool_op : Operator
                プーリング処理後の出力オペレーター
        """
        # Variable の名前空間（スコープ定義）
        with tf.variable_scope( name, reuse = reuse ):
            # Max Pooling 演算
            pool_op = tf.nn.max_pool(
                          value = input_tsr,
                          ksize = [1, 2, 2, 1],
                          strides = [1, 2, 2, 1],
                          padding = "SAME",
                          name = name
                      )

        return pool_op
    ```

- 次に、ベースネットワークの後段に続くレイヤーを構築する。<br>
    ```python
    [SingleShotMultiBoxDetector.py]
    class SingleShotMultiBoxDetector( NeuralNetworkBase ):
    ...
    def model():
        ...
        #-----------------------------------------------------------------------------
        # layer 6
        #-----------------------------------------------------------------------------
        self.conv6_op = self.convolution_layer( 
                            input_tsr = self.base_vgg16._y_out_op, 
                            filter_height = 3, filter_width = 3,
                            n_strides = 1,
                            n_output_channels = 1024,
                            name = "conv6", 
                            reuse = False
                        )

        self.pool6_op = self.pooling_layer( input_tsr = self.conv6_op, name = "pool6", reuse = False )

        #-----------------------------------------------------------------------------
        # layer 7
        #-----------------------------------------------------------------------------
        self.conv7_op = self.convolution_layer( 
                            input_tsr = self.pool6_op, 
                            filter_height = 1, filter_width = 1,
                            n_strides = 1,
                            n_output_channels = 1024,
                            name = "conv7", 
                            reuse = False
                        )

        #-----------------------------------------------------------------------------
        # layer 8
        #-----------------------------------------------------------------------------
        self.conv8_1_op = self.convolution_layer( 
                              input_tsr = self.conv7_op, 
                              filter_height = 1, filter_width = 1,
                              n_strides = 1,
                              n_output_channels = 256,
                              name = "conv8_1", 
                              reuse = False
                          )

        self.conv8_2_op = self.convolution_layer( 
                              input_tsr = self.conv8_1_op, 
                              filter_height = 3, filter_width = 3,
                              n_strides = 2,
                              n_output_channels = 512,
                              name = "conv8_2", 
                              reuse = False
                          )

        #-----------------------------------------------------------------------------
        # layer 9
        #-----------------------------------------------------------------------------
        self.conv9_1_op = self.convolution_layer( 
                              input_tsr = self.conv8_2_op, 
                              filter_height = 1, filter_width = 1,
                              n_strides = 1,
                              n_output_channels = 128,
                              name = "conv9_1", 
                              reuse = False
                          )

        self.conv9_2_op = self.convolution_layer( 
                              input_tsr = self.conv9_1_op, 
                              filter_height = 3, filter_width = 3,
                              n_strides = 2,
                              n_output_channels = 256,
                              name = "conv9_2", 
                              reuse = False
                          )

        #-----------------------------------------------------------------------------
        # layer 10
        #-----------------------------------------------------------------------------
        self.conv10_1_op = self.convolution_layer( 
                               input_tsr = self.conv9_2_op, 
                               filter_height = 1, filter_width = 1,
                               n_strides = 1,
                               n_output_channels = 128,
                               name = "conv10_1", 
                               reuse = False
                           )

        self.conv10_2_op = self.convolution_layer( 
                              input_tsr = self.conv10_1_op, 
                              filter_height = 3, filter_width = 3,
                              n_strides = 2,
                              n_output_channels = 256,
                              name = "conv10_2", 
                              reuse = False
                          )

        #-----------------------------------------------------------------------------
        # layer 11
        #-----------------------------------------------------------------------------
        self.conv11_1_op = self.convolution_layer( 
                               input_tsr = self.conv10_2_op, 
                               filter_height = 1, filter_width = 1,
                               n_strides = 1,
                               n_output_channels = 128,
                               name = "conv11_1", 
                               reuse = False
                           )

        self.conv11_2_op = self.convolution_layer( 
                              input_tsr = self.conv11_1_op, 
                              filter_height = 3, filter_width = 3,
                              n_strides = 3,
                              n_output_channels = 256,
                              name = "conv11_2", 
                              reuse = False
                          )
        ...
    ```

- 更に、Extra Feature Maps のレイヤーを構築する。<br>
    - この Extra Feature Maps は、各畳み込み層の出力から物体検出モジュールへの畳み込みで、<br>
    上記アーキテクチャ図の青線部分に対応したものである。<br>
    ```python
    [SingleShotMultiBoxDetector.py]
    class SingleShotMultiBoxDetector( NeuralNetworkBase ):
    ...
    def model():
        ...
        #-----------------------------------------------------------------------------
        # Extra Feature Maps （アーキテクチャ図の青線部分＜各層 → Detections per Classes＞）
        #-----------------------------------------------------------------------------
        self.fmaps = []

        # extra feature map 1
        self.fmaps.append( 
            self.convolution_layer(
                input_tsr = self.base_vgg16._y_out_op,
                filter_height = 3, filter_width = 3,
                n_strides = 1,
                n_output_channels = self.n_boxes[0] * ( self.n_classes + 4 ),
                name = "fmap1", 
                reuse = False
            )
        )

        # extra feature map 2
        self.fmaps.append( 
            self.convolution_layer(
                input_tsr = self.conv7_op,
                filter_height = 3, filter_width = 3,
                n_strides = 1,
                n_output_channels = self.n_boxes[1] * ( self.n_classes + 4 ),
                name = "fmap2", 
                reuse = False
            )
        )

        # extra feature map 3
        self.fmaps.append( 
            self.convolution_layer(
                input_tsr = self.conv8_2_op,
                filter_height = 3, filter_width = 3,
                n_strides = 1,
                n_output_channels = self.n_boxes[2] * ( self.n_classes + 4 ),
                name = "fmap3", 
                reuse = False
            )
        )

        # extra feature map 4
        self.fmaps.append( 
            self.convolution_layer(
                input_tsr = self.conv9_2_op,
                filter_height = 3, filter_width = 3,
                n_strides = 1,
                n_output_channels = self.n_boxes[3] * ( self.n_classes + 4 ),
                name = "fmap4", 
                reuse = False
            )
        )

        # extra feature map 5
        self.fmaps.append( 
            self.convolution_layer(
                input_tsr = self.conv10_2_op,
                filter_height = 3, filter_width = 3,
                n_strides = 1,
                n_output_channels = self.n_boxes[4] * ( self.n_classes + 4 ),
                name = "fmap5", 
                reuse = False
            )
        )

        # extra feature map 6
        self.fmaps.append( 
            self.convolution_layer(
                input_tsr = self.conv11_2_op,
                filter_height = 1, filter_width = 1,
                n_strides = 1,
                n_output_channels = self.n_boxes[5] * ( self.n_classes + 4 ),
                name = "fmap6", 
                reuse = False
            )
        )

        #-----------------------------------------------------------------------------
        # extra feature maps による物体の所属クラスとスコア値の算出
        #-----------------------------------------------------------------------------
        fmaps_reshaped = []
        for i, fmap in zip( range(len(self.fmaps)), self.fmaps ):
            # [batch_size=None, image_height, image_width, n_channles]
            output_shape = fmap.get_shape().as_list()
            
            # extra feature map の高さ、幅
            fmap_height = output_shape[1]
            fmap_width = output_shape[2]
            
            # [batch_size=None, image_height, image_width, n_channles] → [batch_size=None, xxx, self.n_classes + 4 ] に　reshape
            fmap_reshaped = tf.reshape( fmap, [-1, fmap_width * fmap_height * self.n_boxes[i], self.n_classes + 4] )

            #
            fmaps_reshaped.append( fmap_reshaped )

        # reshape した fmap を結合
        # Tensor("concat:0", shape=(?, 8752, 25), dtype=float32)
        # 25 = 21(クラス数) + 4( (xmin, ymin, xmax, ymax) の 4 次元の情報で物体を囲む矩形の位置 )
        fmap_concatenated = tf.concat( fmaps_reshaped, axis = 1 )

        # 特徴マップが含む物体の確信度と予想位置（形状のオフセット）
        # pred_confidences.shape = [None, 8752, 21] | 21: クラス数
        # pred_locations.shape = [None, 8752, 4]  | 4 : (xmin, ymin, xmax, ymax) の 4 次元の情報で物体を囲む矩形の位置
        self.pred_confidences = fmap_concatenated[ :, :, :self.n_classes ]
        self.pred_locations = fmap_concatenated[ :, :, self.n_classes: ]
        ...
    ```

#### 4. デフォルトボックスの生成
各 extra feature map に対応したデフォルトボックスを生成する。<br>
この処理は、`SingleShotMultiBoxDetector` クラスの `generate_default_boxes_in_fmaps(...)` メソッドで行う。 <br>

- 
```python
[SingleShotMultiBoxDetector.py]
class 
```

#### 5. 損失関数の設定
SSD モデルの損失関数を設定する。<br>
この設定は、`SingleShotMultiBoxDetector` クラスの `loss(...)` メソッドにて行う。



#### 6. 構築した SSD モデルによる学習


#### 7. 学習済み SSD モデルによる推論フェイズ


<br>

### コードの実行結果

|パラメータ名|値（実行条件１）|
|---|---|
|xxx|xxx|
|xxx|xxx|

<br>

---

<a id="ID_4"></a>

## 背景理論

<a id="ID_4-1"></a>

## 背景理論１

---

## デバッグメモ

```python
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

```python
BATCH: 1 / EPOCH: 1, LOSS: 77.99568176269531
BATCH: 2 / EPOCH: 1, LOSS: 72.53934478759766
BATCH: 3 / EPOCH: 1, LOSS: 66.68212127685547
BATCH: 4 / EPOCH: 1, LOSS: 57.453941345214844
BATCH: 5 / EPOCH: 1, LOSS: 65.49876403808594
...
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
...
BATCH: 35 / EPOCH: 2, LOSS: 43.423709869384766
BATCH: 36 / EPOCH: 2, LOSS: 44.227413177490234
BATCH: 37 / EPOCH: 2, LOSS: 41.00377655029297
BATCH: 38 / EPOCH: 2, LOSS: 50.184600830078125
```

```python
pred_confs : ndarray
shape [1,8752,21]   [1, デフォルトボックスの総数, クラス数]
    [0][0] デフォルトボックス１の各クラスの所属の確信度
    array([-0.14629412,  0.38752401,  0.63575637, -0.41271916,  0.23870134,
            0.31687331,  0.21811765, -0.03108937,  0.10936093,  0.0263918 ,        0.13831818,  0.24440765, -0.31142211, -0.10909909, -0.46765071,        0.09381628, -0.04382503, -0.42369995,  0.43960947,  0.26494616,       -0.33809745], 
        dtype=float32)

hist : list<クラス数>
    [144, 211, 395, 440, 130, 123, 605, 448, 686, 275, 404, 228, 181, 256, ...]

possibilities : list<float64>
    [0] 0.082660712503914727, 
    [1] 0.065760134891456545, 
    ...
    [8751] 0.051121967036249438

indicies : ndarray
    shape = [200]
    [0] 3521
    [1] 1865
    ...
    [199] 5146

```
