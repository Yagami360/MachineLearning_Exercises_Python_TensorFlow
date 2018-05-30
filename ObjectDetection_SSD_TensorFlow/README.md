## TensorFlow を用いた SSD [Single Shot muitibox Detector] の実装と簡単な応用

> **実装中...（別レポジトリに移行予定）**

ニューラルネットワークによる一般物体検出アルゴリズムの１つである、SSD [Single Shot muitibox Detector] を TensorFlow で実装。（ChainerCV や OpenCV 等にある実装済み or 学習済み SSD モジュールのような高レベル API 使用せずに、TensorFlow でイチから実装している。）<br>

この `README.md` ファイルには、各コードの実行結果、コードの内容の説明を記載しています。<br>
又、分かりやすいように `main.py` ファイル毎に１つの完結した実行コードにしています。<br>

尚、SSD [Single Shot muitibox Detector] に関しての、背景理論は以下のサイトに記載してあります。<br>

- [星の本棚 : ニューラルネットワーク / ディープラーニング](http://yagami12.hatenablog.com/entry/2017/09/17/111935#ID_11-4)


### 項目 [Contents]

1. [使用するデータセット](#ID_2)
1. [コード説明＆実行結果](#ID_3)
    1. [TensorFlow を用いた SSD [Single Shot muitibox Detector] の実装 : `main2.py`](#ID_3-2)
        1. [コードの内容説明](#ID_3-2-2)
            1. [Poscal VOC2007 データセットにある、画像、物体情報の読み込み＆抽出](#ID_3-2-2-1)
            1. [SSD モデルの各種パラメーターの設定](#ID_3-2-2-2)
            1. [SSD モデルの構築](#ID_3-2-2-3)
            1. [デフォルトボックスの生成](#ID_3-2-2-4)
            1. [損失関数の設定](#ID_3-2-2-5)
            1. [Optimizer の設定](#ID_3-2-2-6)
            1. [構築した SSD モデルの学習](#ID_3-2-2-7)
            1. [学習済み SSD モデルによる推論フェイズ](#ID_3-2-2-8)
        1. [コードの実行結果](#ID_3-2-3)
1. [背景理論](#ID_4)
1. [参考サイト](#ID_5)


<a id="ID_1"></a>

---

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

|label index|object name|
|---|---|
|0|'aeroplane'|
|1|'bicycle'|
|2|'bird'|
|3|'boat'|
|4|'bottle'|
|5|'bus'|
|6|'car'|
|7|'cat'|
|8|'chair'|
|9|'cow'|
|10|'diningtable'|
|11|'dog'|
|12|'horse'|
|13|'motorbike'|
|14|'person'|
|15|'pottedplant'|
|16|'sheep'|
|17|'sofa'|
|18|'train'|
|19|'tvmonitor|
|20|Unknown|

- [Microsoft COCO](http://mscoco.org/explore/)
    - 80種類のカテゴリーからなる物体検出用のデータセット

‘person‘, ‘bicycle‘, ‘car‘, ‘motorcycle‘, ‘airplane‘, ‘bus‘, ‘train‘, ‘truck’, ‘boat‘, ‘traffic light’, ‘fire hydrant’, ‘stop sign’, ‘parking meter’, ‘bench’, ‘bird‘, ‘cat‘, ‘dog‘, ‘horse‘, ‘sheep‘, ‘cow‘, ‘elephant’, ‘bear’, ‘zebra’, ‘giraffe’, ‘backpack’, ‘umbrella’, ‘handbag’, ‘tie’, ‘suitcase’, ‘frisbee’, ‘skis’, ‘snowboard’, ‘sports ball’, ‘kite’, ‘baseball bat’, ‘baseball glove’, ‘skateboard’, ‘surfboard’, ‘tennis racket’, ‘bottle‘, ‘wine glass’, ‘cup’, ‘fork’, ‘knife’, ‘spoon’, ‘bowl’, ‘banana’, ‘apple’, ‘sandwich’, ‘orange’, ‘broccoli’, ‘carrot’, ‘hot dog’, ‘pizza’, ‘donut’, ‘cake’, ‘chair‘, ‘couch‘, ‘potted plant‘, ‘bed’, ‘dining table‘, ‘toilet’, ‘tv‘, ‘laptop’, ‘mouse’, ‘remote’, ‘keyboard’, ‘cell phone’, ‘microwave’, ‘oven’, ‘toaster’, ‘sink’, ‘refrigerator’, ‘book’, ‘clock’, ‘vase’, ‘scissors’, ‘teddy bear’, ‘hair drier’, ‘toothbrush’

- MS COCO API : 

- Open Images Dataset V4
    - https://storage.googleapis.com/openimages/web/download.html


---

<a id="ID_3"></a>

## コード説明＆実行結果

<a id="ID_3-2"></a>

## TensorFlow を用いた SSD [Single Shot muitibox Detector] の実装 : `main2.py`
TensorFlow を用いた SSD [Single Shot muitibox Detector] の実装。<br>
ChainerCV や OpenCV 等にある実装済み or 学習済み SSD モジュールのような高レベル API 使用せずに、TensorFlow で実装している。<br>

<!--
<a id="ID_3-2-1"></a>

### ☆ 使用するライブラリ

- TensorFlow ライブラリ
    - xxx

- その他ライブラリ
    - xxx

<br>
-->

<a id="ID_3-2-2"></a>

### **☆ コードの内容説明**
以下、コードの説明。<br>

<a id="ID_3-2-2-1"></a>

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

<br>

<a id="ID_3-2-2-2"></a>

#### 2. SSD モデルの各種パラメーターの設定
SSD モデルの各種パラメーターの設定を行う。<br>
この設定は、`SingleShotMultiBoxDetector` クラスのインスタンス作成時の引数にて行う。<br>

```python
[main2.py]
def main():
    ...
    ssd = SingleShotMultiBoxDetector(
              session = tf.Session(),
              epochs = 20,                      
              batch_size = 10,                  
              eval_step = 1,                    
              save_step = 100,                  
              image_height = 300,               
              image_width = 300,                
              n_channels = 3,                   
              n_classes = 21,                   
              n_boxes = [ 4, 6, 6, 6, 6, 6 ]     
      )
```
- 引数 `epochs` は、後の学習時 `ssd.fit(...)` での総エポック数。
- 引数 `batch_size` は、後の学習時 `ssd.fit(...)` でのミニバッチサイズ。
- 引数 `image_height` は、入力画像データの高さ（ピクセル単位）<br>
- 引数 `image_width` は、入力画像データの幅（ピクセル単位）<br>
- 引数 `n_channels` は、入力画像データのチャンネル数（ピクセル単位）<br>
  ⇒ 本コードでは、300 × 300 の画像で実施。
- 引数 `n_classes` は、識別クラス数。<br>
  但し、どの物体にも属さないこと示す値として、識別物体数に + 1 された値となることに注意。<br>
- 引数 `n_boxes` は、各特徴マップにおけるデフォルトボックス数。<br>

<br>

<a id="ID_3-2-2-3"></a>

### 3. SSD モデルの構築
SSD モデルを構築する。<br>
より詳細には、以下のアーキテクチャ図に従って、マルチスケール特徴マップのための各種畳み込み層の構築を行う。<br>
![image](https://user-images.githubusercontent.com/25688193/39536606-0e4f1416-4e72-11e8-91e2-82b516706bae.png)<br>

この処理は、`SingleShotMultiBoxDetector` クラスの `model()` メソッドで行う。 <br>

```python
def main():
    ...
    #======================================================================
    # モデルの構造を定義する。
    # Define the model structure.
    # ex) add_op = tf.add(tf.mul(x_input_holder, weight_matrix), b_matrix)
    #======================================================================
    ssd.model()
```

- SSD モデルの構築では、まず初めにベースネットワークとなる VGG-16 モデルを構築する。<br>
    ```python
    [SingleShotMultiBoxDetector.py / class SingleShotMultiBoxDetector]
    def model():
        #-----------------------------------------------------------------------------
        # ベースネットワーク
        #-----------------------------------------------------------------------------
        self.base_vgg16.model()
        ...
    ```
    - この SSD のベースネットワークとしての VGG16 は、従来の VGG16 における全結合層を、畳み込み層に置き換えたモデルであり、以下のように `BaseNetworkVGG16` クラスの `model()` メソッドで定義されたモデルである。<br>
    ```python
    [BaseNetwork.py / class BaseNetworkVGG16]
    def model( self ):
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
    - 尚、上記 `model()` メソッド内でコールされている畳み込み処理関数 `convolution_layer(...)` は、以下のように定義されている。
    ```python
    [BaseNetwork.py / class BaseNetworkVGG16]
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
    - 同様に、上記 `model()` メソッド内でコールされているプーリング処理関数 `pooling_layer(...)` は、以下のように定義されている。
    ```python
    [BaseNetwork.py / class BaseNetworkVGG16]
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
    [SingleShotMultiBoxDetector.py / class SingleShotMultiBoxDetector]
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
    [SingleShotMultiBoxDetector.py / class SingleShotMultiBoxDetector]
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
        # pred_confs.shape = [None, 8752, 21] | 21: クラス数
        # pred_locs.shape = [None, 8752, 4]  | 4 : (xmin, ymin, xmax, ymax) の 4 次元の情報で物体を囲む矩形の位置
        self.pred_cons = fmap_concatenated[ :, :, :self.n_classes ]
        self.pred_locs = fmap_concatenated[ :, :, self.n_classes: ]
        ...
    ```
<br>

<a id="ID_3-2-2-3"></a>

#### 4. デフォルトボックスの生成
各 extra feature map に対応した一連のデフォルトボックス群を生成する。<br>
この処理は、`SingleShotMultiBoxDetector` クラスの `generate_default_boxes_in_fmaps(...)` メソッドで行う。 <br>

```python
def main():
    ...
    # 特徴マップに対応した一連のデフォルト群の生成
    ssd.generate_default_boxes_in_fmaps()
```

デフォルトボックスに関するアスペクト比のマップ `aspect_set` 、及びスケール値の最大値 `scale_max`、最小値 `scale_min` といったパラメータの設定は、このメソッド内で行っている。<br>

```python
[SingleShotMultiBoxDetector.py / class SingleShotMultiBoxDetector]
def generate_default_boxes_in_fmaps( self ):
    """
    各 extra feature map に対応したデフォルトボックスを生成する。

    [Output]
        self._default_box_set : DefaultBoxSet
            生成した 一連のデフォルトボックス群を表すクラス DefaultBoxSet のオブジェクト
    """
    # extra feature map の形状（ピクセル単位）
    fmap_shapes = [ fmap.get_shape().as_list() for fmap in self.fmaps ]

    # 各 extra feature maps に対応した、各デフォルトボックスのアスペクト比
    aspect_set = [
                     [1.0, 1.0, 2.0, 1.0/2.0],                 # extra fmap 1
                     [1.0, 1.0, 2.0, 1.0/2.0, 3.0, 1.0/3.0],   # extra fmap 2
                     [1.0, 1.0, 2.0, 1.0/2.0, 3.0, 1.0/3.0],   #
                     [1.0, 1.0, 2.0, 1.0/2.0, 3.0, 1.0/3.0],
                     [1.0, 1.0, 2.0, 1.0/2.0, 3.0, 1.0/3.0],
                     [1.0, 1.0, 2.0, 1.0/2.0, 3.0, 1.0/3.0],
                 ]
    ...
```

そして、各特徴マップ、アスペクト比、スケール値に対応した一連のデフォルトボックス群は、このメソッド内で処理される、`DefaultBoxSet` クラスのオブジェクト `self._default_box_set` として表現され、<br>
実際の一連のデフォルトボックス群の生成は、このクラス `DefaultBoxSet` の `generate_boxes(...)` メソッドで行なう。<br>

```python
[SingleShotMultiBoxDetector.py / class SingleShotMultiBoxDetector]
def generate_default_boxes_in_fmaps( self ):
    ...
    # 一連のデフォルトボックス群を表すクラス DefaultBoxSet のオブジェクトを生成
    self._default_box_set = DefaultBoxSet( scale_min = 0.2, scale_max = 0.9 )
        
    # 一連のデフォルトボックス群を生成
    self._default_box_set.generate_boxes( fmaps_shapes = fmap_shapes, aspect_set = aspect_set )

    return self._default_box_set
```

- この `generate_boxes(...)` メソッドでは、以下の処理が行われる。<br>
    1. 各特徴マップ（のサイズ `fmaps_shape` ）`k` に対して 、スケール値 `s_k` を計算。<br>
    ```python
    [DefaultBox.py / class DefaultBoxSet]
    def generate_boxes( self, fmaps_shapes, aspect_set ):
        ...
        for k, fmap_shape in enumerate( fmaps_shapes ):
            s_k = self.calc_scale( k )
    ```
        
    2. 各アスペクト比 `aspects` と、各特徴マップ k の高さ `fmap_height`、幅 `fmap_width` から構成される各セルのグリッド（１×１ピクセル）`x`, `y` に対して、長方形の中心座標 `center_x`, `center_y`、アスペクト比 `aspect`、デフォルトボックスの高さ `box_height`、幅 `box_width` を抽出 or 計算する。<br>
    ここで、デフォルトボックスの各特徴マップ `k` 、及び各スケール値 `s_k` に対する、幅と高さは、以下の式で算出する。<br>
        ![image](https://user-images.githubusercontent.com/25688193/40353511-a20ebee8-5dec-11e8-99d2-8b5d8bb8e96f.png)<br>
    ```python
    [DefaultBox.py / class DefaultBoxSet]
    def generate_boxes( self, fmaps_shapes, aspect_set ):
        ...
        # fmap_shape[0] にはバッチサイズが入力されている。
        fmap_width  = fmap_shape[1]
        fmap_height = fmap_shape[2]

        # 引数で指定されたアスペクト比の集合 aspect_set から、
        # 特徴マップ k に対してのアスペクト比のリスト aspects を抽出        
        aspects = aspect_set[k]

        # 特徴マップ k に対してのアスペクト比のリスト aspects から各アスペクト値 aspect を抽出
        for aspect in aspects:
            # 特徴マップのセルのグリッド（1 pixcel）に関してのループ処理
            for y in range( fmap_height ):
                # セルのグリッドの中央を 0.5 として計算 
                center_y = ( y + 0.5 ) / float( fmap_height )

                for x in range( fmap_width ):
                    center_x = ( x + 0.5 ) / float( fmap_width )

                    # 各特徴マップ k とスケール値 s_k から、デフォルトボックスの幅と高さを計算
                    box_width = s_k * np.sqrt( aspect )
                    box_height = s_k / np.sqrt( aspect )
    ```

    3. これら中心座標 `center_x`,`center_y`、ボックスの高さ `box_hight` 、幅 `box_width` 、スケール値 `s_k`、アスペクト比 `aspect` を属性にもつデフォルトボックス `default_box` を生成する。<br>（ここで、`group_id` と `id` は、各デフォルトボックスを識別するための便宜上の ID で、`id` 値は各デフォルトボックスに固有の値、同様の `group_id` 値は、同様の特徴マップ `k` を元に生成したデフォルトボックスであることを示している。）<br>
    ```python
    [DefaultBox.py / class DefaultBoxSet]
    def generate_boxes( self, fmaps_shapes, aspect_set ):
        ...
        default_box = DefaultBox(
                          group_id = k + 1,
                          id = id,
                          center_x = center_x, center_y = center_y,
                          width = box_width, height = box_height, 
                          scale = s_k,
                        aspect = aspect
                      )
    ```

    4. 生成したデフォルトボックスをリスト `self._default_boxes` に追加する。<br>
    ```python
    [DefaultBox.py / class DefaultBoxSet]
    def generate_boxes( self, fmaps_shapes, aspect_set ):
        ...
        self.add_default_box( default_box )
    ```
    ```python
    [DefaultBox.py / class DefaultBoxSet]
    def add_default_box( self, default_box ):
        """
        引数で指定されたデフォルトボックスを、一連のデフォルトボックスのリストに追加する。

        [Input]
            default_box : DefaultBox
                デフォルトボックスのクラス DefaultBox のオブジェクト

        """
        self._default_boxes.append( default_box )

        return
    ```
        

- ここで、バウンディングボックスの形状回帰のためのスケール値 `s_k` の計算は、`DefaultBoxSet` クラスのメソッド `calc_scale(...)` で行われる。<br>
    具体的には、各特徴マップ k (=1~6) についてのデフォルトボックスのスケール `s_k` を、特徴マップ `k` 、及び、`DefaultBoxSet` クラスのオブジェクト作成時に設定したスケール値の最大値 `scale_max`、最小値 `scale_min` に基づき、以下のように計算している。<br>
    ![image](https://user-images.githubusercontent.com/25688193/40351479-7a5fe87c-5de7-11e8-89bf-192c07e89e0a.png)<br>

    ```python
    [DefaultBox.py / class DefaultBoxSet]
    def calc_scale( self, k ):
        """
        BBOX の形状回帰のためのスケール値を計算する。
        具体的には、各特徴マップ k (=1~6) についてのデフォルトボックスのスケール s_k は、以下のようにして計算される。
        s_k = s_min + (s_max - s_min) * (k - 1.0) / (m - 1.0), m = 6

        [Input]
            k : int
                特徴マップ fmap の番号。1 ~ self._n_fmaps の間の数
        [Output]
            s_k : float
                指定された番号の特徴マップのスケール値
        """
        s_k = self._scale_min + ( self._scale_max - self._scale_min ) * k / ( self._n_fmaps - 1.0 )
        
        return s_k
    ```

- 尚、本コードのパラメータにおけるデフォルトボックスの総数は、`8752` 個となる。<br>

- 動作確認として、生成したデフォルトボックスの内、同様の `group_id` をもつデフォルトボックスを表示。
    - 特徴マップのセル（グリッド）に従って、特徴マップを隙間なく敷き詰め [tile] されていることが分かる。
    ![image](https://user-images.githubusercontent.com/25688193/40359986-7dc090c6-5dff-11e8-9f36-17a63b0f714a.png)<br>

<br>


<a id="ID_3-2-2-5"></a>

#### 5. 損失関数の設定
SSD モデルの損失関数を設定する。<br>
この設定は、`SingleShotMultiBoxDetector` クラスの `loss(...)` メソッドにて行う。<br>

```python
def main():
    ...
    #======================================================================
    # 損失関数を設定する。
    # Declare the loss functions.
    #======================================================================
    ssd.loss( nnLoss = None )
```

- この `loss(...)` メソッド内では、以下の処理が行われる。<br>
    1. 位置特定誤差 `loss_loc_op` は、予想されたボックス（l）と正解ボックス（g）の間の Smooth L1 誤差（関数）であり、<br>
    以下の式で与えられる。<br>
    ![image](https://user-images.githubusercontent.com/25688193/40358451-424b88b6-5dfa-11e8-935e-a36eaba9d4b1.png)<br>
        
        ```python
        [SingleShotMultiBoxDetector.py / class SingleShotMultiBoxDetector]
        def loss(...):
            ...
            #---------------------------------------------------------------------------
            # 位置特定誤差 L_loc
            # L_loc = Σ_(i∈pos) Σ_(m) { x_ij^k * smoothL1( predbox_i^m - gtbox_j^m ) }
            #---------------------------------------------------------------------------
            smoothL1_op = smooth_L1( x = ( self.gt_boxes_holder - self.pred_locs ) )
            loss_loc_op = tf.reduce_sum( smoothL1_op, reduction_indices = 2 ) * self.pos_holder
        
            loss_loc_op = tf.reduce_sum( loss_loc_op, reduction_indices = 1 ) / ( 1e-5 + tf.reduce_sum( self.pos_holder, reduction_indices = 1 ) )
        ```

        - ここで、Smooth L1 損失関数は、このメソッド `loss(...)` 内で以下のように定義されている。

        ```python
        [SingleShotMultiBoxDetector.py / class SingleShotMultiBoxDetector]
        def loss(...):
            def smooth_L1( x ):
                """
                smooth L1 loss func

                smoothL1 = 0.5 * x^2 ( if |x| < 1 )
                         = |x| -0.5 (otherwise)
                """
                # 0.5 * x^2
                sml1 = tf.multiply( 0.5, tf.pow(x, 2.0) )

                # |x| - 0.5
                sml2 = tf.subtract( tf.abs(x), 0.5 )
            
                # 条件 : |x| < 1
                cond = tf.less( tf.abs(x), 1.0 )

                return tf.where( cond, sml1, sml2 )    
        ```

    2. 確信度誤差 `loss_conf_op` は、所属クラスのカテゴリ（c）に対する softmax cross entropy 誤差（関数）であり、<br>
    以下の式で与えられる。<br>
    ![image](https://user-images.githubusercontent.com/25688193/40358707-238920e0-5dfb-11e8-9a83-84808c19a875.png)<br>

        ```python
        [SingleShotMultiBoxDetector.py / class SingleShotMultiBoxDetector]
        def loss(...):
            ...
            #---------------------------------------------------------------------------
            # 確信度誤差 L_conf
            # L_conf = Σ_(i∈pos) { x_ij^k * log( softmax(c) ) }, c = カテゴリ、ラベル
            #---------------------------------------------------------------------------
            loss_conf_op = tf.nn.sparse_softmax_cross_entropy_with_logits( 
                               logits = self.pred_confs, 
                               labels = self.gt_labels_holder 
                           )

            loss_conf_op = loss_conf_op * ( self.pos_holder + self.neg_holder )

            loss_conf_op = tf.reduce_sum( loss_conf_op, reduction_indices = 1 ) / ( 1e-5 + tf.reduce_sum( ( self.pos_holder + self.neg_holder ), reduction_indices = 1) )
        ```

    3. SSD の損失関数 `self._loss_op` は、この位置特定誤差 `loss_loc_op` と確信度誤差 `loss_conf_op` の重み付き和であり、<br>
    （SSD の学習は、複数の物体カテゴリーを扱うことを考慮して行われるため２つの線形和をとる。）<br>
    以下の式で与えられる。<br>
    ![image](https://user-images.githubusercontent.com/25688193/40358172-605e3548-5df9-11e8-8f75-4cdedb9cc931.png)<br>

        ```python
        [SingleShotMultiBoxDetector.py / class SingleShotMultiBoxDetector]
        def loss(...):
            ...
            #---------------------------------------------------------------------------
            # 合計誤差 L
            #---------------------------------------------------------------------------
            self._loss_op = tf.reduce_sum( loss_conf_op + loss_loc_op )
        ```

<br>

<a id="ID_3-2-2-6"></a>

#### 6. Optimizer の設定
最適化アルゴリズム Optimizer として、Adam アルゴリズム を使用する。<br>

- Optimizer の設定は、`SingleShotMultiBoxDetector` クラスの `optimizer(...)` メソッドで行う。
- 学習率 : `learning_rate = 0.0001`で検証。減衰項は、`adam_beta1 = 0.9` , `adam_beta2 = 0.999`

```python
[main2.py]
def main():
    ...
    ssd.optimizer( Adam( learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999 ) )
```

```python
[NNOptimizer.py]
class Adam( NNOptimizer ):
    """
    Adam アルゴリズムを表すクラス
    NNOptimizer クラスの子クラスとして定義
    """
    def __init__( self, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.99, node_name = "Adam_Optimizer" ):
        self._learning_rate = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._node_name = node_name
        self._optimizer = self.optimizer()
        self._train_step = None

        return
    
    def optimizer( self ):
        self._optimizer = tf.train.AdamOptimizer( 
                              learning_rate = self._learning_rate, 
                              beta1 = self._beta1,
                              beta2 = self._beta2
                          )

        return self._optimizer

    def train_step( self, loss_op ):
        self._train_step = self._optimizer.minimize( loss_op )
        return self._train_step
```

<br>

<a id="ID_3-2-2-7"></a>

#### 7. 構築した SSD モデルによる学習
- トレーニング用データ `X_train`, `y_train` に対し、fitting 処理（モデルのトレーニングデータでの学習）を行う。
- この fitting 処理は、`SingleShotMultiBoxDetector` クラスの `fit(...)` メソッドで行う。

```python
[main2.py]
def main():
    ...
    ssd.fit( X_train, y_train )
```

- この `fit(...)` メソッド内では、以下の処理が行われる。<br>
    1. Variable の初期化＆セッションの run<br>
    ```python
    [SingleShotMultiBoxDetector.py / class SingleShotMultiBoxDetector]
    def fit( self, X_train, y_train ):
        ...
        #----------------------------------------------------------
        # 学習開始処理
        #----------------------------------------------------------
        # Variable の初期化オペレーター
        self._init_var_op = tf.global_variables_initializer()

        # Session の run（初期化オペレーター）
        self._session.run( self._init_var_op )
        ...
    ```
    2. ミニバッチ処理<br>
    ```python
    [SingleShotMultiBoxDetector.py / class SingleShotMultiBoxDetector]
    def fit( self, X_train, y_train ):
        ...
        # ミニバッチの繰り返し回数
        n_batches = len( X_train ) // self._batch_size       # バッチ処理の回数
        n_minibatch_iterations = self._epochs * n_batches    # ミニバッチの総繰り返し回数
        n_minibatch_iteration = 0                            # ミニバッチの現在の繰り返し回数

        #----------------------------------------------------------
        # 学習処理
        #----------------------------------------------------------
        # for ループでエポック数分トレーニング
        for epoch in range( 1, self._epochs + 1 ):
            # ミニバッチサイズ単位で for ループ
            # エポック毎に shuffle し直す。
            gen_minibatch = generate_minibatch( 
                X = X_train, y = y_train , 
                batch_size = self._batch_size, 
                bSuffle = True, random_seed = 12 
            )

            # n_batches 回のループ
            for i ,(batch_x, batch_y) in enumerate( gen_minibatch, 1 ):
                n_minibatch_iteration += 1
        ...
    ```
    ```python
    def generate_minibatch( X, y, batch_size, bSuffle = True, random_seed = 12 ):
        """
        指定された（トレーニング）データから、ミニバッチ毎のデータを生成する。
        （各 Epoch 処理毎に呼び出されることを想定している。）
        """
        # 各 Epoch 度に shuffle し直す。
        if( bSuffle == True ):
            idxes = np.arange( len(y) )   # 0 ~ y.shape[0] の連番 idxes を生成

            # メルセンヌツイスターというアルゴリズムを使った擬似乱数生成器。
            # コンストラクタに乱数の種(シード)を設定。
            random_state = np.random.RandomState( random_seed )
            random_state.shuffle( idxes )
                
            # shuffle された連番 idxes 値のデータに置き換える。
            X_ = [] 
            y_ = []
            for idx in idxes:
                X_.append( X[idx] )
                y_.append( y[idx] )

        # 0 ~ 行数まで batch_size 間隔でループ
        for i in range( 0, len(X_), batch_size ):
            # mini batch data
            batch_X_ = X_[i:i+batch_size]
            batch_y_ = y_[i:i+batch_size]

            # yield 文で逐次データを return（関数の処理を一旦停止し、値を返す）
            # メモリ効率向上のための処理
            yield ( batch_X_, batch_y_ )
    ```
    3. デフォルトボックスのクラス所属の確信度、長方形位置を取得。<br>
   ```python
    [SingleShotMultiBoxDetector.py / class SingleShotMultiBoxDetector]
    def fit( self, X_train, y_train ):
        ...
        #----------------------------------------------------------------------
        # デフォルトボックスの物体のクラス所属の確信度、長方形位置を取得
        #----------------------------------------------------------------------
        f_maps, pred_confs, pred_locs = \
        self._session.run(
            [ self.fmaps, self.pred_confs, self.pred_locs ], 
            feed_dict = { self.base_vgg16.X_holder: batch_x }
        )
        ...
    ```
    4. 教師データに含まれる、物体数、所属クラス、長方形位置座標の抽出とコンバート処理<br>
    ```python
    [SingleShotMultiBoxDetector.py / class SingleShotMultiBoxDetector]
    def fit( self, X_train, y_train ):
        ...
        # batch_size 分のループ
        for i in range( len(batch_x) ):
            actual_labels = []
            actual_loc_rects = []

            #------------------------------------------------------------------
            # 教師データの物体のクラス所属の確信度、長方形位置のフォーマットを変換
            #------------------------------------------------------------------
            # 教師データから物体のクラス所属の確信度、長方形位置情報を取り出し
            # 画像に存在する物体の数分ループ処理
            for obj in batch_y[i]:
                # 長方形の位置情報を取り出し
                loc_rect = obj[:4]

                # 所属クラス情報を取り出し＆ argmax でクラス推定
                label = np.argmax( obj[4:] )

                # 位置情報のフォーマットをコンバート
                # [ top_left_x, top_left_y, bottom_right_x, bottom_right_y ] 
                # → [ top_left_x, top_left_y, width, height ]
                width = loc_rect[2] - loc_rect[0]
                height = loc_rect[3] - loc_rect[1]
                loc_rect = np.array( [ loc_rect[0], loc_rect[1], width, height ] )

                # [ top_left_x, top_left_y, width, height ] → [ center_x, center_y, width, height ]
                center_x = ( 2 * loc_rect[0] + loc_rect[2] ) * 0.5
                center_y = ( 2 * loc_rect[1] + loc_rect[3] ) * 0.5
                loc_rect = np.array( [ center_x, center_y, abs(loc_rect[2]), abs(loc_rect[3]) ] )
                
                #
                actual_loc_rects.append( loc_rect )
                actual_labels.append( label )

        ...

    ```
    4. （デフォルトボックスと正解ボックスの）マッチング戦略<br>
    ```python
    [SingleShotMultiBoxDetector.py / class SingleShotMultiBoxDetector]
    def fit( self, X_train, y_train ):
        ...
        # reset eval
        positives = []      # self.pos_holder に供給するデータ : 正解ボックスとデフォルトボックスの一致
        negatives = []      # self.neg_holder に供給するデータ : 正解ボックスとデフォルトボックスの不一致
        ex_gt_labels = []   # self.gt_labels_holder に供給するデータ : 正解ボックスの所属クラスのラベル
        ex_gt_boxes = []    # self.gt_boxes_holder に供給するデータ : 正解ボックス

        # batch_size 文のループ
        for i in range( len(batch_x) ):
            ...
            #----------------------------------------------------------------------
            # デフォルトボックスと正解ボックスのマッチング処理（マッチング戦略）
            #----------------------------------------------------------------------
            pos_list, neg_list, expanded_gt_labels, expanded_gt_locs = \
            self._matcher.match( 
                pred_confs, pred_locs, actual_labels, actual_loc_rects
            )

            # マッチング結果を追加
            positives.append( pos_list )
            negatives.append( neg_list )
            ex_gt_labels.append( expanded_gt_labels )
            ex_gt_boxes.append( expanded_gt_locs )
        ...
    ```
    ```python
    [BBoxMatcher.py / class BBoxMatcher]

    ```
    5. トレーニングステップでの学習と loss 値の計算 & 取得<br>
    ```python
    [SingleShotMultiBoxDetector.py / class SingleShotMultiBoxDetector]
    def fit( self, X_train, y_train ):
        ...
        # for ループでエポック数分トレーニング
        for epoch in range( 1, self._epochs + 1 ):
            ...
            # n_batches = X_train.shape[0] // self._batch_size 回のループ
            for i ,(batch_x, batch_y) in enumerate( gen_minibatch, 1 ):
                #------------------------------------------------------------------
                # 設定された最適化アルゴリズム Optimizer でトレーニング処理を run
                #------------------------------------------------------------------
                loss, _, = self._session.run(
                               [ self._loss_op, self._train_step ],
                               feed_dict = {
                                   self.base_vgg16.X_holder: batch_x,
                                   self.pos_holder: positives,
                                   self.neg_holder: negatives,
                                   self.gt_labels_holder: ex_gt_labels,
                                   self.gt_boxes_holder: ex_gt_boxes

                               }
                           )
        ...
    ```

<br>

<a id="ID_3-2-2-8"></a>

#### 8. 学習済み SSD モデルによる推論フェイズ
> **実装中...**

- 学習済み SSD モデルから、各デフォルトボックスの属するクラス、及び、各デフォルトボックスの座標値の推論(予想）データを取得する。<br>
```python
[SingleShotMultiBoxDetector.py / class SingleShotMultiBoxDetector]
def predict( self, image ):
    """
    学習済み SSD モデルから、各デフォルトボックスの所属クラスと位置座標の推論（予想）を行う。

    [Input]
        image : ndarray / shape = [image_haight, image_width, n_channels]
            物体検出の推論をしたい画像データ
    [Output]
        pred_confs : ndarry / shape = [デフォルトボックスの総数, クラス数]
            デフォルトボックスの属するクラスの予想値
        pred_locs : ndarry / shape = [デフォルトボックスの総数, 座標値の４次元]
            デフォルトボックスの座標の予想値
    """
    feature_maps, pred_confs, pred_locs = \
    self._session.run( 
        [ self.fmaps, self.pred_confs, self.pred_locs ], 
        feed_dict = { self.base_vgg16.X_holder: [image] }   # [] でくくって、shape を [300,300,3] → [,300,300,3] に reshape
    )

    # 余計な次元を削除して、
    # [1, デフォルトボックスの総数, クラス数] → [デフォルトボックスの総数, クラス数] に reshape
    # [1, デフォルトボックスの総数, 座標値の４次元] → [デフォルトボックスの総数, 座標値の４次元] に reshape
    pred_confs = np.squeeze( pred_confs )
    pred_locs = np.squeeze( pred_locs )

    return pred_confs, pred_locs
```

- 次に、取得した各デフォルトボックスの属するクラス、及び、各デフォルトボックスの座標値の推論(予想）データから、クラスの確信度が高いデフォルトボックスを検出する。<br>
- この処理は、`detect_object(...)` メソッドで行われる。

```python

```

- クラス所属の確信度の上位 `n_top_prob` 個（引数で与えられる）を抽出する。（top-k filtering アルゴリズム）<br>
- 推論されたデータに対し、バウンディングボックスの重複防止のために non-maximum suppression アルゴリズムを適用する。<br>

<br>

#### 9. TensorBoard の計算グラフ
このモデルの TensorBorad で描写した計算グラフは以下のようになる。<br>

![graph_large_attrs_key _too_large_attrs limit_attr_size 1024 run](https://user-images.githubusercontent.com/25688193/40402857-fe5f8ece-5e88-11e8-8e44-911914433eed.png)<br>

![graph_large_attrs_key _too_large_attrs limit_attr_size 1024 run 1](https://user-images.githubusercontent.com/25688193/40402956-a2efc99a-5e89-11e8-91a7-2bf31b8cf0b6.png)<br>

---

<a id="ID_3-2-3"></a>

### ☆ コードの実行結果

|パラメータ名|値（実行条件１）|
|---|---|
|xxx|xxx|
|xxx|xxx|

<br>

---

<a id="ID_4"></a>

## 背景理論

- [星の本棚 : ニューラルネットワーク / ディープラーニング](http://yagami12.hatenablog.com/entry/2017/09/17/111935#ID_11-4)


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

[18/05/23]
```python
pred_confs : shape = [1, DBOX の総数(8752), クラス数(21)]
    [0] array([-0.15612692, -0.29669559, -0.15306497,  0.09254545, -0.22070014,       -0.22384059, -0.78873271, -0.08331621,  0.01295507,  0.16571039,       -0.63299227,  0.25336158, -0.19712901, -0.50648594, -0.43001437,        0.04846162, -0.00387031, -0.34417355, -0.38269293, -0.01045942,        0.34372237], dtype=float32)
    [1] array([-0.68770736, -0.6439172 , -0.94381034, -0.31826934, -0.9395144 ,       -1.3598454 , -0.99760568, -0.89838231, -0.86218756, -0.41516352,       -1.32084978, -0.7376709 , -0.72060591, -0.61360884, -0.85756791,       -0.93495744, -0.4175511 , -1.51470733, -0.80751765, -0.98726952,       -0.09816164], dtype=float32)
    ...
    [8751]



idxs : 
[14  0  3 23 22 21 20 19 18 11 12 10  5 17 16 13 15  6 24  4 25  7  2  1  9 8]
idxs2 : 
[16 21 24 25  4 17 22  9 23 13  5 12 20 15  2  0 19  7  8 11 18  6  3  1 10 14]




loc : [ 0.03156713  1.83142224  0.72907385  2.41800079]
pt1 : (11,915)
pt2 : (273,1209)


```

```python
[180526_epoch8batch495 / 000488.jpg]

pred_confs : shape = (1, 8752, 21)
[0][0] 
array([ 0.37168849, -1.54700267, -1.19724   , -1.16353285, -1.97406149,       -1.3798728 , -1.25823641, -1.56293559, -0.76379716, -0.7348628 ,       -2.67191076, -1.48244298, -1.40735316, -1.22580874, -1.11963832,       -1.09645462, -1.23183715, -2.18374658, -1.98043728, -1.62870884,        
1.4447335 ], dtype=float32)
[0][8751]


hist : [536, 60, 420, 662, 108, 165, 685, 71, 244, 164, 49, 153, 44, 88, 2282, 172, 269, 84, 93, 145, 2258]


    array([201, 205, 209, 213, 197, 193, 217, 173, 221, 189, 225, 229, 233,       237, 185, 241, 181, 245, 169, 249, 177], dtype=int64)

y2 : shape = (n_class)
    array([ 7.2680769 ,  7.42006707,  7.5573051 ,  7.64983058,  7.12237597,
        6.96000552,  7.67807817,  5.90138412,  7.66387224,  6.75598383,
        7.62829614,  7.58795476,  7.53211498,  7.45248079,  6.52060533,
        7.34095216,  6.29373741,  7.10960817,  5.58761716,  6.55509424,
        6.09716964])
idxs : 
    [18  7 20 16 14 19  9  5 17  4  0 15  1 13 12  2 11 10  3  8  6]


label : 20.0
loc : [ 0.01285222  0.95629692  0.10695443  4.3835454 ]
pt1 : (6,358)
pt2 : (53,1643)

```

