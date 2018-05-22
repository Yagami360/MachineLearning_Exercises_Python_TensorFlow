## TensorFlow を用いた SSD [Single Shot muitibox Detector] の実装と簡単な応用

> **実装中...（別レポジトリに移行予定）**

ニューラルネットワークによる一般物体検出アルゴリズムの１つである、SSD [Single Shot muitibox Detector] を TensorFlow で実装。（ChainerCV や OpenCV 等にある実装済み or 学習済み SSD モジュールのような高レベル API 使用せずに、TensorFlow でイチから実装している。）<br>

この `README.md` ファイルには、各コードの実行結果、コードの内容の説明を記載しています。<br>
分かりやすいように `main.py` ファイル毎に１つの完結した実行コードにしています。<br>

尚、SSD [Single Shot muitibox Detector] に関しての、背景理論は以下のサイトに記載してあります。<br>

- [星の本棚 : ニューラルネットワーク / ディープラーニング](http://yagami12.hatenablog.com/entry/2017/09/17/111935#ID_11-4)


### 項目 [Contents]

1. [使用するデータセット](#ID_2)
1. [コード説明＆実行結果](#ID_3)
    1. [TensorFlow を用いた SSD [Single Shot muitibox Detector] の実装 : `main2.py`](#ID_3-2)
        1. [使用するライブラリ](#3-2-1)
        1. [コードの内容説明](#3-2-2)
            1. [Poscal VOC2007 データセットにある、画像、物体情報の読み込み＆抽出](#3-2-2-1)
            1. [SSD モデルの各種パラメーターの設定](#3-2-2-2)
            1. [SSD モデルの構築](#3-2-2-3)
            1. [デフォルトボックスの生成](#3-2-2-4)
            1. [損失関数の設定](#3-2-2-5)
            1. [Optimizer の設定](#3-2-2-6)
            1. [構築した SSD モデルの学習](#3-2-2-7)
            1. [学習済み SSD モデルによる推論フェイズ](#3-2-2-8)
        1. [コードの実行結果](#3-2-3)
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

<!--
- Open Images Dataset V4
    - https://storage.googleapis.com/openimages/web/download.html
-->

---

<a id="ID_3"></a>

## コード説明＆実行結果

<a id="ID_3-2"></a>

## TensorFlow を用いた SSD [Single Shot muitibox Detector] の実装 : `main2.py`
> **実装中...**

TensorFlow を用いた SSD [Single Shot muitibox Detector] の実装。<br>
ChainerCV や OpenCV 等にある実装済み or 学習済み SSD モジュールのような高レベル API 使用せずに、TensorFlow で実装している。<br>

<a id="ID_3-2-1"></a>

### ☆ 使用するライブラリ

- TensorFlow ライブラリ
    - xxx

- その他ライブラリ
    - xxx

<br>


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
    - この SSD のベースネットワークとしての VGG16 は、従来の VGG16 における全結合層を、畳み込み層に置き換えたモデルであり、`BaseNetworkVGG16` クラスの `model()` メソッドで定義されたモデルである。<br>
    ```python
    [BaseNetwork.py]
    class BaseNetworkVGG16( BaseNetwork ):
    ...
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
    - 同様に、上記 `model()` メソッド内でコールされているプーリング処理関数 `pooling_layer(...)` は、以下のように定義されている。
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
<br>

<a id="ID_3-2-2-3"></a>

#### 4. デフォルトボックスの生成
各 extra feature map に対応した一連のデフォルトボックス群を生成する。<br>
この処理は、`SingleShotMultiBoxDetector` クラスの `generate_default_boxes_in_fmaps(...)` メソッドで行う。 <br>
（デフォルトボックスに関するアスペクト比のマップ `aspect_set` 、及びスケール値の最大値 `scale_max`、最小値 `scale_min` といったパラメータの設定も、このメソッド内で行っている。）<br>

```python
[SingleShotMultiBoxDetector.py]
class SingleShotMultiBoxDetector( NeuralNetworkBase ):
    ...
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

        # 一連のデフォルトボックス群を表すクラス DefaultBoxSet のオブジェクトを生成
        self._default_box_set = DefaultBoxSet( scale_min = 0.2, scale_max = 0.9 )
        
        # 一連のデフォルトボックス群を生成
        self._default_box_set.generate_boxes( fmaps_shapes = fmap_shapes, aspect_set = aspect_set )

        return self._default_box_set
```

この各特徴マップ、アスペクト比、スケール値に対応した一連のデフォルトボックス群は、`DefaultBoxSet` クラスのオブジェクト `self._default_box_set` として表現され、<br>
実際の一連のデフォルトボックス群の生成は、このクラス `DefaultBoxSet` の `generate_boxes(...)` メソッドで行なう。

```python
[DefaultBox.py]
class DefaultBoxSet( object ):
    ...
    def generate_boxes( self, fmaps_shapes, aspect_set ):
        """
        generate default boxes based on defined number
        
        [Input]
            fmaps_shapes : nadarry( [][] )
                extra feature map の形状（ピクセル単位）
                the shape is [  first-feature-map-boxes ,
                                second-feature-map-boxes ,
                                    ...
                                sixth-feature-map-boxes , ]
                    ==> ( total_boxes_number x defined_size )
                
                feature map sizes per output such as...
                [ 
                    [ None, 19, 19, ],      # extra feature-map-shape 1 [batch_size, fmap_height, fmap_width]
                    [ None, 19, 19, ],      # extra feature-map-shape 2
                    [ None, 10, 10 ],       # extra feature-map-shape 3
                    [ None, 5, 5, ],        # extra feature-map-shape 4
                    [ None, 3, 3, ],        # extra feature-map-shape 5
                    [ None, 1, 1, ],        # extra feature-map-shape 6
                ]

            aspect_set : nadarry( [][] )
                extra feature map に対してのアスペクト比
                such as...
                [1.0, 1.0, 2.0, 1.0/2.0],                   # extra feature map 1
                [1.0, 1.0, 2.0, 1.0/2.0, 3.0, 1.0/3.0],     # extra feature map 2
                [1.0, 1.0, 2.0, 1.0/2.0, 3.0, 1.0/3.0],
                [1.0, 1.0, 2.0, 1.0/2.0, 3.0, 1.0/3.0],
                [1.0, 1.0, 2.0, 1.0/2.0, 3.0, 1.0/3.0],
                [1.0, 1.0, 2.0, 1.0/2.0, 3.0, 1.0/3.0],

        [Output]
            self._default_boxes : list<DefaultBox>
                generated default boxes list

        """
        self._n_fmaps = len( fmaps_shapes )

        id = 0
        for k, fmap_shape in enumerate( fmaps_shapes ):
            s_k = self.calc_scale( k )

            fmap_width  = fmap_shape[1]
            fmap_height = fmap_shape[2]
            
            aspects = aspect_set[k]

            for aspect in aspects:
                # 特徴マップのセルのグリッド（1 pixcel）に関してのループ処理
                for y in range( fmap_height ):
                    # セルのグリッドの中央を 0.5 として計算 
                    center_y = ( y + 0.5 ) / float( fmap_height )

                    for x in range( fmap_width ):
                        center_x = ( x + 0.5 ) / float( fmap_width )

                        box_width = s_k * np.sqrt( aspect )
                        box_height = s_k / np.sqrt( aspect )

                        id += 1
                        default_box = DefaultBox(
                                          group_id = k + 1,
                                          id = id,
                                          center_x = center_x, center_y = center_y,
                                          width = box_width, height = box_height, 
                                          scale = s_k,
                                          aspect = aspect
                                      )

                        self.add_default_box( default_box )

        return self._default_boxes
```

- このメソッドでは、以下の処理が行われる。<br>
    1. 各特徴マップ（のサイズ `fmaps_shape` ）`k` に対して 、スケール値 `s_k` を計算。<br>
    ```python
    [DefaultBox.py]
    class DefaultBoxSet( object ):
    ...
    def generate_boxes( self, fmaps_shapes, aspect_set ):
        ...
        for k, fmap_shape in enumerate( fmaps_shapes ):
            s_k = self.calc_scale( k )
    ```
        
    2. 各アスペクト比 `aspects` と、各特徴マップ k の高さ `fmap_height`、幅 `fmap_width` から構成される各セルのグリッド（１×１ピクセル）`x`, `y` に対して、長方形の中心座標 `center_x`, `center_y`、アスペクト比 `aspect`、デフォルトボックスの高さ `box_height`、幅 `box_width` を抽出 or 計算する。<br>
    ここで、デフォルトボックスの各特徴マップ `k` 、及び各スケール値 `s_k` に対する、幅と高さは、以下の式で算出する。<br>
        ![image](https://user-images.githubusercontent.com/25688193/40353511-a20ebee8-5dec-11e8-99d2-8b5d8bb8e96f.png)<br>
    ```python
    [DefaultBox.py]
    class DefaultBoxSet( object ):
    ...
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
    [DefaultBox.py]
    class DefaultBoxSet( object ):
    ...
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
    [DefaultBox.py]
    class DefaultBoxSet( object ):
    ...
    def generate_boxes( self, fmaps_shapes, aspect_set ):
        ...
        self.add_default_box( default_box )
    ```
    ```python
    [DefaultBox.py]
    class DefaultBoxSet( object ):
    ...
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
    具体的には、各特徴マップ k (=1~6) についてのデフォルトボックスのスケール `s_k` を、以下のようにして計算している。<br>
    ![image](https://user-images.githubusercontent.com/25688193/40351479-7a5fe87c-5de7-11e8-89bf-192c07e89e0a.png)<br>
```python
[DefaultBox.py]
class DefaultBoxSet( object ):
    ...
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


<br>


<a id="ID_3-2-2-5"></a>

#### 5. 損失関数の設定
SSD モデルの損失関数を設定する。<br>
この設定は、`SingleShotMultiBoxDetector` クラスの `loss(...)` メソッドにて行う。

```python
[SingleShotMultiBoxDetector.py]
class SingleShotMultiBoxDetector( NeuralNetworkBase ):
    ...
    def loss( self, nnLoss ):
        """
        損失関数（誤差関数、コスト関数）の定義を行う。
        SSD の損失関数は、位置特定誤差（loc）と確信度誤差（conf）の重み付き和であり、
        （SSD の学習は、複数の物体カテゴリーを扱うことを考慮して行われるため２つの線形和をとる。）
        以下の式で与えられる。
        
        Loss = (Loss_conf + a*Loss_loc) / N

        [Input]
            nnLoss : NNLoss クラスのオブジェクト
            
        [Output]
            self._loss_op : Operator
                損失関数を表すオペレーター
        """
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

        # 生成したデフォルトボックスの総数
        total_boxes = len( self._default_box_set._default_boxes )
        #print( "total_boxes", total_boxes )     # 8752

        #---------------------------------------------------------------------------
        # 各種 Placeholder の生成
        #---------------------------------------------------------------------------
        # ground truth label （正解ボックスの所属クラス）の placeholder
        self.gt_labels_holder = tf.placeholder( shape = [None, total_boxes], dtype = tf.int32, name = "gt_labels_holder" )

        # ground truth boxes （正解ボックス）の placeholder
        self.gt_boxes_holder = tf.placeholder( shape = [None, total_boxes, 4], dtype = tf.float32, name = "gt_boxes_holder"  )

        # positive (デフォルトボックスと正解ボックスのマッチングが正) list の placeholder
        # negative (デフォルトボックスと正解ボックスのマッチングが負) list の placeholder
        self.pos_holder = tf.placeholder( shape = [None, total_boxes], dtype = tf.float32, name = "pos_holder"  )
        self.neg_holder = tf.placeholder( shape = [None, total_boxes], dtype = tf.float32, name = "neg_holder"  )

        #---------------------------------------------------------------------------
        # 位置特定誤差 L_loc
        # L_loc = Σ_(i∈pos) Σ_(m) { x_ij^k * smoothL1( predbox_i^m - gtbox_j^m ) }
        #---------------------------------------------------------------------------
        smoothL1_op = smooth_L1( x = ( self.gt_boxes_holder - self.pred_locations ) )
        loss_loc_op = tf.reduce_sum( smoothL1_op, reduction_indices = 2 ) * self.pos_holder
        
        loss_loc_op = tf.reduce_sum( loss_loc_op, reduction_indices = 1 ) / ( 1e-5 + tf.reduce_sum( self.pos_holder, reduction_indices = 1 ) )
        
        #---------------------------------------------------------------------------
        # 確信度誤差 L_conf
        # L_conf = Σ_(i∈pos) { x_ij^k * log( softmax(c) ) }, c = カテゴリ、ラベル
        #---------------------------------------------------------------------------
        loss_conf_op = tf.nn.sparse_softmax_cross_entropy_with_logits( 
                           logits = self.pred_confidences, 
                           labels = self.gt_labels_holder 
                       )

        loss_conf_op = loss_conf_op * ( self.pos_holder + self.neg_holder )
        
        loss_conf_op = tf.reduce_sum( loss_conf_op, reduction_indices = 1 ) / ( 1e-5 + tf.reduce_sum( ( self.pos_holder + self.neg_holder ), reduction_indices = 1) )

        #---------------------------------------------------------------------------
        # 合計誤差 L
        #---------------------------------------------------------------------------
        self._loss_op = tf.reduce_sum( loss_conf_op + loss_loc_op )

        return self._loss_op

```

- SSD の損失関数 `self._loss_op` は、位置特定誤差 `loss_loc_op` と確信度誤差 `loss_conf_op` の重み付き和であり、<br>
    （SSD の学習は、複数の物体カテゴリーを扱うことを考慮して行われるため２つの線形和をとる。）<br>
    以下の式で与えられる。<br>
    ![image](https://user-images.githubusercontent.com/25688193/40358172-605e3548-5df9-11e8-8f75-4cdedb9cc931.png)<br>
    ```python
    [SingleShotMultiBoxDetector.py]
    class SingleShotMultiBoxDetector( NeuralNetworkBase ):
    ...
    def loss(...):
        ...
        #---------------------------------------------------------------------------
        # 合計誤差 L
        #---------------------------------------------------------------------------
        self._loss_op = tf.reduce_sum( loss_conf_op + loss_loc_op )
    ```

- 位置特定誤差 `loss_loc_op` は、予想されたボックス（l）と正解ボックス（g）の間の Smooth L1 誤差（関数）であり、<br>
    以下の式で与えられる。<br>
    ![image](https://user-images.githubusercontent.com/25688193/40358451-424b88b6-5dfa-11e8-935e-a36eaba9d4b1.png)<br>
    ```python
    [SingleShotMultiBoxDetector.py]
    class SingleShotMultiBoxDetector( NeuralNetworkBase ):
    ...
    def loss(...):
        ...
        #---------------------------------------------------------------------------
        # 位置特定誤差 L_loc
        # L_loc = Σ_(i∈pos) Σ_(m) { x_ij^k * smoothL1( predbox_i^m - gtbox_j^m ) }
        #---------------------------------------------------------------------------
        smoothL1_op = smooth_L1( x = ( self.gt_boxes_holder - self.pred_locations ) )
        loss_loc_op = tf.reduce_sum( smoothL1_op, reduction_indices = 2 ) * self.pos_holder
        
        loss_loc_op = tf.reduce_sum( loss_loc_op, reduction_indices = 1 ) / ( 1e-5 + tf.reduce_sum( self.pos_holder, reduction_indices = 1 ) )
    ```

    - ここで、Smooth L1 損失関数は、このメソッド `loss(...)` 内で以下のように定義されている。
    ```python
    [SingleShotMultiBoxDetector.py]
    class SingleShotMultiBoxDetector( NeuralNetworkBase ):
    ...
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

- 確信度誤差 `loss_conf_op` は、所属クラスのカテゴリ（c）に対する softmax cross entropy 誤差（関数）であり、<br>
    以下の式で与えられる。<br>
    ![image](https://user-images.githubusercontent.com/25688193/40358707-238920e0-5dfb-11e8-9a83-84808c19a875.png)<br>
    ```python
    [SingleShotMultiBoxDetector.py]
    class SingleShotMultiBoxDetector( NeuralNetworkBase ):
    ...
    def loss(...):
        ...
        #---------------------------------------------------------------------------
        # 確信度誤差 L_conf
        # L_conf = Σ_(i∈pos) { x_ij^k * log( softmax(c) ) }, c = カテゴリ、ラベル
        #---------------------------------------------------------------------------
        loss_conf_op = tf.nn.sparse_softmax_cross_entropy_with_logits( 
                           logits = self.pred_confidences, 
                           labels = self.gt_labels_holder 
                       )

        loss_conf_op = loss_conf_op * ( self.pos_holder + self.neg_holder )
        
        loss_conf_op = tf.reduce_sum( loss_conf_op, reduction_indices = 1 ) / ( 1e-5 + tf.reduce_sum( ( self.pos_holder + self.neg_holder ), reduction_indices = 1) )
    ```


<br>

<a id="ID_3-2-2-6"></a>

#### 6. Optimizer の設定

- Adam
- 学習率 : `learning_rate = 0.0001`
- 減衰項 : `adam1 = 0.9` , `adam2 = 0.999`

<br>

<a id="ID_3-2-2-7"></a>

#### 7. 構築した SSD モデルによる学習

- ミニバッチ処理
- 教師データに含まれる、物体数、所属クラス、長方形位置座標の抽出とコンバート処理
- （デフォルトボックスと正解ボックスの）マッチング戦略
- トレーニングステップでの学習
- loss 値の計算 & 取得

<br>

<a id="ID_3-2-2-8"></a>

#### 8. 学習済み SSD モデルによる推論フェイズ

- ハードネガティブマイニング
- non-maximum suppression アルゴリズム
    - 推論されたデータに対し、バウンディングボックスのかぶり防止のために non-maximum suppression アルゴリズムを適用する。

<br>

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
