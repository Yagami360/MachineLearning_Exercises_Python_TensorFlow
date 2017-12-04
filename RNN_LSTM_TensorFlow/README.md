## TensorFlow を用いた LSTM [Long short-term memory] の実装と簡単な応用

TensorFlow を用いた、LSTM [Long short-term memory] による時系列モデルの予想、画像識別、自然言語処理の練習用実装コード集。

この `README.md` ファイルには、各コードの実行結果、概要、RNN の背景理論の説明を記載しています。
分かりやすいように `main.py` ファイル毎に１つの完結した実行コードにしています。

### 項目 [Contents]

1. [使用するライブラリ](#ID_1)
1. [使用するデータセット](#ID_2)
1. [コードの説明＆実行結果](#ID_3)
    1. [LSTM によるノイズ付き sin 波形（時系列データ）からの長期の波形の予想（生成）処理 : `main1.py`](#ID_3-1)
        1. [コードの内容説明](#ID_3-1-1)
        1. [コードの実行結果](#ID_3-1-2)
    1. [LSTM による Adding Problem に対する長期予想性とその評価処理 : `main2.py`](#ID_3-2)
        1. [コードの内容説明](#ID_3-2-1)
        1. [コードの実行結果](#ID_3-2-2)
    1. LSTM によるシェイクスピア作品のワード予想処理 : `main3.py`
    1. 複数の LSTM 層によるシェイクスピア作品のワード予想処理 : `main4.py`
1. [背景理論](#ID_4)
    1. [リカレントニューラルネットワーク [RNN : Recursive Neural Network]<br>＜階層型ニューラルネットワーク＞](#ID_5)
        1. [長・短期記憶（LSTM [long short-term memory]）モデル](#ID_5-2)
            1. [CEC [constant error carousel]](#ID_5-2-1)
            1. [重み衝突 [weight conflict] と入力ゲート [input gate]、出力ゲート [output gate]](#ID_5-2-2)
            1. [忘却ゲート [forget gate]](#ID_5-2-3)
            1. [覗き穴結合 [peephole connections]](#ID_5-2-4)
            1. [LSTM モデルの定式化](#ID_5-2-5)


<a id="ID_1"></a>

### 使用するライブラリ

> TensorFlow ライブラリ <br>
>> `tf.contrib.rnn.LSTMRNNCell(...)` : <br>
>> 時系列に沿った LSTM 構造を提供するクラス `LSTMCell` の `cell` を返す。<br>
>> この `cell` は、内部（プロパティ）で state（隠れ層の状態）を保持しており、これを次の時間の隠れ層に順々に渡していくことで、時間軸の逆伝搬を実現する。<br>
>>> https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMCell<br>

> その他ライブラリ
>>

<br>

<a id="ID_2"></a>

### 使用するデータセット

- ノイズ付き sin 波形（時系列データとして利用）
![rnn_1-1](https://user-images.githubusercontent.com/25688193/33367977-a1f6a1b0-d533-11e7-8daa-d6a51e5d9eb7.png)

- Adding Problem に対応するデータ（時系列データに対する長期の予想評価として利用）
    - Adding Problem データの内、1, 2, 9999, 10000 つ目のシーケンスのデータを表示した図
![rnnlm_2-2-2](https://user-images.githubusercontent.com/25688193/33543425-6098de0a-d91a-11e7-8c95-ea65825b157a.png)

<br>

<a id="ID_3"></a>

## コードの説明＆実行結果

<a id="ID_3-1"></a>

## LSTM によるノイズ付き sin 波形（時系列データ）からの長期の波形の予想（生成）処理 : `main1.py`

<a id="ID_3-1-1"></a>

LSTM モデルによる時系列データの取り扱いの簡単な例として、先の [`./RNN_TensorFlow/main1.py`](https://github.com/Yagami360/MachineLearning_Exercises_Python_TensorFlow/tree/master/RNN_TensorFlow#rnn-によるノイズ付き-sin-波形時系列データからの波形の予想生成処理--main1py) で行った処理と同じ、ノイズ付き sin 波形（時系列データとみなす）の予想（生成）を考える。

- 先の [`./RNN_TensorFlow/main1.py`](https://github.com/Yagami360/MachineLearning_Exercises_Python_TensorFlow/tree/master/RNN_TensorFlow#rnn-によるノイズ付き-sin-波形時系列データからの波形の予想生成処理--main1py) で使用した通常の RNN モデルで、`tf.contrib.rnn.BasicRNNCell(...)` としていた箇所を、`tf.contrib.rnn.LSTMCell(...)` に変更する。
    ```python
    [RecurrentNNLSTM.py]
    def model( self ):
        ...
        #--------------------------------------------------------------
        # 入力層 ~ 隠れ層
        #--------------------------------------------------------------
        # tf.contrib.rnn.LSTMCell(...) : 時系列に沿った RNN 構造を提供するクラス `LSTMCell` のオブジェクト cell を返す。
        # この cell は、内部（プロパティ）で state（隠れ層の状態）を保持しており、
        # これを次の時間の隠れ層に順々に渡していくことで、時間軸の逆伝搬を実現する。
        cell = tf.contrib.rnn.LSTMCell( 
                   num_units = self._n_hiddenLayer     # int, The number of units in the RNN cell.
               )

        # 最初の時間 t0 では、過去の隠れ層がないので、
        # cell.zero_state(...) でゼロの状態を初期設定する。
        initial_state_tsr = cell.zero_state( self._batch_size_holder, tf.float32 )

        #-----------------------------------------------------------------
        # 過去の隠れ層の再帰処理
        #-----------------------------------------------------------------
        self._rnn_states.append( initial_state_tsr )

        with tf.variable_scope('RNN-LSTM'):
            for t in range( self._n_in_sequence ):
                if (t > 0):
                    # tf.get_variable_scope() : 名前空間を設定した Variable にアクセス
                    # reuse_variables() : reuse フラグを True にすることで、再利用できるようになる。
                    tf.get_variable_scope().reuse_variables()

                # LSTMCellクラスの `__call__(...)` を順次呼び出し、
                # 各時刻 t における出力 cell_output, 及び状態 state を算出
                cell_output, state_tsr = cell( inputs = self._X_holder[:, t, :], state = self._rnn_states[-1] )

                # 過去の隠れ層の出力をリストに追加
                self._rnn_cells.append( cell_output )
                self._rnn_states.append( state_tsr )

        # 最終的な隠れ層の出力
        output = self._rnn_cells[-1]

        # 隠れ層 ~ 出力層
        self._weights.append( self.init_weight_variable( input_shape = [self._n_hiddenLayer, self._n_outputLayer] ) )
        self._biases.append( self.init_bias_variable( input_shape = [self._n_outputLayer] ) )
    ```
- その他の処理は、 先の [`./RNN_TensorFlow/main1.py`](https://github.com/Yagami360/MachineLearning_Exercises_Python_TensorFlow/tree/master/RNN_TensorFlow#rnn-によるノイズ付き-sin-波形時系列データからの波形の予想生成処理--main1py) で使用した通常の RNN モデルと同様になる。
- 尚、この LSTM モデルを TensorBoard で描写した計算グラフは以下のようになる。
![graph_large_attrs_key _too_large_attrs limit_attr_size 1024 run 4](https://user-images.githubusercontent.com/25688193/33547356-331fd9f8-d927-11e7-90c1-65027f1cdf7b.png)
![graph_large_attrs_key _too_large_attrs limit_attr_size 1024 run 5](https://user-images.githubusercontent.com/25688193/33547357-336b9f32-d927-11e7-8902-315b0e335a93.png)

<br>

<a id="ID_3-1-2"></a>

### コードの実行結果

### 損失関数のグラフ

- ｛入力層：１ノード、隠れ層：<span style="color:red">**20**</span> ノード、出力層：１ノード｝の LSTM モデルと通常の RNN モデルの比較
    - 各シーケンス長 : **25** 個
    - 学習率 0.001, 最適化アルゴリズム : Adam ( 減衰項 : beta1 = 0.9, beta2 = 0.999 )
    - トレーニング用データ : 90 %、テスト用データ : 10% に分割
    - １枚目の図が、LSTM モデルでの損失関数のグラフ。２枚目の図が、通常の RNN モデルでの損失関数のグラフ
![rnn-lstm_1-1-3-h20](https://user-images.githubusercontent.com/25688193/33553583-b75ce19c-d93c-11e7-980c-af30df4654a9.png)
![rnn_1-2-20](https://user-images.githubusercontent.com/25688193/33424393-151103c2-d5ff-11e7-9de3-4993be4767d8.png)

- ｛入力層：１ノード、隠れ層：<span style="color:red">**50**</span> ノード、出力層：１ノード｝の LSTM モデルと通常の RNN モデルの比較
    - 各シーケンス長 : **25** 個
    - 学習率 0.001, 最適化アルゴリズム : Adam ( 減衰項 : beta1 = 0.9, beta2 = 0.999 )
    - トレーニング用データ : 90 %、テスト用データ : 10% に分割
    - １枚目の図が、LSTM モデルでの損失関数のグラフ。２枚目の図が、通常の RNN モデルでの損失関数のグラフ
![rnn-lstm_1-1-3-h50](https://user-images.githubusercontent.com/25688193/33553772-5bb39f6a-d93d-11e7-82eb-549af7240967.png)
![rnn_1-2-50](https://user-images.githubusercontent.com/25688193/33424608-a70bf372-d5ff-11e7-8472-46bf4c47ea1c.png)

<br>

### 予想出力値と元データの波形図（時系列データ）

- ｛入力層：１ノード、隠れ層：<span style="color:red">**20**</span> ノード、出力層：１ノード｝の LSTM モデルと通常の RNN モデルの比較
    - 各シーケンス長 : **25** 個
    - 学習率 0.001, 最適化アルゴリズム : Adam ( 減衰項 : beta1 = 0.9, beta2 = 0.999 )
    - トレーニング用データ : 90 %、テスト用データ : 10% に分割
    - １枚目の図が LSTM モデルでの予想波形、２枚目の図が、通常の RNN モデルでの予想波形
![rnn-lstm_1-2-h20](https://user-images.githubusercontent.com/25688193/33553610-cdf9870c-d93c-11e7-9b6f-d3d8998aa14b.png)
![rnn_1-3-20](https://user-images.githubusercontent.com/25688193/33424647-c8b09c80-d5ff-11e7-8626-151369d31e83.png)

- ｛入力層：１ノード、隠れ層：<span style="color:red">**50**</span> ノード、出力層：１ノード｝の LSTM モデル
    - 各シーケンス長 : **25** 個
    - 学習率 0.001, 最適化アルゴリズム : Adam ( 減衰項 : beta1 = 0.9, beta2 = 0.999 )
    - トレーニング用データ : 90 %、テスト用データ : 10% に分割
    - １枚目の図が LSTM モデルでの予想波形、２枚目の図が、通常の RNN モデルでの予想波形
![rnn-lstm_1-2-h50](https://user-images.githubusercontent.com/25688193/33553775-5f1f09fa-d93d-11e7-8702-501d0d9d9a81.png)
![rnn_1-3-50](https://user-images.githubusercontent.com/25688193/33424650-c905ced0-d5ff-11e7-8c80-743fd0319046.png)

> 先の通常の RNN モデルより、長期の予想が改善していることが分かる。


<br>

---

<a id="ID_3-2"></a>

## LSTM による Adding Problem に対する長期予想性とその評価処理 : `main2.py`

<a id="ID_3-2-1"></a>

先の ノイズ付き sin 波形の LSTM での予想処理で、LSTM が通常の RNN より、精度の高い予想が出来ていることを確認したが、
より一般的に LSTM の長期依存性の学習評価を確認するために、Adding Problem というトイ・プロブレムで LSTM 長期依存性を評価する。

このデータは、時系列データである入力データ x(t) が、シグナルデータ s(t) とマスクデータ m(t) の２種類からなるデータで、<br>
シグナルデータ s(t) が、 0 ~ 1 の一様乱数分布 U(0,1) に従って発生させたデータであり、<br>
マスクデータ m(t) が、{ 0 or 1 } の値となるが、与えられた時間 t の範囲で、ランダムに選ばれた 2 つのデータのみ 1 を取り、その他は全て 0 となるようなデータである。

式で書くと、
>![image](https://user-images.githubusercontent.com/25688193/33545118-7a2030d4-d920-11e7-9a2a-c13c494588b3.png)<br>
>
>![image](https://user-images.githubusercontent.com/25688193/33545485-b521c82c-d921-11e7-8102-3064c87ee9c9.png)<br>
>
>この入力 x(t) に対しての、出力 y は、<br>
>![image](https://user-images.githubusercontent.com/25688193/33545649-3423b5cc-d922-11e7-8067-de1e051dbefe.png)<br>

<!--
表で書くと、
> 記載中...
-->

一部を図示すると、
> Adding Problem データの内、1, 2, 9999, 10000 つ目のシーケンスのデータを表示した図
>![rnnlm_2-2-2](https://user-images.githubusercontent.com/25688193/33543425-6098de0a-d91a-11e7-8c95-ea65825b157a.png)

- まず、Adding Problem のデータセットに対応するデータを生成する。
    - この処理は、`generate_adding_problem(...)` 関数で行い、以下のようなコードになる。
    ```python
    [MLPreProcess.py]
    @staticmethod
    def generate_adding_problem( t, n_sequence, seed = 12 ):
        numpy.random.seed( seed = seed )
        
        # 0~1 の間の一様ランダムからなるシグナル（シーケンス）× シグナル数（シーケンス数）作成
        singnals = numpy.random.uniform( low = 0.0, high = 1.0, size = ( n_sequence, t ) )
        
        #-----------------------------
        # 0 or 1 からなるマスクの作成
        #-----------------------------
        # まず全体を 0 で埋める
        masks = numpy.zeros( shape = ( n_sequence, t ) )

        for i in range( n_sequence ):
            # マスクの値 0 or 1
            mask = numpy.zeros( shape = ( t ) )
            # numpy.random.permutation(...) : 配列をランダムに入れ替え
            inidices = numpy.random.permutation( numpy.arange(t) )[:2]
            mask[inidices] = 1
            masks[i] = mask
        
        #-----------------------------
        # シグナル×マスクの作成
        #-----------------------------
        # まず全体を 0 で埋める
        adding_data = numpy.zeros( shape = ( n_sequence, t, 2 ) )
        
        # シグナルの配列
        adding_data[ :, :, 0 ] = singnals
        
        # マスクの配列
        adding_data[ :, :, 1 ] = masks

        # 出力
        adding_targets = ( singnals * masks ).sum( axis = 1 ).reshape( n_sequence, 1 )

        return adding_data, adding_targets
    ```
    - そして、main 側でこの関数 `generate_adding_problem(...)` を呼び出し、N = 10,000 個のデータを時間 t = 250 の幅で生成する。
    ```python
    [main2.py]
    X_features, y_labels = MLPreProcess.generate_adding_problem( t = 250, n_sequence = 10000, seed = 12 )
    ```
- データセットを、トレーニング用データセットと、テスト用データセットに分割する。
分割割合は、トレーニング用データ 90%、テスト用データ 10%
    ```python
    [main2.py]
    X_train, X_test, y_train, y_test \
    = MLPreProcess.dataTrainTestSplit( X_input = X_features, y_input = y_labels, ratio_test = 0.1, input_random_state = 1 )
    ```
- LSTM モデルの各種パラメーターの設定を行う。
    - この設定は、`RecurrectNNLSTM` クラスのインスタンス作成時の引数にて行う。
        - 入力層のノード数 `n_inputLayer` は **2** 個（入力データが、シグナルとマスクデータから成るので）、隠れ層のノード数 `n_hiddenLayer` は 100 個で検証、出力層のノード数 `n_outputLayer` は 1 個（ 推定器 Estimiter なので）
        - １つのシーケンスの長さ `n_in_sequence` は 250 個
        - エポック数は `epochs` 500, ミニバッチサイズ `batch_size`は 10
    ```python
    [main2.py]
    rnn1 = RecurrentNNLSTM(
               session = tf.Session( config = tf.ConfigProto(log_device_placement=True) ),
               n_inputLayer = len( X_features[0][0] ),
               n_hiddenLayer = 100,
               n_outputLayer = len( y_labels[0] ),
               n_in_sequence = X_features.shape[1],
               epochs = 500,
               batch_size = 10,
               eval_step = 1
           )
    ```
- 損失関数として、L2ノルムを使用する。
    ```python
    [main2.py]
    rnn1.loss( L2Norm() )
    ```
- 最適化アルゴリズム Optimizer として、Adam アルゴリズムを使用する。
    - 学習率 `learning_rate` は、可変な値（0.001 等）で検証。減衰項は `beta1 = 0.9`, `beta1 = 0.999`
    ```python
    [main2.py]
    rnn1.optimizer( Adam( learning_rate = learning_rate1, beta1 = adam_beta1, beta2 = adam_beta2 ) )
    ```
- トレーニング用データ `X_train`, `y_train` に対し、fitting 処理を行う。
    ```python
    [main2.py]
    rnn1.fit( X_train, y_train )
    ```
- fitting 処理 `fit(...)` 後のモデルで、時系列データの予想を行う。
    - 関数 `predict(...)` を、以下のように main 処理側で呼び出し、一連の時系列データの予想値を取得する。
    ```python
    [main2.py]
    predicts1 = rnn1.predict( X_features )
    ```
- その他の処理は、 先の [./RNN_LSTM_TensorFlow/main1.py](https://github.com/Yagami360/MachineLearning_Exercises_Python_TensorFlow/tree/master/RNN_LSTM_TensorFlow#lstm-によるノイズ付き-sin-波形時系列データからの長期の波形の予想生成処理--main1py) で使用した LSTM モデルと同様になる。
- 尚、この RNN モデルを TensorBoard で描写した計算グラフは以下のようになる。
![graph_large_attrs_key _too_large_attrs limit_attr_size 1024 run 3](https://user-images.githubusercontent.com/25688193/33543980-62943bb2-d91c-11e7-9046-c376f1f1c18f.png)


<a id="ID_3-1-2"></a>

### コードの実行結果

### 損失関数のグラフ

- ｛入力層：２ノード、隠れ層：**100** ノード、出力層：１ノード｝の LSTM と通常の RNN モデルの比較
    - 各シーケンス長 : 250 個
    - 学習率 0.001, 最適化アルゴリズム : Adam ( 減衰項 : beta1 = 0.9, beta2 = 0.999 )
    - トレーニング用データ : 90 %、テスト用データ : 10% に分割 
    - １枚目の図が LSTM モデルでの損失関数のグラフ。２枚目の図が、通常の RNN モデルでの損失関数のグラフ
![rnn-lstm_2-1-2-h20](https://user-images.githubusercontent.com/25688193/33553385-da7fba92-d93b-11e7-98ce-10d10ff01e12.png)
> コード実装中...

<!--
> LSTM モデルでは、損失関数の値が 0 付近に収束しており、うまく学習出来ていることが見て取れる。<br>
> 一方、通常の RNN モデルでは、損失関数の値が収束しておらず、うまく学習出来ていないことが分かる。
-->

### 予想出力値と元データの波形図（時系列データ）

- ｛入力層：２ノード、隠れ層：**100** ノード、出力層：１ノード｝の LSTM モデルと通常の RNN モデルの比較
    - 各シーケンス長 : 250 個
    - 学習率 0.001, 最適化アルゴリズム : Adam ( 減衰項 : beta1 = 0.9, beta2 = 0.999 )
    - トレーニング用データ : 90 %、テスト用データ : 10% に分割 
    - １枚目の図が LSTM モデルでの予想波形のグラフ。２枚目の図が、通常の RNN モデルでの予想波形のグラフ
> コード実装中...

<br>

---


<a id="ID_4"></a>

## 背景理論

<a id="ID_5"></a>

## リカレントニューラルネットワーク [RNN : Recursive Neural Network]<br>＜階層型ニューラルネットワーク＞の概要
![image](https://user-images.githubusercontent.com/25688193/30980712-f06a0906-a4bc-11e7-9b15-4c46834dd6d2.png)
![image](https://user-images.githubusercontent.com/25688193/30981066-22f53124-a4be-11e7-9111-9514f04aed7c.png)

<a id="ID_5-2"></a>

### 長・短期記憶（LSTM [long short-term memory]）モデル

<a id="ID_5-2-1"></a>

#### CEC [constant error carousel]
![image](https://user-images.githubusercontent.com/25688193/31226189-2d62a892-aa10-11e7-93e5-b32902d83702.png)
![image](https://user-images.githubusercontent.com/25688193/31226163-0eb9927a-aa10-11e7-9d06-306e4443c5a8.png)
![image](https://user-images.githubusercontent.com/25688193/31235831-6fa44284-aa2d-11e7-9377-845ea30837c5.png)
![image](https://user-images.githubusercontent.com/25688193/31226906-eb4288bc-aa12-11e7-9f16-621ed4d50063.png)

<a id="ID_5-2-2"></a>

#### 重み衝突 [weight conflict] と入力ゲート [input gate]、出力ゲート [output gate]
![image](https://user-images.githubusercontent.com/25688193/31236796-16687124-aa30-11e7-89b5-2da158274de7.png)
![image](https://user-images.githubusercontent.com/25688193/31246908-ed52d18e-aa49-11e7-946f-44f3fa177eb3.png)
![image](https://user-images.githubusercontent.com/25688193/31246932-fa855dc2-aa49-11e7-882d-462dd22be03d.png)

<a id="ID_5-2-3"></a>

#### 忘却ゲート [forget gate]
![image](https://user-images.githubusercontent.com/25688193/31247911-036bc036-aa4d-11e7-9f5f-117eaab0b738.png)
![image](https://user-images.githubusercontent.com/25688193/31247928-130b98b8-aa4d-11e7-89aa-ac27b1667666.png)
![image](https://user-images.githubusercontent.com/25688193/31248855-2cf3eb7e-aa50-11e7-99b7-4c81a093f679.png)
![image](https://user-images.githubusercontent.com/25688193/31249125-2453757e-aa51-11e7-9ce2-715edddf8232.png)

<a id="ID_5-2-4"></a>

#### 覗き穴結合 [peephole connections]
![image](https://user-images.githubusercontent.com/25688193/31272328-83122b86-aac5-11e7-84db-6a52bd8d2c44.png)
![image](https://user-images.githubusercontent.com/25688193/31272347-8f9d67bc-aac5-11e7-9fda-640bdb6a9d7f.png)
![image](https://user-images.githubusercontent.com/25688193/31279596-941088d2-aae4-11e7-9e30-dc28771800c4.png)

<a id="ID_5-2-5"></a>

#### LSTM モデルの定式化
![image](https://user-images.githubusercontent.com/25688193/31278352-91da316c-aadf-11e7-8ad6-963e7e235852.png)
![image](https://user-images.githubusercontent.com/25688193/31283264-169b4f16-aaf0-11e7-9f19-976dc2e09bc9.png)
![image](https://user-images.githubusercontent.com/25688193/31284097-8a2e6e84-aaf2-11e7-8e7d-df00110c5bf6.png)
![image](https://user-images.githubusercontent.com/25688193/31293857-b20586f6-ab13-11e7-85b2-460f9bab5e62.png)
![image](https://user-images.githubusercontent.com/25688193/31294053-706d763a-ab14-11e7-8aed-1fed8327d58c.png)

---

デバッグメモ
