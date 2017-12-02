## TensorFlow を用いた LSTM [Long short-term memory] の実装と簡単な応用

TensorFlow を用いた、LSTM [Long short-term memory] による時系列モデルの予想、画像識別、自然言語処理の練習用実装コード集。

この README.md ファイルには、各コードの実行結果、概要、RNN の背景理論の説明を記載しています。
分かりやすいように main.py ファイル毎に１つの完結した実行コードにしています。

### 項目 [Contents]

1. [使用するライブラリ](#ID_1)
1. [使用するデータセット](#ID_2)
1. [コードの説明＆実行結果](#ID_3)
    1. [LSTM によるノイズ付き sin 波形（時系列データ）からの長期の波形の予想（生成）処理 : `main1.py`](#ID_3-1)
        1. [コードの内容説明](#ID_3-1-1)
        1. [コードの実行結果](#ID_3-1-2)
    1. LSTM による Adding Problem 対する長期予想性とその評価処理 : `main2.py`
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

>> `tf.variable_scope(...)` : Variable に名前空間を与える。<br>
>>> https://www.tensorflow.org/api_docs/python/tf/variable_scope<br>

>> `tf.get_variable(...)` : <br>
>> 変数名の識別子（新規か？重複がないか？）を管理しながら変数の名前空間の定義を行い、必ず `tf.variable_scope()` とセットで使う。<br>
>>> https://qiita.com/TomokIshii/items/ffe999b3e1a506c396c8


> その他ライブラリ
>>

<br>

<a id="ID_2"></a>

### 使用するデータセット

> ノイズ付き sin 波形（時系列データとして利用）
![rnn_1-1](https://user-images.githubusercontent.com/25688193/33367977-a1f6a1b0-d533-11e7-8daa-d6a51e5d9eb7.png)

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
![graph_large_attrs_key _too_large_attrs limit_attr_size 1024 run 16](https://user-images.githubusercontent.com/25688193/33520066-08f105c0-d7f7-11e7-8939-e067401b527d.png)
![graph_large_attrs_key _too_large_attrs limit_attr_size 1024 run](https://user-images.githubusercontent.com/25688193/33520067-091a298c-d7f7-11e7-8060-42512d7241bf.png)
![graph_large_attrs_key _too_large_attrs limit_attr_size 1024 run 1](https://user-images.githubusercontent.com/25688193/33520068-09417a78-d7f7-11e7-9711-39ae30ed39b5.png)

<br>

<a id="ID_3-1-2"></a>

### コードの実行結果

### 損失関数のグラフ

- ｛入力層：１ノード、隠れ層：<span style="color:red">**20**</span> ノード、出力層：１ノード｝、各シーケンス長 : 25 個の LSTM モデル 
![rnn_2-1-20](https://user-images.githubusercontent.com/25688193/33446785-886412c0-d644-11e7-8b61-63a068403f02.png)

- ｛入力層：１ノード、隠れ層：<span style="color:red">**50**</span> ノード、出力層：１ノード｝、各シーケンス長 : 25 個の LSTM モデル 
![rnn_2-1-50](https://user-images.githubusercontent.com/25688193/33447976-1d99f1e0-d648-11e7-9688-f6dec4b3219f.png)

### 予想出力値と元データの波形図（時系列データ）

- ｛入力層：１ノード、隠れ層：<span style="color:red">**20**</span> ノード、出力層：１ノード｝、各シーケンス長 : 25 個の LSTM モデル 
![rnn_2-2-20](https://user-images.githubusercontent.com/25688193/33446787-89d01aaa-d644-11e7-9288-fe6998441568.png)

- ｛入力層：１ノード、隠れ層：<span style="color:red">**50**</span> ノード、出力層：１ノード｝、各シーケンス長 : 25 個の LSTM モデル 
![rnn_2-2-50](https://user-images.githubusercontent.com/25688193/33447978-1eef3690-d648-11e7-82e1-29166b519557.png)

> 先の通常の RNN モデルより、長期の予想が改善していることが分かる。


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
