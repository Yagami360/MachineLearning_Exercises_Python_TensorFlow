## TensorFlow を用いたリカレントニューラルネットワーク（RNN）の実装と簡単な応用

TensorFlow を用いた、リカレントニューラルネットワーク（RNN）による時系列モデルの予想、画像識別、自然言語処理の練習用実装コード集。

この README.md ファイルには、各コードの実行結果、概要、RNN の背景理論の説明を記載しています。
分かりやすいように main.py ファイル毎に１つの完結した実行コードにしています。

### 項目 [Contents]

1. [使用するライブラリ](#ID_1)
1. [使用するデータセット](#ID_2)
1. [コードの説明＆実行結果](#ID_3)
    1. [RNN によるノイズ付き sin 波形（時系列データ）からの波形の予想（生成）処理 : `main1.py`](#ID_3-1)
        1. [コードの内容説明](#ID_3-1-1)
        1. [コードの実行結果](#ID_3-1-2)
    1. [RNN によるテキストデータからのスパム文章の確率予想処理 : `main2.py`](#ID_3-2)
        1. [コードの内容説明](#ID_3-2-1)
        1. [コードの実行結果](#ID_3-2-2)
    1. [LSTM によるノイズ付き sin 波形（時系列データ）からの長期の波形の予想（生成）処理 : `main3.py`](#ID_3-3)
        1. [コードの内容説明](#ID_3-3-1)
        1. [コードの実行結果](#ID_3-3-2)
    1. GNU による sin 波形（時系列データ）の生成処理 : `main4.py`
    1. 双方向 RNN による MNIST データセットの識別処理 : `main5.py`
    1. RNN Encoder-Decoder による自然言語処理（足し算の応答） : `main6.py` 
1. [背景理論](#ID_4)
    1. [リカレントニューラルネットワーク [RNN : Recursive Neural Network]<br>＜階層型ニューラルネットワーク＞](#ID_5)
        1. [リカレントニューラルネットワークのアーキテクチャの種類](#ID_5-1)
            1. [隠れ層間で回帰構造をもつネットワーク](#ID_5-1-1)
                1. [通時的誤差逆伝搬法 [BPTT : back-propagation through time]](#ID_5-1-1-1)
        1. [長・短期記憶（LSTM [long short-term memory]）モデル](#ID_5-2)
            1. [CEC [constant error carousel]](#ID_5-2-1)
            1. [重み衝突 [weight conflict] と入力ゲート [input gate]、出力ゲート [output gate]](#ID_5-2-2)
            1. [忘却ゲート [forget gate]](#ID_5-2-3)
            1. [覗き穴結合 [peephole connections]](#ID_5-2-4)
            1. [LSTM モデルの定式化](#ID_5-2-5)
        1. [GRU [gated recurrent unit]](#ID_5-3)
        1. [双方向 RNN [BiRNN : Bidirectional RNN]](#ID_5-4)
        1. [RNN Encoder-Decoder (Seqenence-to-sequence models)](#ID_5-5)


<a id="ID_1"></a>

### 使用するライブラリ

> TensorFlow ライブラリ <br>
>> `tf.contrib.rnn.BasicRNNCell(...)` : <br>
>> 時系列に沿った RNN 構造を提供するクラス `BasicRNNCell` の `cell` を返す。<br>
>> この `cell` は、内部（プロパティ）で state（隠れ層の状態）を保持しており、これを次の時間の隠れ層に順々に渡していくことで、時間軸の逆伝搬を実現する。<br>
>>> https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicRNNCell<br>

>> `tf.contrib.rnn.LSTMCell(...)` : Long short-term memory unit (LSTM) recurrent network cell.<br>
>>> https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMCell<br>

>> `tf.variable_scope(...)` : Variable に名前空間を与える。<br>
>>> https://www.tensorflow.org/api_docs/python/tf/variable_scope<br>

>> `tf.get_variable(...)` : <br>
>> 変数名の識別子（新規か？重複がないか？）を管理しながら変数の名前空間の定義を行い、必ず `tf.variable_scope()` とセットで使う。<br>
>>> https://qiita.com/TomokIshii/items/ffe999b3e1a506c396c8

>> `tf.contrib.learn.preprocessing.VocabularyProcessor(...)` : テキスト情報を数値インデックスのリストに変換する。<br>
>>> https://orajavasolutions.wordpress.com/2016/11/22/how-to-extract-vocabulary-from-tensorflow-vocabularyprocessor-object/<br>

> その他ライブラリ
>>


<br>

<a id="ID_2"></a>

### 使用するデータセット

- ノイズ付き sin 波形
    - 時系列データの簡単な例として利用する。
![rnn_1-1](https://user-images.githubusercontent.com/25688193/33367977-a1f6a1b0-d533-11e7-8daa-d6a51e5d9eb7.png)

- SMS Spam Collection データセット
    - URL : http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip
    - 自然言語処理の簡単な例として、テキストデータからスパムの確率予想を行うために使用する。
    - The SMS Spam Collection v.1 (text file: smsspamcollection) has a total of 4,827 SMS legitimate messages (86.6%) and a total of 747 (13.4%) spam messages.
    - The files contain one message per line. Each line is composed by two columns: one with label (ham or spam) and other with the raw text.
![image](https://user-images.githubusercontent.com/25688193/33476677-2881be66-d6c6-11e7-82c9-1c5a3e502270.png)


<br>


<a id="ID_3"></a>

## コードの説明＆実行結果

<a id="ID_3-1"></a>

## RNN によるノイズ付き sin 波形（時系列データ）からの波形の予想（生成）処理 : `main1.py`

<a id="ID_3-1-1"></a>

RNN による時系列モデルの取り扱いの簡単な例として、ノイズ付き sin 波形（時系列データとみなす）の予想（生成）を考える。

- まず、以下のような、ノイズ付き sin 波形を生成する。<br>
    - 周期 `T = 100`, ノイズ幅 `noize_size = 0.05` 
![rnn_1-1](https://user-images.githubusercontent.com/25688193/33367977-a1f6a1b0-d533-11e7-8daa-d6a51e5d9eb7.png)
    ```python
    [main1.py]
    T1 = 100                # ノイズ付き sin 波形の周期
    noize_size1 = 0.05      # ノイズ付き sin 波形のノイズ幅
    times = numpy.arange( 2.5 * T1 + 1 )    # 時間 t の配列
    x_dat, y_dat = MLPreProcess.generate_sin_with_noize( t = times, T = T1, noize_size = noize_size1, seed = 12 )
    ```
    ```python
    [MLPreProcess.py]
    @staticmethod
    def generate_sin_with_noize( t, T = 100, noize_size = 0.05, seed = 12 ):
        """
        ノイズ付き sin 波形（時系列データ）を生成する
        [Input]
            t : array
                時間のリスト
            T : float
                波形の周期
            noize_size : float
                ノイズ幅の乗数
        """
        numpy.random.seed( seed = seed )

        # numpy.random.uniform(...) : 一様分布に従う乱数
        noize = noize_size * numpy.random.uniform( low = -1.0, high = 1.0, size = len(t),  )
        sin = numpy.sin( 2.0 * numpy.pi * (t / T) )

        x_dat = t
        y_dat = sin + noize

        return x_dat, y_dat
    ```
- 通時的誤差逆伝搬法 [BPTT : back-propagation through time] での計算負荷の関係上、
時系列データを一定間隔 τ でシーケンス状に区切る。
![image](https://user-images.githubusercontent.com/25688193/33429345-cf0e245c-d60f-11e7-9b6c-1b2ebc6cee12.png)
    - 具体的には、サイズが τ で<br>
     `{ f(t=1), f(t=2), ... , f(t=τ) }` , <br>
     `{ f(t=2), f(t=3), ... , f(t=τ+1) }` , <br>
     ... ,<br>
      `{ f(t-τ), f(t-τ+1), ... , f(t) }` <br>
      のシーケンスから成る合計 `t - τ + 1` 個のデータセット
    - これにより、長期の時間依存性に関する情報
    （ `{ f(t-τ), f(t-τ+1), ... , f(t) }` に対しては、`{ f(1), f(2), ... , f(t-τ-1) }` の情報）が抜け落ちてしまうが、データセットとしては、t - τ + 1 個の決まった数になるので、学習＆テストデータとして扱いやすくなる。
    ```python
    [main1.py]
    # BPTT での計算負荷の関係上、時系列データを一定間隔に区切る
    len_sequences = len( x_dat )      # 全時系列データの長さ
    n_in_sequence = 25                # １つの時系列データのシーケンスの長さ τ

    data = []       # 区切った時系列データの f(t=1~τ), f(t=2~τ+1), ... 値のリスト（各要素はベクトル）
    targets = []    # 区切った各 data の１つの要素 f(t) に対応する目的値 f(t+1) のリスト

    for i in range( 0, len_sequences - len_in_sequence ):
        data.append( y_dat[ i : i + len_in_sequence ] )
        targets.append( y_dat[ i + len_in_sequence ] )

    # 一般の次元のデータの場合にでも対応できるように、shape を
    # shape = (n_sample) → (n_data, len_in_sequence, 1) に reshape
    X_features = numpy.array( data ).reshape( len(data), len_in_sequence, 1 )
    y_labels = numpy.array( targets ).reshape( len(targets), 1 )
    ```
- データセットを、トレーニング用データセットと、テスト用データセットに分割する。
    - 分割割合は、トレーニング用データ 90%、テスト用データ 10%
    ```python
    [main1.py]
    X_train, X_test, y_train, y_test \
    = MLPreProcess.dataTrainTestSplit( X_input = X_features, y_input = y_labels, ratio_test = 0.1, input_random_state = 1 )
    ```
- 時系列に沿った、過去の隠れ層のモデルを構築する。
    - そのために、`tf.contrib.rnn.BasicRNNCell(...)` を用いて、時系列に沿った RNN 構造を提供するクラス `BasicRNNCell` の `cell` を取得する。
    - この `cell` は、内部（プロパティ）で state（隠れ層の状態）を保持しており、これを次の時間の隠れ層に順々に渡していくことで、時間軸の逆伝搬を実現する。
    ```python
    [RecurrentNN.py]
    def model():
        ...
        cell = tf.contrib.rnn.BasicRNNCell( num_units = self._n_hiddenLayer )
    ```
    - 尚、最初の時間 t0 では、過去の隠れ層がないので、`cell.zero_state(...)` でゼロの状態を初期設定する。
    ```python
    [RecurrentNN.py]
    def model():
        ...
        initial_state = cell.zero_state( self._batch_size_tsr, tf.float32 )
    ```
    -  １つの時系列データの長さ `self._n_in_sequence` ぶん for ループし、
    隠れ層の再帰構造を定義する。
        - この際、過去を表す Variable にアクセスできるようにするために、<br>
        `with tf.variable_scope('RNN'):` 構文で、Variable の名前空間を定義し、<br>
        `tf.get_variable_scope()` で名前空間を設定した Variable にアクセス、<br>
        `tf.get_variable_scope().reuse_variables()` で reuse フラグを True にすることで、再利用できるようにする。
    ```python
    [RecurrentNN.py]
    def model():
        ...
        #-----------------------------------------------------------------
        # 過去の隠れ層の再帰処理
        #-----------------------------------------------------------------
        self._rnn_states.append( initial_state_tsr )

        with tf.variable_scope('RNN'):
            for t in range( self._n_in_sequence ):
                if (t > 0):
                    # tf.get_variable_scope() : 名前空間を設定した Variable にアクセス
                    # reuse_variables() : reuse フラグを True にすることで、再利用できるようになる。
                    tf.get_variable_scope().reuse_variables()

                # BasicRNNCellクラスの `__call__(...)` を順次呼び出し、
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
    - 最終的なモデルの出力は、出力層への入力をそのまま出力する、線形活性とする。
    ```python
    [RecurrentNN.py]
    def model():
        ...
        #--------------------------------------------------------------
        # 出力層への入力
        #--------------------------------------------------------------
        y_in_op = tf.matmul( output, self._weights[-1] ) + self._biases[-1]

        #--------------------------------------------------------------
        # モデルの出力
        #--------------------------------------------------------------
        # 線形活性
        self._y_out_op = y_in_op

        return self._y_out_op
    ```
- 損失関数として、L2ノルムを使用する。
    ```python
    [main1.py]
    rnn1.loss( L2Norm() )
    ```
- 最適化アルゴリズム Optimizer として、Adam アルゴリズムを使用する。
    ```python
    [main1.py]
    rnn1.optimizer( Adam( learning_rate = learning_rate1, beta1 = adam_beta1, beta2 = adam_beta2 ) )
    ```
- トレーニング用データ `X_train`, `y_train` に対し、fitting 処理を行う。
    ```python
    [main1.py]
    rnn1.fit( X_train, y_train )
    ```
- fitting 処理 `fit(...)` 後のモデルで、時系列データの予想（ノイズ付き sin 波形の生成）を行う。
    - まず、元の時系列データの１データだけ切り出し、次の時刻の時系列データを予想する。
    - そして、この処理を繰り返すことで、任意の時刻の時系列データの予想値を算出する流れとなる。
    - 実装コードでは、まず以下のようにして、指定された時系列データの最初の一部を抜き出す。
    ```python
    [RecurrentNN.py]
    def predict( self, X_test ):
        ...
        # 元データの最初の一部 τ 文だけを切り出し、
        # 後に、τ+1 を予想 → τ+2 を予想 ...
        X_t = X_test[:1]    # X(t=1) ~ X(t=τ)
    ```
    - その後、for ループで逐次予想値を append する。
    ```python
    [RecurrentNN.py]
    def predict( self, X_test ):
        ...
        # 予想値のリスト（時系列データ）
        # t=1～τ までの予想値はないので None とする。
        predicts = [ None for i in range(self._n_in_sequence) ]

        # 指定した時系列データの総数(t)
        n_sequences = len( X_test[:,1] )

        # サイズが τ で、
        # { f(t=1), f(t=2), ... , f(t=τ) }, { f(t=2), f(t=3), ... , f(t=τ+1) }, ... , { f(t-τ), f(t-τ+1), ... , f(t) }
        #  の合計 t - τ + 1 個のデータセットに対応したループ処理
        # n_sequences - self._n_in_sequence + 1 : 時系列データの総数(t) - シーケンス内のデータ数(τ) + 1 
        for i in range( n_sequences - self._n_in_sequence + 1):
            # 最後の時系列データを抽出
            # ２回目以降のループでは、new_sequence
            X_t_last = X_t[-1:]

            # 最後の時系列データ X_t_last から未来 prob を予測
            prob = self._session.run(
                       self._y_out_op,
                       feed_dict = { 
                           self._X_holder: X_t_last,
                           self._batch_size_holder: 1
                       }
                   )

            # 予測結果 prob を用いて新しい時系列データ new_sequence を生成
            # numpy.concatenate(...) : ２個以上の配列を軸指定して結合
            # x_last + prob
            # X_t_last.reshape(self._n_in_sequence, self._n_inputLayer)[1:] : 
            # shape = [25,1] に reshape し、[1:] で2番目~最後のシーケンス（各々サイズ _n_in_sequence のベクトル）指定
            new_sequence = numpy.concatenate(
                               ( X_t_last.reshape(self._n_in_sequence, self._n_inputLayer)[1:], prob ), axis = 0
                           ).reshape( 1, self._n_in_sequence, self._n_inputLayer )

            # new_sequence を append した新たな X_t とする。
            X_t = numpy.append( X_t, new_sequence, axis = 0 )
            
            # prob[0][0] : prob = [[xxx]] を shape=1 に reshape し
            # array(xxx, dtype=float32) の値 xxx を格納
            predicts.append( prob[0][0] )

        return predicts
    ```
    - そして、この関数 `predict(...)` を、以下のように main 処理側で呼び出し、一連の時系列データの予想値を取得する。
    ```python
    [main1.py]
    predicts1 = rnn1.predict( X_features )
    ```
- 入力層：１ノード、出力層：１ノードで、隠れ層のノード数を変えたモデルでそれぞれ性能評価する。
- TensorBoard での計算グラフ
![graph_large_attrs_key _too_large_attrs limit_attr_size 1024 run 1](https://user-images.githubusercontent.com/25688193/33439281-648d17f2-d630-11e7-9ace-31d3c78ec783.png)
> わかりやすくなるように、モデルのスコープ・変数名修正中...

<br>

<a id="ID_3-1-2"></a>

### コードの実行結果

### 損失関数のグラフ

- ｛入力層：１ノード、隠れ層：<span style="color:red">**20**</span> ノード、出力層：１ノード｝、各シーケンス長 : 25 個の RNN モデル
![rnn_1-2-20](https://user-images.githubusercontent.com/25688193/33424393-151103c2-d5ff-11e7-9de3-4993be4767d8.png)

- ｛入力層：１ノード、隠れ層：<span style="color:red">**30**</span> ノード、出力層：１ノード｝、各シーケンス長 : 25 個のの RNN モデル
![rnn_1-2-30](https://user-images.githubusercontent.com/25688193/33424585-9c44a07e-d5ff-11e7-9182-64c30b40a22c.png)

- ｛入力層：１ノード、隠れ層：<span style="color:red">**50**</span> ノード、出力層：１ノード｝、各シーケンス長 : 25 個のの RNN モデル
![rnn_1-2-50](https://user-images.githubusercontent.com/25688193/33424608-a70bf372-d5ff-11e7-8472-46bf4c47ea1c.png)

- ｛入力層：１ノード、隠れ層：<span style="color:red">**100**</span> ノード、出力層：１ノード｝、各シーケンス長 : 25 個の RNN モデル
![rnn_1-2-100](https://user-images.githubusercontent.com/25688193/33424610-a85f0fc0-d5ff-11e7-96e1-35116462eb54.png)

> 隠れ層のノード数を変えた各 RNN モデル（各シーケンス長 : 25 個）での損失関数のグラフ。<br>
> 損失関数として、L2ノルムを使用。又、最適化アルゴリズムとして Adam アルゴリズムを使用。<br>
> 各モデルともに 0 付近の値に収束しており、うまく学習出来ていることが分かる。<br>
> 又、隠れ層のノード数が多いモデルほど、早期に収束していることが見て取れる。

### 予想出力値と元データの波形図（時系列データ）

- ｛入力層：１ノード、隠れ層：<span style="color:red">**20**</span> ノード、出力層：１ノード｝、各シーケンス長 : 25 個の RNN モデル<br>
![rnn_1-3-20](https://user-images.githubusercontent.com/25688193/33424647-c8b09c80-d5ff-11e7-8626-151369d31e83.png)

- ｛入力層：１ノード、隠れ層：<span style="color:red">**30**</span> ノード、出力層：１ノード｝、各シーケンス長 : 25 個の RNN モデル<br>
![rnn_1-3-30](https://user-images.githubusercontent.com/25688193/33424649-c8ddc8a4-d5ff-11e7-8d4c-c35bdb327eac.png)

- ｛入力層：１ノード、隠れ層：<span style="color:red">**50**</span> ノード、出力層：１ノード｝、各シーケンス長 : 25 個の RNN モデル<br>
![rnn_1-3-50](https://user-images.githubusercontent.com/25688193/33424650-c905ced0-d5ff-11e7-8c80-743fd0319046.png)

- ｛入力層：１ノード、隠れ層：<span style="color:red">**100**</span> ノード、出力層：１ノード｝、各シーケンス長 : 25 個の RNN モデル<br>
![rnn_1-3-100](https://user-images.githubusercontent.com/25688193/33424651-c92f8036-d5ff-11e7-9d66-39ab0aef9d41.png)

> 隠れ層のノード数を変えた各 RNN モデル（各シーケンス長 : 25 個）での時系列データの予想値のグラフ。<br>
> 損失関数として、L2ノルムを使用。又、最適化アルゴリズムとして Adam アルゴリズムを使用。<br>
> 黒字が、学習データであるノイズ付き sin 波（時系列データ）。<br>
> 赤線が、この黒線のノイズ付き sin 波（時系列データ）に基づき、隠れ層のノード数を変えた各 RNN モデルで予想した時系列データのプロット図。<br>
> 隠れ層のノード数が少ないモデルでは、短期的なデータに対しては、うまく元の時系列データを予測できているが、長期的なデータに対しては、うまく予測できていないことが分かる。<br>
> 一方、隠れ層のノード数が多いモデルでは、長期的なデータに対しても、長期の予測が可能になっていることが分かる。<br>
> 但し、隠れ層のノード数が多すぎると、誤差が拡大が目立つようになり、長期の予想の精度が下がっていく傾向も見て取れる。因みに、この挙動は、力学系における初期値鋭敏依存性と同様の振る舞い。

<br>

- ｛入力層：１ノード、隠れ層：<span style="color:red">**51**</span> ノード、出力層：１ノード｝、各シーケンス長 : <span style="color:red">**50**</span> 個の RNN モデル
![rnn_1-2-51-seq50](https://user-images.githubusercontent.com/25688193/33427984-87dd4c2a-d60a-11e7-8bd4-3d0892d58a86.png)
![rnn_1-3-51-seq50](https://user-images.githubusercontent.com/25688193/33427986-8bc8751c-d60a-11e7-8a48-13d6834d1a96.png)
> 検証した条件の内、最もよく近似できている。（隠れ層のノード数＝各シーケンス長＋１）


<br>

---

<a id="ID_3-2"></a>

## RNN によるテキストデータからのスパム文章の確率予想処理 : `main2.py`
> 実装中...

<a id="ID_3-2-1"></a>

### コードの内容説明

通常の RNN による自然言語処理の例として、テキストデータからのスパム文章の確率予想処理を行う。<br>
この例で使用するデータは、スパム文章か否かの正解ラベル付き ("ham" or "spam") の SMS Spam Collection データセットである。

![image](https://user-images.githubusercontent.com/25688193/33476677-2881be66-d6c6-11e7-82c9-1c5a3e502270.png)

予測に使用する RNN モデルは、埋め込みテキストから入力用シーケンスを RNN モデルへの入力といて扱い、RNNモデルの最後の出力をスパムか否かを判定する予想値 0 or 1 ( "ham" or "spam" ) に対応させる。

- まずは、SMS Spam Collection データセットからテキストデータを読み込み、取得する。
    - Unicode 形式で、各行の文字全体を読み込み、リストに append していく
    ```python
    [MLPreProcess.py]
    def load_sms_spam_collection( ... ):
        ...
        text_data = []

        #----------------------------------------------------
        # codecs.open() 関数と with 構文で文字データの読み込む
        # "r" : 文字のまま読み込み
        #----------------------------------------------------
        with codecs.open( path, "r", "utf-8" ) as file:
           # txt ファイルの各行に関してのループ処理
           for row in file:
               # 各行の文字列全体（特殊文字、空白込 : \t　\n）を格納
               text_data.append( row )

        # 最後の行の除去処理
        # text_data[:-1] : 最初の行の文字列 ~ 最後の行 -1 の文字列の配列
        # ['ham\tGo until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...\n', 
        # 'ham\tOk lar... Joking wif u oni...\n', ... 
        # "ham\tThe guy did some bitching but I acted like i'd be interested in buying something else next week and he gave it to us for free\n" ]
        text_data = text_data[:-1]            
    ```
    - 水平タブを表すエスケープシーケンス `\t` 部分で別の配列に分割する。
        - 文字の分割には `split(...)` 関数を使用する。
    ```python
    [MLPreProcess.py]
    def load_sms_spam_collection( ... ):
        ...
        # 水平タブを表すエスケープシーケンス `\t` 部分で別の配列に分割する。
        # split(...) : 文字列の分割
        # [ ['ham', 'Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...\n'], 
        #   ['ham', 'Ok lar... Joki ..\n'], ...
        #   ['ham', "The guy did some bitching but I acted like i'd be interested in buying something else next week and he gave it to us for free\n"]
        # ]
        text_data = [ str.split( "\t" ) for str in text_data if len(str) >= 1 ]
    ```
    - `"ham"`, `"spam"` の文字列の部分を教師データ `text_data_labels` と本文データ `test_data_feature` に切り分ける。
        - この処理は、以下のコードのように、リストの内包表記と `list(...)` 合わせて行い、`zip(*list)` で `text_data` の内容が "ham" or "spam" , と "本文" 部分に unpack（展開）解釈されることにより実現する。
    ```python
    [MLPreProcess.py]
    def load_sms_spam_collection( ... ):
        ...
        # "ham", "spam" の文字列の部分を教師データに切り分け
        # list(...) : リストの内包表記と合わせて text_data_labels と test_data_feature のリストの生成
        # zip(*text_data) → zip(*list) : 関数の前のアスタリスクは展開されて解釈されるので、
        # text_data の内容が "ham" or "spam" , と "本文" 部分に unpack（展開）解釈される。
        text_data_labels, text_data_features = [ list(str) for str in zip(*text_data) ]
    ```
    - クリーニング処理を行う。具体的には、文字量を減らすために、特殊文字と余分なホワイトスペースを取り除く。
        - まず、`re.sub(...)` を使用して、正規表現で文字列を別の文字列（文字なし）に置換する。
        - `join(...)` を使用して、空白文字 `" "` を区切り文字として、分割→結合する。
        - 最後に、`lower()` を使用して、大文字を小文字に変換する。
    ```python
    [MLPreProcess.py]
    def load_sms_spam_collection( ... ):
        ...
        if ( bCleaning == True ):
            def clean_text( str ):
                # re.sub() : 正規表現で文字列を別の文字列で置換
                str = re.sub( 
                          pattern = r"([^\s\w]|_|[0-9])+",  # 正規表現 : []集合, |和集合（または）()グループ化
                          repl = "",                        # 置換する文字列 : "" なので文字なしに置換
                          string = str                      # 置換される文字列
                      )
                #print( "re.sub(...) :", str )

                # sep.join(seq) : sepを区切り文字として、seqを連結してひとつの文字列にする。
                # 空白文字 " " を区切りにして分割処理
                str = " ".join( str.split() )
                #print( "sep.join( str.split() ) :", str )

                # リスト中の大文字→小文字に変換
                str = str.lower()

                return str

            text_data_features = [ clean_text(str) for str in text_data_features ]
    ```
- TensorFlow の組み込み関数を用いて、テキスト情報を数値インデックスのリストに変換する。
    - この際、テキストの長さは最大で `n_max_in_sequence = 25` 個の単語数とし、これよりも長いテキスト（シーケンス）は、この長さで打ち切り、それよりも短いテキスト（シーケンス）は 0 で埋める。（つまり、シーケンスなしとする）
    - 又、語彙に `min_word_freq = 10` 回以上出現する単語のみを考慮し、それらの単語をサイズが `embedding_size = 50` のトレーニング可能なベクトルに埋め込む。
    - そして、これらの処理には、`tf.contrib.learn.preprocessing.VocabularyProcessor(...)` を使用する。
    ```python
    [MLPreProcess.py]
    def def text_vocabulary_processing( ... ):
        ...
        # テキストの長さは最大で `n_max_in_sequence` 個の単語数とし、
        # これよりも長いテキスト（シーケンス）は、この長さで打ち切り、
        # それよりも短いテキスト（シーケンス）は 0 で埋める。（つまり、シーケンスなしとする）
        # 又、語彙に `min_word_freq` 回以上出現する単語のみを考慮し、それらの単語をサイズが `embedding_size` のトレーニング可能なベクトルに埋め込む。
        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
                              max_document_length = n_max_in_sequence, 
                              min_frequency = min_word_freq
                          )
        
        # Transform the text using the vocabulary.
        # VocabularyProcessor.fit_transform(...) : <generator object VocabularyProcessor.transform at 0x000001FAF79EF4C0>
        numpy.array( list( vocab_processor.fit_transform( text_data ) ) )
    ```
    ```python
    [main1.py]
    X_features = MLPreProcess.text_vocabulary_processing( text_data = text_data_features, n_max_in_sequence = 25, min_word_freq = 10 )
    ```
    - テキストデータ状の教師データ `text_data_labels` の方は、単純に `"ham"` → `1`, `"spam"` → `0` に変換する。
    ```python
    [main2.py]
    y_labels = numpy.array( [1 if label_str=='ham' else 0 for label_str in text_data_labels] )
    ```
- データ `X_features`, `y_labels` をシャッフルする。
    ```python
    [main2.py]
    shuffled_idx = numpy.random.permutation( numpy.arange( len(y_labels) ) )
    X_features_shuffled = X_features[ shuffled_idx ]
    y_labels_shuffled = y_labels[ shuffled_idx ]
    ```
- データをトレーニング用データ ( `X_train`, `y_train` ) とテスト用データ ( `X_test`, `y_test` ) に分割する。
    - 分割割合は、トレーニング用 80 % 、テスト用 20 % とする。
    ```python
    [main2.py]
    X_train, X_test, y_train, y_test \
    = MLPreProcess.dataTrainTestSplit( X_input = X_features, y_input = y_labels, ratio_test = 0.2, input_random_state = 1 )
    ```
- RNN モデルの各種パラメーターの設定を行う。
    - xxx
- RNN モデルの構造を定義する。
    - xxx

#### 補足（参考URL）
- 文字列操作
    - https://python.civic-apps.com/string-split-join/
    - http://www.yukun.info/blog/2008/08/python-upper-lower-case-letters-converted.html
- `zip(*list)` の挙動
    - https://qiita.com/7of9/items/e23bdd6e8d4d7997104a
- 正規表現について
    - http://uxmilk.jp/41416


<br>

<a id="ID_3-2-2"></a>

### コードの実行結果
> 実装中...





<br>

---

<a id="ID_3-3"></a>

## LSTM によるノイズ付き sin 波形（時系列データ）からの長期の波形の予想（生成）処理 : `main3.py`

<a id="ID_3-3-1"></a>

LSTM モデルによる時系列データの取り扱いの簡単な例として、先と同じ、ノイズ付き sin 波形（時系列データとみなす）の予想（生成）を考える。

- 先の通常の RNN モデルで、`tf.contrib.rnn.BasicRNNCell(...)` としていた箇所を、`tf.contrib.rnn.LSTMCell(...)` に変更する。
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
- その他の処理は、先の通常の RNN モデルと同様になる。
- TensorBoard で表示される計算グラフ
![graph_large_attrs_key _too_large_attrs limit_attr_size 1024 run lstm](https://user-images.githubusercontent.com/25688193/33447507-e63608de-d646-11e7-93e4-4bf43ee891b4.png)
![graph_large_attrs_key _too_large_attrs limit_attr_size 1024 run 2](https://user-images.githubusercontent.com/25688193/33447555-07d8a8ac-d647-11e7-8562-9942a7f7fe28.png)

<br>

<a id="ID_3-3-2"></a>

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

<a id="ID_5-1"></a>

### リカレントニューラルネットワーク（RNN）のアーキテクチャの種類

<a id="ID_5-1-1"></a>

#### 隠れ層間で回帰構造をもつネットワーク
![image](https://user-images.githubusercontent.com/25688193/31013864-0baf82cc-a553-11e7-9296-870f600f0381.png)
![image](https://user-images.githubusercontent.com/25688193/31013877-185db822-a553-11e7-9c5f-625acace78f8.png)
![image](https://user-images.githubusercontent.com/25688193/31016204-bcbc0cb0-a55e-11e7-86df-3ba574fa8bc2.png)
![image](https://user-images.githubusercontent.com/25688193/31017867-f379d312-a564-11e7-9d67-fc79a7dda26d.png)

<a id="ID_5-1-1-1"></a>

##### 通時的誤差逆伝搬法 [BPTT : back-propagation through time]
![image](https://user-images.githubusercontent.com/25688193/31018664-dacf44d4-a567-11e7-8014-34523646bfca.png)
![image](https://user-images.githubusercontent.com/25688193/31019688-7725288c-a56b-11e7-919d-d0be44d4be33.png)

<a id="ID_5-1-1-1-1"></a>

###### 通時的誤差逆伝搬法によるパラメータの更新式（誤差関数が最小２乗誤差）
![image](https://user-images.githubusercontent.com/25688193/31189494-64bc3a80-a973-11e7-90a8-348e97f93f47.png)
![image](https://user-images.githubusercontent.com/25688193/31189294-c48db61a-a972-11e7-9673-a7805c53eaf5.png)
![image](https://user-images.githubusercontent.com/25688193/31190398-f0e5337a-a975-11e7-8eff-ff74adf3a6ff.png)
![image](https://user-images.githubusercontent.com/25688193/31190919-835e326e-a977-11e7-966e-d3675cb83452.png)
![image](https://user-images.githubusercontent.com/25688193/31211718-661ae058-a9d6-11e7-96ae-075f35981fd1.png)


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


<a id="ID_5-3"></a>

### GRU [gated recurrent unit]
![image](https://user-images.githubusercontent.com/25688193/31338072-e514030c-ad38-11e7-908c-2446c32b60c6.png)
![image](https://user-images.githubusercontent.com/25688193/31306417-cfa02a3c-ab8a-11e7-8fb1-0579fe5aa0be.png)
![image](https://user-images.githubusercontent.com/25688193/31307146-b1ce34fa-ab98-11e7-862a-b139d330222e.png)
![image](https://user-images.githubusercontent.com/25688193/31308026-2bd77ff2-abaa-11e7-967e-04cff1579a36.png)


<a id="ID_5-4"></a>

### 双方向 RNN [BiRNN : Bidirectional RNN]
![image](https://user-images.githubusercontent.com/25688193/31332068-edadd682-ad1f-11e7-9f11-e7374b83465e.png)
![image](https://user-images.githubusercontent.com/25688193/31334064-78437f7a-ad27-11e7-84f2-decd65d1599d.png)
![image](https://user-images.githubusercontent.com/25688193/31335870-68a806d2-ad2f-11e7-9cd2-36648536cc64.png)
![image](https://user-images.githubusercontent.com/25688193/31335226-9d1c925a-ad2c-11e7-8f79-dccd9d931c41.png)
![image](https://user-images.githubusercontent.com/25688193/31335735-d0a5b780-ad2e-11e7-82ae-17cd33f2546c.png)


<a id="ID_5-5"></a>

### RNN Encoder-Decoder (Seqenence-to-sequence models)
![image](https://user-images.githubusercontent.com/25688193/31340555-7cd2efac-ad41-11e7-85f0-d70f0f9c7bee.png)
![image](https://user-images.githubusercontent.com/25688193/31370123-203bf512-adc4-11e7-8bc1-d65df760a43f.png)
![image](https://user-images.githubusercontent.com/25688193/31370130-2c510356-adc4-11e7-9a59-d2b93cfa4698.png)
![image](https://user-images.githubusercontent.com/25688193/31370139-372bbfd2-adc4-11e7-965c-96bc88661505.png)
![image](https://user-images.githubusercontent.com/25688193/31371878-45210ec6-adce-11e7-9096-3bbd77dee065.png)
![image](https://user-images.githubusercontent.com/25688193/31376678-b29f4ff0-ade0-11e7-9988-88602f28b32c.png)



### デバッグメモ
[17/11/30]
```python
Enter main()

len_sequences : 251
len_one_sequence : 25
X_features.shape : (226, 25, 1)
y_labels.shape : (226, 1)

X_train.shape : (203, 25, 1)
y_train.shape : (203, 1)

-----
after __init__()
RecurrentNN(batch_size=None, epochs=None, eval_step=None, n_hiddenLayer=None,
      n_in_sequence=None, n_inputLayer=None, n_outputLayer=None,
      session=None)
_session :  <tensorflow.python.client.session.Session object at 0x0000015696FFEF98>
_init_var_op :
 None
_loss_op :  None
_optimizer :  None
_train_step :  None
_y_out_op :  None
_n_inputLayer :  1
_n_hiddenLayer :  20
_n_outputLayer :  1
_n_in_sequence :  25
_epoches :  500
_batch_size :  10
_eval_step :  1
_X_holder :  Tensor("Placeholder:0", shape=(?, 25, 1), dtype=float32)
_t_holder :  Tensor("Placeholder_1:0", shape=(?, 1), dtype=float32)
_keep_prob_holder :  Tensor("Placeholder_2:0", dtype=float32)
_batch_size_holder :  Tensor("Placeholder_3:0", shape=(), dtype=int32)
_weights : 
 []
[]
_biases : 
 []
[]
----------------------------------


cell : <tensorflow.python.ops.rnn_cell_impl.BasicRNNCell object at 0x00000156A0DA8CF8>
initial_state_tsr : Tensor("BasicRNNCellZeroState/zeros:0", shape=(?, 20), dtype=float32)

```