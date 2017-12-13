## TensorFlow を用いた RNN Encoder-Decoder / Sequence-to-Seuqence の実装と簡単な応用

TensorFlow を用いた、RNN Encoder-Decoder / Sequence-to-Seqence による自然言語処理（質問応答、ワード予想等）の練習用実装コード集。

この `README.md` ファイルには、各コードの実行結果、概要、RNN の背景理論の説明を記載しています。
分かりやすいように `main.py` ファイル毎に１つの完結した実行コードにしています。

### 項目 [Contents]

1. [使用するライブラリ](#ID_1)
1. [使用するデータセット](#ID_2)
1. [コードの説明＆実行結果](#ID_3)
    1. [RNN Encoder-Decoder（LSTM 使用） による簡単な質問応答（足し算）処理 : `main1.py`](#ID_3-1)
        1. [コードの内容説明](#ID_3-1-1)
        1. [コードの実行結果](#ID_3-1-2)
    1. [RNN Encoder-Decoder（LSTM 使用） による英文学作品のワード予想処理 : `main2.py`](#ID_3-2)
        1. [コードの内容説明](#ID_3-2-1)
        1. [コードの実行結果](#ID_3-2-2)
    1. RNN Encoder-Decoder（複数の LSTM 層使用） による英文学作品のワード予想処理 : `main3.py`
1. [背景理論](#ID_4)
    1. [リカレントニューラルネットワーク [RNN : Recursive Neural Network]<br>＜階層型ニューラルネットワーク＞](#ID_5)
        1. [RNN Encoder-Decoder (Seqenence-to-sequence models)](#ID_5-5)

<a id="ID_1"></a>

### 使用するライブラリ

> TensorFlow ライブラリ <br>
>> `tf.contrib.rnn.LSTMRNNCell(...)` : <br>
>> 時系列に沿った LSTM 構造を提供するクラス `LSTMCell` の `cell` を返す。<br>
>> この `cell` は、内部（プロパティ）で state（隠れ層の状態）を保持しており、これを次の時間の隠れ層に順々に渡していくことで、時間軸の逆伝搬を実現する。<br>
>>> https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMCell<br>

>> `tf.einsum(...)` : Tensor 積等の Tensor 間演算をアインシュタイン縮約記法で指定する。<br>
>>> https://www.tensorflow.org/api_docs/python/tf/einsum<br>

> その他ライブラリ
>>

<br>

<a id="ID_2"></a>

### 使用するデータセット

- Project Gutenberg ( https://www.gutenberg.org/ ) にある文学作品のデータ
    - シェイクスピア作品
        - http://www.gutenberg.org/ebooks/100?msg=welcome_stranger
        - UTF-8 / txt 形式 : http://www.gutenberg.org/cache/epub/100/pg100.txt
        - HTML 形式 : http://www.gutenberg.org/cache/epub/100/pg100-images.html


<br>

<a id="ID_3"></a>

## コードの説明＆実行結果

<a id="ID_3-1"></a>

## RNN Encoder-Decoder（LSTM 使用） による簡単な質問応答（足し算）処理 : `main1.py`

RNN Encoder-Decoder（LSTM 使用） による自然言語処理の応用例として、質問応答（QA）があるが、ここでは、この質問応答（QA）の簡単な例として、指定された数字の足し算を答える応答問題を考える。


<a id="ID_3-1-1"></a>

以下、コードの説明。

- まず、指定された数字の足し算を答える応答用に、整数の加算演算データセットを生成する関数 `MLPreProcess.generate_add_uint_operation_dataset(...)` を実装する。<br>
![image](https://user-images.githubusercontent.com/25688193/33876304-7fd5dcc4-df68-11e7-9534-f33e4a12c194.png)<br>
![image](https://user-images.githubusercontent.com/25688193/33877497-e442e23a-df6b-11e7-9210-5c230548f28d.png)
    - この関数の処理は、以下のコードのようになる。<br>
    - 指定された桁数の整数字をランダムに生成する処理を、関数ブロックで以下のコードのように実装しておく。（後述の処理で使用）
    ```python
    [MLPreProcess.py]
    def generate_add_uint_operation_dataset( ... ):
        ...
        def generate_number_uint( digits ):
            """
            指定された桁数の整数字をランダムに生成する。
            """
            number = ""

            # 1 ~ digit 番目の桁数に関してのループ
            for i in range( numpy.random.randint(1, digits+1) ):
                number += numpy.random.choice( list("0123456789") )
    
            return int(number)
    ```
    - そして、各シーケンスの空白文字の Padding 処理による桁合わせ処理を、関数ブロックで以下のコードのように実装する。（後述の処理で使用）
    ```python
    [MLPreProcess.py]
    def generate_add_uint_operation_dataset( ... ):
        ...
        def padding( str, max_len ):
            """
            空白文字の Padding 処理による桁合わせ
            """
            # 空白 × 埋め合わせ数
            str_padding = str + " " * ( max_len - len(str) )
            
            return str_padding
    ```
    - ランダムに生成した、桁数 `digits` からなる２つの整数の足し算の式 `uint_x + uint_y` と、この式の解に対応するデータ（応答データ）を、指定されたサンプル数 `n_samples` 個ぶん作成し、それらを空白文字 `" "` で padding 処理してシーケンス長を揃えておく。
    ```python
    [MLPreProcess.py]
    def generate_add_uint_operation_dataset( ... ):
        ...
        # 入力桁数
        input_digit = digits * 2 + 1     # 123+456
        # 出力桁数
        output_digit = digits + 1        # 500+500=1000 のような桁上りのケースを考慮

        # 
        dat_x = []
        dat_y = []

        # 指定されたサンプル数ぶんループ処理
        for i in range( n_samples ):
            uint_x = generate_number_uint( digits )
            uint_y = generate_number_uint( digits )

            train = "{}+{}".format( uint_x, uint_y )
            train = padding( train, input_digit )
            dat_x.append( train )

            target = "{}".format( uint_x + uint_y )
            target = padding( target, output_digit )
            dat_y.append( target )
    ```
    - 文字からインデックスへのディクショナリ `dict_str_to_idx` にもとづき、各文字を one-hot encode 処理する。
    ```python
    [MLPreProcess.py]
    def generate_add_uint_operation_dataset( ... ):
        ...
        map_str = "0123456789+ "  # map 作成用の使用する文字列
        # 文字からインデックスへの map
        dict_str_to_idx = { key: idx for (idx,key) in enumerate( map_str ) }
        
        # インデックスから文字への map
        dict_idx_to_str = { idx: key for (key,idx) in dict_str_to_idx.items() }

        # one-hot encode されたデータ shape = (n_sample, sequence, one-hot encodeed vector size)
        X_features = numpy.zeros( ( len(dat_x), input_digit, len(map_str) ), dtype = numpy.int )
        y_labels = numpy.zeros( ( len(dat_x), output_digit, len(map_str) ), dtype = numpy.int )

        for i in range( n_samples ):
            for (j, str) in enumerate( dat_x[i] ):
                X_features[ i, j, dict_str_to_idx[str] ] = 1     # one-hot encode の 1 の部分
            for (j, str) in enumerate( dat_y[i] ):
                y_labels[ i, j, dict_str_to_idx[str] ] = 1     # one-hot encode の 1 の部分

        return X_features, y_labels
    ```
- データセットを、トレーニング用データセットと、テスト用データセットに分割する。
    - 分割割合は、トレーニング用データ 90%、テスト用データ 10%
    ```python
    X_train, X_test, y_train, y_test \
    = MLPreProcess.dataTrainTestSplit( X_input = X_features, y_input = y_labels, ratio_test = 0.1, input_random_state = 1 )
    ```
- この自然言語処理（NLP）に対応した、RNN Encoder-Decoder モデルの各種パラメーターの設定を行う。
    - この設定は、`RecurrectNNEncoderDecoderLSTM` クラスのインスタンス作成時の引数にて行う。
    ```python
    [main1.py]
    rnn1 = RecurrectNNEncoderDecoderLSTM(
               session = tf.Session(),
               n_inputLayer = 12,                   # 12 : "0123456789+ " の 12 文字
               n_hiddenLayer = 128,                 # rnn の cell 数と同じ
               n_outputLayer = 12,                  # 12 : "0123456789+ " の 12 文字
               n_in_sequence_encoder = 7,           # エンコーダー側のシーケンス長 / 足し算の式のシーケンス長 : "123 " "+" "456 " の計 4+1+4=7 文字
               n_in_sequence_decoder = 4,           # デコーダー側のシーケンス長 / 足し算の式の結果のシーケンス長 : "1000" 計 4 文字
               epochs = 4000,
               batch_size = 200,
               eval_step = 1
           )
    ```
- RNN Encoder-Decoder （LSTM使用） モデルの構造を定義する。
![image](https://user-images.githubusercontent.com/25688193/31370123-203bf512-adc4-11e7-8bc1-d65df760a43f.png)
    - この処理は、`RecurrectNNEncoderDecoderLSTM` クラスの `model()` メソッドにて行う。
    - まず、Encoder 側のモデルを RNN の再帰構造に従って構築していく。
    ```python
    [RecurrectNNEncoderDecoderLSTM.py]
    def model():
        ...
        #--------------------------------------------------------------
        # Encoder
        #--------------------------------------------------------------
        # tf.contrib.rnn.LSTMCell(...) : 時系列に沿った RNN 構造を提供するクラス `LSTMCell` のオブジェクト cell を返す。
        # この cell は、内部（プロパティ）で state（隠れ層の状態）を保持しており、
        # これを次の時間の隠れ層に順々に渡していくことで、時間軸の逆伝搬を実現する。
        cell_encoder = tf.contrib.rnn.LSTMCell( 
                           num_units = self._n_hiddenLayer,     # int, The number of units in the RNN cell.
                           forget_bias = 1.0                    # 忘却ゲートのバイアス項 / Default : 1.0  in order to reduce the scale of forgetting at the beginning of the training.
                       )
        #self._rnn_cells_encoder.append( cell_encoder ) # 後述の処理で同様の処理が入るので不要

        # 最初の時間 t0 では、過去の隠れ層がないので、
        # cell.zero_state(...) でゼロの状態を初期設定する。
        initial_state_encoder_tsr = cell_encoder.zero_state( self._batch_size_holder, tf.float32 )
        self._rnn_states_encoder.append( initial_state_encoder_tsr )

        # Encoder の過去の隠れ層の再帰処理
        with tf.variable_scope('Encoder'):
            for t in range( self._n_in_sequence_encoder ):
                if (t > 0):
                    # tf.get_variable_scope() : 名前空間を設定した Variable にアクセス
                    # reuse_variables() : reuse フラグを True にすることで、再利用できるようになる。
                    tf.get_variable_scope().reuse_variables()

                # LSTMCellクラスの `__call__(...)` を順次呼び出し、
                # 各時刻 t における出力 cell_output, 及び状態 state を算出
                cell_encoder_output, state_encoder_tsr = cell_encoder( inputs = self._X_holder[:, t, :], state = self._rnn_states_encoder[-1] )

                # 過去の隠れ層の出力をリストに追加
                self._rnn_cells_encoder.append( cell_encoder_output )
                self._rnn_states_encoder.append( state_encoder_tsr )

        # 最終的な Encoder の出力
        #output_encoder = self._rnn_cells_encoder[-1]
    ```
    - 次に、Decoder 側のモデルを RNN の再帰構造に従って構築していく。
        - Decoder の初期状態は Encoder の最終出力なので、これに従って初期状態を定める。
    ```python
    [RecurrectNNEncoderDecoderLSTM.py]
    def model():
        ...
    ```
    - xxx
    ```python
    [RecurrectNNEncoderDecoderLSTM.py]
    def model():
        ...
    ```
    - 最終的なモデルの出力は、隠れ層から出力層への入力を softmax して出力する。
    ```python
    [RecurrectNNEncoderDecoderLSTM.py]
    def model():
        ...
        #--------------------------------------------------------------
        # モデルの出力
        #--------------------------------------------------------------
        # softmax
        self._y_out_op = Softmax().activate( input = y_in_op )

        return self._y_out_op
    ```
- 損失関数として、ソフトマックス・エントロピー関数を使用する。
    ```python
    [main1.py]
    rnn1.loss( SoftmaxCrossEntropy() )
    ```
- 最適化アルゴリズム Optimizer として、Adam アルゴリズム を使用する。
    - 学習率 `learning_rate` は、0.001 で検証。減衰項は `beta1 = 0.9`, `beta1 = 0.999`
    ```python
    [main1.py]
    rnn1.optimizer( Adam( learning_rate = learning_rate1, beta1 = adam_beta1, beta2 = adam_beta2 ) )
    ```
- トレーニング用データ `X_train`, `y_train` に対し、fitting 処理を行う。
    ```python
    [main1.py]
    rnn1.fit( X_train, y_train )
    ```
- fitting 処理 `fit(...)` 後のモデルで予想を行い、正解率を算出する。
    - xxx
- 尚、このモデルの TensorBorad で描写した計算グラフは以下のようになる。
![graph_large_attrs_key _too_large_attrs limit_attr_size 1024 run](https://user-images.githubusercontent.com/25688193/33880754-feae2346-df75-11e7-982a-28f4f805da71.png)
![graph_large_attrs_key _too_large_attrs limit_attr_size 1024 run 1](https://user-images.githubusercontent.com/25688193/33880755-fed72192-df75-11e7-9fab-4bfc5b9459fb.png)



<a id="ID_3-1-2"></a>

### コードの実行結果

### 実行条件１
- ｛入力層 : 12 ノード、隠れ層 : **128** ノード、出力層 : 12 ノード｝の RNN Encoder-Decoder モデル
    - エンコーダーのシーケンス長 : **7** 個, デコーダーのシーケンス長 : **4** 個
    - 学習率 **0.001**, 最適化アルゴリズム : Adam ( 減衰項 : beta1 = 0.9, beta2 = 0.999 )
    - サンプル数 **20000** 個（トレーニング用データ : 90 %、テスト用データ : 10% に分割）
    - エポック数 20000 回, ミニバッチサイズ **100**

##### 損失関数のグラフ（実行条件１）
![rnn_encoder-decoder_1-1-1 _samples20000-batch100-adam](https://user-images.githubusercontent.com/25688193/33927863-1e8f299c-e027-11e7-83b5-5a1f80649781.png)
> エポック数 15000 回程度で 0 に収束しており、うまく学習出来ていることが見て取れる。

##### 学習済みモデルでの正解率（実行条件１）

|data|Acuraccy|サンプル数|
|---|---|---|---|
|total|0.969|20000|
|train data|0.989|18000 (90.0%)|
|test data|0.794|2000 (10.0%)|

> トレーニング用データ、テスト用データともに高い正解率となっており、<br>
> この自然言語処理の問題に対し、このモデルはうまく適用出来ていることが分かる。

##### 学習済みモデルでのテスト用データでの応答処理結果（実行条件１）

||1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|質問 Q|3+881  |830+95  |19+5   |1+21   |34+910 |52+14  |7+1    |522+3  |9+8    |433+77 |244+396|0+71   |87+613 |119+573|568+4  |98+2   |287+302|87+613 |119+573|568+4  |
|応答 A|884 |916 |24  |22  |944 |67  |8   |525 |17  |510 |631 |71  |700 |612 |572 |100 |598 |700 |612 |572 |
|正誤|○|×|○|○|○|×|○|○|○|○|×|○|○|×|○|○|×|○|×|○|

> 足し算の応答結果が誤っている場合でも、全く正解と離れた値ではなく、正解に近い値になっている傾向が見て取れる。<br>
> これは、このモデル (RNN Encoder-Decoder) が、足し算のルール（入出力パターン）をうまく学習出来ているためと考えられる。<br>
> 尚、応答結果が誤っている場合において、足し算の値が大きい場合のほうが、正解との値の差は大きい傾向も見て取れる。<br>


<br>

---


<a id="ID_3-2"></a>

## RNN Encoder-Decoder（LSTM 使用） による英文学作品のワード予想処理 : `main2.py`
> 実装中...

RNN Encoder-Decoder による自然言語処理（NLP）の一例として、英文学作品のシェイクスピア作品のテキストデータ ( http://www.gutenberg.org/cache/epub/100/pg100.txt ) を用いて、RNN Encoder-Decoder （LSTM 使用）モデルで学習し、特定のワード（"thus"（それ故）, "more"（更には） 等）の後に続くワードを予想する処理を実装する。

<a id="ID_3-2-1"></a>

以下、コードの説明

- まず、The Project Gutenberg EBook にある、シェイクスピア作品のテキストデータの読み込み＆抽出処理を行う。
    - この処理は、`MLPreProcess.load_textdata_by_shakespeare_from_theProjectGutenbergEBook(...)` 関数にて行う。
        - 具体的には、まず、指定されたテキストデータの各行を Unicode 形式で読み込み、データを格納する。
        ```python
        [MLPreProcess.py]
        def load_textdata_by_shakespeare_from_theProjectGutenbergEBook( ... ):
            text_data = []

            #--------------------------------------------------------
            # codecs.open() 関数と with 構文でテキストデータの読み込む
            # "r" : 文字のまま読み込み
            #--------------------------------------------------------
            with codecs.open( path, "r", "utf-8" ) as file:
                # txt ファイルの各行に関してのループ処理
                for row in file:
                    # 各行の文字列全体（特殊文字、空白込 : \t　\n）を格納
                    text_data.append( row )
        ```
        - 読み込むテキストデータの内容上、本文とは関係ない説明文を除外する。
        ```python
        [MLPreProcess.py]
        def load_textdata_by_shakespeare_from_theProjectGutenbergEBook( ... ):
            ...
            # EBook のテキストファイルに含まれている、最初の説明文の段落部分を除外
            text_data = text_data[ n_DeleteParagraph : ]
        ```
        - 改行, 先頭に復帰の特殊文字 \n, \r を削除する
        ```python
        [MLPreProcess.py]
        def load_textdata_by_shakespeare_from_theProjectGutenbergEBook( ... ):
            ...
            text_data = [ str.replace( "\r\n", "" ) for str in text_data ]
            text_data = [ str.replace( "\n", "" ) for str in text_data ]
        ```
        - クリーニング処理を実施する。<br>
        これは、文字量を減らすために、各種句読点、余分なホワイトスペースを削除する処理となる。<br>
        但し、ハイフン "-" と、アポストロフィ "'" は残す。（この作品文章が文章内容を繋ぐのに、頻繁に使用されているため）
        ```python
        [MLPreProcess.py]
        def load_textdata_by_shakespeare_from_theProjectGutenbergEBook( ... ):
            ...
            if ( bCleaning == True ):
                # string.punctuation : 文字と文字の間の句読点、括弧などをまとめたもの
                # 置換表現で「!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~」等
                punctuation = string.punctuation

                # punctuation から ハイフン "-" と、アポストロフィ "'" を除外
                # sep.join(seq) : sepを区切り文字として、seqを連結してひとつの文字列にする。
                punctuation = ''.join( [ x for x in punctuation if x not in ['-', "'"] ] )

                def clean_text( str ):
                    # ハイフン "-" と、アポストロフィ "'" 以外の特殊文字をホワイトスペース " " に置換
                    # re.sub() : 正規表現で文字列を別の文字列で置換
                    str = re.sub( 
                              pattern = r"[{}]".format( punctuation ),  # 正規表現 : []集合, |和集合（または）()グループ化
                              repl = " ",                               # 置換する文字列 : " " なのでホワイトスペースに置換
                              string = str                              # 置換される文字列
                          )

                    # 任意の空白文字 \s = [\t\n\r\f\v] をホワイトスペース " " に置換
                    # + : １回以上の繰り返し（正規表現）
                    str = re.sub( 
                              pattern = "\s+",      # 正規表現 : []集合, |和集合（または）()グループ化
                              repl = " ",           # 置換する文字列 : " " なのでホワイトスペースに置換
                            string = str          # 置換される文字列
                        )

                    # ホワイトスペース " " に置換したものを一斉に除外
                    # str.strip() : 引数を指定しないとホワイトスペースを除去する
                    str = str.strip()

                    # リスト中の大文字→小文字に変換
                    str = str.lower()

                    return str

            text_data = [ clean_text(str) for str in text_data ]
        ```
- 抽出したテキストデータから、出現頻度の高い単語をディクショナリに登録する。<br>
（出現頻度の高い単語のみ学習の対象とする。出現頻度の低い単語は除外）
    - この処理は `MLPreProcess.text_vocabulary_processing_without_tensorflow(...)` にて行う。
        - 前段階として、str からなる list → １つの str に変換したものを、空白スペースで split し、単語単位の配列に変換しておく。
        ```python
        [MLPreProcess]
        def text_vocabulary_processing_without_tensorflow(...):
            ...
            # list<str> → １つの str に変換
            text_data = "".join( text_data )
            
            # 空白スペースで split し、単語単位の配列に変換
            text_data = text_data.split( " " )
        ```
        - `collections` モジュールの `collections.Counter(...)` を用いて、抽出したテキストデータから、単語の出現頻度を数える
        ```python
        [MLPreProcess]
        def text_vocabulary_processing_without_tensorflow(...):
            ...
            word_counts = collections.Counter( text_data )
        ```
        - 抽出した単語の出現頻度から、出現頻度の高い (`min_word_freq` 値以上の) 単語をディクショナリに登録する。出現頻度の低い単語は除外
        ```python
        [MLPreProcess]
        def text_vocabulary_processing_without_tensorflow(...):
            ...
            word_counts = { key: count for (key,count) in word_counts.items() if count > min_word_freq }    # ディクショナリの内包表現
        ```
        - 語彙 "xxx" → インデックスへの map を作成する。
        ```python
        [MLPreProcess]
        def text_vocabulary_processing_without_tensorflow(...):
            ...
            # dict.keys() : ディクショナリから key を取り出し
            dict_keys_words = word_counts.keys()
            dict_vcab_to_idx = { key: (idx+1) for (idx,key) in enumerate( dict_keys_words ) }
            # 不明な key (=vocab) のインデックスとして 0 を登録
            dict_vcab_to_idx[ "unknown" ] = 0

        ```
        - インデックス → 語彙 "xxx" への map を作成する。
        ```python
        [MLPreProcess]
        def text_vocabulary_processing_without_tensorflow(...):
            ...
            dict_idx_to_vocab = { idx: key for (key,idx) in dict_vcab_to_idx.items() }
        ```
        - 更に、抽出したテキストデータを、このディクショナリに基づき、数値インデックス情報に変換する。
        ```python
        [MLPreProcess]
        def text_vocabulary_processing_without_tensorflow(...):
            ...
            #---------------------------------------------------------
            # テキストデータのインデックス配列
            #---------------------------------------------------------
            text_data_idx = []
        
            # テキストから抽出した単語単位の配列 text_data に関してのループ
            for (idx,words) in enumerate( text_data ):
                try:
                    text_data_idx.append( dict_vcab_to_idx[words] )
                except:
                    text_data_idx.append( 0 )

            # list → ndarray に変換
            text_data_idx = numpy.array( text_data_idx )

            # 単語の数
            n_vocab = len( dict_idx_to_vocab ) + 1
        ```
- この自然言語処理（NLP）に対応した、RNN Encoder-Decoder モデルの各種パラメーターの設定を行う。
    - この設定は、`RecurrectNNEncoderDecoderLSTM` クラスのインスタンス作成時の引数にて行う。
    - xxx
- xxx




<br>



---


<a id="ID_4"></a>

## 背景理論

<a id="ID_5"></a>

## リカレントニューラルネットワーク [RNN : Recursive Neural Network]<br>＜階層型ニューラルネットワーク＞の概要
![image](https://user-images.githubusercontent.com/25688193/30980712-f06a0906-a4bc-11e7-9b15-4c46834dd6d2.png)
![image](https://user-images.githubusercontent.com/25688193/30981066-22f53124-a4be-11e7-9111-9514f04aed7c.png)

<a id="ID_5-5"></a>

### RNN Encoder-Decoder (Seqenence-to-sequence models)
![image](https://user-images.githubusercontent.com/25688193/31340555-7cd2efac-ad41-11e7-85f0-d70f0f9c7bee.png)
![image](https://user-images.githubusercontent.com/25688193/31370123-203bf512-adc4-11e7-8bc1-d65df760a43f.png)
![image](https://user-images.githubusercontent.com/25688193/31370130-2c510356-adc4-11e7-9a59-d2b93cfa4698.png)
![image](https://user-images.githubusercontent.com/25688193/31370139-372bbfd2-adc4-11e7-965c-96bc88661505.png)
![image](https://user-images.githubusercontent.com/25688193/31371878-45210ec6-adce-11e7-9096-3bbd77dee065.png)
![image](https://user-images.githubusercontent.com/25688193/31376678-b29f4ff0-ade0-11e7-9988-88602f28b32c.png)


<br>

## デバッグメモ

[17/12/13]
```python
    def predict( self, X_test ):
        prob = self._session.run(
                   self._y_out_op,
                   feed_dict = { 
                       self._X_holder: X_test,
                       self._batch_size_holder: 1
                   }
               )

tensorflow.python.framework.errors_impl.InvalidArgumentError: ConcatOp : Dimensions of inputs should match: shape[0] = [10,12] vs. shape[1] = [1,128]
	 [[Node: Encoder/Encoder/lstm_cell/concat = ConcatV2[N=2, T=DT_FLOAT, Tidx=DT_INT32, _device="/job:localhost/replica:0/task:0/device:CPU:0"](Encoder/strided_slice, LSTMCellZeroState/zeros, Decoder/Eval_root/Decoder/lstm_cell/split_2/split_dim)]]

X_test.shape : (10, 7, 12)
self._y_out_op : shape = TensorShape([Dimension(None), Dimension(4), Dimension(12)])
self._X_holder :.shape = TensorShape([Dimension(None), Dimension(7), Dimension(12)])

----------------------------------
after model()
RecurrectNNEncoderDecoderLSTM(batch_size=None, epochs=None, eval_step=None,
               n_hiddenLayer=None, n_in_sequence_decoder=None,
               n_in_sequence_encoder=None, n_inputLayer=None,
               n_outputLayer=None, session=None)
_session :  <tensorflow.python.client.session.Session object at 0x0000029D73B43F60>
_init_var_op :
 None
_loss_op :  None
_optimizer :  None
_train_step :  None
_y_out_op :  Tensor("Eval_root/Reshape:0", shape=(?, 4, 12), dtype=float32)
_n_inputLayer :  12
_n_hiddenLayer :  128
_n_outputLayer :  12
_n_in_sequence_encoder : 7
_n_in_sequence_decoder : 4
_epoches :  100
_batch_size :  20
_eval_step :  1
_X_holder : Tensor("X_holder:0", shape=(?, 7, 12), dtype=float32)
_t_holder : Tensor("t_holder:0", shape=(?, 4, 12), dtype=float32)
_dropout_holder : Tensor("dropout_holder:0", dtype=float32)
_batch_size_holder : Tensor("batch_size_holder:0", shape=(), dtype=int32)
_bTraing_holder : Tensor("bTraining_holder:0", dtype=bool)
_rnn_cells_encoder : 
 [<tf.Tensor 'Encoder/Encoder/lstm_cell/mul_2:0' shape=(?, 128) dtype=float32>, <tf.Tensor 'Encoder/Encoder/lstm_cell/mul_5:0' shape=(?, 128) dtype=float32>, <tf.Tensor 'Encoder/Encoder/lstm_cell/mul_8:0' shape=(?, 128) dtype=float32>, <tf.Tensor 'Encoder/Encoder/lstm_cell/mul_11:0' shape=(?, 128) dtype=float32>, <tf.Tensor 'Encoder/Encoder/lstm_cell/mul_14:0' shape=(?, 128) dtype=float32>, <tf.Tensor 'Encoder/Encoder/lstm_cell/mul_17:0' shape=(?, 128) dtype=float32>, <tf.Tensor 'Encoder/Encoder/lstm_cell/mul_20:0' shape=(?, 128) dtype=float32>]
_rnn_states_encoder : 
 [LSTMStateTuple(c=<tf.Tensor 'LSTMCellZeroState/zeros:0' shape=(?, 128) dtype=float32>, h=<tf.Tensor 'LSTMCellZeroState/zeros_1:0' shape=(?, 128) dtype=float32>), LSTMStateTuple(c=<tf.Tensor 'Encoder/Encoder/lstm_cell/add_1:0' shape=(?, 128) dtype=float32>, h=<tf.Tensor 'Encoder/Encoder/lstm_cell/mul_2:0' shape=(?, 128) dtype=float32>), LSTMStateTuple(c=<tf.Tensor 'Encoder/Encoder/lstm_cell/add_3:0' shape=(?, 128) dtype=float32>, h=<tf.Tensor 'Encoder/Encoder/lstm_cell/mul_5:0' shape=(?, 128) dtype=float32>), LSTMStateTuple(c=<tf.Tensor 'Encoder/Encoder/lstm_cell/add_5:0' shape=(?, 128) dtype=float32>, h=<tf.Tensor 'Encoder/Encoder/lstm_cell/mul_8:0' shape=(?, 128) dtype=float32>), LSTMStateTuple(c=<tf.Tensor 'Encoder/Encoder/lstm_cell/add_7:0' shape=(?, 128) dtype=float32>, h=<tf.Tensor 'Encoder/Encoder/lstm_cell/mul_11:0' shape=(?, 128) dtype=float32>), LSTMStateTuple(c=<tf.Tensor 'Encoder/Encoder/lstm_cell/add_9:0' shape=(?, 128) dtype=float32>, h=<tf.Tensor 'Encoder/Encoder/lstm_cell/mul_14:0' shape=(?, 128) dtype=float32>), LSTMStateTuple(c=<tf.Tensor 'Encoder/Encoder/lstm_cell/add_11:0' shape=(?, 128) dtype=float32>, h=<tf.Tensor 'Encoder/Encoder/lstm_cell/mul_17:0' shape=(?, 128) dtype=float32>), LSTMStateTuple(c=<tf.Tensor 'Encoder/Encoder/lstm_cell/add_13:0' shape=(?, 128) dtype=float32>, h=<tf.Tensor 'Encoder/Encoder/lstm_cell/mul_20:0' shape=(?, 128) dtype=float32>)]
_rnn_cells_decoder : 
 [<tf.Tensor 'Encoder/Encoder/lstm_cell/mul_20:0' shape=(?, 128) dtype=float32>, <tf.Tensor 'Decoder/Eval_root/Decoder/lstm_cell/mul_2:0' shape=(?, 128) dtype=float32>, <tf.Tensor 'Decoder/Eval_root/Decoder/lstm_cell/mul_5:0' shape=(?, 128) dtype=float32>, <tf.Tensor 'Decoder/Eval_root/Decoder/lstm_cell/mul_8:0' shape=(?, 128) dtype=float32>]
_rnn_states_decoder : 
 [LSTMStateTuple(c=<tf.Tensor 'Encoder/Encoder/lstm_cell/add_13:0' shape=(?, 128) dtype=float32>, h=<tf.Tensor 'Encoder/Encoder/lstm_cell/mul_20:0' shape=(?, 128) dtype=float32>), LSTMStateTuple(c=<tf.Tensor 'Decoder/Eval_root/Decoder/lstm_cell/add_1:0' shape=(?, 128) dtype=float32>, h=<tf.Tensor 'Decoder/Eval_root/Decoder/lstm_cell/mul_2:0' shape=(?, 128) dtype=float32>), LSTMStateTuple(c=<tf.Tensor 'Decoder/Eval_root/Decoder/lstm_cell/add_3:0' shape=(?, 128) dtype=float32>, h=<tf.Tensor 'Decoder/Eval_root/Decoder/lstm_cell/mul_5:0' shape=(?, 128) dtype=float32>), LSTMStateTuple(c=<tf.Tensor 'Decoder/Eval_root/Decoder/lstm_cell/add_5:0' shape=(?, 128) dtype=float32>, h=<tf.Tensor 'Decoder/Eval_root/Decoder/lstm_cell/mul_8:0' shape=(?, 128) dtype=float32>)]
_weights : 
 [<tf.Variable 'init_weight_var:0' shape=(128, 12) dtype=float32_ref>]
_biases : 
 [<tf.Variable 'init_bias_var:0' shape=(12,) dtype=float32_ref>]
----------------------------------

```

```python

epochs : 200
n_batches : 90

==========
Epoch: 0
==========
validation loss: 0.605214
validation acc:  0.35025

==========
Epoch: 110
==========
validation loss: 0.0786947
validation acc:  0.9175

==========
Epoch: 158
==========
validation loss: 0.0823262
validation acc:  0.92675
```

```python

```