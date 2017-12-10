## TensorFlow を用いた RNN Encoder-Decoder の実装と簡単な応用

TensorFlow を用いた、RNN Encoder-Decoder による自然言語処理（質問応答、ワード予想等）の練習用実装コード集。

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
        1.[]

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
> 実装中...

<!--

<a id="ID_3-1-1"></a>

以下、コードの説明。

- xxx

<a id="ID_3-1-2"></a>

### コードの実行結果

### 損失関数のグラフ

-->

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

<a id="ID_5-2"></a>

