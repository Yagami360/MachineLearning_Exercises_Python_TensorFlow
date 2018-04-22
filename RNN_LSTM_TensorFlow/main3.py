# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境 (TensorFlow インストール済み)
#     <Anaconda Prompt>
#     conda create -n tensorflow python=3.5
#     activate tensorflow
#     pip install --ignore-installed --upgrade tensorflow
#     pip install --ignore-installed --upgrade tensorflow-gpu

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter     # 出現回数をカウントする辞書型オブジェクト
import string

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops

# 自作クラス
from MLPreProcess import MLPreProcess
from MLPlot import MLPlot

import NNActivation                                     # ニューラルネットワークの活性化関数を表すクラス
from NNActivation import NNActivation
from NNActivation import Sigmoid
from NNActivation import Relu
from NNActivation import Softmax

import NNLoss                                           # ニューラルネットワークの損失関数を表すクラス
from NNLoss import L1Norm
from NNLoss import L2Norm
from NNLoss import BinaryCrossEntropy
from NNLoss import CrossEntropy
from NNLoss import SigmoidCrossEntropy
from NNLoss import SoftmaxCrossEntropy
from NNLoss import SparseSoftmaxCrossEntropy

import NNOptimizer                                      # ニューラルネットワークの最適化アルゴリズム Optimizer を表すクラス
from NNOptimizer import GradientDecent
from NNOptimizer import GradientDecentDecay
from NNOptimizer import Momentum
from NNOptimizer import NesterovMomentum
from NNOptimizer import Adagrad
from NNOptimizer import Adadelta
from NNOptimizer import Adam

from NeuralNetworkBase import NeuralNetworkBase
#from RecurrectNNEncoderDecoderLSTM import RecurrectNNEncoderDecoderLSTM
#from Seq2SeqRNNEncoderDecoderLSTM import Seq2SeqRNNEncoderDecoderLSTM
from Many2OneMultiRNNLSTM import Many2OneMultiRNNLSTM


def SaveCsvFileFromIMDbDataset( basepath ):
    #import pypind
    import pandas as pd
    import os

    labels = { "pos":1, "neg":0 }   # 肯定的、否定的の評価ラベル
    
    #---------------------------------------------
    # ファイルを読み込み pandas DataFrame に格納
    #---------------------------------------------
    df = pd.DataFrame()

    for str1 in ( "test", "train" ):
        for str2 in ( "pos", "neg" ):
            path = os.path.join( basepath, str1, str2 )
            
            # listdir関数でディレクトリとファイルの一覧を取得する
            files = os.listdir( path )

            for file in files:
                with open( os.path.join(path,file), "r", encoding = "utf-8" ) as infile:
                    txt = infile.read()
                
                df = df.append( [ [txt,labels[str2]] ], ignore_index=True )

    df.columns = [ "review", "sentiment" ]

    #---------------------------------------------
    # csv ファイルに書き込み
    #---------------------------------------------
    # 行の順番をシャッフルしておく。（過学習対策）
    np.random.seed(0)
    df = df.reindex( np.random.permutation(df.index) )

    df.to_csv( "movie_data.csv", index=False, encoding="utf-8" )

    return



def main():
    """
    TensorFlow を用いた Many-to-one な RNN （LSTM 使用）によるIMDb映画評論の感情分析
    """
    print("Enter main()")

    # Reset graph
    #ops.reset_default_graph()

    #======================================================================
    # データセットを読み込み or 生成
    # Import or generate data.
    #======================================================================
    #SaveCsvFileFromIMDbDataset( "C:\Data\MachineLearning_DataSet\\aclImdb" )
    dfIMb = pd.read_csv( "movie_data.csv", encoding="utf-8" )
    
    counts = Counter()

    #-------------------------------------------------
    # クリーニング & 出現単語カウント処理
    #-------------------------------------------------
    for (i,review) in enumerate( dfIMb["review"] ):
        # string.punctuation : 文字と文字の間の句読点、括弧などをまとめたもの
        # 置換表現で「!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~」等
        punctuation = string.punctuation

        # sep.join(seq) : sepを区切り文字として、seqを連結してひとつの文字列にする。
        # "".join(...) : list<str> → １つの str に変換
        # " " + str + " " : 空白スペースで分割
        text = "".join( [str if str not in punctuation else " "+str+" " for str in review ] )

        # リスト中の大文字→小文字に変換
        text = text.lower()

        # loc() : 行ラベル、 列ラベルで pandas DataFrame のデータを参照
        dfIMb.loc[i, "review"] = text

        # Counter.update() : 辞書オブジェクトに辞書オブジェクトを後ろから連結。
        # string.split(“ 区切り文字 ”) : 区切り文字で区切られたリストを得る
        counts.update( text.split() )

    # counts Counter({'the': 667950, '.': 650520, ',': 544818, 'and': 324383, 'a': 322941, 'of': 289412, ...
    # 'japanes': 1, 'germinates': 1, 'aspidistra': 1, 'bonhomie': 1, 'horlicks': 1})
    #print( "counts", counts)

    #======================================================================
    # データを変換、正規化
    # Transform and normalize data.
    # ex) data = tf.nn.batch_norm_with_global_normalization(...)
    #======================================================================
    #-------------------------------------------------
    # 読み込みデータの単語 → 整数型へのマッピング処理
    #-------------------------------------------------
    # key = counts.get（単語の出現回数）に従ってソート
    word_counts = sorted( counts, key=counts.get, reverse=True)
    print( "word_counts", word_counts[:5] )         # ['the', '.', ',', 'and', 'a']

    dict_word2int = { word: ii for (ii,word) in enumerate(word_counts,1) }

    mapped_reviews =[]  # レビュー文を（出現単語回数の）数値インデックス情報に変換したデータ
    for review in dfIMb["review"]:
        mapped_reviews.append( [ dict_word2int[word] for word in review.split() ] )

    print( "mapped_review", mapped_reviews[:10] )   # mapped_review [[15, 5646, 3, 1, 2160, 3977, 26959 ...
    
    #-------------------------------------------------
    # Zero padding
    #-------------------------------------------------
    sequence_length = 200   # シーケンス長（RNN の T に対応）
    sequences = np.zeros( 
                    shape=( len(mapped_reviews), sequence_length ), 
                    dtype=int 
                )
    
    for (i,row) in enumerate( mapped_reviews ):
        review_arr = np.array( row )
        sequences[i, -len(row):] = review_arr[-sequence_length:]     # 配列の右側から詰めていく
    

    #======================================================================
    # データセットをトレーニングデータ、テストデータ、検証データセットに分割
    #======================================================================
    X_train = sequences[:25000, :]
    y_train = dfIMb.loc[:25000, "sentiment"].values

    X_test = sequences[25000:, :]
    y_test = dfIMb.loc[25000:, "sentiment"].values

    #======================================================================
    # アルゴリズム（モデル）のパラメータを設定
    # Set algorithm parameters.
    # ex) learning_rate = 0.01  iterations = 1000
    #======================================================================
    n_words = max( list(dict_word2int.values()) ) + 1
    print( "n_words :", n_words )

    learning_rate1 = 0.001
    adam_beta1 = 0.9        # For the Adam optimizer
    adam_beta2 = 0.999      # For the Adam optimizer
    rnn = Many2OneMultiRNNLSTM(
              session = tf.Session(),
              #n_inputLayer = 1,
              #n_hiddenLayer = 128,                 # １つの LSTM ブロック中に集約されている隠れ層のノード数
              #n_outputLayer = 1,
              n_in_sequence_encoder = sequence_length,         # エンコーダー側のシーケンス長 / 
              n_vocab = n_words,                                   #
              epochs = 2000,
              batch_size = 100,
              eval_step = 1,
              save_step = 500
          )

    #======================================================================
    # 変数とプレースホルダを設定
    # Initialize variables and placeholders.
    # TensorFlow は, 損失関数を最小化するための最適化において,
    # 変数と重みベクトルを変更 or 調整する。
    # この変更や調整を実現するためには, 
    # "プレースホルダ [placeholder]" を通じてデータを供給（フィード）する必要がある。
    # そして, これらの変数とプレースホルダと型について初期化する必要がある。
    # ex) a_tsr = tf.constant(42)
    #     x_input_holder = tf.placeholder(tf.float32, [None, input_size])
    #     y_input_holder = tf.placeholder(tf.fload32, [None, num_classes])
    #======================================================================
    

    #======================================================================
    # モデルの構造を定義する。
    # Define the model structure.
    # ex) add_op = tf.add(tf.mul(x_input_holder, weight_matrix), b_matrix)
    #======================================================================
    rnn.model()

    #======================================================================
    # 損失関数を設定する。
    # Declare the loss functions.
    #======================================================================
    rnn.loss( SigmoidCrossEntropy() )

    #======================================================================
    # モデルの最適化アルゴリズム Optimizer を設定する。
    # Declare Optimizer.
    #======================================================================
    rnn.optimizer( Adam( learning_rate = learning_rate1, beta1 = adam_beta1, beta2 = adam_beta2 ) )

    #======================================================================
    # モデルの初期化と学習（トレーニング）
    # ここまでの準備で, 実際に, 計算グラフ（有向グラフ）のオブジェクトを作成し,
    # プレースホルダを通じて, データを計算グラフ（有向グラフ）に供給する。
    # Initialize and train the model.
    #
    # ex) 計算グラフを初期化する方法の１つの例
    #     with tf.Session( graph = graph ) as session:
    #         ...
    #         session.run(...)
    #         ...
    #     session = tf.Session( graph = graph )  
    #     session.run(…)
    #======================================================================
    # TensorBoard 用のファイル（フォルダ）を作成
    #rnn.write_tensorboard_graph()

    #rnn.fit( X_train, y_train )

    #======================================================================
    # モデルの評価
    # (Optional) Evaluate the model.
    #======================================================================
    #---------------------------------------------------------
    # 損失関数を plot
    #---------------------------------------------------------


    print("Finish main()")

    return


if __name__ == '__main__':
     main()

