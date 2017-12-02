# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境 (TensorFlow インストール済み)
#     <Anaconda Prompt>
#     conda create -n tensorflow python=3.5
#     activate tensorflow
#     pip install --ignore-installed --upgrade tensorflow
#     pip install --ignore-installed --upgrade tensorflow-gpu

import numpy
import pandas
import matplotlib.pyplot as plt

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
from RecurrectNNLanguageModel import RecurrectNNLanguageModel


def main():
    """
    TensorFlow を用いた RNN によるテキストデータからのスパムの確率予想処理
    """
    print("Enter main()")

    # Reset graph
    #ops.reset_default_graph()

    # Session の設定
    #session = tf.Session()

    #======================================================================
    # データセットを読み込み or 生成
    # Import or generate data.
    #======================================================================
    # SMS Spam Collection データセットのファイルへのパス
    spam_datset_path = "C:\Data\MachineLearning_DataSet\SMS_Spam_Collection\smsspamcollection\SMSSpamCollection.txt"

    # SMS Spam Collection データセットからテキストデータを読み込み、取得する。
    text_data_features, text_data_labels = MLPreProcess.load_sms_spam_collection( path = spam_datset_path, bCleaning = True )
    print( "len( text_data_features ) :", len( text_data_features ) )   # len( text_data_features ) : 5573
    print( "len( text_data_labels ) :", len( text_data_labels ) )       # len( text_data_labels ) : 5573

    #======================================================================
    # データを変換、正規化
    # Transform and normalize data.
    # ex) data = tf.nn.batch_norm_with_global_normalization(...)
    #======================================================================
    # テキスト情報を数値インデックスのリストに変換する。
    X_features, n_vocab = MLPreProcess.text_vocabulary_processing( text_data = text_data_features, n_max_in_sequence = 25, min_word_freq = 10 )
    y_labels = numpy.array( [1 if label_str=='ham' else 0 for label_str in text_data_labels] )

    print( "X_features.shape :", X_features.shape )     # X_features.shape : (5573, 25)
    print( "y_labels.shape :", y_labels.shape )         # y_labels.shape : (5573,)
    print( "n_vocab", n_vocab )                         # n_vocab 934

    # データをシャッフルする。
    shuffled_idx = numpy.random.permutation( numpy.arange( len(y_labels) ) )
    X_features_shuffled = X_features[ shuffled_idx ]
    y_labels_shuffled = y_labels[ shuffled_idx ]
    
    print( "X_features_shuffled.shape :", X_features_shuffled.shape )   # X_features_shuffled.shape : (5573, 25)
    print( "y_labels_shuffled.shape :", y_labels_shuffled.shape )       # y_labels_shuffled.shape : (5573,)

    #======================================================================
    # データセットをトレーニングデータ、テストデータ、検証データセットに分割
    #======================================================================
    X_train, X_test, y_train, y_test \
    = MLPreProcess.dataTrainTestSplit( X_input = X_features, y_input = y_labels, ratio_test = 0.2, input_random_state = 1 )

    print( "X_train.shape :", X_train.shape )   # X_train.shape : (4458, 25)
    print( "y_train.shape :", y_train.shape )   # y_train.shape : (4458,)
    print( "X_train :", X_train )               # [[ 12 324  43 ...,  49  11   0] [469 851 418 ...,   0   0   0] ... [ 11  93 440 ...,   0   0   0]]
    print( "y_train :", y_train )               # [1 0 1 ..., 1 0 1]

    #======================================================================
    # アルゴリズム（モデル）のパラメータを設定
    # Set algorithm parameters.
    # ex) learning_rate = 0.01  iterations = 1000
    #======================================================================
    learning_rate1 = 0.0005
    learning_rate2 = 0.0005
    adam_beta1 = 0.9        # For the Adam optimizer
    adam_beta2 = 0.999      # For the Adam optimizer

    rnn1 = RecurrectNNLanguageModel(
               session = tf.Session( config = tf.ConfigProto(log_device_placement=True) ),
               n_inputLayer = 1,
               n_hiddenLayer = 10,
               n_outputLayer = 1,
               n_in_sequence = 25,
               n_vocab = n_vocab,           # 934
               n_in_embedding_vec = 50,
               epochs = 500,
               batch_size = 10,
               eval_step = 1
           )

    rnn2 = RecurrectNNLanguageModel(
               session = tf.Session( config = tf.ConfigProto(log_device_placement=True) ),
               n_inputLayer = 1,
               n_hiddenLayer = 30,
               n_outputLayer = 1,
               n_in_sequence = 25,
               n_vocab = n_vocab,           # 934
               n_in_embedding_vec = 50,
               epochs = 500,
               batch_size = 10,
               eval_step = 1
           )

    rnn1.print( "after __init__()" )

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
    rnn1.model()
    #rnn2.model()

    #======================================================================
    # 損失関数を設定する。
    # Declare the loss functions.
    #======================================================================
    rnn1.loss( SparseSoftmaxCrossEntropy() )
    #rnn2.loss( SparseSoftmaxCrossEntropy() )

    #======================================================================
    # モデルの最適化アルゴリズム Optimizer を設定する。
    # Declare Optimizer.
    #======================================================================
    rnn1.optimizer( Adam( learning_rate = learning_rate1, beta1 = adam_beta1, beta2 = adam_beta2 ) )
    #rnn2.optimizer( Adam( learning_rate = learning_rate2, beta1 = adam_beta1, beta2 = adam_beta2 ) )

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
    #rnn1.fit( X_train, y_train )
    #rnn2.fit( X_train, y_train )

    #rnn1.print( "after fitting" )

    #======================================================================
    # モデルの評価
    # (Optional) Evaluate the model.
    #======================================================================
    # TensorBoard 用のファイル（フォルダ）を作成
    #rnn1.write_tensorboard_graph()

    # 時系列データの予想値を取得
    #predicts1 = rnn1.predict( X_features )
    #predicts2 = rnn2.predict( X_features )

    #print( "predicts1 :\n", predicts1 )

    #---------------------------------------------------------
    # 損失関数を plot
    #---------------------------------------------------------
    """
    plt.clf()

    plt.plot(
        range( rnn1._epochs ), rnn1._losses_train,
        label = 'RNN1 = [%d - %d - %d], learning_rate = %0.3f' % ( rnn1._n_inputLayer, rnn1._n_hiddenLayer, rnn1._n_outputLayer, learning_rate1 ) ,
        linestyle = '-',
        #linewidth = 2,
        color = 'red'
    )
    """
    """
    plt.plot(
        range( rnn2._epochs ), rnn2._losses_train,
        label = 'RNN2 = [%d - %d - %d], learning_rate = %0.3f' % ( rnn2._n_inputLayer, rnn2._n_hiddenLayer, rnn2._n_outputLayer, learning_rate2 ) ,
        linestyle = '--',
        #linewidth = 2,
        color = 'glue'
    )
    """
    """
    plt.title( "loss / L2 Norm (MSE)" )
    plt.legend( loc = 'best' )
    #plt.ylim( [0, 1.05] )
    plt.xlabel( "Epocs" )
    plt.grid()
    plt.tight_layout()
    
    plt.savefig("RNN_1-2.png", dpi = 300, bbox_inches = "tight" )
    #MLPlot.saveFigure( fileName = "RNN_1-1.png" )
    plt.show()
    """
    
    #======================================================================
    # ハイパーパラメータのチューニング (Optional)
    #======================================================================


    #======================================================================
    # デプロイと新しい成果指標の予想 (Optional)
    #======================================================================


    print("Finish main()")
    return
    

if __name__ == '__main__':
     main()
