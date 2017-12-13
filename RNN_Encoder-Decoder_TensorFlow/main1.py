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
from RecurrectNNEncoderDecoderLSTM import RecurrectNNEncoderDecoderLSTM


def main():
    """
    TensorFlow を用いた RNN Encoder-Decoder（LSTM 使用） による簡単な質問応答（足し算）処理
    """
    print("Enter main()")

    # Reset graph
    #ops.reset_default_graph()

    #======================================================================
    # データセットを読み込み or 生成
    # Import or generate data.
    #======================================================================
    X_features, y_labels, dict_str_to_idx, dict_idx_to_str = MLPreProcess.generate_add_uint_operation_dataset( n_samples = 20000, digits = 3, seed = 12 )
    print( "X_features.shape :", X_features.shape )
    print( "y_labels :", y_labels.shape )
    print( "dict_str_to_idx :", dict_str_to_idx )
    print( "dict_idx_to_str :", dict_idx_to_str )

    #======================================================================
    # データを変換、正規化
    # Transform and normalize data.
    # ex) data = tf.nn.batch_norm_with_global_normalization(...)
    #======================================================================

    
    #======================================================================
    # データセットをトレーニングデータ、テストデータ、検証データセットに分割
    #======================================================================
    X_train, X_test, y_train, y_test \
    = MLPreProcess.dataTrainTestSplit( X_input = X_features, y_input = y_labels, ratio_test = 0.1, input_random_state = 1 )

    #======================================================================
    # アルゴリズム（モデル）のパラメータを設定
    # Set algorithm parameters.
    # ex) learning_rate = 0.01  iterations = 1000
    #======================================================================
    learning_rate1 = 0.001
    adam_beta1 = 0.9        # For the Adam optimizer
    adam_beta2 = 0.999      # For the Adam optimizer

    rnn1 = RecurrectNNEncoderDecoderLSTM(
               session = tf.Session(),
               n_inputLayer = 12,                   # 12 : "0123456789+ " の 12 文字
               n_hiddenLayer = 128,                 # rnn の cell 数と同じ
               n_outputLayer = 12,                  # 12 : "0123456789+ " の 12 文字
               n_in_sequence_encoder = 7,           # エンコーダー側のシーケンス長 / 足し算の式のシーケンス長 : "123 " "+" "456 " の計 4+1+4=7 文字
               n_in_sequence_decoder = 4,           # デコーダー側のシーケンス長 / 足し算の式の結果のシーケンス長 : "1000" 計 4 文字
               epochs = 20000,
               batch_size = 100,
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
    rnn1.print( "after model()" )

    #======================================================================
    # 損失関数を設定する。
    # Declare the loss functions.
    #======================================================================
    rnn1.loss( CrossEntropy() )

    #======================================================================
    # モデルの最適化アルゴリズム Optimizer を設定する。
    # Declare Optimizer.
    #======================================================================
    rnn1.optimizer( Adam( learning_rate = learning_rate1, beta1 = adam_beta1, beta2 = adam_beta2 ) )

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
    #rnn1.write_tensorboard_graph()

    # fitting 処理を行う
    rnn1.fit( X_train, y_train )
    #rnn1.print( "after fitting" )

    #======================================================================
    # モデルの評価
    # (Optional) Evaluate the model.
    #======================================================================
    #---------------------------------------------------------
    # 損失関数を plot
    #---------------------------------------------------------
    plt.clf()
    plt.plot(
        range( rnn1._epochs ), rnn1._losses_train,
        label = 'RNN Encoder-Decoder - %s = [%d - %d - %d], learning_rate = %0.3f' % ( type(rnn1) , rnn1._n_inputLayer, rnn1._n_hiddenLayer, rnn1._n_outputLayer, learning_rate1 ) ,
        linestyle = '-',
        linewidth = 0.1,
        color = 'red'
    )
    plt.title( "loss / cross-entropy" )
    plt.legend( loc = 'best' )
    plt.ylim( ymin = 0.0 )
    plt.xlabel( "Epocs" )
    plt.grid()
    plt.tight_layout()
    
    MLPlot.saveFigure( fileName = "RNN_Encoder-Decoder_1-1.png" )
    plt.show()
    
    #---------------------------------------------------------
    # 予想値
    #---------------------------------------------------------
    # 予想値を取得
    predicts1 = rnn1.predict( X_test )
    print( "predicts1 :", predicts1 )
    
    # 正解率を取得
    accuracy_total1 = rnn1.accuracy( X_features, y_labels )
    accuracy_train1 = rnn1.accuracy( X_train, y_train )
    accuracy_test1 = rnn1.accuracy( X_test, y_test )
    print( "accuracy_total1 : {} / n_sample : {}".format( accuracy_total1,  len(X_features[:,0,0]) ) )
    print( "accuracy_train1 : {} / n_sample : {}".format( accuracy_train1,  len(X_train[:,0,0]) ) )
    print( "accuracy_test1 : {} / n_sample : {}".format( accuracy_test1,  len(X_test[:,0,0]) ) )

    #---------------------------------------------------------
    # 質問＆応答処理
    #---------------------------------------------------------
    #print( "numpy.argmax( X_test[0,:,:], axis = -1 ) :", numpy.argmax( X_test[0,:,:], axis = -1 ) )
    # 質問文の数
    n_questions = min( 100, len(X_test[:,0,0]) )

    for q in range( n_questions ):
        answer = rnn1.question_answer_responce( question = X_test[q,:,:], dict_idx_to_str = dict_idx_to_str )
        
        # one-hot encoding → 対応する数値インデックス → 対応する文字に変換
        question = numpy.argmax( X_test[q,:,:], axis = -1 )
        question = "".join( dict_idx_to_str[i] for i in question )

        print( "-------------------------------" )
        print( "n_questions = {}".format( q ) )
        print( "Q : {}".format( question ) )
        print( "A : {}".format( answer ) )

        # 正解データ（教師データ）をone-hot encoding → 対応する数値インデックス → 対応する文字に変換
        target = numpy.argmax( y_test[q,:,:], axis = -1 )
        target = "".join( dict_idx_to_str[i] for i in target )

        if ( answer == target ):
            print( "T/F : T" )
        else:
            print( "T/F : F" )
        print( "-------------------------------" )


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
