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
from RecurrentNN import RecurrentNN
from RecurrentNNLSTM import RecurrentNNLSTM


def main():
    """
    TensorFlow を用いた LSTM による Adding Problem 対する長期予想性とその評価処理
    """
    print("Enter main()")

    # Reset graph
    #ops.reset_default_graph()

    #======================================================================
    # データセットを読み込み or 生成
    # Import or generate data.
    #======================================================================
    X_features, y_labels = MLPreProcess.generate_adding_problem( t = 250, n_sequence = 10000, seed = 12 )
    print( "X_features.shape :", X_features.shape )     # X_features.shape : (10000, 250, 2)
    print( "y_labels.shape :", y_labels.shape )         # y_labels.shape : (10000, 1)
    
    #---------------------------------------------------------
    # Adding Problem 波形を plot
    #---------------------------------------------------------
    """
    plt.clf()
    
    plt.subplot( 2, 2, 1 )
    plt.scatter(
        range( len(X_features[0,:,0]) ), X_features[0,:,0],
        label = 'signals / u(0,1)',
        marker = 'o',
        s = 5,
        color = 'black'
    )
    plt.scatter(
        range( len(X_features[0,:,1]) ), X_features[0,:,1],
        label = 'masks / {0 or 1}, Σ_t mask = 2',
        marker = 'x',
        s = 5,
        color = 'red'
    )
    plt.plot(
        range( len(X_features[0,:,1]) ), X_features[0,:,0] * X_features[0,:,1],
        label = 'adding_data',
        linestyle = '-',
        #linewidth = 2,
        color = 'red'
    )
    plt.title( "Adding Problem data / n_sequence = 1" )
    plt.legend( loc = 'best' )
    plt.ylim( [0, 1.05] )
    plt.xlabel( "t" )
    plt.grid()
    plt.tight_layout()
    
    plt.subplot( 2, 2, 2 )
    plt.scatter(
        range( len(X_features[1,:,0]) ), X_features[1,:,0],
        label = 'signals / u(0,1)',
        marker = 'o',
        s = 5,
        color = 'black'
    )
    plt.scatter(
        range( len(X_features[1,:,1]) ), X_features[1,:,1],
        label = 'masks / {0 or 1}, Σ_t mask = 2',
        marker = 'x',
        s = 5,
        color = 'red'
    )
    plt.plot(
        range( len(X_features[1,:,1]) ), X_features[1,:,0] * X_features[1,:,1],
        label = 'adding_data',
        linestyle = '-',
        #linewidth = 2,
        color = 'red'
    )
    plt.title( "Adding Problem data / n_sequence = 2" )
    plt.legend( loc = 'best' )
    plt.ylim( [0, 1.05] )
    plt.xlabel( "t" )
    plt.grid()
    plt.tight_layout()

    plt.subplot( 2, 2, 3 )
    plt.scatter(
        range( len(X_features[1,:,0]) ), X_features[-2,:,0],
        label = 'signals / u(0,1)',
        marker = 'o',
        s = 5,
        color = 'black'
    )
    plt.scatter(
        range( len(X_features[1,:,1]) ), X_features[-2,:,1],
        label = 'masks / {0 or 1}, Σ_t mask = 2',
        marker = 'x',
        s = 5,
        color = 'red'
    )
    plt.plot(
        range( len(X_features[1,:,1]) ), X_features[-2,:,0] * X_features[-2,:,1],
        label = 'adding_data',
        linestyle = '-',
        #linewidth = 2,
        color = 'red'
    )
    plt.title( "Adding Problem data / n_sequence = 9999" )
    plt.legend( loc = 'best' )
    plt.ylim( [0, 1.05] )
    plt.xlabel( "t" )
    plt.grid()
    plt.tight_layout()

    plt.subplot( 2, 2, 4 )
    plt.scatter(
        range( len(X_features[1,:,0]) ), X_features[-1,:,0],
        label = 'signals / u(0,1)',
        marker = 'o',
        s = 5,
        color = 'black'
    )
    plt.scatter(
        range( len(X_features[1,:,1]) ), X_features[-1,:,1],
        label = 'masks / {0 or 1}, Σ_t mask = 2',
        marker = 'x',
        s = 5,
        color = 'red'
    )
    plt.plot(
        range( len(X_features[1,:,1]) ), X_features[-1,:,0] * X_features[-1,:,1],
        label = 'adding_data',
        linestyle = '-',
        #linewidth = 2,
        color = 'red'
    )
    plt.title( "Adding Problem data / n_sequence = 10000" )
    plt.legend( loc = 'best' )
    plt.ylim( [0, 1.05] )
    plt.xlabel( "t" )
    plt.grid()
    plt.tight_layout()
    
    MLPlot.saveFigure( fileName = "RNN-LSTM_2-2.png" )
    plt.show()
    """

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

    print( "X_train.shape :", X_train.shape )
    print( "y_train.shape :", y_train.shape )

    #======================================================================
    # アルゴリズム（モデル）のパラメータを設定
    # Set algorithm parameters.
    # ex) learning_rate = 0.01  iterations = 1000
    #======================================================================
    learning_rate1 = 0.001
    adam_beta1 = 0.9        # For the Adam optimizer
    adam_beta2 = 0.999      # For the Adam optimizer

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

    """
    rnn2 = RecurrentNN(
               session = tf.Session( config = tf.ConfigProto(log_device_placement=True) ),
               n_inputLayer = len( X_features[0][0] ),
               n_hiddenLayer = 100,
               n_outputLayer = len( y_labels[0] ),
               n_in_sequence = X_features.shape[1],
               epochs = 500,
               batch_size = 10,
               eval_step = 1
           )
    """

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
    #rnn1.print( "after model()" )

    #======================================================================
    # 損失関数を設定する。
    # Declare the loss functions.
    #======================================================================
    rnn1.loss( L2Norm() )
    #rnn2.loss( L2Norm() )

    #======================================================================
    # モデルの最適化アルゴリズム Optimizer を設定する。
    # Declare Optimizer.
    #======================================================================
    rnn1.optimizer( Adam( learning_rate = learning_rate1, beta1 = adam_beta1, beta2 = adam_beta2 ) )
    #rnn2.optimizer( Adam( learning_rate = learning_rate1, beta1 = adam_beta1, beta2 = adam_beta2 ) )

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
    #rnn2.write_tensorboard_graph()

    # fitting 処理を行う
    rnn1.fit( X_train, y_train )
    #rnn2.fit( X_train, y_train )
    rnn1.print( "after fitting" )

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
        label = 'RNN - %s = [%d - %d - %d], learning_rate = %0.3f' % ( type(rnn1) , rnn1._n_inputLayer, rnn1._n_hiddenLayer, rnn1._n_outputLayer, learning_rate1 ) ,
        linestyle = '-',
        linewidth = 1,
        color = 'red'
    )
    """
    plt.plot(
        range( rnn2._epochs ), rnn2._losses_train,
        label = 'RNN1 = [%d - %d - %d], learning_rate = %0.3f' % ( rnn2._n_inputLayer, rnn2._n_hiddenLayer, rnn2._n_outputLayer, learning_rate1 ) ,
        linestyle = '--',
        linewidth = 1,
        color = 'blue'
    )
    """
    plt.title( "loss / L2 Norm (MSE)" )
    plt.legend( loc = 'best' )
    plt.ylim( ymin = 0.0 )
    plt.xlabel( "Epocs" )
    plt.grid()
    plt.tight_layout()
    
    MLPlot.saveFigure( fileName = "RNN-LSTM_2-2.png" )
    plt.show()

    #---------------------------------------------------------
    # 時系列データの予想値と元の Adding Problem 波形を plot
    #---------------------------------------------------------
    # 時系列データの予想値を取得
    #predicts1 = rnn1.predict( X_features )
    #print( "predicts1 :\n", predicts1 )

    """
    plt.clf()

    x_dat_with_noize, y_dat_with_noize = MLPreProcess.generate_sin_with_noize( t = times, T = T1, noize_size = noize_size1, seed = 12 )
    plt.plot(
        x_dat_with_noize, y_dat_with_noize,
        label = 'noize = %0.3f' % noize_size1,
        linestyle = '-',
        #linewidth = 2,
        color = 'black'
    )

    x_dat_without_noize, y_dat_without_noize = MLPreProcess.generate_sin_with_noize( t = times, T = T1, noize_size = 0, seed = 12 )
    plt.plot(
        x_dat_without_noize, y_dat_without_noize,
        label = 'without noize',
        linestyle = '--',
        #linewidth = 2,
        color = 'black'
    )

    plt.plot(
        x_dat, predicts1,
        label = 'predict1 : RNN-LSTM1 = [%d - %d - %d], learning_rate = %0.3f' % ( rnn1._n_inputLayer, rnn1._n_hiddenLayer, rnn1._n_outputLayer, learning_rate1 ),
        linestyle = '-',
        #linewidth = 2,
        color = 'red'
    )

    plt.title( "time series / Adding Problem" )
    plt.legend( loc = 'best' )
    plt.ylim( [-1.10, 1.10] )
    plt.xlabel( "t [time]" )
    plt.grid()
    plt.tight_layout()
   
    MLPlot.saveFigure( fileName = "RNN-LSTM_2-3.png" )
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
