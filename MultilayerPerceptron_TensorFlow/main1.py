# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境 (TensorFlow インストール済み)
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
from MLPlot import MLPlot
from MLPreProcess import MLPreProcess

from MultilayerPerceptron import MultilayerPerceptron

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
from NNOptimizer import Momentum
from NNOptimizer import NesterovMomentum
from NNOptimizer import Adagrad
from NNOptimizer import Adadelta


def main():
    """
    多層パーセプトロンを用いた、データの識別（２クラスの分類問題）
    """
    print("Enter main()")

    #======================================================================
    # データセットを読み込み or 生成
    # Import or generate data.
    #======================================================================
    X_features, y_labels = MLPreProcess.generateCirclesDataSet( input_n_samples = 300, input_noize = 0.1 )
    
    #======================================================================
    # データを変換、正規化
    # Transform and normalize data.
    # ex) data = tf.nn.batch_norm_with_global_normalization(...)
    #======================================================================
    #y_labels.reshape( 300, 1 )
    
    #======================================================================
    # データセットをトレーニングデータ、テストデータ、検証データセットに分割
    #======================================================================
    X_train, X_test, y_train, y_test \
    = MLPreProcess.dataTrainTestSplit( X_input = X_features, y_input = y_labels, ratio_test = 0.2, input_random_state = 1 )

    print( "X_train :\n", X_train )
    print( "y_train :\n", y_train )

    #======================================================================
    # アルゴリズム（モデル）のパラメータを設定
    # Set algorithm parameters.
    # ex) learning_rate = 0.01  iterations = 1000
    #======================================================================
    # 多層パーセプトロンクラスのオブジェクト生成
    mlp1 = MultilayerPerceptron(
               session = tf.Session(),
               n_inputLayer = len(X_features[0]), 
               n_hiddenLayers = [3],
               n_outputLayer = 1,
               activate_hiddenLayer = Sigmoid(),
               activate_outputLayer = Sigmoid(),
               epochs = 500,
               batch_size = 20
           )

    mlp2 = MultilayerPerceptron(
               session = tf.Session(),
               n_inputLayer = len(X_features[0]), 
               n_hiddenLayers = [3,3],
               n_outputLayer = 1,
               activate_hiddenLayer = Sigmoid(),
               activate_outputLayer = Sigmoid(),
               epochs = 500,
               batch_size = 20
           )
    
    #======================================================================
    # 変数とプレースホルダを設定
    # Initialize variables and placeholders.
    # TensorFlow は, 損失関数を最小化するための最適化において,
    # 変数と重みベクトルを変更 or 調整する。
    # この変更や調整を実現するためには, 
    # "プレースホルダ [placeholder]" を通じてデータを供給（フィード）する必要がある。
    # そして, これらの変数とプレースホルダと型について初期化する必要がある。
    # ex) a_var = tf.constant(42)
    #     x_input_holder = tf.placeholder(tf.float32, [None, input_size])
    #     y_input_holder = tf.placeholder(tf.fload32, [None, num_classes])
    #======================================================================


    #======================================================================
    # モデルの構造を定義する。
    # Define the model structure.
    # ex) add_op = tf.add(tf.mul(x_input_holder, weight_matrix), b_matrix)
    #======================================================================
    mlp1.model()
    mlp2.model()

    #======================================================================
    # 損失関数を設定する。
    # Declare the loss functions.
    #======================================================================
    mlp1.loss( BinaryCrossEntropy() )
    mlp2.loss( BinaryCrossEntropy() )

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
    # モデルの最適化アルゴリズムを設定
    mlp1.optimizer( GradientDecent( learning_rate = 0.05 ) )
    mlp2.optimizer( GradientDecent( learning_rate = 0.05 ) )

    # トレーニングデータで fitting 処理
    mlp1.fit( X_train, y_train )
    mlp2.fit( X_train, y_train )

    mlp1.print( "after fit()" )
    mlp2.print( "after fit()" )
    #print( mlp1._session.run( mlp1._weights[0] ) )

    #======================================================================
    # モデルの評価
    # (Optional) Evaluate the model.
    #======================================================================
    predict1 = mlp1.predict( X_test )
    print( "predict1 :\n", predict1 )

    # テストデータでの正解率
    accuracy1 = mlp1.accuracy( X_test, y_test )
    accuracy2 = mlp2.accuracy( X_test, y_test )

    print( "accuracy1 [test data] : ", accuracy1 )
    print( "accuracy2 [test data] : ", accuracy2 )
    
    # トレーニング回数に対する loss 値の plot
    plt.clf()
    plt.subplot( 1, 2, 1 )
    plt.plot(
        range( 0, 500 ), mlp1._losses_train,
        label = 'train data : MLP = 2-3-1',
        linestyle = '-',
        #linewidth = 2,
        color = 'red'
    )
    plt.title( "loss" )
    plt.legend( loc = 'best' )
    #plt.ylim( [0, 1.05] )
    plt.xlabel( "Epocs" )
    plt.tight_layout()

    plt.subplot( 1, 2, 2 )
    plt.plot(
        range( 0, 500 ), mlp2._losses_train,
        label = 'train data : MLP = 2-3-3-1',
        linestyle = '-',
        #linewidth = 2,
        color = 'red'
    )
    plt.title( "loss" )
    plt.legend( loc = 'best' )
    #plt.ylim( [0, 1.05] )
    plt.xlabel( "Epocs" )
    plt.tight_layout()
    
    MLPlot.saveFigure( fileName = "MultilayerPerceptron_1-1.png" )
    plt.show()
    

    # 識別結果＆境界の plot
    plt.clf()
    plt.subplot( 1, 2, 1 )
    MLPlot.drawDiscriminantRegions( X_features, y_labels, classifier = mlp1 )
    plt.title( "Mulutiplelayer Perceptron : 2-3-1" )
    plt.legend( loc = 'best' )

    plt.subplot( 1, 2, 2 )
    MLPlot.drawDiscriminantRegions( X_features, y_labels, classifier = mlp2 )
    plt.title( "Mulutiplelayer Perceptron : 2-3-3-1" )
    plt.legend( loc = 'best' )

    MLPlot.saveFigure( fileName = "MultilayerPerceptron_1-2.png" )
    plt.show()

    
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
