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
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# 自作クラス
from MLPlot import MLPlot                               # 機械学習の plot 処理群を表すクラス
from MLPreProcess import MLPreProcess                   # 機械学習の前処理群を表すクラス

from MultilayerPerceptron import MultilayerPerceptron   # 多層パーセプトロン MLP を表すクラス
from ConvolutionalNN import ConvolutionalNN             # 畳み込みニューラルネットワーク CNN を表すクラス

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
    CNNを用いた、MNIST 画像データの識別
    """
    print("Enter main()")

    #======================================================================
    # データセットを読み込み or 生成
    # Import or generate data.
    #======================================================================
    #======================================================================
    # データセットをトレーニングデータ、テストデータ、検証データセットに分割
    #======================================================================
    # MNIST データが格納されているフォルダへのパス
    mnist_path = "D:\Data\MachineLearning_DataSet\MNIST"

    X_train, y_train = MLPreProcess.load_mnist( mnist_path, "train" )
    X_test, y_test = MLPreProcess.load_mnist( mnist_path, "t10k" )

    X_train = numpy.array( [numpy.reshape(x, (28,28)) for x in X_train] )
    X_test = numpy.array( [numpy.reshape(x, (28,28)) for x in X_test] )

    """
    # TensorFlow のサポート関数を使用して, MNIST データを読み込み
    mnist = read_data_sets( mnist_path )
    print( "mnist :\n", mnist )
    X_train = numpy.array( [numpy.reshape(x, (28,28)) for x in mnist.train.images] )
    X_test = numpy.array( [numpy.reshape(x, (28,28)) for x in mnist.test.images] )
    y_train = mnist.train.labels
    y_test = mnist.test.labels
    """

    print( "X_train.shape : ", X_train.shape )
    print( "y_train.shape : ", y_train.shape )
    print( "X_test.shape : ", X_test.shape )
    print( "y_test.shape : ", y_test.shape )

    print( "X_train : \n", X_train )
    print( "y_train : \n", y_train )
    
    #======================================================================
    # データを変換、正規化
    # Transform and normalize data.
    # ex) data = tf.nn.batch_norm_with_global_normalization(...)
    #======================================================================
    # One -hot encoding
    #y_train_encoded = numpy.eye(10)[ y_train.astype(int) ]
    #y_test_encoded = numpy.eye(10)[ y_test.astype(int) ]

    session = tf.Session()
    encode_holder = tf.placeholder(tf.int64, [None])
    y_oneHot_enoded_op = tf.one_hot( encode_holder, depth=10, dtype=tf.float32 ) # depth が 出力層のノード数に対応
    session.run( tf.global_variables_initializer() )
    y_train_encoded = session.run( y_oneHot_enoded_op, feed_dict = { encode_holder: y_train } )
    y_test_encoded = session.run( y_oneHot_enoded_op, feed_dict = { encode_holder: y_test } )
    print( "y_train_encoded.shape : ", y_train_encoded.shape )
    print( "y_train_encoded.dtype : ", y_train_encoded.dtype )
    print( "y_test_encoded.shape : ", y_test_encoded.shape )

    #======================================================================
    # アルゴリズム（モデル）のパラメータを設定
    # Set algorithm parameters.
    # ex) learning_rate = 0.01  iterations = 1000
    #======================================================================
    # CNN クラスのオブジェクト生成
    cnn1 = ConvolutionalNN(
               session = tf.Session( config = tf.ConfigProto(log_device_placement=True) ),
               epochs = 500,
               batch_size = 100,
               eval_step = 1,
               image_height = 28,                   # 28 pixel
               image_width = 28,                    # 28 pixel
               n_channels = 1,                      # グレースケール
               n_ConvLayer_featuresMap = [25, 50],  # conv1 : 25 枚, conv2 : 50 枚
               n_ConvLayer_kernels = [4, 4],        # conv1 : 4*4, conv2 : 4*4
               n_strides = 1,
               n_pool_wndsize = 2,
               n_pool_strides = 2,
               n_fullyLayers = 100,
               n_labels = 10
           )

    #cnn1._t_holder = tf.placeholder( tf.int32, [None] )

    cnn2 = ConvolutionalNN(
               session = tf.Session( config = tf.ConfigProto(log_device_placement=True) ),
               epochs = 500,
               batch_size = 100,
               eval_step = 1,
               image_height = 28,                   # 28 pixel
               image_width = 28,                    # 28 pixel
               n_channels = 1,                      # グレースケール
               n_ConvLayer_featuresMap = [25, 50],  # conv1 : 25 枚, conv2 : 50 枚
               n_ConvLayer_kernels = [4, 4],        # conv1 : 4*4, conv2 : 4*4
               n_strides = 1,
               n_pool_wndsize = 2,
               n_pool_strides = 2,
               n_fullyLayers = 100,
               n_labels = 10
           )

    #cnn2._t_holder = tf.placeholder( tf.int32, [None] )

    #cnn1.print( "after __init__()" )

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
    cnn1.model()
    cnn2.model()
    cnn1.print( "after model()" )

    #======================================================================
    # 損失関数を設定する。
    # Declare the loss functions.
    #======================================================================
    cnn1.loss( SoftmaxCrossEntropy() )
    cnn2.loss( SoftmaxCrossEntropy() )

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
    cnn1.optimizer( Momentum( learning_rate = 0.0001, momentum = 0.9 ) )
    cnn2.optimizer( Momentum( learning_rate = 0.0005, momentum = 0.9 ) )

    # トレーニングデータで fitting 処理
    cnn1.fit( X_train, y_train_encoded )
    cnn2.fit( X_train, y_train_encoded )

    cnn1.print( "after fit()" )
    #print( mlp1._session.run( mlp1._weights[0] ) )

    #======================================================================
    # モデルの評価
    # (Optional) Evaluate the model.
    #======================================================================
    #-------------------------------------------------------------------
    # トレーニング回数に対する loss 値の plot
    #-------------------------------------------------------------------
    plt.clf()
    plt.plot(
        range( 0, 500 ), cnn1._losses_train,
        label = 'train data : CNN1 = [25,50,100], learning_rate = 0.0001',
        linestyle = '-',
        #linewidth = 2,
        color = 'red'
    )
    plt.plot(
        range( 0, 500 ), cnn2._losses_train,
        label = 'train data : CNN2 = [25,50,100], learning_rate = 0.0005',
        linestyle = '--',
        #linewidth = 2,
        color = 'blue'
    )
    plt.title( "loss" )
    plt.legend( loc = 'best' )
    #plt.ylim( [0, 1.05] )
    plt.xlabel( "Epocs" )
    plt.tight_layout()
   
    MLPlot.saveFigure( fileName = "CNN_1-1.png" )
    plt.show()

    #--------------------------------------------------------------------
    # テストデータでの正解率
    #--------------------------------------------------------------------
    accuracy1 = cnn1.accuracy( X_test, y_test )
    accuracy2 = cnn2.accuracy( X_test, y_test )
    print( "accuracy1 [test data] : %0.3f" % accuracy1 )
    print( "accuracy2 [test data] : %0.3f" % accuracy2 )

    print( "accuracy1 labels [test data]" )
    accuracys1 = cnn1.accuracy_labels( X_test, y_test )
    for i in range( len(accuracys1) ):
        print( "label %d : %.3f" % ( i, accuracys1[i] ) )

    print( "accuracy2 labels [test data]" )
    accuracys2 = cnn2.accuracy_labels( X_test, y_test )
    for i in range( len(accuracys2) ):
        print( "label %d : %.3f" % ( i, accuracys2[i] ) )

    #-------------------------------------------------------------------
    # 正解画像＆誤識別画像の plot
    #-------------------------------------------------------------------
    predict1 = cnn1.predict( X_test )
    predict2 = cnn2.predict( X_test )
    print( "predict1 : ", predict1 )
    print( "predict2 : ", predict2 )

    # 正解・不正解のリスト [True or False]
    corrects1 = numpy.equal( predict1, y_test )
    corrects2 = numpy.equal( predict2, y_test )
    print( "corrects1 :", corrects1 )
    print( "corrects2 :", corrects2 )

    figure1, axis1 = plt.subplots( 
                        nrows = 5, ncols = 8,
                        sharex = True, sharey = True     # x,y 軸をシャアする
                     )
    
    # ２次元配列を１次元に変換
    axis1 = axis1.flatten()

    # 正解画像の plot のための loop
    #plt.clf()
    for (idx, image) in enumerate( X_test[ corrects1 ][0:40] ):
        #print( "idx", idx )
        image = image.reshape(28,28)        # １次元配列を shape = [28 ,28] に reshape
        axis1[idx].imshow(
            image,
            cmap = "Greys",
            interpolation = "nearest"   # 補間方法
        )
        axis1[idx].set_title( "Actual: " + str( y_test[corrects1][idx] ) + " Pred: " + str( predict1[corrects1][idx] ), fontsize = 8 )

    axis1[0].set_xticks( [] )
    axis1[0].set_yticks( [] )
    #plt.tight_layout()
    MLPlot.saveFigure( fileName = "CNN_1-2.png" )
    plt.show()
    

    # 誤識別画像の plot のための loop
    figure2, axis2 = plt.subplots( 
                        nrows = 5, ncols = 8,
                        sharex = True, sharey = True     # x,y 軸をシャアする
                     )

    # ２次元配列を１次元に変換
    axis2 = axis2.flatten()

    for (idx, image) in enumerate( X_test[ ~corrects1 ][0:40] ):
        image = image.reshape(28,28)        # １次元配列を shape = [28 ,28] に reshape
        axis2[idx].imshow(
            image,
            cmap = "Greys",
            interpolation = "nearest"   # 補間方法
        )
        axis2[idx].set_title( "Actual: " + str( y_test[~corrects1][idx] ) + " Pred: " + str( predict1[~corrects1][idx] ), fontsize = 8 )

    axis2[0].set_xticks( [] )
    axis2[0].set_yticks( [] )
    #plt.tight_layout()
    MLPlot.saveFigure( fileName = "CNN_1-3.png" )
    plt.show()

    #---------------------------------------------------------------------
    # MNIST 画像を plot
    #---------------------------------------------------------------------
    # 先頭の 0~9 のラベルの画像データを plot
    """
    # plt.subplots(...) から,
    # Figure クラスのオブジェクト、Axis クラスのオブジェクト作成
    figure, axis = plt.subplots( 
                       nrows = 2, ncols = 5,
                       sharex = True, sharey = True     # x,y 軸をシャアする
                   )
    # 2 × 5 配列を１次元に変換
    axis = axis.flatten()
    # 数字の 0~9 の plot 用の for ループ
    for i in range(10):
        image = X_train[y_train == i][0]    #
        image = image.reshape(28,28)        # １次元配列を shape = [28 ,28] に reshape
        axis[i].imshow(
            image,
            cmap = "Greys",
            interpolation = "nearest"   # 補間方法
        )
    axis[0].set_xticks( [] )
    axis[0].set_yticks( [] )
    plt.tight_layout()
    MLPlot.saveFigure( fileName = "MultilayerPerceptron_3-1.png" )
    plt.show()
    """

    """
    # 特定のラベルの 25 枚の画像データを plot
    figure, axis = plt.subplots( nrows = 5, ncols = 5, sharex = True, sharey = True )
    axis = axis.flatten()
    for i in range(25):
        image = X_train[y_train == 7][i].reshape(28,28)    
        axis[i].imshow( image, cmap = "Greys", interpolation = "nearest" )
    
    axis[0].set_xticks( [] )
    axis[0].set_yticks( [] )
    plt.tight_layout()
    MLPlot.saveFigure( fileName = "MultilayerPerceptron_3-2.png" )
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