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
from NNOptimizer import GradientDecentDecay
from NNOptimizer import Momentum
from NNOptimizer import NesterovMomentum
from NNOptimizer import Adagrad
from NNOptimizer import Adadelta


def main():
    """
    CNNを用いた、CIFAR-10 画像データの識別
    """
    print("Enter main()")

    #======================================================================
    # データセットを読み込み or 生成
    # Import or generate data.
    #======================================================================
    #======================================================================
    # データセットをトレーニングデータ、テストデータ、検証データセットに分割
    #======================================================================
    # CIFAR-10 のラベル値とカテゴリーのディクショナリ
    cifar10_labels_dict = {
        0 : "airplane",
        1 : "automoblie",
        2 : "bird",
        3 : "cat",
        4 : "deer",
        5 : "dog",
        6 : "frog",
        7 : "horse",
        8 : "ship",
        9 : "truck",
    }
    
    # CIFAR-10 データが格納されているフォルダへのパス
    cifar10_path = "D:\Data\MachineLearning_DataSet\CIFAR\cifar-10-batches-bin"

    X_train, y_train = MLPreProcess.load_cifar10_train( cifar10_path, fileName = "data_batch_1.bin" )
    X_test, y_test = MLPreProcess.load_cifar10_test( cifar10_path )
    
    # [n_channel, image_height, image_width] = [3,32,32] に reshape
    X_train = numpy.array( [numpy.reshape(x, (3,32,32)) for x in X_train] )
    X_test = numpy.array( [numpy.reshape(x, (3,32,32)) for x in X_test] )
    y_train = numpy.reshape( y_train, 10000 )
    y_test = numpy.reshape( y_test, 10000 )

    # imshow(), fit()で読める ([1]height, [2]width, [0] channel) の順番に変更するために
    # numpy の transpose() を使って次元を入れ替え
    X_train = numpy.array( [ numpy.transpose( x, (1, 2, 0) ) for x in X_train] )             
    X_test = numpy.array( [ numpy.transpose( x, (1, 2, 0) ) for x in X_test] )
    
    print( "X_train.shape : ", X_train.shape )
    print( "y_train.shape : ", y_train.shape )
    print( "X_test.shape : ", X_test.shape )
    print( "y_test.shape : ", y_test.shape )

    print( "X_train[0] : \n", X_train[0] )
    print( "y_train[0] : \n", y_train[0] )
    print( "[y_train == 0] : \n", [ y_train == 0 ] )
    
    #---------------------------------------------------------------------
    # CIFAR-10 画像を plot
    #---------------------------------------------------------------------
    # 先頭の 0~9 のラベルの画像データを plot
    # plt.subplots(...) から,
    # Figure クラスのオブジェクト、Axis クラスのオブジェクト作成
    figure, axis = plt.subplots( 
                       nrows = 8, ncols = 8,
                       sharex = True, sharey = True     # x,y 軸をシャアする
                   )
    # 2 dim 配列を１次元に変換
    axis = axis.flatten()

    # ラベルの 0~9 の plot 用の for ループ
    for i in range(64):
        image = X_train[i]
        axis[i].imshow( image )
        axis[i].set_title( "Actual: " + cifar10_labels_dict[ y_train[i] ], fontsize = 8 )

    axis[0].set_xticks( [] )
    axis[0].set_yticks( [] )
    plt.tight_layout()
    MLPlot.saveFigure( fileName = "CNN_2-1.png" )
    plt.show()

    # 特定のラベルの画像データを plot
    figure, axis = plt.subplots( nrows = 8, ncols = 8, sharex = True, sharey = True )
    axis = axis.flatten()
    for i in range(64):
        image = X_train[y_train == 0][i]
        axis[i].imshow( image )
        axis[i].set_title( "Actual: " + cifar10_labels_dict[ y_train[0] ], fontsize = 8 )

    axis[0].set_xticks( [] )
    axis[0].set_yticks( [] )
    plt.tight_layout()
    MLPlot.saveFigure( fileName = "CNN_2-2.png" )
    plt.show()

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
               image_height = 32,                   # 32 pixel
               image_width = 32,                    # 32 pixel
               n_channels = 3,                      # RGB の 3 チャンネル
               n_ConvLayer_featuresMap = [64, 64],  # conv1 : 64*64, conv2 : 64*64
               n_ConvLayer_kernels = [5, 5],        # conv1 : 5*5, conv2 : 5*5
               n_strides = 1,
               n_pool_wndsize = 3,
               n_pool_strides = 2,
               n_fullyLayers = 384,
               n_labels = 10
           )

    # 入力画像データの [n_channels, image_height, image_width] の shape に対応するように placeholder の形状を reshape
    #cnn1._X_holder = tf.placeholder( tf.float32, shape = [ None, 3, 32, 32 ] )

    cnn2 = ConvolutionalNN(
               session = tf.Session( config = tf.ConfigProto(log_device_placement=True) ),
               epochs = 500,
               batch_size = 100,
               eval_step = 1,
               image_height = 32,                   # 32 pixel
               image_width = 32,                    # 32 pixel
               n_channels = 3,                      # RGB の 3 チャンネル
               n_ConvLayer_featuresMap = [64, 64],  # conv1 : 64*64, conv2 : 64*64
               n_ConvLayer_kernels = [5, 5],        # conv1 : 5*5, conv2 : 5*5
               n_strides = 1,
               n_pool_wndsize = 3,
               n_pool_strides = 2,
               n_fullyLayers = 384,
               n_labels = 10
           )

    # 入力画像データの [n_channels, image_height, image_width] の shape に対応するように placeholder の形状を reshape
    #cnn2._X_holder = tf.placeholder( tf.float32, shape = [ None, 3, 32, 32 ] )

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
    #cnn1.print( "after model()" )

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
    learning_rate1 = 0.0001
    learning_rate2 = 0.0005
    cnn1.optimizer( GradientDecent( learning_rate = learning_rate1 ) )
    cnn2.optimizer( GradientDecent( learning_rate = learning_rate2 ) )
    #cnn1.optimizer( GradientDecentDecay( learning_rate = learning_rate1, n_generation = 500, n_gen_to_wait = 5, lr_recay = 0.1 ) )
    #cnn2.optimizer( GradientDecentDecay( learning_rate = learning_rate2, n_generation = 500, n_gen_to_wait = 5, lr_recay = 0.1 ) )

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
        label = 'train data : CNN1 = [50 - 50 - 384], learning_rate = %0.4f' % learning_rate1,
        linestyle = '-',
        #linewidth = 2,
        color = 'red'
    )
    plt.plot(
        range( 0, 500 ), cnn2._losses_train,
        label = 'train data : CNN1 = [50 - 50 - 384], learning_rate = %0.4f' % learning_rate2,
        linestyle = '--',
        #linewidth = 2,
        color = 'blue'
    )
    plt.title( "loss" )
    plt.legend( loc = 'best' )
    #plt.ylim( [0, 1.05] )
    plt.xlabel( "Epocs" )
    plt.tight_layout()
   
    MLPlot.saveFigure( fileName = "CNN_2-3.png" )
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
    print( "corrects1 : ", corrects1 )
    print( "corrects2 : ", corrects2 )


    figure, axis = plt.subplots( 
                        nrows = 5, ncols = 8,
                        sharex = True, sharey = True     # x,y 軸をシャアする
                     )
    
    # ２次元配列を１次元に変換
    axis = axis.flatten()

    # 正解画像の plot のための loop
    #plt.clf()
    for (idx, image) in enumerate( X_test[ corrects1 ][0:40] ):
        #print( "idx", idx )
        image = image.reshape(32,32,3)        # １次元配列を shape = [32, 32, 3] に reshape
        axis[idx].imshow( image )
        axis[idx].set_title( 
            "Actual: " + cifar10_labels_dict[ y_test[corrects1][idx] ] + " / " +
            "Pred: " + cifar10_labels_dict[ predict1[corrects1][idx] ], 
            fontsize = 8 
        )

    axis[0].set_xticks( [] )
    axis[0].set_yticks( [] )
    #plt.tight_layout()
    MLPlot.saveFigure( fileName = "CNN_2-4.png" )
    plt.show()
    

    # 誤識別画像の plot のための loop
    figure, axis = plt.subplots( 
                        nrows = 5, ncols = 8,
                        sharex = True, sharey = True     # x,y 軸をシャアする
                     )

    # ２次元配列を１次元に変換
    axis = axis.flatten()

    for (idx, image) in enumerate( X_test[ ~corrects1 ][0:40] ):
        image = image.reshape(32,32,3)        # １次元配列を shape = [32, 32 ,3] に reshape
        axis[idx].imshow( image )
        axis[idx].set_title( 
            "Actual: " + cifar10_labels_dict[ y_test[~corrects1][idx] ] + " / " +
            "Pred: " + cifar10_labels_dict[ predict1[~corrects1][idx] ], 
            fontsize = 8 
        )

    axis[0].set_xticks( [] )
    axis[0].set_yticks( [] )
    #plt.tight_layout()
    MLPlot.saveFigure( fileName = "CNN_2-5.png" )
    plt.show()
    
    #
    figure, axis = plt.subplots( 
                        nrows = 5, ncols = 8,
                        sharex = True, sharey = True     # x,y 軸をシャアする
                     )
    
    # ２次元配列を１次元に変換
    axis = axis.flatten()

    # 正解画像の plot のための loop
    #plt.clf()
    for (idx, image) in enumerate( X_test[ corrects2 ][0:40] ):
        #print( "idx", idx )
        image = image.reshape(32,32,3)        # １次元配列を shape = [32 ,32, 3] に reshape
        axis[idx].imshow( image )
        axis[idx].set_title( 
            "Actual: " + cifar10_labels_dict[ y_test[corrects2][idx] ] + " / " +
            "Pred: " + cifar10_labels_dict[ predict2[corrects2][idx] ], 
            fontsize = 8 
        )

    axis[0].set_xticks( [] )
    axis[0].set_yticks( [] )
    #plt.tight_layout()
    MLPlot.saveFigure( fileName = "CNN_2-6.png" )
    plt.show()
    

    # 誤識別画像の plot のための loop
    figure, axis = plt.subplots( 
                        nrows = 5, ncols = 8,
                        sharex = True, sharey = True     # x,y 軸をシャアする
                     )

    # ２次元配列を１次元に変換
    axis = axis.flatten()

    for (idx, image) in enumerate( X_test[ ~corrects2 ][0:40] ):
        image = image.reshape(32,32,3)        # １次元配列を shape = [32 ,32, 3] に reshape
        axis[idx].imshow( image )
        axis[idx].set_title( 
            "Actual: " + cifar10_labels_dict[ y_test[~corrects2][idx] ] + " / " +
            "Pred: " + cifar10_labels_dict[ predict2[~corrects2][idx] ], 
            fontsize = 8 
        )

    axis[0].set_xticks( [] )
    axis[0].set_yticks( [] )
    #plt.tight_layout()
    MLPlot.saveFigure( fileName = "CNN_2-7.png" )
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
