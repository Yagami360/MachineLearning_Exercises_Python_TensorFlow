# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境 (TensorFlow インストール済み)
#     <Anaconda Prompt>
#     conda create -n tensorflow python=3.5
#     activate tensorflow
#     pip install --ignore-installed --upgrade tensorflow
#     pip install --ignore-installed --upgrade tensorflow-gpu

import numpy as np
import pandas
import matplotlib.pyplot as plt

import argparse                     # コマンドライン引数の解析モジュール

import matplotlib.animation as animation
import pickle

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops

# 自作クラス
from MLPreProcess import MLPreProcess
#from MLPlot import MLPlot

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
from DeepConvolutionalGAN import DeepConvolutionalGAN


def main():
    """
    TensorFlow を用いた DCGAN による MNIST データの自動生成
    """
    print("Enter main()")

    #======================================================================
    # コマンドライン引数
    #======================================================================
    """
    # パーサーの作成
    arg_parser = argparse.ArgumentParser(
                     prog = "main1.py",                   # プログラム名
                     usage = "TensorFlow を用いた DCGAN による MNIST データの自動生成",  # プログラムの利用方法
                     description = 'pre description',     # 引数のヘルプの前に表示する文
                     epilog = "epilog",                   # 引数のヘルプの後で表示する文
                     add_help = True                      # -h/–help オプションの追加
                 )

    # コマンドライン引数を追加
    # 位置引数 : 関数に対して必須となる引数
    # オプション引数: 与えても与えなくてもどちらでも良い引数
    # オプション引数には接頭辞「–」を付ける必要があり、実行時に引数を指定する場所はどこでも構いません。
    # 接頭辞「–」で短い略語を指定し、接頭辞「—-」で長い名称
    # オプション引数以外の引数は位置引数として扱われ、実行時の引数の位置は決まっています。
    arg_parser.add_argument( '--learning_rate', dest = 'learning_rate', type = float, default = 0.001 )

    # コマンドライン引数を解析する。
    arg_parser.parse_args()
    """

    #======================================================================
    # データセットを読み込み or 生成
    # Import or generate data.
    #======================================================================
    #======================================================================
    # データセットをトレーニングデータ、テストデータ、検証データセットに分割
    #======================================================================
    # MNIST データが格納されているフォルダへのパス
    mnist_path = "C:\Data\MachineLearning_DataSet\MNIST"

    X_train, y_train = MLPreProcess.load_mnist( mnist_path, "train" )
    X_test, y_test = MLPreProcess.load_mnist( mnist_path, "t10k" )

    # データは shape = [n_sample, image_width=28, image_height=28] の形状に reshape
    X_train = np.array( [np.reshape(x, (28,28)) for x in X_train] )
    X_test = np.array( [np.reshape(x, (28,28)) for x in X_test] )
    
    print( "X_train.shape : ", X_train.shape )
    print( "y_train.shape : ", y_train.shape )
    print( "X_test.shape : ", X_test.shape )
    print( "y_test.shape : ", y_test.shape )

    #print( "X_train : \n", X_train )
    #print( "y_train : \n", y_train )

    #======================================================================
    # データを変換、正規化
    # Transform and normalize data.
    # ex) data = tf.nn.batch_norm_with_global_normalization(...)
    #======================================================================
    # One -hot encoding
    #y_train_encoded = np.eye(10)[ y_train.astype(int) ]
    #y_test_encoded = np.eye(10)[ y_test.astype(int) ]
    """
    session = tf.Session()
    encode_holder = tf.placeholder(tf.int64, [None])
    y_oneHot_enoded_op = tf.one_hot( encode_holder, depth=10, dtype=tf.float32 ) # depth が 出力層のノード数に対応
    session.run( tf.global_variables_initializer() )
    y_train_encoded = session.run( y_oneHot_enoded_op, feed_dict = { encode_holder: y_train } )
    y_test_encoded = session.run( y_oneHot_enoded_op, feed_dict = { encode_holder: y_test } )
    print( "y_train_encoded.shape : ", y_train_encoded.shape )
    print( "y_train_encoded.dtype : ", y_train_encoded.dtype )
    print( "y_test_encoded.shape : ", y_test_encoded.shape )
    session.close()
    """

    #======================================================================
    # アルゴリズム（モデル）のパラメータを設定
    # Set algorithm parameters.
    # ex) learning_rate = 0.01  iterations = 1000
    #======================================================================
    epochs = 200
    #epochs = 5000
    #epochs = 7000
    #epochs = 20000

    eval_step = 50
    batch_size = 32
    learning_rate = 0.0001
    beta1 = 0.5
    beta2 = 0.99

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
    # DCGAN クラスのオブジェクト生成
    dcgan = DeepConvolutionalGAN(
                session = tf.Session( config = tf.ConfigProto(log_device_placement=True) ),
                epochs = epochs,
                batch_size = batch_size,
                eval_step = eval_step,
                image_height = 28,                      # 28 pixel
                image_width = 28,                       # 28 pixel
                n_channels = 1,                         # グレースケール
                n_G_deconv_featuresMap = [128, 64, 1],  # 
                n_D_conv_featuresMap = [1, 64, 128],    #
                n_labels = 2
           )

    dcgan.print( "after init" )
    

    #======================================================================
    # モデルの構造を定義する。
    # Define the model structure.
    # ex) add_op = tf.add(tf.mul(x_input_holder, weight_matrix), b_matrix)
    #======================================================================
    dcgan.model()

    #======================================================================
    # 損失関数を設定する。
    # Declare the loss functions.
    #======================================================================
    dcgan.loss()
    dcgan.optimizer( 
        nnOptimizerG = Adam( learning_rate = learning_rate, beta1 = beta1, beta2 = beta2 ),
        nnOptimizerD = Adam( learning_rate = learning_rate, beta1 = beta1, beta2 = beta2 )
    )

    dcgan.print( "after building model & loss & optimizer" )

    # TensorBoard
    #dcgan.write_tensorboard_graph()
    
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
    # トレーニングデータで fitting 処理
    dcgan.fit( X_train, y_train = None )

    # モデルの保存
    #dcgan.save_model()
    #dcgan.load_model()

    #======================================================================
    # モデルの評価
    # (Optional) Evaluate the model.
    #======================================================================
    #-------------------------------------------------------------------
    # トレーニング回数に対する loss 値の plot
    #-------------------------------------------------------------------
    # loss 値の list に対応させる x 軸データ用の list
    loss_axis_x = [ (idx+1) * eval_step for idx in range( len(dcgan._losses_train) ) ]
    #print( "loss_axis_x :", loss_axis_x )

    plt.clf()
    plt.plot(
        loss_axis_x, dcgan._losses_train,
        label = ' loss (total) : train data',
        linestyle = '-',
        #linewidth = 2,
        color = 'black'
    )
    plt.plot(
        loss_axis_x, dcgan._losses_G_train,
        label = ' loss (generator) : train data',
        linestyle = '--',
        #linewidth = 2,
        color = 'red'
    )
    plt.plot(
        loss_axis_x, dcgan._losses_D_train,
        label = ' loss (discriminator) : train data',
        linestyle = '--',
        #linewidth = 2,
        color = 'blue'
    )
    plt.title( "loss / sparse softmax cross-entory" )
    plt.legend( loc = 'best' )
    plt.xlim( 1, epochs )
    #plt.ylim( [0, 1.05] )
    plt.xlabel( "Epochs / batch_size = {}".format(batch_size) )
    plt.grid()
    plt.tight_layout()
   
    #MLPlot.saveFigure( fileName = "GAN_DCGAN_1-1.png" )
    plt.savefig( "GAN_DCGAN_1-1_epoch{}.png".format(epochs), dpi = 300, bbox_inches = "tight" )
    #plt.show()

    #-------------------------------------------------------------------
    # 学習過程で生成しておいた画像の animation gif
    #-------------------------------------------------------------------
    fig = plt.figure( figsize = (4,6) )
    print( "len(dcgan._images_evals[0])", len(dcgan._images_evals[0]) )

    images = []
    for imgs in dcgan._images_evals:
        # row : 1
        subplot = fig.add_subplot( 1, 1, 1 )
        subplot.set_xticks([])
        subplot.set_yticks([])

        image = np.hstack( 
                    np.array( imgs ) 
                )
        image = np.reshape( image, (28*4, 28*8) )

        image = subplot.imshow(
                    image, 
                    vmin = 0, vmax = 1,
                    cmap = plt.cm.gray_r
                )
        images.append( [image] )

    ani = animation.ArtistAnimation( 
              fig, images, 
              interval = 10  # ms 単位
          )
    
    ani.save('./output_image/DCGAN_fitting_epoch{}.gif'.format(epochs), writer='imagemagick', fps = 100 )
    plt.show()
    #-------------------------------------------------------------------
    # 学習済み DCGAN に対し、入力ノイズ自動画像生成
    #-------------------------------------------------------------------
    # Generator に入力する初期ノイズ
    input_noize1 = np.random.rand( batch_size, 64 ) * 2.0 - 1.0
    inputs_noize_tsr1 = tf.constant( 
                           np.array( input_noize1 ), 
                           dtype = tf.float32 
                       )

    # result : 0.0 ~ 1.0    
    result1 = dcgan._session.run( 
                 dcgan.generator( input = inputs_noize_tsr1, reuse = True ) 
             )

    images1 = []
    for i in range( result1.shape[0] ):
        images1.append( result1[i,:,:,0] )

    fig = plt.figure( figsize = (4,8) )
    k = 0
    for i in range(8):
        for j in range(4):
            k += 1
            subplot = fig.add_subplot( 4, 8, k )
            subplot.set_xticks([])
            subplot.set_yticks([])
            subplot.imshow(
                images1[k-1], 
                vmin=0, vmax=1,
                cmap = plt.cm.gray_r
            )
    plt.savefig( "GAN_DCGAN_1-2_epoch{}.png".format(epochs), dpi = 300, bbox_inches = "tight" )

    #-------------------------------------------------------------------
    # 学習済み DCGAN に対し、入力ノイズを動かし Animation gif
    #-------------------------------------------------------------------
    # Generator に入力する初期ノイズ
    input_noize = np.random.rand( 3, 64 ) * 2.0 - 1.0
    
    morphing_inputs = []

    # 球の表面上の回転
    theta1, theta2 = 0, 0
    for _ in range(32):     # batch_size
        theta1 += 2*np.pi / 32 * 2
        theta2 += 2*np.pi / 32
        morphing_inputs.append(
            np.cos(theta1) * input_noize[0] \
            + np.sin(theta1)*( np.cos(theta2)*input_noize[1] + np.sin(theta2)*input_noize[2] )
        )

    inputs_noize_tsr = tf.constant( 
                           np.array( morphing_inputs ), 
                           dtype = tf.float32 
                       )

    # result : 0.0 ~ 1.0    
    result = dcgan._session.run( 
                 dcgan.generator( input = inputs_noize_tsr, reuse = True ) 
             )
    
    images = []
    fig = plt.figure(figsize=(4,4))
    for i in range( result.shape[0] ):
        subplot = fig.add_subplot(1,1,1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        image = subplot.imshow(
                    result[i,:,:,0], 
                    vmin = 0, vmax = 1, 
                    cmap = plt.cm.gray_r
                )
        images.append( [image] )  # i=63 : ValueError: outfile must be *.htm or *.html

    ani = animation.ArtistAnimation( 
              fig, images, 
              interval = 10  # ms 単位
          )
    
    ani.save('./output_image/DCGAN_morphing1_epoch{}.gif'.format(epochs), writer='imagemagick', fps = 100 )
    
    #---------------------------------------
    # Generator に入力する初期ノイズ
    input_noize = np.random.rand( 3, 64 ) * 2.0 - 1.0
    
    morphing_inputs = []

    # 球の表面上の回転
    theta1, theta2 = 0, 0
    for _ in range(32):     # batch_size
        theta1 += 2*np.pi / 32
        theta2 += 2*np.pi / 32
        morphing_inputs.append(
            np.cos(theta1) * input_noize[0] \
            + np.sin(theta1)*( np.cos(theta2)*input_noize[1] + np.sin(theta2)*input_noize[2] )
        )

    inputs_noize_tsr = tf.constant( 
                           np.array( morphing_inputs ), 
                           dtype = tf.float32 
                       )

    # result : 0.0 ~ 1.0    
    result = dcgan._session.run( 
                 dcgan.generator( input = inputs_noize_tsr, reuse = True ) 
             )
    
    images = []
    fig = plt.figure(figsize=(4,4))
    for i in range( result.shape[0] ):
        subplot = fig.add_subplot(1,1,1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        image = subplot.imshow(
                    result[i,:,:,0], 
                    vmin = 0, vmax = 1, 
                    cmap = plt.cm.gray_r
                )
        images.append( [image] )  # i=63 : ValueError: outfile must be *.htm or *.html

    ani = animation.ArtistAnimation( 
              fig, images, 
              interval = 10  # ms 単位
          )
    
    ani.save('./output_image/DCGAN_morphing2_epoch{}.gif'.format(epochs), writer='imagemagick', fps = 100 )
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