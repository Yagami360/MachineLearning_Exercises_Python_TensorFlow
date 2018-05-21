# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境 
# + TensorFlow 1.4.0 インストール済み
#     <Anaconda Prompt>
#     conda create -n tensorflow python=3.5
#     activate tensorflow
#     pip install --ignore-installed --upgrade tensorflow
#     pip install --ignore-installed --upgrade tensorflow-gpu
# + OpenCV 3.3.1 インストール済み
#     pip install opencv-python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from scipy.misc import imread, imresize

import datetime

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops

# OpenCV ライブラリ
import cv2

# 自作モジュール
from util.MLPreProcess import MLPreProcess
from util.MLPlot import MLPlot

from model.NNActivation import NNActivation              # ニューラルネットワークの活性化関数を表すクラス
from model.NNActivation import Sigmoid
from model.NNActivation import Relu
from model.NNActivation import Softmax

from model.NNLoss import NNLoss                          # ニューラルネットワークの損失関数を表すクラス
from model.NNLoss import L1Norm
from model.NNLoss import L2Norm
from model.NNLoss import BinaryCrossEntropy
from model.NNLoss import CrossEntropy
from model.NNLoss import SoftmaxCrossEntropy
from model.NNLoss import SparseSoftmaxCrossEntropy

from model.NNOptimizer import NNOptimizer                # ニューラルネットワークの最適化アルゴリズム Optimizer を表すクラス
from model.NNOptimizer import GradientDecent
from model.NNOptimizer import GradientDecentDecay
from model.NNOptimizer import Momentum
from model.NNOptimizer import NesterovMomentum
from model.NNOptimizer import Adagrad
from model.NNOptimizer import Adadelta
from model.NNOptimizer import Adam

from model.NeuralNetworkBase import NeuralNetworkBase
from model.VGG16Network import VGG16Network

from model.BaseNetwork import BaseNetwork
from model.BaseNetwork import BaseNetworkVGG16
from model.BaseNetwork import BaseNetworkResNet

from model.DefaultBox import DefaultBox
from model.DefaultBox import DefaultBoxSet
from model.BoundingBox import BoundingBox

from model.SingleShotMultiBoxDetector import SingleShotMultiBoxDetector


def load_image_voc2007( path ):
    """
    load specified image

    Args: image path
    Return: resized image, its size and channel
    """
    from scipy.misc import imread, imresize

    img = imread( path )
    h, w, c = img.shape
    img = imresize( img, (300, 300) )
    img = img[:, :, ::-1].astype('float32')
    img /= 255.
    return img, w, h, c


def main():
    """
    TensorFlow を用いた SSD [Single Shot muitibox Detector] の実装
    """
    print("Enter main()")

    # ライブラリのバージョン確認
    print( "TensorFlow version :", tf.__version__ )
    print( "OpenCV version :", cv2.__version__ )

    # Reset graph
    ops.reset_default_graph()

    # Session の設定
    #session = tf.Session()
    
    #======================================================================
    # データセットを読み込み or 生成
    # Import or generate data.
    #======================================================================
    # VOC2007 のデータセットへのパス
    dataset_path = "C:\Data\MachineLearning_DataSet\VOC2007\\"

    #------------------------------------------------------------------------------------------------------
    # load pickle data set annotation
    # data : [ image_filename, N x 24 の 2 次元配列 ] 0.0 ~ 1.0
    # N は、画像の中にある検出物体の数で、画像によって異なる。
    # 24 というのは、位置とクラス名のデータを合わせたデータを表すベクトルになっていて、
    # この内、(xmin, ymin, xmax, ymax) の 4 次元の情報で物体を囲む矩形の位置を表し、残りの 20 次元でクラス名を表します。
    # クラス名が20次元あるということは20種類の物体を見分けたい、ということになります。
    # 教師データ : data[ keys[idxes] ]
    #------------------------------------------------------------------------------------------------------
    with open(dataset_path + 'VOC2007.pkl', 'rb') as f:
        data = pickle.load(f)
        keys = sorted(data.keys())

    print( "len( data ) :", len( data ) )   # 4952
    print( "keys :", keys[0:10] )

    # 処理負荷軽減のためデータ数カット（デバッグ用途）
    #data = data[0:1000]

    #======================================================================
    # データセットをトレーニングデータ、テストデータ、検証データセットに分割
    #======================================================================
    # 80% をトレーニングデータにする。
    n_trains = int( round(0.8 * len(keys)) )

    # トレーニングデータとテストデータの key
    train_keys = keys[:n_trains]
    test_keys = keys[n_trains:]
    n_tests = len( test_keys )
    print( "len(train_keys) :", len(train_keys) )
    print( "len(test_keys) :", len(test_keys) )

    # 計算負荷軽減（デバッグ用途）
    n_trains = 100
    n_tests = int( round(0.8 * n_trains) )
    print( "n_trains :", n_trains )
    print( "n_tests :", n_tests )

    # トレーニングデータとテストデータの抽出 
    X_train = []
    y_train = []
    for key in train_keys[0:n_trains]:
        # image dataのみ取得。高さ、幅、チャンネル数情報は除外
        image, _, _, _ = load_image_voc2007(  dataset_path + key )
        
        X_train.append( image )
        y_train.append( data[key] )

    X_test = []
    y_test = []
    for key in test_keys[0:n_tests]:
        # image dataのみ取得。高さ、幅、チャンネル数情報は除外
        image, _, _, _ = load_image_voc2007( dataset_path + key )

        X_test.append( image )
        y_test.append( data[key] )

    # list → numpy に変換
    #X_train = np.array( X_train )
    #X_test = np.array( X_test )

    # X_train.shape : [ n_trains, (img, w, h, c) ]
    print( "X_train.shape : (%d,%d)" % ( len(X_train), len(X_train[0]) ) )        # (n_trains, 4)
    print( "y_train.shape : (%d,%d)" % ( len( y_train ), len( y_train[0] ) ) )    # (n_trains, 2, 24)
    #print( "X_test.shape : (%d,%d)" % ( len( X_test ), len( X_test[0] ) ) )

    #======================================================================
    # データを変換、正規化
    # Transform and normalize data.
    # ex) data = tf.nn.batch_norm_with_global_normalization(...)
    #======================================================================
    
    #======================================================================
    # アルゴリズム（モデル）のパラメータを設定
    # Set algorithm parameters.
    # ex) learning_rate = 0.01  iterations = 1000
    #======================================================================
    learning_rate1 = 0.001
    adam_beta1 = 0.9            # For the Adam optimizer
    adam_beta2 = 0.999          # For the Adam optimizer

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
    ssd = SingleShotMultiBoxDetector(
              session = tf.Session(),
              epochs = 2,
              batch_size = 10,
              eval_step = 1,
              save_step = 100,
              image_height = 300,
              image_width = 300,
              n_channels = 3,
              n_classes = 21,
              n_boxes = [ 4, 6, 6, 6, 6, 6 ]
          )

    #======================================================================
    # モデルの構造を定義する。
    # Define the model structure.
    # ex) add_op = tf.add(tf.mul(x_input_holder, weight_matrix), b_matrix)
    #======================================================================
    ssd.model()
    #ssd.print( "after model()" )

    # 特徴マップに対応した一連のデフォルト群の生成
    ssd.generate_default_boxes_in_fmaps()
    #ssd.print( "after generate_default_boxes_in_fmaps()" )

    #======================================================================
    # 損失関数を設定する。
    # Declare the loss functions.
    #======================================================================
    ssd.loss( nnLoss = None )

    #======================================================================
    # モデルの最適化アルゴリズム Optimizer を設定する。
    # Declare Optimizer.
    #======================================================================
    ssd.optimizer( Adam( learning_rate = learning_rate1, beta1 = adam_beta1, beta2 = adam_beta2 ) )
    ssd.print( "after optimizer()" )

    #-------------------------------------------------------------------
    # 生成したデフォルトボックス群の表示（学習処理前）
    #-------------------------------------------------------------------
    """
    image = np.full( (300, 300, 3), 256, dtype=np.uint8 )
    image = ssd._default_box_set.draw_rects( image, group_id = 1 )
    
    cv2.namedWindow( "image", cv2.WINDOW_NORMAL)
    cv2.imshow( "image", image )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

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
    #ssd.fit( X_train, y_train )
    #ssd.print( "after fitting" )

    ssd.load_model()

    #======================================================================
    # モデルの評価
    # (Optional) Evaluate the model.
    #======================================================================
    ssd.write_tensorboard_graph()

    #---------------------------------------------------------
    # 損失関数を plot
    #---------------------------------------------------------
    plt.clf()
    plt.plot(
        range( len(ssd._losses_train) ), ssd._losses_train,
        label = 'SSD (base - VGG16), learning_rate = %0.3f' % ( learning_rate1 ),
        linestyle = '-',
        linewidth = 0.2,
        color = 'red'
    )
    plt.title( "loss" )
    plt.legend( loc = 'best' )
    plt.xlim( xmin = 0, xmax = len(ssd._losses_train) )
    plt.ylim( ymin = 0.0 )
    plt.xlabel( "minibatch iteration" )
    plt.grid()
    plt.tight_layout()
    MLPlot.saveFigure( fileName = "SSD_2-1.png" )
    #plt.show()

    #-------------------------------------------------------------------
    # 生成したデフォルトボックス群の表示（学習処理後）
    #-------------------------------------------------------------------
    fontType = cv2.FONT_HERSHEY_SIMPLEX

    #image = np.full( (300, 300, 3), 256, dtype=np.uint8 )
    #image = ssd._default_box_set.draw_rects( image, group_id = 1 )

    #
    image = X_test[0]
    pred_confs, pred_locs = ssd.predict( image = [image] )  # [] でくくって、shape を [300,300,3] → [,300,300,3]
    locs, labels = ssd.detect_objects( pred_confs, pred_locs )

    # image のフォーマットを元の高さ、幅情報付きに戻す。
    # x = x[:, :, ::-1]
    image *= 255.
    image = np.clip(image, 0, 255).astype('uint8')
    image = imresize(image, (300, 300))


    if len(labels) and len(locs):
        for label, loc in zip(labels, locs):
            # loc のフォーマットをもとに戻す。
            #loc = center2corner(loc)
            corner_x = loc[0] - loc[2] * 0.5
            corner_y = loc[1] - loc[3] * 0.5
            loc = np.array( [corner_x, corner_y, abs(loc[2]), abs(loc[3])] )

            #loc = convert2diagonal_points(loc)
            loc = np.array( [loc[0], loc[1], loc[0]+loc[2], loc[1]+loc[3]] )

            cv2.rectangle(
                image, 
                ( int(loc[0]*300), int(loc[1]*300) ), 
                ( int(loc[2]*300), int(loc[3]*300) ), 
                (0, 0, 255),
                1
            )

            cv2.putText(
                image, 
                str(int(label)), 
                ( int(loc[0]*300), int(loc[1]*300) ), 
                fontType, 
                0.7, 
                (0, 0, 255), 
                1
            )
    #str( datetime.datetime.now() )
    
    cv2.imwrite( "./evaluated/test.jpg", image )
    cv2.namedWindow( "image", cv2.WINDOW_NORMAL)
    cv2.imshow( "image", image )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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