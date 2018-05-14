# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境 (TensorFlow 1.4.0 インストール済み)
#     <Anaconda Prompt>
#     conda create -n tensorflow python=3.5
#     activate tensorflow
#     pip install --ignore-installed --upgrade tensorflow
#     pip install --ignore-installed --upgrade tensorflow-gpu

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops

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
#from model.VGG16Network import VGG16Network

from model.DefaultBox import DefaultBox
from model.DefaultBox import DefaultBoxes
from model.BoundingBox import BoundingBox

from model.BaseNetwork import BaseNetwork
from model.BaseNetwork import BaseNetworkVGG16
from model.BaseNetwork import BaseNetworkResNet



def main():
    """
    TensorFlow を用いた SSD [Single Shot muitibox Detector] の実装
    """
    print("Enter main()")

    # Reset graph
    ops.reset_default_graph()

    # Session の設定
    #session = tf.Session()

    #-----------------------------------------------------------
    # 各自作クラスの動作確認（デバッグ用）
    #-----------------------------------------------------------
    default_box1 = DefaultBox(
                      group_id = 1, id = 1,
                      center_x = 0.5, center_y = 0.5,
                      width = 1, height = 1,
                      scale = 1
                  )
    
    default_box1.print( "" )

    #
    default_boxes1 = DefaultBoxes(
                         group_id = 1,
                         n_fmaps = 6,
                         scale_min = 0.2, scale_max = 0.9
                     )

    default_boxes1.print( "after __init__()" )
    default_boxes1.add_default_box( default_box1 )
    default_boxes1.print( "after add_default_box()" )

    #
    """
    bbox = BoundingBox(
               label = 1,
               position = [ 1, 1 ]
           )

    bbox.print()
    """

    #======================================================================
    # データセットを読み込み or 生成
    # Import or generate data.
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
    cifar10_path = "C:\Data\MachineLearning_DataSet\CIFAR\cifar-10-batches-bin"

    #======================================================================
    # データセットをトレーニングデータ、テストデータ、検証データセットに分割
    #======================================================================
    """
    #X_train, y_train = MLPreProcess.load_cifar10_train( cifar10_path, fileName = "data_batch_1.bin" )
    X_train, y_train = MLPreProcess.load_cifar10_trains( cifar10_path )
    X_test, y_test = MLPreProcess.load_cifar10_test( cifar10_path )
    """
    # 処理負荷軽減のためデータ数カット（デバッグ用途）
    """
    n_train_data = 1000
    n_test_data = 500
    X_train = X_train[0:n_train_data]
    y_train = y_train[0:n_train_data]
    X_test = X_test[0:n_train_data]
    y_test = y_test[0:n_train_data]
    """
    """
    print( "X_train.shape : ", X_train.shape )
    print( "y_train.shape : ", y_train.shape )
    print( "X_test.shape : ", X_test.shape )
    print( "y_test.shape : ", y_test.shape )

    #print( "X_train[0] : \n", X_train[0] )
    #print( "y_train[0] : \n", y_train[0] )
    #print( "[y_train == 0] : \n", [ y_train == 0 ] )
    """

    #======================================================================
    # データを変換、正規化
    # Transform and normalize data.
    # ex) data = tf.nn.batch_norm_with_global_normalization(...)
    #======================================================================
    """
    session = tf.Session()
    encode_holder = tf.placeholder(tf.int64, [None])
    y_oneHot_enoded_op = tf.one_hot( encode_holder, depth=10, dtype=tf.float32 )    # depth が 出力層のノード数に対応
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
    learning_rate1 = 0.001

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

    #======================================================================
    # 損失関数を設定する。
    # Declare the loss functions.
    #======================================================================

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

    #======================================================================
    # モデルの評価
    # (Optional) Evaluate the model.
    #======================================================================
    
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