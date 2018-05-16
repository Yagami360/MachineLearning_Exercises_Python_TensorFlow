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
from model.DefaultBox import DefaultBoxes
from model.BoundingBox import BoundingBox

from model.SingleShotMultiBoxDetector import SingleShotMultiBoxDetector


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

    #-----------------------------------------------------------
    # 各自作クラスの動作確認（デバッグ用）
    #-----------------------------------------------------------
    default_box1 = DefaultBox(
                      group_id = 1, id = 1,
                      center_x = 0.5, center_y = 0.5,
                      width = 1.5, height = 1,
                      scale = 1,
                      aspect = 1
                  )
    
    default_box1.print( "" )

    #
    default_boxes1 = DefaultBoxes(
                         group_id = 1,
                         n_fmaps = 6,
                         scale_min = 0.2, scale_max = 0.9
                     )

    #default_boxes1.print( "after __init__()" )
    #default_boxes1.add_default_box( default_box1 )
    #default_boxes1.print( "after add_default_box()" )

    #
    """
    bbox = BoundingBox(
               label = 1,
               position = [ 1, 1 ]
           )

    bbox.print()
    """

    image = np.full( (300, 300, 3), 256, dtype=np.uint8 )
    image = default_box1.draw_rect( image, color = (0,0,255), thickness = 2 )

    image = default_boxes1.draw_rects( image, group_id = 1 )

    #cv2.namedWindow( "image", cv2.WINDOW_NORMAL)
    #cv2.imshow( "image", image )
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #======================================================================
    # データセットを読み込み or 生成
    # Import or generate data.
    #======================================================================

    #======================================================================
    # データセットをトレーニングデータ、テストデータ、検証データセットに分割
    #======================================================================

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
    """
    base_vgg16 = BaseNetworkVGG16(
                     session = tf.Session(),
                     image_height = 300,
                     image_width = 300,
                     n_channels = 3
                 )
    """

    ssd = SingleShotMultiBoxDetector(
              session = tf.Session(),
              image_height = 300,
              image_width = 300,
              n_channels = 3
          )

    #======================================================================
    # モデルの構造を定義する。
    # Define the model structure.
    # ex) add_op = tf.add(tf.mul(x_input_holder, weight_matrix), b_matrix)
    #======================================================================
    #base_vgg16.model()
    #base_vgg16.print( "after model()" )

    ssd.model()
    ssd.print( "after model()" )

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