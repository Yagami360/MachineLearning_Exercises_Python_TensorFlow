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

from CNNStyleNet import CNNStyleNet


def main():
    """
    TensorFlow を用いた CNN-StyleNet / NeuralStyle（ニューラルスタイル）による画像生成処理
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


    #======================================================================
    # データを変換、正規化
    # Transform and normalize data.
    # ex) data = tf.nn.batch_norm_with_global_normalization(...)
    #======================================================================


    #======================================================================
    # データセットをトレーニングデータ、テストデータ、検証データセットに分割
    #======================================================================


    #======================================================================
    # アルゴリズム（モデル）のパラメータを設定
    # Set algorithm parameters.
    # ex) learning_rate = 0.01  iterations = 1000
    #======================================================================
    image_content_path1 = "C:\Data\MachineLearning_DataSet\CNN-StyleNet\image_content\\neko-sensei.jpg"
    image_style_path1 = "C:\Data\MachineLearning_DataSet\CNN-StyleNet\image_style\starry_night.jpg"
    learning_rate1 = 0.01
    adam_beta1 = 0.9        # For the Adam optimizer
    adam_beta2 = 0.999      # For the Adam optimizer

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
    styleNet1 = CNNStyleNet(
                    image_content_path = image_content_path1,
                    image_style_path = image_style_path1,
                    vgg_mat_file = "C:\Data\MachineLearning_DataSet\CNN-StyleNet\imagenet-vgg-verydeep-19.mat",
                    session = tf.Session( config = tf.ConfigProto(log_device_placement=True) ),
                    epochs = 10000,
                    eval_step = 50,
                    weight_image_content = 200.0,
                    weight_image_style = 200.0,
                    weight_regularization = 100,
                    n_strides = 1,
                    n_pool_wndsize = 2,
                    n_pool_strides = 2
                )
    
    styleNet1.print( "" )
    
    #======================================================================
    # モデルの構造を定義する。
    # Define the model structure.
    # ex) add_op = tf.add(tf.mul(x_input_holder, weight_matrix), b_matrix)
    #======================================================================
    styleNet1.model()
    styleNet1.print( "after model()" )

    #======================================================================
    # 損失関数を設定する。
    # Declare the loss functions.
    #======================================================================
    styleNet1.loss()

    #======================================================================
    # モデルの最適化アルゴリズム Optimizer を設定する。
    # Declare Optimizer.
    #======================================================================
    styleNet1.optimizer( Adam( learning_rate = learning_rate1, beta1 = adam_beta1, beta2 = adam_beta2 ) )
    #styleNet1._session.run( tf.global_variables_initializer() )
    #styleNet1.optimizer( GradientDecent( learning_rate = learning_rate1 ) )

    # TensorBoard 用のファイル（フォルダ）を作成
    styleNet1.write_tensorboard_graph()

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
    styleNet1.run()
    
    #======================================================================
    # モデルの評価
    # (Optional) Evaluate the model.
    #======================================================================
    #styleNet1.show_output_image()
    #styleNet1.save_output_image( "", "CNN-StyleNet_output_image_1.png" )
    #styleNet1.save_output_image_gif( "", "CNN-StyleNet_output_image_1.gif" )


    #-------------------------------------------------------------------
    # トレーニング回数に対する loss 値の plot
    #-------------------------------------------------------------------
    plt.clf()
    plt.plot(
        range( len(styleNet1._losses_train) ), styleNet1._losses_train,
        label = "losses",
        linestyle = '-',
        #linewidth = 2,
        color = 'black'
    )
    plt.plot(
        range( len(styleNet1._losses_content_train) ), styleNet1._losses_content_train,
        label = "losses_content",
        linestyle = '--',
        #linewidth = 2,
        color = 'red'
    )
    plt.plot(
        range( len(styleNet1._losses_style_train) ), styleNet1._losses_style_train,
        label = "losses_style",
        linestyle = '--',
        #linewidth = 2,
        color = 'blue'
    )
    plt.plot(
        range( len(styleNet1._losses_total_var_train) ), styleNet1._losses_total_var_train,
        label = "losses_total_var",
        linestyle = '--',
        #linewidth = 2,
        color = 'green'
    )
    plt.title( "loss : AdamOptimizer" )
    plt.legend( loc = 'best' )
    #plt.ylim( [0, 1.05] )
    plt.xlabel( "Epocs %d / eval_step %d" % ( styleNet1._epochs, styleNet1._eval_step ) )
    plt.tight_layout()
   
    plt.savefig("CNN_StyleNet_1-1.png", dpi = 300, bbox_inches = "tight" )
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