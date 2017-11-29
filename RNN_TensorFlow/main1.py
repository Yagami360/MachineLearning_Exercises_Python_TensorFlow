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




def main():
    """
    TensorFlow を用いた RNN によるノイズ付き sin 波形（時系列データ）からの波形の予想（生成）処理
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
    T1 = 100                # ノイズ付き sin 波形の周期
    noize_size1 = 0.05      # ノイズ付き sin 波形のノイズ幅

    times = numpy.arange( 2.5 * T1 + 1 )    # 時間 t の配列

    #print( "X_dat :", X_dat )

    X_features, y_labels = MLPreProcess.generate_sin_with_noize( t = times, T = T1, noize_size = noize_size1, seed = 12 )
    print( "X_features", X_features )
    print( "y_labels", y_labels )

    # ノイズ付き sin 波形を plot
    plt.clf()

    plt.plot(
        X_features, y_labels,
        label = 'noize = %0.3f' % noize_size1,
        linestyle = '-',
        #linewidth = 2,
        color = 'red'
    )

    plt.title( "sin with noize" )
    plt.legend( loc = 'best' )
    plt.ylim( [-1.05, 1.05] )
    plt.xlabel( "t [time]" )
    plt.grid()
    plt.tight_layout()
   
    plt.savefig("RNN_1-1.png", dpi = 300, bbox_inches = "tight" )
    #MLPlot.saveFigure( fileName = "RNN_1-1.png" )
    plt.show()

    #======================================================================
    # データを変換、正規化
    # Transform and normalize data.
    # ex) data = tf.nn.batch_norm_with_global_normalization(...)
    #======================================================================
    # BPTT での計算負荷の関係上、時系列データを一定間隔に区切る
    len_sequences = len( X_features )   # 全時系列データの長さ
    len_one_sequence = 25               # １つの時系列データの長さ τ
    
    print( "len_sequences :", len_sequences )
    print( "len_one_sequence :", len_one_sequence )

    data = []       # 区切った時系列データの t 値（ t=1~τ, t=2~τ+1, ... ）のリスト
    targets = []    # 

    for i in range( 0, len_sequences - len_one_sequence + 1 ):
        data.append( X_features[ i : i + len_one_sequence ] )
        data.append( X_features[ i + len_one_sequence ] )

    print( "data.shape :\n", data.shape )
    print( "targets.shape :\n", targets.shape )
    print( "data :\n", data )
    print( "targets :\n", targets )
    

    #======================================================================
    # データセットをトレーニングデータ、テストデータ、検証データセットに分割
    #======================================================================


    #======================================================================
    # アルゴリズム（モデル）のパラメータを設定
    # Set algorithm parameters.
    # ex) learning_rate = 0.01  iterations = 1000
    #======================================================================
    learning_rate1 = 0.01

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
    # モデルの最適化アルゴリズム Optimizer を設定する。
    # Declare Optimizer.
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
    

    #-------------------------------------------------------------------
    # トレーニング回数に対する loss 値の plot
    #-------------------------------------------------------------------
    """
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