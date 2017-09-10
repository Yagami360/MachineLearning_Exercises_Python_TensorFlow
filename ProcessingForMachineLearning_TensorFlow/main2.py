# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境 (TensorFlow インストール済み)
#     [Anaconda Prompt]
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

# scikit-learn ライブラリ関連


# 自作クラス
from MLPlot import MLPlot


def main():
    """
    損失関数の実装
    """
    print("Enter main()")

    #======================================================================
    # 損失関数の実装
    #======================================================================
    #-------------------------------------------------------------------
    # 回帰問題の為の損失関数
    #-------------------------------------------------------------------
    # Reset graph
    ops.reset_default_graph()

    # Session の設定
    session = tf.Session()

    # 一連の予想値
    x_predicts_tsr = tf.linspace( -1. , 1. , 500 )     # -1 ~ +1 の範囲の 500 個のシーケンス Tensor
    print( "x_predicts_tsr :\n", x_predicts_tsr )

    # 目的値
    target_tsr = tf.constant( 0. )   #  定数 Tensor で値（目的値）を 0 としている。
    print( "target_tsr :\n", target_tsr )
    
    # 回帰問題の為の損失関数のオペレーション作成
    l2_loss_op = tf.square( target_tsr - x_predicts_tsr )   # L2 正則化の損失関数のオペレーション
    l1_loss_op = tf.abs( target_tsr - x_predicts_tsr )      # L1 正則化の損失関数のオペレーション
    
    print( "l2_loss_op = tf.square( target - x_predicts_tsr ) :\n", l2_loss_op )
    print( "l1_loss_op = tf.abs( target - x_predicts_tsr ) :\n", l1_loss_op )

    # Session を run してオペレーションを実行
    axis_x_list = session.run( x_predicts_tsr )     # 損失関数のグラフ化のためのグラフ化のための x 軸の値のリスト 
    output_l2_loss = session.run( l2_loss_op )      # L2 正則化の損失関数の値 （グラフの y 軸値の list）
    output_l1_loss = session.run( l1_loss_op )      # L1 正則化の損失関数の値 （グラフの y 軸値の list）
    
    # TensorBoard 用のファイル（フォルダ）を作成
    # Add summaries to tensorboard
    #merged = tf.summary.merge_all( key='summaries' )
    # tensorboard --logdir=${PWD}
    #summary_writer = tf.summary.FileWriter( "./TensorBoard", graph = session.graph )

    session.close()
    
    #---------------------------------------
    # plot loss functions
    #---------------------------------------
    plt.clf()
 
    # plot L2 loss function
    plt.plot( 
        axis_x_list, output_l2_loss, 
        label='L2 loss',
        linestyle = ':',
        #linewidth = 2,
        color = 'red'
    )
    # plot L1 loss function
    plt.plot( 
        axis_x_list, output_l1_loss, 
        label='L1 loss',
        linestyle = '--',
        #linewidth = 2,
        color = 'blue'
    )
    plt.title( "loss functions" )
    plt.legend( loc = 'best' )
    plt.ylim( [-0.05, 1.0] )
    plt.tight_layout()

    
    MLPlot.saveFigure( fileName = "ProcessingForMachineLearning_TensorFlow_2-1.png" )
    plt.show()


    print("Finish main()")
    return
    

if __name__ == '__main__':
     main()