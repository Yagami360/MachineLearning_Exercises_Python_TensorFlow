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
    ニューラルネットにおける活性化関数の実装
    """
    print("Enter main()")

    #======================================================================
    # ニューラルネットにおける活性化関数の実装
    #======================================================================
    # 描写する x 軸の値の list
    axis_x_list = numpy.linspace( start = -10., stop = 10., num = 100 )
    #print( "axis_x_list :\n", axis_x_list )
    
    # Reset graph
    ops.reset_default_graph()

    # Session の設定
    session = tf.Session()

    # 活性化関数のオペレーション作成
    relu_op = tf.nn.relu( features = axis_x_list )
    relu6_op = tf.nn.relu6( features = axis_x_list )
    softplus_op = tf.nn.softplus( features = axis_x_list )
    elu_op = tf.nn.elu( features = axis_x_list )

    print( "tf.nn.relu(...) :\n", relu_op )
    print( "tf.nn.relu6(...) :\n", relu6_op )
    print( "tf.nn.softplus(...) :\n", softplus_op )
    print( "tf.nn.elu(...) :\n", elu_op )

    # Session を run してオペレーションを実行
    output_relu = session.run( relu_op )
    output_relu6 = session.run( relu6_op )
    output_softplus = session.run( softplus_op )
    output_elu = session.run( elu_op )

    #print( "session.run( relu_op ) :\n", output_relu )
    
    # TensorBoard 用のファイル（フォルダ）を作成
    # Add summaries to tensorboard
    #merged = tf.summary.merge_all( key='summaries' )
    # tensorboard --logdir=${PWD}
    #summary_writer = tf.summary.FileWriter( "./TensorBoard", graph = session.graph )

    session.close()
    
    #---------------------------------------
    # plot activate functions 1
    #---------------------------------------
    plt.clf()
 
    # plot Relu function
    plt.subplot( 2, 2, 1 )
    plt.plot( 
        axis_x_list, output_relu, 
        label='ReLU' 
        #linestyle = ':',
        #linewidth = 2,
        #color = 'red'
    )
    plt.title( "Relu [Rectified Liner Unit] \n activate function" )
    plt.legend( loc = 'best' )
    plt.ylim( [-1.50, 10.0] )
    #plt.tight_layout()

    # plot Relu6 function
    plt.subplot( 2, 2, 2 )
    plt.plot( 
        axis_x_list, output_relu6, 
        label='ReLU6'
        #linestyle = '--',
        #linewidth = 2,
        #color = 'blue'
    )
    plt.title( "Relu6 \n activate function" )
    plt.legend( loc = 'best' )
    plt.ylim( [-1.50, 10.0] )
    #plt.tight_layout()

    # plot softplus function
    plt.subplot( 2, 2, 3 )
    plt.plot( 
        axis_x_list, output_softplus, 
        label='softplus'
        #linestyle = '--',
        #linewidth = 2,
        #color = 'blue'
    )
    plt.title( "softplus \n activate function" )
    plt.legend( loc = 'best' )
    plt.ylim( [-1.50, 10.0] )
    #plt.tight_layout()

    # plot ELU function
    plt.subplot( 2, 2, 4 )
    plt.plot( 
        axis_x_list, output_elu, 
        label='ELU'
        #linestyle = '--',
        #linewidth = 2,
        #color = 'blue'
    )
    plt.title( "ELU [Exponetial Liner Unit] \n activate function" )
    plt.legend( loc = 'best' )
    plt.ylim( [-1.50, 10.0] )
    #plt.tight_layout()



    MLPlot.saveFigure( fileName = "ProcessingForMachineLearning_TensorFlow_1-1.png" )
    plt.show()

    #---------------------------------------
    # plot activate functions 2
    #---------------------------------------
    plt.clf()

    # plot sigmoid function
    plt.subplot( 2, 2, 1 )

    MLPlot.saveFigure( fileName = "ProcessingForMachineLearning_TensorFlow_1-2.png" )
    #plt.show()


    print("Finish main()")
    return
    

if __name__ == '__main__':
     main()