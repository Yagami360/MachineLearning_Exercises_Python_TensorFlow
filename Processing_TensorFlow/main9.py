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


def main():
    """
    計算グラフでの複数の層の追加、操作
    """
    print("Enter main()")

    #======================================================================
    # 計算グラフでの複数の層の追加、操作
    #======================================================================
    # Reset graph
    ops.reset_default_graph()

    # Session の設定
    session = tf.Session()

    #-------------------------------------------------
    # 各種 通常の変数、Tensor、placeholder の作成
    #--------------------------------------------------

    # 画像のデータを供給するための placeholder
    # 画像の形状 
    # shape[0] : 画像の数
    # shape[1] : 画像の height
    # shape[2] : 画像の width
    # shape[3] : 画像の Channel 数
    image_shape = [ 1, 4, 4, 1 ] 
    image_holder = tf.placeholder( tf.float32, shape = image_shape )
    print( "image_holder :", image_holder )
    
    # ランダムな画像を生成するためのランダム変数
    random_value = numpy.random.uniform( size = image_shape )
    print( "random_value : ", random_value )
    
    # 画像のウインドウのフィルタ値、スライド値
    filer_const = tf.constant( 0.25, shape = [2, 2, 1, 1] )
    stride_value = [1, 2, 2, 1]

    #----------------------------------------------------------------------
    # layer（Opノード）の作成  
    #----------------------------------------------------------------------
    # 画像のウインドウに定数を "畳み込む"（） 関数
    # 画像のウインドウの要素（ピクセル）毎に、指定したフィルタで積を求め、
    # 又、画像のウインドウを上下左右方向にスライドさせる。
    mov_avg_layer_op = tf.nn.conv2d(
                           input = image_holder,       #
                           filter = filer_const,       #
                           strides = stride_value,     #
                           padding = "SAME",           #
                           name = "Moving_Ave_Window"  # 層の名前
                       )
    
    print( "filer_const : ", filer_const )
    print( "mov_avg_layer_op : ", mov_avg_layer_op )

    # 画像のウインドウの移動平均の２×２出力を行うカスタム層を作成する。
    # 練習用コードの可読性のため、main() 関数内にて関数定義する。
    def costom_layer( input_matrix ):
        retun
    
    #----------------------------------------------------------------------
    # 計算グラフの実行
    #----------------------------------------------------------------------
    output1 = session.run( mov_avg_layer_op, feed_dict = { image_holder : random_value } )

    print( "session.run( mov_avg_layer_op, feed_dict = { image_holder : random_value } ) : \n", output1 )   


    # TensorBoard 用のファイル（フォルダ）を作成
    # Add summaries to tensorboard
    #merged = tf.summary.merge_all( key='summaries' )
    # tensorboard --logdir=${PWD}
    #summary_writer = tf.summary.FileWriter( "./TensorBoard", graph = session.graph )

    session.close()
    

    print("Finish main()")
    return
    

if __name__ == '__main__':
     main()