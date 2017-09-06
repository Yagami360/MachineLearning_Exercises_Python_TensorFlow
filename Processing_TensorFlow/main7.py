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
from tensorflow.python.framework import ops     # ?


def main():
    """
    計算グラフでの演算（オペレーション、Opノード）の設定、実行
    """
    print("Enter main()")

    #======================================================================
    # 計算グラフでの演算（オペレーション、Opノード）の設定、実行
    #======================================================================
    # Reset graph
    ops.reset_default_graph()

    # Session の設定
    session = tf.Session()

    # 各種 通常の変数、Tensor、placeholder の作成
    float_list = numpy.array( [1., 3., 5, 7, 9.] )
    float_holder = tf.placeholder( tf.float32 )
    float_const = tf.constant( 3. )

    # オペレーション（Opノード）の作成
    multipy_op = tf.multiply( float_holder, float_const )

    # 入力値の list を for ループして、list の各値に対し、オペレーション実行
    for value in float_list:
        # Session を run して、
        # 計算グラフに追加した placeholder をfeed_dict を通じて、オペレーション実行
        output = session.run( multipy_op, feed_dict = { float_holder: value } )
        
        # Session を run した結果（Output）を print 出力
        print( output )    


    # TensorBoard 用のファイル（フォルダ）を作成
    # Add summaries to tensorboard
    #merged = tf.summary.merge_all()
    # tensorboard --logdir=${PWD}
    #summary_writer = tf.summary.FileWriter( "./TensorBoard", graph = session.graph )

    session.close()
    

    print("Finish main()")
    return
    

if __name__ == '__main__':
     main()