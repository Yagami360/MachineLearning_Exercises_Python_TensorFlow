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
    変数とプレースホルダの設定、及び計算グラフとセッション
    TensorBoard でのコードの結果の計算グラフを可視化
    """
    print("Enter main()")

    #======================================================================
    # 変数の設定からOpノード、計算グラフの構築
    #======================================================================
    # Reset graph
    ops.reset_default_graph()

    # Session の設定
    session = tf.Session()

    # Tensor の設定
    zeros_tsr = tf.zeros( [3, 2] )
    
    # Variable の設定
    zeros_var = tf.Variable( zeros_tsr )
    print( "tf.Variable() :", zeros_var )
    
    # Opノード [op : operation] の作成 変数の初期化
    init_op = tf.global_variables_initializer()
    print( "tf.global_variables_initializer() :\n", init_op )

    # Session を run
    output = session.run( init_op )
    print( "session.run( init_op ) :\n", output )
    
    # TensorBoard 用のファイル（フォルダ）を作成
    merged = tf.summary.merge_all() # Add summaries to tensorboard
    summary_writer = tf.summary.FileWriter( "./TensorBoard", graph = session.graph )    # tensorboard --logdir=${PWD}

    #session.close()
        
    #======================================================================
    # 変数の設定からOpノード、計算グラフの構築
    #======================================================================

    print("Finish main()")
    return
    

if __name__ == '__main__':
     main()