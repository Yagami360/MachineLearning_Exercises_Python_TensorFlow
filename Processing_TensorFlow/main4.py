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
    TensorFlow における行列の操作

    """
    print("Enter main()")

    #======================================================================
    # 行列の作成と操作（加算、減算等）
    #======================================================================
    # Reset graph
    ops.reset_default_graph()

    # Session の設定
    session = tf.Session()

    # 各種行列 Tensor の作成
    Identity_matrix = tf.diag( [1.0, 1.0, 1.0] )    # tf.diag(...) : list から対角行列を作成
    A_matrix = tf.truncated_normal( [2, 3] )        # tf.truncated_normal(...) : 
    B_matrix = tf.fill( [2,3], 5.0)                 # 
    C_matrix = tf.random_uniform( [3,2] )           # tf.random_uniform(...) :
    D_matrix = tf.convert_to_tensor(                # tf.convert_to_tensor(...) : numpy.array から Tensor を作成
                   numpy.array( [[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]] )
               )

    print( "Identity_matrix <Tensor型> : ", Identity_matrix )
    print( "A_matrix <Tensor型> : ", A_matrix )
    print( "B_matrix <Tensor型> : ", B_matrix )
    print( "C_matrix <Tensor型> : ", C_matrix )
    print( "D_matrix <Tensor型> : ", D_matrix )

    # Session を run して値を設定後 print 出力
    print( "session.run( Identity_matrix ) :\n", session.run( Identity_matrix ) )
    print( "session.run( A_matrix ) :\n", session.run( A_matrix ) )
    print( "session.run( B_matrix ) :\n", session.run( B_matrix ) )
    print( "session.run( C_matrix ) :\n", session.run( C_matrix ) )
    print( "session.run( D_matrix ) :\n", session.run( D_matrix ) )

    # 行列の加算をして print
    print( "A_matrix + B_marix : \n", session.run( A_matrix + B_matrix) )

    # 行列の減算をして print
    print( "A_matrix - B_marix : \n", session.run( A_matrix - B_matrix) )

    # 行列の乗算をして print
    print( 
        "tf.matmul( B_matrix, Identity_matrix ) : \n", 
        session.run( 
            tf.matmul( B_matrix, Identity_matrix ) 
        ) 
    )

    # TensorBoard 用のファイル（フォルダ）を作成
    #merged = tf.summary.merge_all() # Add summaries to tensorboard
    #summary_writer = tf.summary.FileWriter( "./TensorBoard", graph = session.graph )    # tensorboard --logdir=${PWD}

    session.close()
    

    print("Finish main()")
    return
    

if __name__ == '__main__':
     main()