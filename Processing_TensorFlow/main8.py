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
    計算グラフでの入れ子の演算の階層化
    """
    print("Enter main()")

    #======================================================================
    # 計算グラフでの入れ子の演算の階層化
    #======================================================================
    # Reset graph
    ops.reset_default_graph()

    # Session の設定
    session = tf.Session()

    # 各種 通常の変数、Tensor、placeholder の作成
    dim3_list = numpy.array( 
                    [
                        [ 1.,  3., 5,  7,  9.], 
                        [-2.,  0., 2., 4., 6.],
                        [-6., -3., 0., 3., 6.]
                    ]
                )
    
    # 後の for ループでの処理のため、
    # 入力を２つ（２回ループ）にするため、配列を複製
    values = numpy.array( [dim3_list, dim3_list + 1] )

    # 3*5 のデータを供給するための placeholder
    dim3_holder = tf.placeholder( tf.float32, shape = (3,5) )
    
    # 行列の演算用のConst 値
    const1 = tf.constant( [ [1.],[0.],[-1.],[2.],[4.] ] )
    const2 = tf.constant( [ [2.] ] )
    const3 = tf.constant( [ [10.] ] )

    print( "const1 : ", const1 )
    print( "const2 : ", const2 )
    print( "const3 : ", const3 )

    # オペレーション（Opノード）の作成＆連結
    matmul_op1 = tf.matmul( dim3_holder, const1 )
    matmul_op2 = tf.matmul( matmul_op1, const2 )
    add_op3 = tf.add( matmul_op2, const3 )

    print( "tf.matmul( dim3_holder, const1 ) : ", matmul_op1 )
    print( "tf.matmul( matmul_op1, const2 ) : ", matmul_op2 )
    print( "tf.add( matmul_op2, const1 ) : ", add_op3 )

    # 入力値の list を for ループして、list の各値に対し、
    # 構築した計算グラフのオペレーション実行
    for value in values:
        # Session を run して、
        # Placeholder から feed_dict を通じて、データを供給しながら、オペレーション実行
        output = session.run( add_op3, feed_dict = { dim3_holder : value } )
        
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