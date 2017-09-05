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
    TensorFlow における演算（オペレーション、Opノード）の設定、実行処理
    """
    print("Enter main()")

    #======================================================================
    # 演算（オペレーション、Opノード）の設定、実行
    #======================================================================
    #----------------------------------------------------------------------
    # 単一の演算（オペレーション、Opノード）
    #----------------------------------------------------------------------
    # Reset graph
    ops.reset_default_graph()

    # Session の設定
    session = tf.Session()

    # オペレーション div 
    # 何種類かの 割り算演算 div の動作を確認する。
    div_op      = tf.div( 3, 4 )        # tf.div(x,y) : 整数演算での div
    truediv_op  = tf.truediv( 3, 4 )    # tf.truediv(x,y) : 浮動小数点数に対しての div
    floordiv_op = tf.floordiv( 3, 4 )   # 浮動小数点数であるが、整数での演算を行いたい場合の div

    # Session を run してオペレーションを実行後 print 出力
    # 何種類かの 割り算演算 div の動作を確認する。
    print( "session.run( div_op ) :\n", session.run( div_op ) )
    print( "session.run( truediv_op ) :\n", session.run( truediv_op ) )
    print( "session.run( truediv_op ) :\n", session.run( floordiv_op ) )
    

    # TensorBoard 用のファイル（フォルダ）を作成
    #merged = tf.summary.merge_all() # Add summaries to tensorboard
    #summary_writer = tf.summary.FileWriter( "./TensorBoard", graph = session.graph )    # tensorboard --logdir=${PWD}

    session.close()
    
    #----------------------------------------------------------------------
    # 複数の演算（オペレーション、Opノード）の組み合わせ
    #----------------------------------------------------------------------
    # Reset graph
    ops.reset_default_graph()

    # Session の設定
    session = tf.Session()

    # オペレーションの組み合わせ 
    comb_tan_op = tf.div( 
                      tf.sin( 3.1416/4. ), 
                      tf.cos( 3.1416/4. ) 
                  )

    cusmom_polynormal_op = cusmom_polynormal( x = 10 )
    
    # Session を run してオペレーションを実行後 print 出力
    output1 = session.run( comb_tan_op )
    output2 = session.run( cusmom_polynormal_op )

    print( "session.run( comb_tan_op ) : ", output1 )
    print( "session.run( cusmom_polynormal_op ) : ", output2 )

    session.close()


    print("Finish main()")
    return
    
def cusmom_polynormal( x ):
    '''
    f(x) = 3 * x^2 - x + 10
    '''
    cusmom_polynormal_op = ( tf.subtract( 3*tf.square(x), x) + 10 )
    return cusmom_polynormal_op


if __name__ == '__main__':
     main()