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
    テンソルの設定、及び計算グラフとセッション 
    """
    print("Enter main()")


    #======================================================================
    # テンソルの設定
    #======================================================================
    #--------------------------------------
    # 固定のテンソルを作成
    #--------------------------------------
    # tf.zeros(...) : 全ての要素が 0 からなる Tensor を作成する
    zeros_tsr = tf.zeros( [3, 2] )
    
    # Session を生成せず、そのまま Tensor を print()
    print(  " tf.zeros(...) の Tensor 型 : \n", zeros_tsr )   # <出力> 
                                                              # Tensor("zeros:0", shape=(3, 2), dtype=float32)
                                                              # Tensor オブジェクトが出力されている。
                                                              # これは session を実行していないため。

    # Session を生成＆run() して、Tensor を print()
    session = tf.Session()
    print( session.run( zeros_tsr ) )   # <出力>
                                        # [[ 0.  0.]
                                        #  [ 0.  0.]
                                        #  [ 0.  0.]
                                        # 全ての要素が 0 からなる Tensor（この場合、２次配列）が出力されている。
    
    session.close()


    # tf.ones(...) : 全ての要素が 1 からなる Tensor を作成する
    ones_tsr = tf.ones( [3, 2] )
    # Session を生成せず、そのまま Tensor を print()
    print(  " tf.ones(...) の Tensor 型 : ", ones_tsr )

    # Session を生成＆run() して、Tensor を print()
    session = tf.Session()
    print( "tf.ones(...) の value : ", session.run( ones_tsr ) )    # <出力>
                                                                    # [[ 1.  1.]
                                                                    #  [ 1.  1.]
                                                                    #  [ 1.  1.]
    session.close()


    # tf.fill(...) : 指定した定数で埋められた Tensor を作成する。
    filled_tsr = tf.fill( [3, 2], "const" )
    print( "tf.fill(...) の Tensor 型 : ", filled_tsr )

    session = tf.Session()
    print( "tf.fill(...) の value : ", session.run( filled_tsr ) )
    session.close()


    # tf.constant(...) : 指定した既存の定数から Tensor を作成する。
    const_tsr = tf.constant( [1,2,3] )
    print( "tf.constant(...) の Tensor 型 : ", const_tsr )

    session = tf.Session()
    print( "tf.constant(...) の value \n: ", session.run( const_tsr ) )
    session.close()


    #--------------------------------------
    # シーケンステンソルを作成
    #--------------------------------------
    # tf.linspace(...) : stop 値のあるシーケンステンソル
    liner_tsr = tf.linspace( start = 0.0, stop = 1.0, num = 3 )
    print( "tf.linspace(...) の Tensor 型 : ", liner_tsr )
    
    session = tf.Session()
    print( "tf.linspace(...) の value : \n", session.run( liner_tsr ) )
    session.close()

    # tf.range(...) : stop 値のないシーケンステンソル
    int_seq_tsr = tf.range( start = 1, limit = 15, delta = 3 )
    print( "tf.range(...) の Tensor 型 : ", int_seq_tsr )
    
    session = tf.Session()
    print( "tf.range(...) の value : \n", session.run( int_seq_tsr ) )
    session.close()

    #--------------------------------------
    # ランダムテンソルを作成
    #--------------------------------------



    print("Finish main()")
    return
    

if __name__ == '__main__':
     main()