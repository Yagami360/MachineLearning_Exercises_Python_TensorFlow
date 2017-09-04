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
    TensorFlow を用いた基本的な処理のフレームワーク
    """
    print("Enter main()")

    #======================================================================
    # テンソルを作成
    #======================================================================
    #hello_tsr = tf.constant( "Hello, TensorFlow!" )

    # 固定のテンソルを作成
    zero_tsr = tf.zeros( [3, 2] )
    
    # Session を生成せず、そのまま Tensor を print()
    print( zero_tsr )                   # <出力> 
                                        # Tensor("zeros:0", shape=(3, 2), dtype=float32)
                                        # Tensor オブジェクトが出力されている。
                                        # これは session を実行していないため。

    # Session を生成＆run() して、Tensor を print()
    session = tf.Session()
    print( session.run( zero_tsr ) )    # <出力>
                                        # [[ 0.  0.]
                                        #  [ 0.  0.]
                                        #  [ 0.  0.]
                                        # 全ての要素が 0 からなるテンソル（この場合、２次配列）が出力されている。
    

    #======================================================================
    # 変数とプレースホルダを設定
    #======================================================================
    
    
    print("Finish main()")
    return
    

if __name__ == '__main__':
     main()