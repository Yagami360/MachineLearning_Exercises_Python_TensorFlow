# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境 (TensorFlow インストール済み)

"""
    更新情報
    [17/10/14] : 新規作成
    [17/xx/xx] : 
               : 
"""

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops

import numpy

# 親クラス（自作クラス）
from NeuralNetworkBase import NeuralNetworkBase


class MultilayerPerceptron( NeuralNetworkBase ):
    """
    多層パーセプトロンを表すクラス（自作クラス）
    ニューラルネットワークの基底クラス NeuralNetworkBase （自作の基底クラス）を継承している。
    
    ----------------------------------------------------------------------------------------------------

    [protedted] protedted な使用法を想定 

    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """

    def __init__( self ):
        """
        コンストラクタ（厳密にはイニシャライザ）
        """

        return