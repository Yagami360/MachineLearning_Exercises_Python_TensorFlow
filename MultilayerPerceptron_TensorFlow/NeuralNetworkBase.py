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

class NeuralNetworkBase(object):
    """
    ニューラルネットワークの基底クラス（自作クラス）
    TensorFlow ライブラリを使用
    
    ニューラルネットワークの基本的なフレームワークを想定した仮想メソッドからなる抽象クラス。
    実際のニューラルネットワークを表すクラスの実装は、このクラスを継承することで行うことを想定している。
    
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

