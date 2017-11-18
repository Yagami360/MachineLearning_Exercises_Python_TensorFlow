# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境 (TensorFlow インストール済み)

"""
    更新情報
    [17/10/21] : 新規作成
    [17/11/18] : NNActivation クラスの仕様変更。NNActivation クラスの子クラス定義
               : 
"""

import numpy

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops


class NNActivation( object ):
    """
    ニューラルネットワークの活性化関数を表すクラス
    実際の活性化関数を表すクラスの実装は、このクラスを継承し、オーバーライドするを想定している。
    ------------------------------------------------------------------------------------------------
    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
        _activate_op : 活性化関数の Operater
        _node_name : str
             この Operator ノードの名前
             
    [protedted] protedted な使用法を想定 
        
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( self, node_name = "Activation_op" ):
        self._activate_op = None
        self._node_name = node_name
        
        return

    def print( str ):
        print( "NNActivation" )
        print( self )
        print( str )
        print( "_activate_op :", self._activate_op )
        print( "_node_name :", self._node_name )

        return

    def activate( self, input, node_name = "Activation_op" ):
        """
        活性化関数のオペレーターを返す。

        [Input]
            input : Operator or placeholder
                活性化関数の入力となる Operatore、又は placeholder

        [Output]
            活性化関数のオペレーター
        """
        return self._activate_op


class Sigmoid( NNActivation ):
    """
    シグモイド関数の活性化関数
    NNActivation クラスの子クラスとして定義
    """
    def __init__( self, node_name = "Activate_Sigmoid_op" ):
        self._activate_op = None
        self._node_name = node_name

        return

    def activate( self, input, node_name = "Activate_Sigmoid_op" ):
        self._activate_op = tf.nn.sigmoid( input, name = node_name )
        self._node_name = node_name

        return self._activate_op


class Relu( NNActivation ):
    """
    Relu 関数の活性化関数
    NNActivation クラスの子クラスとして定義
    """
    def __init__( self, node_name = "Activate_Relu_op" ):
        self._activate_op = None
        self._node_name = node_name

        return

    def activate( self, input, node_name = "Activate_Relu_op" ):
        self._activate_op = tf.nn.relu( input, name = node_name )
        self._node_name = node_name

        return self._activate_op


class Softmax( NNActivation ):
    """
    softmax 関数の活性化関数
    NNActivation クラスの子クラスとして定義
    """
    def __init__( self, node_name = "Activate_Softmax_op" ):
        self._activate_op = None
        self._node_name = node_name

        return

    def activate( self, input, node_name = "Activate_Softmax_op" ):
        self._activate_op = tf.nn.softmax( input, name = node_name )
        self._node_name = node_name

        return self._activate_op

