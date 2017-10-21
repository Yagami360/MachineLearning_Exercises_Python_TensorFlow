# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境 (TensorFlow インストール済み)

"""
    更新情報
    [17/10/21] : 新規作成
    [17/xx/xx] : 
               : 
"""

import numpy

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops


class NNActivation(object):
    """
    ニューラルネットワークの活性化関数を表すクラス

    ------------------------------------------------------------------------------------------------
    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
        _activate_type : str
            活性化関数の種類
            "original" : 独自の活性化関数を指定する場合に設定
            "sigmoid" : シグモイド関数 tf.nn.sigmoid(...)
            "relu" : Relu 関数 tf.nn.relu(...)
            "softmax" : softmax 関数 tf.nn.softmax(...)

    [protedted] protedted な使用法を想定 

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """

    def __init__( self, activate_type = "sigmoid" ):
        self._activate_type = activate_type

        return

    def activate( self, input, original_activate_op = None ):
        """
        指定されている activate_type に応じた、
        活性化関数のオペレーターを返す。

        [Input]
            input : Operator or placeholder
                活性化関数の入力となる Operatore、又は placeholder

            original_activate_op : Operator (Default : None)
                外部で定義した独自の活性化関数を指定したい場合に設定

        [Output]
            指定されている activate_type に応じた、活性化関数のオペレーター
        """
        # 外部で定義した独自の活性化関数
        if ( self._activate_type == "original" ):
            activate_op = original_activate_op
        # シグモイド関数
        elif ( self._activate_type == "sigmoid" ):
            activate_op = tf.nn.sigmoid( input )
        # Relu 関数
        elif ( self._activate_type == "relu" ):
            activate_op = tf.nn.relu( input )
        # softmax 関数
        elif ( self._activate_type == "softmax" ):
            activate_op = tf.nn.softmax( input )
        else:
            activate_op = tf.nn.sigmoid( input )

        return activate_op

