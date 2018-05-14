# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境 (TensorFlow 1.4.0 インストール済み)

"""
    更新情報
    [18/05/14] : 新規作成
    [xx/xx/xx] : 

"""

import numpy as np

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops


class BoundingBox( object ):
    """
    SSD [Single Shot muitibox Detector] で使用する、バウンディングボックスを表すクラス。
    Bouding Box is the result of comparison with default box.
    bouding box has loc (position) and class's label.

    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
        _label : int
            バウンディングボックスの所属クラスのクラスラベル

        _position : list<float?>
            バウンディングボックスの中心位置座標。
            x,y 座標 ?

    [protedted] protedted な使用法を想定 

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( self, label, position ):
        self._label = label
        self._position = position

        return


    def print( self, str = None ):
        print( "----------------------------------" )
        print( str )
        print( self )

        print( "_label :", self._label )
        print( "_position :", self._position )

        print( "----------------------------------" )

        return
