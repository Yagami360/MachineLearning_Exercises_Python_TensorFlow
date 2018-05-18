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

# openCV ライブラリ
import cv2


class BoundingBox( object ):
    """
    SSD [Single Shot muitibox Detector] で使用する、バウンディングボックスを表すクラス。
    Bouding Box is the result of comparison with default box.
    bouding box has loc (position) and class's label.

    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
        _label : int
            バウンディングボックスの所属クラスのクラスラベル

        _rect_loc : list<float>
            バウンディングボックスの長方形座標。
            [ center_x, center_y, width, height ]

    [protedted] protedted な使用法を想定 

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( self, label, rect_loc ):
        self._label = label
        self._rect_loc = rect_loc

        return

    """
    def __init__( self, label, center_x, center_y, height, width ):
        self._label = label
        self._rect_loc = [ center_x, center_y, height, width ]

        return
    """

    def print( self, str = None ):
        print( "----------------------------------" )
        print( str )
        print( self )

        print( "_label :", self._label )
        print( "_rect_loc :", self._rect_loc )

        print( "----------------------------------" )

        return


    def draw_rect( self, image, color = (0,0,255), thickness = 1 ):
        """
        バウンディングボックスの長方形を描写する。
        """
        center_x = image.shape[0] * self._rect_loc[0] - 0.5
        center_y = image.shape[1] * self._rect_loc[1] - 0.5
        height = image.shape[1] * self._rect_loc[2]
        width = image.shape[0] * self._rect_loc[3]

        point1_x = int( center_x - width/2 )   # 長方形の左上 x 座標
        point1_y = int( center_y - height/2 )  # 長方形の左上 y 座標
        point2_x = int( center_x + width/2 )   # 長方形の右下 x 座標
        point2_y = int( center_y + height/2 )  # 長方形の右下 y 座標

        image = cv2.rectangle(
                    img = image,
                    pt1 = ( point1_x, point1_y ),  # 長方形の左上座標
                    pt2 = ( point2_x, point2_y ),  # 長方形の右下座標
                    color = color,                 # BGR
                    thickness = thickness          # 線の太さ（-1 の場合、color で設定した色で塗りつぶし）
                )

        return image