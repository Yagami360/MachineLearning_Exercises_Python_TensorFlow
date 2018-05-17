# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境 
# + TensorFlow 1.4.0 インストール済み
# + OpenCV 3.3.1 インストール済み

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


class DefaultBox( object ):
    """
    SSD [Single Shot muitibox Detector] で使用する、デフォルトボックスを表すクラス。
    Bouding Box is the result of comparison with default box.
    bouding box has loc (position) and label (index).

    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
        _group_id : int
            デフォルトボックス群の識別ID
        _id : int
            デフォルトボックスの識別ID

        _center_x : float
            デフォルトボックスの中心位置座標 x (0.0~1.0)
        _center_y : float
            デフォルトボックスの中心位置座標 y (0.0~1.0)

        _width : float
            デフォルトボックスの幅
        _height : float
            デフォルトボックスの高さ

        _scale : float
            デフォルトボックスのスケール値
        _aspect : float
            デフォルトボックスの縦横のアスペクト比

    [protedted] protedted な使用法を想定 

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( 
            self,
            group_id = None,
            id = None,
            center_x = 0.0,
            center_y = 0.0,
            width = 1,
            height = 1,
            scale = 1,
            aspect = 1
        ):

        self._group_id = group_id        
        self._id = id

        self._center_x = center_x
        self._center_y = center_y

        self._width = width
        self._height = height

        self._scale = scale
        self._aspect = aspect

        return


    def print( self, str = None ):
        print( "----------------------------------" )
        print( str )
        print( self )

        print( "_group_id : %d, _id : %d" % ( self._group_id, self._id ) )
        print( "_center_x : %0.5f, _center_y : %0.5f" % ( self._center_x, self._center_y ) )
        print( "_width : %0.5f, _height : %0.5f" % ( self._width, self._height ) )
        print( "_scale : %0.5f, _aspect : %0.5f" % ( self._scale, self._aspect ) )

        print( "----------------------------------" )

        return


    def draw_rect( self, image, color = (0,0,255), thickness = 1 ):
        """
        デフォルトボックスの長方形を描写する。
        """
        center_x = image.shape[0] * self._center_x - 0.5
        center_y = image.shape[1] * self._center_y - 0.5
        width = image.shape[0] * self._width * self._scale * (1 / np.sqrt( self._aspect ) )
        height = image.shape[1] * self._height * self._scale * np.sqrt( self._aspect )

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


class DefaultBoxSet( object ):
    """
    一連のデフォルトボックス群を表すクラス。

    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
        _n_fmaps : int
            特徴マップの数
        _scale_min : float
            最下位レイヤーのスケール値
        _scale_max : float
            最上位レイヤーのスケール値

        _default_boxes : list <DefaultBox>
            一連のデフォルトボックスのリスト

    [protedted] protedted な使用法を想定 


    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__(
            self,
            scale_min = 0.1,
            scale_max = 0.9
        ):

        self._scale_min = scale_min
        self._scale_max = scale_max
        self._n_fmaps = None
        self._default_boxes = []

        return


    def print( self, str = None ):
        print( "----------------------------------" )
        print( str )
        print( self )

        print( "_scale_min :", self._scale_min )
        print( "_scale_max :", self._scale_max )
        print( "_n_fmaps :", self._n_fmaps )

        print( "_default_boxes :", self._default_boxes )
        if( self._default_boxes != [] ):
            for i in range( len(self._default_boxes) ):
                self._default_boxes[i].print()

        print( "----------------------------------" )

        return


    def calc_scale( self, k ):
        """
        BBOX の形状回帰のためのスケール値を計算する。
        具体的には、各特徴マップ k (=1~6) についてのデフォルトボックスのスケール s_k は、以下のようにして計算される。
        s_k = s_min + (s_max - s_min) * (k - 1.0) / (m - 1.0), m = 6

        [Input]
            k : int
                特徴マップ fmap の番号。1 ~ self._n_fmaps の間の数
        [Output]
            s_k : float
                指定された番号の特徴マップのスケール値

        """
        s_k = self._scale_min + ( self._scale_max - self._scale_min ) * k / ( self._n_fmaps - 1.0 )
        
        return s_k


    def generate_boxes( self, fmaps_shapes, aspect_set ):
        """
        generate default boxes based on defined number
        
        [Input]
            fmaps_shapes : nadarry( [][] )
                extra feature map の形状（ピクセル単位）
                the shape is [  first-feature-map-boxes ,
                                second-feature-map-boxes ,
                                    ...
                                sixth-feature-map-boxes , ]
                    ==> ( total_boxes_number x defined_size )
                
                feature map sizes per output such as...
                [ 
                    [ None, 19, 19, ],      # extra feature-map-shape 1 [batch_size, fmap_height, fmap_width]
                    [ None, 19, 19, ],      # extra feature-map-shape 2
                    [ None, 10, 10 ],       # extra feature-map-shape 3
                    [ None, 5, 5, ],        # extra feature-map-shape 4
                    [ None, 3, 3, ],        # extra feature-map-shape 5
                    [ None, 1, 1, ],        # extra feature-map-shape 6
                ]

            aspect_set : nadarry( [][] )
                extra feature map に対してのアスペクト比
                such as...
                [1.0, 1.0, 2.0, 1.0/2.0],                   # extra feature map 1
                [1.0, 1.0, 2.0, 1.0/2.0, 3.0, 1.0/3.0],     # extra feature map 2
                [1.0, 1.0, 2.0, 1.0/2.0, 3.0, 1.0/3.0],
                [1.0, 1.0, 2.0, 1.0/2.0, 3.0, 1.0/3.0],
                [1.0, 1.0, 2.0, 1.0/2.0, 3.0, 1.0/3.0],
                [1.0, 1.0, 2.0, 1.0/2.0, 3.0, 1.0/3.0],

        [Output]
            self._default_boxes : list<DefaultBox>
                generated default boxes list

        """
        self._n_fmaps = len( fmaps_shapes )

        id = 0
        for k, fmap_shape in enumerate( fmaps_shapes ):
            s_k = self.calc_scale( k )

            fmap_width  = fmap_shape[1]
            fmap_height = fmap_shape[2]
            
            aspects = aspect_set[k]

            for aspect in aspects:

                # 特徴マップのセルのグリッド（1 pixcel）に関してのループ処理
                for y in range( fmap_height ):
                    # セルのグリッドの中央を 0.5 として計算 
                    center_y = ( y + 0.5 ) / float( fmap_height )

                    for x in range( fmap_width ):
                        center_x = ( x + 0.5 ) / float( fmap_width )

                        box_width = s_k * np.sqrt( aspect )
                        box_height = s_k / np.sqrt( aspect )

                        id += 1
                        default_box = DefaultBox(
                                          group_id = k + 1,
                                          id = id,
                                          center_x = center_x, center_y = center_y,
                                          width = box_width, height = box_height, 
                                          scale = s_k,
                                          aspect = aspect
                                      )

                        self.add_default_box( default_box )

        return self._default_boxes


    def add_default_box( self, default_box ):
        """
        引数で指定されたデフォルトボックスを、一連のデフォルトボックスのリストに追加する。

        [Input]
            default_box : DefaultBox
                デフォルトボックスのクラス DefaultBox のオブジェクト

        """
        self._default_boxes.append( default_box )

        return


    def draw_rects( self, image, group_id = 1 ):
        """
        指定されたグループ ID の各デフォルトボックスの長方形を描写する。

        """
        if( self._default_boxes == [] ):
            return

        colors_map = [ 
                         (0,0,255),     # 特徴マップ１（グループID１）の色 BGR（赤）
                         (0,255,0),     # 特徴マップ２（グループID２）の色 BGR（）
                         (255,0,0), 
                         (0,0,0), 
                         (0,0,0), 
                         (0,0,0)
                     ]


        for i, default_box in enumerate( self._default_boxes ):
            if( default_box._group_id == group_id ):
                image = default_box.draw_rect( image, color = colors_map[ group_id ], thickness = 1 )

        return image
