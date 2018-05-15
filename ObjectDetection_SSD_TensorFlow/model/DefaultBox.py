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


class DefaultBoxes( object ):
    """
    一連のデフォルトボックス群を表すクラス。

    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
        _group_id : int
            デフォルトボックス群の識別ID

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
            group_id = None,
            n_fmaps = 5,
            scale_min = 0.2,
            scale_max = 0.9
        ):

        self._group_id = group_id
        self._n_fmaps = n_fmaps
        self._scale_min = scale_min
        self._scale_max = scale_max
        self._default_boxes = []

        # 各特徴マップに対応した、各デフォルトボックスのアスペクト比
        self.aspects = [ 1.0, 2.0, 3.0, 1.0/2.0, 1.0/3.0 ]

        # 各特徴マップに対応した一連のデフォルトボックスを生成
        # extra feature maps
        self._fmap_shapes = [
                          [ 19, 19, ],      # feature-map-shape 1 [width, height]
                          [ 10, 10 ],       # feature-map-shape 2
                          [ 5, 5, ],        # feature-map-shape 3
                          [ 3, 3, ],        # feature-map-shape 4
                          [ 1, 1, ],        # feature-map-shape 5
                      ]

        """
        fmaps = []
        fmaps.append( convolution(self.base, 'fmap1') )
        fmaps.append( convolution(self.conv7, 'fmap2') )
        fmaps.append( convolution(self.conv8_2, 'fmap3') )
        fmaps.append( convolution(self.conv9_2, 'fmap4') )
        fmaps.append( convolution(self.conv10_2, 'fmap5') )
        fmaps.append( convolution(self.conv11_2, 'fmap6') )

        fmap_shapes = [map.get_shape().as_list() for map in fmaps]
        """

        self.generate_boxes( self._fmap_shapes )

        return


    def print( self, str = None ):
        print( "----------------------------------" )
        print( str )
        print( self )

        print( "_group_id :", self._group_id )
        print( "_n_fmaps :", self._n_fmaps )
        print( "_scale_min :", self._scale_min )
        print( "_scale_max :", self._scale_max )
        
        print( "_fmap_shapes :", self._fmap_shapes )

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


    def generate_boxes( self, fmap_shapes ):
        """
        generate default boxes based on defined number
        
        [Input]
            fmap_shapes : list
                the shape is [  first-feature-map-boxes ,
                                second-feature-map-boxes ,
                                    ...
                                sixth-feature-map-boxes , ]
                    ==> ( total_boxes_number x defined_size )
                
                feature map sizes per output such as...
                [ 
                    [ 19, 19, ],      # feature-map-shape 1
                    [ 19, 19, ],      # feature-map-shape 2
                    [ 10, 10 ],       # feature-map-shape 3
                    [ 5, 5, ],        # feature-map-shape 4
                    [ 3, 3, ],        # feature-map-shape 5
                    [ 1, 1, ],        # feature-map-shape 6
                ]

        [Output]
            self._default_boxes : list<DefaultBox>
                generated default boxes list

        """
        id = 0
        for k, map_shape in enumerate( fmap_shapes ):
            s_k = self.calc_scale( k )

            for i, aspect in enumerate( self.aspects ):
                fmap_width  = fmap_shapes[k][0]
                fmap_height = fmap_shapes[k][1]

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
        #self._n_fmaps += 1
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
