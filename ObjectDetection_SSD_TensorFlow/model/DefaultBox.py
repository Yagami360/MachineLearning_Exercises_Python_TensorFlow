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
            デフォルトボックスの中心位置座標 x
        _center_y : float
            デフォルトボックスの中心位置座標 y

        _width : int
            デフォルトボックスの幅（ピクセル数）
        _height : int
            デフォルトボックスの高さ（ピクセル数）

        _scale : float
            デフォルトボックスのスケール値

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
            scale = 1
        ):

        self._group_id = group_id        
        self._id = id

        self._center_x = center_x
        self._center_y = center_y

        self._width = width
        self._height = width

        self._scale = scale

        return


    def print( self, str = None ):
        print( "----------------------------------" )
        print( str )
        print( self )

        print( "_group_id :", self._group_id )
        print( "_id :", self._id )

        print( "_center_x :", self._center_x )
        print( "_center_y :", self._center_y )
        print( "_width :", self._width )
        print( "_height :", self._height )
        print( "_scale :", self._scale )

        print( "----------------------------------" )

        return




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

        # 各デフォルトボックスのアスペクト比
        self.aspects = [ 1.0, 2.0, 3.0, 1.0/2.0, 1.0/3.0 ]

        # 各特徴マップに対応した一連のデフォルトボックスを生成
        # extra feature maps
        self._fmap_shapes = [
                          [ 19, 19, ],      # feature-map-shape 1
                          [ 10, 10 ],       # feature-map-shape 3
                          [ 5, 5, ],        # feature-map-shape 4
                          [ 3, 3, ],        # feature-map-shape 5
                          [ 1, 1, ],        # feature-map-shape 6
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
        for k, map_shape in enumerate( fmap_shapes ):
            s_k = self.calc_scale( k )
            aspect = self.aspects[k]
            
            width  = s_k * np.sqrt( aspect )
            height = s_k / np.sqrt( aspect )

            x = float( fmap_shapes[k][0] )
            y = float( fmap_shapes[k][1] )

            center_x = ( x + 0.5 ) / float( width )
            center_y = ( y + 0.5 ) / float( height )

            default_box = DefaultBox(
                              group_id = 1,
                              id = k + 1,
                              center_x = center_x, center_y = center_y,
                              width = width, height = height, 
                              scale = aspect
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
