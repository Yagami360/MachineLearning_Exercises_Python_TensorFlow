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

# 自作モジュール
from model.NeuralNetworkBase import NeuralNetworkBase
from model.BaseNetwork import BaseNetwork
from model.BaseNetwork import BaseNetworkVGG16
from model.BaseNetwork import BaseNetworkResNet

from model.DefaultBox import DefaultBox
from model.DefaultBox import DefaultBoxes
from model.BoundingBox import BoundingBox


class SingleShotMultiBoxDetector( NeuralNetworkBase ):
    """
    SSD [Single Shot muitibox Detector] を表すクラス。

    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
    
    [protedted] protedted な使用法を想定 
        image_height : int
            入力画像データの高さ（ピクセル単位）
        image_width : int
            入力画像データの幅（ピクセル単位）
        n_channels : int
            入力画像データのチャンネル数
            1 : グレースケール画像

        base_vgg16 : BaseNetworkVGG16
            SSD のベースネットワークとしての VGG16 を表す BaseNetworkVGG16 クラスのオブジェクト

        conv6_op : Operator
        pool6_op : Operator
        conv7_op : Operator
        conv8_1_op : Operator
        conv8_2_op : Operator
        conv9_1_op : Operator
        conv9_2_op : Operator
        conv10_1_op : Operator
        conv10_2_op : Operator
        conv11_1_op : Operator
        conv11_2_op : Operator

        f_maps : list<>
            特徴マップのリスト

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( 
            self,
            session = tf.Session(),
            image_height = 32,
            image_width = 32,
            n_channels = 1,
        ):
        
        super().__init__( session )

        # 各パラメータの初期化
        self.image_height = image_height
        self.image_width = image_width
        self.n_channels = n_channels

        # SDD モデルの各オペレーター
        self.base_vgg16 = BaseNetworkVGG16( 
                              session = self._session,
                              image_height = self.image_height,
                              image_width = self.image_width,
                              n_channels = self.n_channels
                          )

        self.conv6_op = None
        self.pool6_op = None
        self.conv7_op = None
        self.conv8_1_op = None
        self.conv8_2_op = None
        self.conv9_1_op = None
        self.conv9_2_op = None
        self.conv10_1_op = None
        self.conv10_2_op = None
        self.conv11_1_op = None
        self.conv11_2_op = None

        #
        self.fmaps = []

        return


    def print( self, str ):
        print( "----------------------------------" )
        print( str )
        print( self )

        print( "_session :", self._session )
        print( "_init_var_op : \n", self._init_var_op )
        print( "_y_out_op :", self._y_out_op )

        print( "_loss_op :", self._loss_op )
        print( "_optimizer :", self._optimizer )
        print( "_train_step :", self._train_step )

        print( "image_height : " , self.image_height )
        print( "image_width : " , self.image_width )
        print( "n_channels : " , self.n_channels )

        self.base_vgg16.print( "base network" )
        print( "conv6_op :", self.conv6_op )
        print( "pool6_op :", self.pool6_op )
        print( "conv7_op :", self.conv7_op )
        print( "conv8_1_op :", self.conv8_1_op )
        print( "conv8_2_op :", self.conv8_2_op )
        print( "conv9_1_op :", self.conv9_1_op )
        print( "conv9_2_op :", self.conv9_2_op )
        print( "conv10_1_op :", self.conv9_1_op )
        print( "conv10_2_op :", self.conv9_2_op )
        print( "conv11_1_op :", self.conv9_1_op )
        print( "conv11_2_op :", self.conv9_2_op )

        print( "----------------------------------" )

        return


    def init_weight_variable( self, input_shape, name = "init_weight_var" ):
        """
        重みの初期化を行う。
        重みは TensorFlow の Variable で定義することで、
        学習過程（最適化アルゴリズム Optimizer の session.run(...)）で自動的に TensorFlow により、変更される値となる。
        [Input]
            input_shape : [int,int]
                重みの Variable を初期化するための Tensor の形状
        [Output]
            正規分布に基づく乱数で初期化された重みの Variable 
            session.run(...) はされていない状態。
        """

        # ゼロで初期化すると、うまく重みの更新が出来ないので、正規分布に基づく乱数で初期化
        # tf.truncated_normal(...) : Tensor を正規分布なランダム値で初期化する
        init_tsr = tf.truncated_normal( shape = input_shape, stddev = 0.01 )

        # 重みの Variable
        weight_var = tf.Variable( init_tsr, name = name )
        
        return weight_var


    def init_bias_variable( self, input_shape, name = "init_bias_var" ):
        """
        バイアス項 b の初期化を行う。
        バイアス項は TensorFlow の Variable で定義することで、
        学習過程（最適化アルゴリズム Optimizer の session.run(...)）で自動的に TensorFlow により、変更される値となる。
        [Input]
            input_shape : [int,int]
                バイアス項の Variable を初期化するための Tensor の形状
        [Output]
            ゼロ初期化された重みの Variable 
            session.run(...) はされていない状態。
        """

        init_tsr = tf.random_normal( shape = input_shape )

        # バイアス項の Variable
        bias_var = tf.Variable( init_tsr, name = name )

        return bias_var


    def convolution_layer( 
            self, 
            input_tsr, 
            filter_height, filter_width,
            n_strides,
            n_input_channels, n_output_channels, 
            name = "conv", reuse = False
        ):
        """
        畳み込み層を構築する。
        
        [Input]
            input_tsr : Tensor / Placeholder
                畳み込み層への入力 Tensor
            filter_height : int
                フィルターの高さ（カーネル行列の行数）
            filter_width : int
                フィルターの幅（カーネル行列の列数）
            n_strides : int
                畳み込み処理（特徴マップ生成）でストライドさせる pixel 数
            n_input_channels : int
                入力データ（画像）のチャンネル数
            n_output_channels : int
                畳み込み処理後のデータのチャンネル数

        [Output]
            out_op : Operator
                畳み込み処理後の出力オペレーター
        """
        
        # Variable の名前空間（スコープ定義）
        with tf.variable_scope( name, reuse = reuse ):
            # 畳み込み層の重み（カーネル）を追加
            # この重みは、畳み込み処理の画像データに対するフィルタ処理（特徴マップ生成）に使うカーネルを表す Tensor のことである。
            # kernel_shape : [ [(filterの高さ) , (filterの幅) , (入力チャネル数) , (出力チャネル数) ]
            kernel = self.init_weight_variable( input_shape = [filter_height, filter_width, n_input_channels, n_output_channels] )
            bias = self.init_bias_variable( input_shape = [n_output_channels] )

            # 畳み込み演算
            conv_op = tf.nn.conv2d(
                          input = input_tsr,
                          filter = kernel,
                          strides = [1, n_strides, n_strides, 1],   # strides[0] = strides[3] = 1. とする必要がある
                          padding = "SAME",
                          name = name
                      )
            
            # 活性化関数として Relu で出力
            out_op = tf.nn.relu( tf.add(conv_op,bias) )

        return out_op


    def pooling_layer( self, input_tsr, name = "pool", reuse = False ):
        """
        VGG16 のプーリング層を構築する。

        [Input]
            input_tsr : Tensor / Placeholder
                畳み込み層への入力 Tensor
        [Output]
            pool_op : Operator
                プーリング処理後の出力オペレーター
        """
        # Variable の名前空間（スコープ定義）
        with tf.variable_scope( name, reuse = reuse ):
            # Max Pooling 演算
            pool_op = tf.nn.max_pool(
                          value = input_tsr,
                          ksize = [1, 2, 2, 1],
                          strides = [1, 2, 2, 1],
                          padding = "SAME",
                          name = name
                      )

        return pool_op


    def model( self ):
        """
        モデルの定義（計算グラフの構築）を行い、
        最終的なモデルの出力のオペレーターを設定する。
        [Output]
            self._y_out_op : Operator
                モデルの出力のオペレーター
        """   
        #-----------------------------------------------------------------------------
        # ベースネットワーク
        #-----------------------------------------------------------------------------
        self.base_vgg16.model()

        #-----------------------------------------------------------------------------
        # layer 6
        #-----------------------------------------------------------------------------
        self.conv6_op = self.convolution_layer( 
                            input_tsr = self.base_vgg16._y_out_op, 
                            filter_height = 3, filter_width = 3,
                            n_strides = 1,
                            n_input_channels = 512, n_output_channels = 1024,
                            name = "conv6", 
                            reuse = False
                        )

        self.pool6_op = self.pooling_layer( input_tsr = self.conv6_op, name = "pool6", reuse = False )

        #-----------------------------------------------------------------------------
        # layer 7
        #-----------------------------------------------------------------------------
        self.conv7_op = self.convolution_layer( 
                            input_tsr = self.conv6_op, 
                            filter_height = 1, filter_width = 1,
                            n_strides = 1,
                            n_input_channels = 1024, n_output_channels = 1024,
                            name = "conv7", 
                            reuse = False
                        )

        #-----------------------------------------------------------------------------
        # layer 8
        #-----------------------------------------------------------------------------
        self.conv8_1_op = self.convolution_layer( 
                              input_tsr = self.conv7_op, 
                              filter_height = 1, filter_width = 1,
                              n_strides = 1,
                              n_input_channels = 1024, n_output_channels = 256,
                              name = "conv8_1", 
                              reuse = False
                          )

        self.conv8_2_op = self.convolution_layer( 
                              input_tsr = self.conv8_1_op, 
                              filter_height = 3, filter_width = 3,
                              n_strides = 2,
                              n_input_channels = 256, n_output_channels = 512,
                              name = "conv8_2", 
                              reuse = False
                          )

        #-----------------------------------------------------------------------------
        # layer 9
        #-----------------------------------------------------------------------------
        self.conv9_1_op = self.convolution_layer( 
                              input_tsr = self.conv8_2_op, 
                              filter_height = 1, filter_width = 1,
                              n_strides = 1,
                              n_input_channels = 512, n_output_channels = 128,
                              name = "conv9_1", 
                              reuse = False
                          )

        self.conv9_2_op = self.convolution_layer( 
                              input_tsr = self.conv9_1_op, 
                              filter_height = 3, filter_width = 3,
                              n_strides = 2,
                              n_input_channels = 128, n_output_channels = 256,
                              name = "conv9_2", 
                              reuse = False
                          )

        #-----------------------------------------------------------------------------
        # layer 10
        #-----------------------------------------------------------------------------
        self.conv10_1_op = self.convolution_layer( 
                               input_tsr = self.conv9_2_op, 
                               filter_height = 3, filter_width = 3,
                               n_strides = 1,
                               n_input_channels = 256, n_output_channels = 128,
                               name = "conv10_1", 
                               reuse = False
                           )

        self.conv10_2_op = self.convolution_layer( 
                              input_tsr = self.conv10_1_op, 
                              filter_height = 3, filter_width = 3,
                              n_strides = 2,
                              n_input_channels = 128, n_output_channels = 256,
                              name = "conv10_2", 
                              reuse = False
                          )

        #-----------------------------------------------------------------------------
        # layer 11
        #-----------------------------------------------------------------------------
        self.conv11_1_op = self.convolution_layer( 
                               input_tsr = self.conv10_2_op, 
                               filter_height = 3, filter_width = 3,
                               n_strides = 1,
                               n_input_channels = 256, n_output_channels = 128,
                               name = "conv11_1", 
                               reuse = False
                           )

        self.conv11_2_op = self.convolution_layer( 
                              input_tsr = self.conv11_1_op, 
                              filter_height = 3, filter_width = 3,
                              n_strides = 3,
                              n_input_channels = 128, n_output_channels = 256,
                              name = "conv11_2", 
                              reuse = False
                          )

        #-----------------------------------------------------------------------------
        # Extra Feature Map
        #-----------------------------------------------------------------------------

        #-----------------------------------------------------------------------------
        # model output
        #-----------------------------------------------------------------------------
        #self._y_out_op = self.pool4_op

        return self._y_out_op
