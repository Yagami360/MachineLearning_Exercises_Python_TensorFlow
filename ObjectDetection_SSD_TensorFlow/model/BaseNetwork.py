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


class BaseNetwork( NeuralNetworkBase ):
    """
    SSD [Single Shot muitibox Detector] でのベースネットワークを表すクラス。

    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
    
    [protedted] protedted な使用法を想定 

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( 
            self,
            session = tf.Session(),
        ):
        
        super().__init__( session )

        return


class BaseNetworkVGG16( BaseNetwork ):
    """
    SSD [Single Shot muitibox Detector] でベースネットワークとして使用する VGG16 を表すクラス。

    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.

    [protedted] protedted な使用法を想定 
        image_height : int
            入力画像データの高さ（ピクセル単位）
        image_width : int
            入力画像データの幅（ピクセル単位）
        n_channels : int
            入力画像データのチャンネル数
            1 : グレースケール画像

        X_holder : placeholder
            入力層に入力データを供給するための placeholder

        conv1_1_op : Operator
        conv1_2_op : Operator
        pool1_op : Operator
        conv2_1_op : Operator
        conv2_2_op : Operator
        pool2_op : Operator
        conv3_1_op : Operator
        conv3_2_op : Operator
        conv3_3_op : Operator
        pool3_op : Operator
        conv4_1_op : Operator
        conv4_2_op : Operator
        conv4_3_op : Operator
        pool4_op : Operator
        conv5_1_op : Operator
        conv5_2_op : Operator
        conv5_3_op : Operator
        pool5_op : Operator

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( 
            self,
            session = tf.Session(),
            image_height = 32,
            image_width = 32,
            n_channels = 1
        ):

        super().__init__( session )

        # 各パラメータの初期化
        self.image_height = image_height
        self.image_width = image_width
        self.n_channels = n_channels

        # VGG-16 モデルの各オペレーター
        self.conv1_1_op = None
        self.conv1_2_op = None
        self.pool1_op = None
        self.conv2_1_op = None
        self.conv2_2_op = None
        self.pool2_op = None
        self.conv3_1_op = None
        self.conv3_2_op = None
        self.conv3_3_op = None
        self.pool3_op = None
        self.conv4_1_op = None
        self.conv4_2_op = None
        self.conv4_3_op = None
        self.pool4_op = None
        self.conv5_1_op = None
        self.conv5_2_op = None
        self.conv5_3_op = None
        self.pool5_op = None
        
        # placeholder の初期化
        # shape の列（横方向）は、各層の次元（ユニット数）に対応させる。
        # shape の行は、None にして汎用性 batch_size を確保
        self.X_holder = tf.placeholder( 
                            tf.float32, 
                            shape = [ None, self.image_height, self.image_width, self.n_channels ],     # [ None, image_height, image_width, n_channels ]
                            name = "X_holder"
                        )
        
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

        print( "conv1_1_op :", self.conv1_1_op )
        print( "conv1_2_op :", self.conv1_2_op )
        print( "pool1_op :", self.pool1_op )
        print( "conv2_1_op :", self.conv2_1_op )
        print( "conv2_2_op :", self.conv2_2_op )
        print( "pool2_op :", self.pool2_op )
        print( "conv3_1_op :", self.conv3_1_op )
        print( "conv3_2_op :", self.conv3_2_op )
        print( "conv3_3_op :", self.conv3_3_op )
        print( "pool3_op :", self.pool3_op )
        print( "conv4_1_op :", self.conv4_1_op )
        print( "conv4_2_op :", self.conv4_2_op )
        print( "conv4_3_op :", self.conv4_3_op )
        print( "pool4_op :", self.pool4_op )
        print( "conv5_1_op :", self.conv5_1_op )
        print( "conv5_2_op :", self.conv5_2_op )
        print( "conv5_3_op :", self.conv5_3_op )
        print( "pool5_op :", self.pool5_op )
        
        print( "X_holder : ", self.X_holder )
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
            filter_height, filter_width, n_input_channels, n_output_channels, 
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
                          strides = [1, 1, 1, 1],
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
        # layer 1
        #-----------------------------------------------------------------------------
        self.conv1_1_op = self.convolution_layer( 
                              input_tsr = self.X_holder, 
                              filter_height = 3, filter_width = 3, n_input_channels = 3, n_output_channels = 64,
                              name = "conv1_1", 
                              reuse = False
                          )

        self.conv1_2_op = self.convolution_layer( 
                              input_tsr = self.conv1_1_op, 
                              filter_height = 3, filter_width = 3, n_input_channels = 64, n_output_channels = 64,
                              name = "conv1_2", 
                              reuse = False
                          )

        self.pool1_op = self.pooling_layer( input_tsr = self.conv1_2_op, name = "pool1", reuse = False )

        #-----------------------------------------------------------------------------
        # layer 2
        #-----------------------------------------------------------------------------
        self.conv2_1_op = self.convolution_layer( 
                              input_tsr = self.pool1_op, 
                              filter_height = 3, filter_width = 3, n_input_channels = 64, n_output_channels = 128,
                              name = "conv2_1",
                              reuse = False
                          )

        self.conv2_2_op = self.convolution_layer( 
                              input_tsr = self.conv2_1_op, 
                              filter_height = 3, filter_width = 3, n_input_channels = 128, n_output_channels = 128,
                              name = "conv2_2",
                              reuse = False
                          )

        self.pool2_op = self.pooling_layer( input_tsr = self.conv2_2_op, name = "pool2", reuse = False )

        #-----------------------------------------------------------------------------
        # layer 3
        #-----------------------------------------------------------------------------
        self.conv3_1_op = self.convolution_layer( 
                              input_tsr = self.pool2_op, 
                              filter_height = 3, filter_width = 3, n_input_channels = 128, n_output_channels = 256,
                              name = "conv3_1",
                              reuse = False
                          )

        self.conv3_2_op = self.convolution_layer( 
                              input_tsr = self.conv3_1_op, 
                              filter_height = 3, filter_width = 3, n_input_channels = 256, n_output_channels = 256,
                              name = "conv3_2",
                              reuse = False
                          )

        self.conv3_3_op = self.convolution_layer( 
                              input_tsr = self.conv3_2_op, 
                              filter_height = 3, filter_width = 3, n_input_channels = 256, n_output_channels = 256,
                              name = "conv3_3",
                              reuse = False
                          )

        self.pool3_op = self.pooling_layer( input_tsr = self.conv3_3_op, name = "pool3", reuse = False )

        #-----------------------------------------------------------------------------
        # layer 4
        #-----------------------------------------------------------------------------
        self.conv4_1_op = self.convolution_layer( 
                              input_tsr = self.pool3_op, 
                              filter_height = 3, filter_width = 3, n_input_channels = 256, n_output_channels = 512,
                              name = "conv4_1",
                              reuse = False
                          )

        self.conv4_2_op = self.convolution_layer( 
                              input_tsr = self.conv4_1_op, 
                              filter_height = 3, filter_width = 3, n_input_channels = 512, n_output_channels = 512,
                              name = "conv4_2",
                              reuse = False
                          )

        """
        self.conv4_3_op = self.convolution_layer( 
                              input_tsr = self.conv4_2_op, 
                              filter_height = 3, filter_width = 3, n_input_channels = 512, n_output_channels = 512,
                              name = "conv4_3",
                              reuse = False
                          )
        
        self.pool4_op = self.pooling_layer( input_tsr = self.conv4_3_op, name = "pool4", reuse = False )
        """

        #-----------------------------------------------------------------------------
        # layer 5
        #-----------------------------------------------------------------------------
        """
        self.conv5_1_op = self.convolution_layer( 
                              input_tsr = self.pool4_op, 
                              filter_height = 3, filter_width = 3, n_input_channels = 512, n_output_channels = 512,
                              name = "conv5_1",
                              reuse = False
                          )

        self.conv5_2_op = self.convolution_layer( 
                              input_tsr = self.conv5_1_op, 
                              filter_height = 3, filter_width = 3, n_input_channels = 512, n_output_channels = 512,
                              name = "conv5_2",
                              reuse = False
                          )

        self.conv5_3_op = self.convolution_layer( 
                              input_tsr = self.conv5_2_op, 
                              filter_height = 3, filter_width = 3, n_input_channels = 512, n_output_channels = 512,
                              name = "conv5_3",
                              reuse = False
                          )

        self.pool5_op = self.pooling_layer( input_tsr = self.conv5_3_op, name = "pool5", reuse = False )
        """

        #-----------------------------------------------------------------------------
        # model output
        #-----------------------------------------------------------------------------
        self._y_out_op = self.conv4_2_op

        return self._y_out_op



class BaseNetworkResNet( BaseNetwork ):
    """
    SSD [Single Shot muitibox Detector] でベースネットワークとして使用する ResNet を表すクラス。

    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.

    [protedted] protedted な使用法を想定 

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）
    
    """
    def __init__( 
            self,
            session = tf.Session(),
        ):

        super().__init__( session )

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
        print( "----------------------------------" )

        return


    def model( self ):
        """
        モデルの定義（計算グラフの構築）を行い、
        最終的なモデルの出力のオペレーターを設定する。
        [Output]
            self._y_out_op : Operator
                モデルの出力のオペレーター
        """   

        return self._y_out_op

