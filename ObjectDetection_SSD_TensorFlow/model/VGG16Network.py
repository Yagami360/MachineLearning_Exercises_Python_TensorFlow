# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境 (TensorFlow 1.4.0 インストール済み)

"""
    更新情報
    [18/05/03] : 新規作成
    [xx/xx/xx] : 

"""

import numpy as np

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops

# 自作モジュール
from model.NeuralNetworkBase import NeuralNetworkBase

from model.NNActivation import NNActivation              # ニューラルネットワークの活性化関数を表すクラス
from model.NNActivation import Sigmoid
from model.NNActivation import Relu
from model.NNActivation import Softmax

from model.NNLoss import NNLoss                          # ニューラルネットワークの損失関数を表すクラス
from model.NNLoss import L1Norm
from model.NNLoss import L2Norm
from model.NNLoss import BinaryCrossEntropy
from model.NNLoss import CrossEntropy
from model.NNLoss import SoftmaxCrossEntropy
from model.NNLoss import SparseSoftmaxCrossEntropy

from model.NNOptimizer import NNOptimizer                # ニューラルネットワークの最適化アルゴリズム Optimizer を表すクラス
from model.NNOptimizer import GradientDecent
from model.NNOptimizer import GradientDecentDecay
from model.NNOptimizer import Momentum
from model.NNOptimizer import NesterovMomentum
from model.NNOptimizer import Adagrad
from model.NNOptimizer import Adadelta
from model.NNOptimizer import Adam


class VGG16Network( NeuralNetworkBase ):
    """
    VGG-16 を表しクラス（自作クラス）
    TensorFlow での VGG-16 の処理をクラス（任意の層に DNN 化可能な柔軟なクラス）でラッピングし、
    scikit-learn ライブラリの classifier, estimator とインターフェイスを共通化することで、
    scikit-learn ライブラリとの互換性のある自作クラス
    
    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.

    [protedted] protedted な使用法を想定 
        vgg_layers_dict : 辞書型
            VGG-16 モデルの構造とパラメータ定義
            size-format:
             "conv": [ [(filterの高さ) , (filterの幅) , (入力チャネル数) , (出力チャネル数) ], [   bias   ] ]
             "relu": [ None ]
             "pool": []
             "fc": [ [ f_h, f_w, in_size, out_size ], [ out_size ] ]

        X_holder : placeholder
            入力層に入力データを供給するための placeholder
        t_holder : placeholder
            出力層に教師データを供給するための placeholder

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）


    """
    def __init__(
            self, 
            session = tf.Session()
        ):

        super().__init__( session )

        # VGG-16 モデルの構造とパラメータ定義
        self.vgg_layers_dict = {
            # layer 1
            "conv1_1": [ [3, 3, 3, 64], [64] ], "relu1_1": [None],
            'conv1_2': [ [3, 3, 64, 64], [64] ], 'relu1_2': [None], 'pool1':[],
            
            # layer 2
            'conv2_1': [ [3, 3, 64, 128], [128] ], 'relu2_1': [None],
            'conv2_2': [ [3, 3, 128, 128], [128] ], 'relu2_2': [], 'pool2': [],
            
            # layer 3
            'conv3_1': [ [3, 3, 128, 256], [256] ], 'relu3_1': [],
            'conv3_2': [ [3, 3, 256, 256], [256] ], 'relu3_2': [],
            'conv3_3': [ [3, 3, 256, 256], [256] ], 'relu3_3': [],
            
            # layer 4
            'conv4_1': [ [3, 3, 256, 512], [512] ], 'relu4_1': [],
            'conv4_2': [ [3, 3, 512, 512], [512] ], 'relu4_2': [],
            'conv4_3': [ [3, 3, 512, 512], [512] ], 'relu4_3': [],
            
            # layer 5
            'conv5_1': [ [3, 3, 512, 512], [512] ], 'relu5_1': [],
            'conv5_2': [ [3, 3, 512, 512], [512] ], 'relu5_2': [],
            'conv5_3': [ [3, 3, 512, 512], [512] ], 'relu5_3': [],

            # 
            'fc6': [ [4096, 0, 0, 0], [4096] ],
            'fc7': [ [4096, 0, 0, 0], [4096] ],
            'fc8': [ [1000, 0, 0, 0], [1000] ]
        }

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
        self.fc6_op = None
        self.fc7_op = None
        self.fc8_op = None

        # placeholder の初期化
        # shape の列（横方向）は、各層の次元（ユニット数）に対応させる。
        # shape の行は、None にして汎用性を確保
        self.X_holder = tf.placeholder( 
                            tf.float32, 
                            shape = [ None, 32, 32, 3 ]     # [ None, image_height, image_width, n_channels ]
                        )

        self.t_holder = tf.placeholder( 
                            tf.float32, 
                            shape = [ None ]  # [n_labels]
                        )

        return


    def print( self, str ):
        print( "----------------------------------" )
        print( str )
        print( self )

        print( "_session :", self._session )
        print( "_init_var_op : \n", self._init_var_op )
        print( "_loss_op :", self._loss_op )
        print( "_y_out_op :", self._y_out_op )

        print( "vgg_layers_dict :", self.vgg_layers_dict )
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
        print( "fc6_op :", self.fc6_op )
        print( "fc7_op :", self.fc7_op )
        print( "fc8_op :", self.fc8_op )

        print( "X_holder : ", self.X_holder )
        print( "t_holder : ", self.t_holder )
        
        print( "----------------------------------" )
        return


    def init_weight_variable( self, input_shape ):
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
        weight_var = tf.Variable( init_tsr, name = "init_weight_var" )
        
        return weight_var


    def init_bias_variable( self, input_shape ):
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
        bias_var = tf.Variable( init_tsr, name = "init_bias_var" )

        return bias_var


    def convolution_layer( self, input_tsr, name = "conv", reuse = False ):
        """
        畳み込み層を構築する。

        [Input]

        [Output]

        """
        
        # Variable の名前空間（スコープ定義）
        with tf.variable_scope( name, reuse = reuse ):
            # VGG16 の構造パラメータを定義した辞書から、畳み込み層のサイズ
            # [ [(filterの高さ) , (filterの幅) , (入力チャネル数) , (出力チャネル数) ], [   bias   ] ] を取得
            size = self.vgg_layers_dict[name]
            
            # 畳み込み層の重み（カーネル）を追加
            # この重みは、畳み込み処理の画像データに対するフィルタ処理（特徴マップ生成）に使うカーネルを表す Tensor のことである。
            kernel = self.init_weight_variable( input_shape = size[0] )
            bias = self.init_bias_variable( input_shape = size[1] )

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


    def fully_conntected_layer( self, input_tsr, nnActivation, name = "", reuse = False ):
        """
        VGG16 の全結合層を構築する。
        """
        # Variable の名前空間（スコープ定義）
        with tf.variable_scope( name, reuse = reuse ):
            # VGG16 の構造パラメータを定義した辞書から、全結合層のサイズ
            # [ [ f_h, f_w, in_size, out_size ], [ out_size ] ] を取得
            size = self.vgg_layers_dict[name]

            # input_tsr の shape を取り出し、list 構造に変換する。
            shape = input_tsr.get_shape().as_list()
            print( "input_tsr / shape", shape )

            # 無名関数 lamda ( x,y が引数 )
            # reduce で畳み込み演算し、fc 層の次元数を算出
            from functools import reduce
            dim = reduce( lambda x, y: x * y, shape[1:] )
            print( "dim :", dim )

            #
            input_reshaped_tsr = tf.reshape( input_tsr, shape = [-1, dim] )
            print( "input_tsr / shape :", input_tsr.get_shape() )
            print( "input_reshaped_tsr / shape :", input_reshaped_tsr.get_shape() )

            #
            weights = self.init_bias_variable( input_shape = [ dim, size[0][0] ] )
            biases = self.init_bias_variable( input_shape = [1] )

            # 全結合層の入力と出力
            fc_in_op = tf.nn.bias_add( tf.matmul(input_reshaped_tsr, weights), biases )
            fc_out_op = nnActivation.activate( fc_in_op )

        return fc_out_op


    def model( self, reuse = False ):
        """
        モデルの定義（計算グラフの構築）を行い、
        最終的なモデルの出力のオペレーターを設定する。
        [Output]
            self._y_out_op : Operator
                モデルの出力のオペレーター
        """
        
        # layer 1
        self.conv1_1_op = self.convolution_layer( input_tsr = self.X_holder, name = 'conv1_1', reuse = False )
        self.conv1_2_op = self.convolution_layer( input_tsr = self.conv1_1_op, name = 'conv1_2', reuse = False )
        self.pool1_op = self.pooling_layer( input_tsr = self.conv1_2_op, name = "pool1", reuse = False )

        # layer 2
        self.conv2_1_op = self.convolution_layer( input_tsr = self.pool1_op, name = 'conv2_1', reuse = False )
        self.conv2_2_op = self.convolution_layer( input_tsr = self.conv2_1_op, name = 'conv2_2', reuse = False )
        self.pool2_op = self.pooling_layer( input_tsr = self.conv2_2_op, name = "pool2", reuse = False )

        # layer 3
        self.conv3_1_op = self.convolution_layer( input_tsr = self.pool2_op, name = 'conv3_1', reuse = False )
        self.conv3_2_op = self.convolution_layer( input_tsr = self.conv3_1_op, name = 'conv3_2', reuse = False )
        self.conv3_3_op = self.convolution_layer( input_tsr = self.conv3_2_op, name = 'conv3_3', reuse = False )
        self.pool3_op = self.pooling_layer( input_tsr = self.conv3_3_op, name = "pool3", reuse = False )

        # layer 4
        self.conv4_1_op = self.convolution_layer( input_tsr = self.pool3_op, name = 'conv4_1', reuse = False )
        self.conv4_2_op = self.convolution_layer( input_tsr = self.conv4_1_op, name = 'conv4_2', reuse = False )
        self.conv4_3_op = self.convolution_layer( input_tsr = self.conv4_2_op, name = 'conv4_3', reuse = False )
        self.pool4_op = self.pooling_layer( input_tsr = self.conv4_3_op, name = "pool4", reuse = False )

        # layer 5
        self.conv5_1_op = self.convolution_layer( input_tsr = self.pool4_op, name = 'conv5_1', reuse = False )
        self.conv5_2_op = self.convolution_layer( input_tsr = self.conv5_1_op, name = 'conv5_2', reuse = False )
        self.conv5_3_op = self.convolution_layer( input_tsr = self.conv5_2_op, name = 'conv5_3', reuse = False )
        self.pool5_op = self.pooling_layer( input_tsr = self.conv5_3_op, name = "pool5", reuse = False )

        # fc6
        #self.fc6_op = self.fully_conntected_layer( input_tsr = self.pool5_op, nnActivation = Relu(), name = "fc6", reuse = False )

        # fc7
        #self.fc7_op = self.fully_conntected_layer( input_tsr = self.fc6_op, nnActivation = Relu(), name = "fc7", reuse = False )

        # fc8
        #self.fc8_op = self.fully_conntected_layer( input_tsr = self.fc7_op, nnActivation = Softmax(), name = "fc8", reuse = False )

        #self._y_out_op = self.fc8_op

        return self._y_out_op


    def loss( self, nnLoss, reuse = False ):
        """
        損失関数の定義を行う。
        
        [Input]
            nnLoss : NNLoss クラスのオブジェクト
            
        [Output]
            self._loss_op : Operator
                損失関数を表すオペレーター
        """

        return self._loss_op


    def optimizer( self, nnOptimizer, reuse = False ):
        """
        モデルの最適化アルゴリズムの設定を行う。
        [Input]
            nnOptimizer : NNOptimizer のクラスのオブジェクト
        [Output]
            optimizer の train_step
        """

        return self._train_step


    def fit( self, X_train, y_train ):
        """
        指定されたトレーニングデータで、モデルの fitting 処理を行う。
        [Input]
            X_train : numpy.ndarray ( shape = [n_samples, n_features] )
                トレーニングデータ（特徴行列）
            
            y_train : numpy.ndarray ( shape = [n_samples] )
                トレーニングデータ用のクラスラベル（教師データ）のリスト
        [Output]
            self : 自身のオブジェクト
        """

        return self._y_out_op

