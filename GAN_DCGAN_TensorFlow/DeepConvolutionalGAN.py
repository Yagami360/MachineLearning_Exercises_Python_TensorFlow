# -*- coding:utf-8 -*-
# Anaconda 5.1.0 環境 (TensorFlow インストール済み)

"""
    更新情報
    [18/02/07] : 新規作成
    [18/xx/xx] : 
               : 
"""
import scipy.misc
import numpy as np

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops

# NN 関連自作クラス
from NeuralNetworkBase import NeuralNetworkBase         # 親クラス

import NNActivation                                     # ニューラルネットワークの活性化関数を表すクラス
from NNActivation import NNActivation
from NNActivation import Sigmoid
from NNActivation import Relu
from NNActivation import Softmax

import NNLoss                                           # ニューラルネットワークの損失関数を表すクラス
from NNLoss import L1Norm
from NNLoss import L2Norm
from NNLoss import BinaryCrossEntropy
from NNLoss import CrossEntropy
from NNLoss import SoftmaxCrossEntropy
from NNLoss import SparseSoftmaxCrossEntropy

import NNOptimizer                                      # ニューラルネットワークの最適化アルゴリズム Optimizer を表すクラス
from NNOptimizer import GradientDecent
from NNOptimizer import GradientDecentDecay
from NNOptimizer import Momentum
from NNOptimizer import NesterovMomentum
from NNOptimizer import Adagrad
from NNOptimizer import Adadelta
from NNOptimizer import Adam


class DeepConvolutionalGAN( NeuralNetworkBase ):
    """
    DCGAN [Deep Convolutional GAN] を表すクラス
    ------------------------------------------------------------------------------------------------
    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の前頭にアンダースコア _ を付ける.
        _G_loss_op : Operator
            Generator の損失関数を表すオペレーター
        _G_optimizer : Optimizer
            Generator のモデルの最適化アルゴリズム
        _G_train_step : 
            Generator のトレーニングステップ
        _G_y_out_op : Operator
            Generator のモデルの出力のオペレーター

        _D_loss_op : Operator
            Descriminator の損失関数を表すオペレーター
        _D_optimizer : Optimizer
            Descriminator のモデルの最適化アルゴリズム
        _D_train_step : 
            Descriminator のトレーニングステップ
        _D_y_out_op1 : Operator
            Descriminator のモデルの出力のオペレーター
            入力が Generator からの出力（ダミー画像）のとき
        _D_y_out_op2 : Operator
            Descriminator のモデルの出力のオペレーター
            入力が本物の画像データのとき

        _weights : list <Variable>
            モデルの各層の重みの Variable からなる list
        _biases : list <Variable>
            モデルの各層のバイアス項の  Variable からなる list

        _epochs : int
            エポック数（トレーニング回数）
        _batch_size : int
            ミニバッチ学習でのバッチサイズ
        _eval_step : int
            学習処理時に評価指数の算出処理を行う step 間隔

        _image_height : int
            入力画像データの高さ（ピクセル単位）
        _image_width : int
            入力画像データの幅（ピクセル単位）
        _n_channels : int
            入力画像データのチャンネル数
            1 : グレースケール画像

        _n_G_dconv_featuresMap : list <int>
            Generator の逆畳み込み層で変換される特徴マップの枚数
            conv1 : _n_G_dconv_featuresMap[0]
            conv2 : _n_G_dconv_featuresMap[1]
        _n_D_conv_featuresMap : list <int>
            Descriminator の畳み込み層で変換される特徴マップの枚数
            conv1 : _n_D_conv_featuresMap[0]
            conv2 : _n_D_conv_featuresMap[1]


        _n_labels : int
            出力ラベル数（= Descriminator の出力層の出力側のノード数）

        _input_noize_holder : placeholder
            Generator に入力ノイズデータを供給するための placeholder
        _image_holder : placeholder
            Descriminator に画像データを供給するための placeholder
        _t_holder : placeholder
            出力層に教師データを供給するための placeholder

        _losses_train : list <float32>
            トレーニングデータでのモデル全体の損失関数の値の list
        _losses_G_train : list <float32>
            トレーニングデータでの Generator の損失関数の値の list
        _losses_D_train : list <float32>
            トレーニングデータでの Discriminator の損失関数の値の list

    [protedted] protedted な使用法を想定 
        
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( 
            self, 
            session = tf.Session(),
            epochs = 20000,
            batch_size = 32,
            eval_step = 1,
            image_height = 28,
            image_width = 28,
            n_channels = 1,
            n_G_deconv_featuresMap = [128, 64, 1],
            n_D_conv_featuresMap = [1, 64, 128],
            n_labels = 2
        ):
        """
        コンストラクタ（厳密にはイニシャライザ）
        """
        super().__init__( session )

        tf.set_random_seed(12)

        # メンバ変数の初期化
        self._session = session
        self._init_var_op = None

        # Genarator 関連
        self._G_loss_op = None
        self._G_optimizer = None
        self._G_train_step = None
        self._G_y_out_op = None

        # Descriminator 関連
        self._D_loss_op = None
        self._D_optimizer = None
        self._D_train_step = None
        self._D_y_out_op1 = None
        self._D_y_out_op2 = None

        # 各パラメータの初期化
        self._weights = []
        self._biases = []

        self._epochs = epochs
        self._batch_size = batch_size
        self._eval_step = eval_step
        
        self._image_height = image_height
        self._image_width = image_width
        self._n_channels = n_channels

        self._n_G_deconv_featuresMap = n_G_deconv_featuresMap
        self._n_D_conv_featuresMap = n_D_conv_featuresMap
        self._n_labels = n_labels

        # placeholder の初期化
        # shape の列（横方向）は、各層の次元（ユニット数）に対応させる。
        # shape の行は、None にして汎用性を確保
        self._input_noize_holder = tf.placeholder( 
                                       tf.float32, 
                                       shape = [ None, image_height * image_width * n_channels ],
                                       name = "input_noizer_holder"
                                   )

        self._image_holder = tf.placeholder( 
                             tf.float32, 
                             shape = [ None, image_height, image_width, n_channels ],
                             name = "image_holder"
                         )

        self._t_holder = tf.placeholder( 
                             tf.int32, 
                             shape = [ None ],
                             #shape = [ self._batch_size, n_labels ],
                             name = "t_holder"
                         )

        self._dropout_holder = tf.placeholder( tf.float32, name = "dropout_holder" )
        self._batch_size_holder = tf.placeholder( tf.int32, shape=[ batch_size ], name = "batch_size_holder" )
        
        # evaluate 関連の初期化
        self._losses_train = []
        self._losses_G_train = []
        self._losses_D_train = []
        
        return


    def print( self, str ):
        print( "-----------------------------------------" )
        print( "DeepConvolutionalGAN" )
        print( self )
        print( str )

        print( "_session : \n", self._session )
        print( "_init_var_op : \n", self._init_var_op )
        
        print( "_epoches : ", self._epochs )
        print( "_batch_size : ", self._batch_size )
        print( "_eval_step : ", self._eval_step )

        print( "_image_height : " , self._image_height )
        print( "_image_width : " , self._image_width )
        print( "_n_channels : " , self._n_channels )

        print( "_n_G_deconv_featuresMap : " , self._n_G_deconv_featuresMap )
        print( "_n_D_conv_featuresMap : " , self._n_D_conv_featuresMap )

        print( "_n_labels : " , self._n_labels )

        print( "_input_noize_holder : ", self._input_noize_holder )
        print( "_image_holder : ", self._image_holder )
        print( "_t_holder : ", self._t_holder )
        print( "_dropout_holder :", self._dropout_holder )
        print( "_batch_size_holder :", self._batch_size_holder )

        print( "_G_loss_op : \n", self._G_loss_op )
        print( "_G_optimizer : \n", self._G_optimizer )
        print( "_G_train_step : \n", self._G_train_step )
        print( "_G_y_out_op : \n", self._G_y_out_op )

        print( "_D_loss_op : \n", self._D_loss_op )
        print( "_D_optimizer : \n", self._D_optimizer )
        print( "_D_train_step : \n", self._D_train_step )
        print( "_D_y_out_op1 : \n", self._D_y_out_op1 )
        print( "_D_y_out_op2 : \n", self._D_y_out_op2 )

        print( "_weights : \n", self._weights )
        if( (self._session != None) and (self._init_var_op != None) ):
            print( self._session.run( self._weights ) )

        print( "_biases : \n", self._biases )
        if( (self._session != None) and (self._init_var_op != None) ):
            print( self._session.run( self._biases ) )

        print( "-----------------------------------------" )

        return

    def init_weight_variable( self, input_shape, train_able = True ):
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
        #weight_var = tf.Variable( init_tsr, name = "weight_var" )
        weight_var = tf.get_variable( 
                         name = "weight_var",
                         #shape = input_shape,
                         initializer = init_tsr, 
                         trainable =  train_able    # session.run() 時のトレーニング変数か否かのフラグ
                     )

        return weight_var


    def init_bias_variable( self, input_shape, train_able = True ):
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

        #init_tsr = tf.zeros( shape = input_shape )
        init_tsr = tf.random_normal( shape = input_shape )

        # バイアス項の Variable
        #bias_var = tf.Variable( init_tsr, name = "bias_var" )
        bias_var = tf.get_variable( 
                       name = "bias_var",
                       #shape = input_shape,
                       initializer = init_tsr, 
                       trainable =  train_able    # session.run() 時のトレーニング変数か否かのフラグ
                   )
        
        return bias_var


    def generator( self, input = None, reuse = False ):
        """
        GAN の Generator 側のモデルを構築する。

        [Input]
            input : Tensor or placeholder
                入力ノイズデータの Tensor or 画像データの placeholder
            reuse : bool
                Variable を共有するか否かのフラグ

        [Output]
            out_G_op : Operator
                Generator の最終的な出力の Operator
        """
        depths = self._n_G_deconv_featuresMap   # Generator の畳み込み層の特徴マップ数
        f_size = int( self._image_height / 2**(len(depths)-1) )    # ?
        i_depth = depths[:-1]                   # 入力 [Input] 側の layer の特徴マップ数
        o_depth = depths[1:]                    # 出力 [Output] 側の layer の特徴マップ数
        z_dim = i_depth[-1]                     # ? ノイズデータの shape
        
        """
        # MNIST データの場合のパラメータ
        depths = [128, 64, 1]                   # 特徴マップの枚数
        f_size = 28 / 2**(len(depths)-1)        #
        i_depth = depths[:-1]                   # 入力 [Input]
        o_depth = depths[1:]                    # 出力 [Output]
        batch_size = 32
        z_dim = 64                              # 
        weight0 = [ z_dim, i_depth[0] * f_size * f_size ]   # 重み
        bias0 = [ i_depth[0] ]

        print( "len(depths) :", len(depths) )   # len(depths) : 3
        print( "f_size :", f_size )             # f_size : 7.0
        print( "i_depth :", i_depth )           # i_depth : [128, 64]
        print( "o_depth :", o_depth )           # o_depth : [64, 1]
        print( "weight0 :", weight0 )           # weight0 :
        """

        #---------------------------------------------------------------------
        # 入力データ（ノイズデータ）を deconv 層へ入力するための reshape
        #---------------------------------------------------------------------
        with tf.variable_scope( "Generator", reuse = reuse ):
            # 入力データ → Generator の deconv 層 への重み
            # ? shape = [z_dim, i_depth[0] * f_size * f_size]
            self._weights.append(
                self.init_weight_variable( 
                    input_shape = [
                        z_dim, 
                        i_depth[0] * f_size * f_size
                    ]
                )
            )

            # 入力データ → Generator の deconv 層 へのバイアス項
            self._biases.append( 
                self.init_bias_variable( input_shape = [ i_depth[0] ] ) 
            )

            tmp_op = tf.matmul( input, self._weights[-1] )
            dc0_op = tf.reshape( tmp_op, [-1, f_size, f_size, i_depth[0]] ) + self._biases[-1]
        
            #print( "tmp_op :", tmp_op )     # Tensor("MatMul:0", shape=(32, 6272), dtype=float32)
            #print( "dc0_op :", dc0_op )     # Tensor("add:0", shape=(32, 7, 7, 128), dtype=float32)

            # batch normarization（ミニバッチごとに平均が0,分散が1）
            # tf.nn.moments(...) : 平均と分散を計算
            # axes = [0, 1, 2] でチャンネル毎の平均と分散を計算
            mean0_op, variance0_op = tf.nn.moments( dc0_op, axes = [0, 1, 2] )
            bn0_op = tf.nn.batch_normalization( dc0_op, mean0_op, variance0_op, None, None, 1e-5 )
            out_G_op = tf.nn.relu( bn0_op )

            #print( "bn0_op :", bn0_op )         # Tensor("batchnorm/add_1:0", shape=(32, 7, 7, 128), dtype=float32)
            #print( "out_G_op :", out_G_op )     # Tensor("Relu:0", shape=(32, 7, 7, 128), dtype=float32)

            #---------------------------------------------------------------------
            # DeConvolution layers
            #---------------------------------------------------------------------
            for layer in range( len(self._n_G_deconv_featuresMap)-1 ):
                with tf.variable_scope( "DeConvLayer_{}".format(layer) ):
                   # 重みの Variable の list に、layer 番目の畳み込み層の重み（カーネル）を追加
                    # この重みは、畳み込み処理の画像データに対するフィルタ処理（特徴マップ生成）に使うカーネルを表す Tensor のことである。
                    self._weights.append( 
                        self.init_weight_variable( 
                            input_shape = [ 
                                5, 5,                           # kernel 行列（フィルタ行列のサイズ） 
                                o_depth[layer], i_depth[layer]  # tf.nn.conv2d_transpose(...) の filter なので、Output, Input の形状
                            ]
                        ) 
                    )
        
                    # バイアス項の Variable の list に、畳み込み層のバイアス項を追加
                    self._biases.append( 
                        self.init_bias_variable( input_shape = [ o_depth[layer] ] ) 
                    )

                    # deconv
                    dc_op = tf.nn.conv2d_transpose(
                                value = out_G_op,
                                filter = self._weights[-1], # 畳込み処理で value で指定した Tensor との積和に使用する filter 行列（カーネル）
                                output_shape = [self._batch_size, f_size*2**(layer+1), f_size*2**(layer+1), o_depth[layer]],    # ?
                                strides = [1, 2, 2, 1]      # strides[0] = strides[3] = 1. とする必要がある
                            )

                    out_G_op = tf.nn.bias_add( dc_op, self._biases[-1] )

                    # batch normarization
                    if( layer < ( len(self._n_G_deconv_featuresMap) - 2 ) ):
                        mean_op, variance_op = tf.nn.moments( out_G_op, axes = [0, 1, 2] )
                        bn_op = tf.nn.batch_normalization( out_G_op, mean_op, variance_op, None, None, 1e-5 )
                        out_G_op = tf.nn.relu( bn_op )

            # tanh
            out_G_op = tf.nn.sigmoid( out_G_op )

        return out_G_op


    def discriminator( self, input = None, reuse = False ):
        """
        GAN の Discriminator 側のモデルを構築する。

        [Input]
            input : Operator or placeholder
                Generator の出力の Operator or 画像データの placeholder
            reuse : bool
                Variable を共有するか否かのフラグ

        [Output]
            self._D_y_out_op : Operator
                Descriminator の最終的な出力の Operator
        """
        depths = self._n_D_conv_featuresMap     # Descriminator の畳み込み層の特徴マップ数
        i_depth = depths[:-1]                   # 入力 [Input] 側の layer の特徴マップ数
        o_depth = depths[1:]                    # 出力 [Output] 側の layer の特徴マップ数

        with tf.variable_scope( "Descriminator", reuse = reuse ):
            out_D_op = input            # 最初の入力は、Generator の出力
            print( "out_D_op_0", out_D_op )

            #----------------------------------------
            # conv layer
            #----------------------------------------
            for layer in range( len(depths) - 1 ):
                with tf.variable_scope( "ConvLayer_{}".format(layer) ):
                    # 重みの Variable の list に、layer 番目の畳み込み層の重み（カーネル）を追加
                    # この重みは、畳み込み処理の画像データに対するフィルタ処理（特徴マップ生成）に使うカーネルを表す Tensor のことである。
                    self._weights.append( 
                        self.init_weight_variable( 
                            input_shape = [ 
                                5, 5,                           # kernel 行列（フィルタ行列のサイズ） 
                                i_depth[layer], o_depth[layer]  # tf.nn.conv2d(...) の filter なので、Input, Output の形状
                            ]
                        ) 
                    )
        
                    # バイアス項の Variable の list に、畳み込み層のバイアス項を追加
                    self._biases.append( 
                        self.init_bias_variable( input_shape = [ o_depth[layer] ] ) 
                    )

                    # conv
                    conv_op = tf.nn.conv2d(
                                  input = out_D_op,             # layer = 0 : Generator の出力 or 入力画像データ, layer = 1~ : 前回の出力
                                  filter = self._weights[-1],   # 畳込み処理で input で指定した Tensor との積和に使用する filter 行列（カーネル）
                                  strides = [1, 2, 2, 1],       # strides[0] = strides[3] = 1. とする必要がある
                                  padding='SAME'                # ゼロパディングを利用する場合はSAMEを指定
                            )

                    out_D_op = tf.nn.bias_add( conv_op, bias = self._biases[-1] )

                    # batch normalization
                    mean_op, variance_op = tf.nn.moments( out_D_op, [0, 1, 2] )
                    bn_op = tf.nn.batch_normalization( out_D_op, mean_op, variance_op, None, None, 1e-5 )

                    # Leaky ReLu
                    out_D_op = tf.maximum( 0.2 * bn_op, bn_op )

                    print( "_weights_{}".format(layer+1), self._weights[-1] )
                    print( "_biases_{}".format(layer+1), self._biases[-1] )
                    print( "out_D_op_{}".format(layer+1), out_D_op )

            #----------------------------------------
            # reshape & fully connected layer
            #----------------------------------------
            with tf.variable_scope( "flatten_fully" ):
                shape = out_D_op.get_shape().as_list()
                dim = shape[1]*shape[2]*shape[3]
                print( "shape :", shape )
                print( "dim :", dim )

                # 一列に平坦化 
                out_flatten = tf.reshape( out_D_op, shape = [-1, dim] )

                # 出力ノード
                # flatten layer → outout layer への重み
                self._weights.append(
                    self.init_weight_variable( input_shape = [ dim, self._n_labels ] )
                )

                # flatten layer → outout layer へのバイアス項
                self._biases.append( 
                    self.init_bias_variable( input_shape = [ self._n_labels ] ) 
                )

                out_D_op = tf.matmul( out_flatten, self._weights[-1] ) + self._biases[-1]

        return out_D_op


    def model( self ):
        """
        モデルの定義を行い、
        最終的なモデルの出力のオペレーターを設定する。

        [Output]
            self._y_out_op : Operator
                モデルの出力のオペレーター
        """
        # 入力データ（ノイズデータ）
        i_depth = self._n_G_deconv_featuresMap[:-1]   # 入力 [Input] 側の layer の特徴マップ数
        z_dim = i_depth[-1]                           # ? ノイズデータの shape

        input_noize_tsr = tf.random_uniform(
                              shape = [self._batch_size, z_dim],
                              minval = -1.0, maxval = 1.0
                          )
        
        # Generator : 入力データは, ノイズデータ
        self._G_y_out_op = self.generator( input = input_noize_tsr, reuse = False )
        
        # Descriminator : 入力データは, Generator の出力
        self._D_y_out_op1 = self.discriminator( input = self._G_y_out_op, reuse = False )
        
        # Descriminator : 入力データは, 画像データ
        self._D_y_out_op2 = self.discriminator( input = self._image_holder, reuse = True )

        # モデルの最終的な出力
        self._y_out_op = self._D_y_out_op2

        return self._y_out_op


    def loss( self, nnLoss ):
        """
        損失関数の定義を行う。
        
        [Input]
            nnLoss : NNLoss クラスのオブジェクト
            
        [Output]
            self._loss_op : Operator
                損失関数を表すオペレーター
        """
        print( "_t_holder :" , self._t_holder )
        print( "_image_holder :" , self._image_holder )
        print( "_G_y_out_op :" , self._G_y_out_op )
        print( "_D_y_out_op1 :" , self._D_y_out_op1 )
        print( "_D_y_out_op2 :" , self._D_y_out_op2 )

        # Generator の損失関数
        self._G_loss_op = nnLoss.loss( t_holder = self._t_holder, y_out_op = self._D_y_out_op1 )
        
        # Descriminator の損失関数
        """
        tmp_D_op0 = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits = self._D_y_out_op1, 
                            labels = tf.zeros( [self._batch_size], dtype=tf.int64 )
                        )
                    )

        tmp_D_op1 = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits = self._D_y_out_op1, 
                            labels = tf.ones( [self._batch_size], dtype=tf.int64 ) 
                        )
                    )
        """
        loss_D_op0 = nnLoss.loss( t_holder = self._t_holder, y_out_op = self._D_y_out_op1 )
        loss_D_op1 = nnLoss.loss( t_holder = self._t_holder, y_out_op = self._D_y_out_op2 )
        self._D_loss_op =  loss_D_op0 + loss_D_op1

        # 仮値を設定
        self._loss_op = self._D_loss_op
        
        return self._loss_op


    def optimizer( self, nnOptimizer ):
        """
        モデルの最適化アルゴリズムの設定を行う。
        [Input]
            nnOptimizer : NNOptimizer のクラスのオブジェクト
        [Output]
            optimizer の train_step
        """
        # Generator, Discriminator の Variable の抽出
        g_vars = [ var for var in tf.trainable_variables() if var.name.startswith('G') ]
        d_vars = [ var for var in tf.trainable_variables() if var.name.startswith('D') ]
        print( "g_vars :", g_vars )
        print( "d_vars :", d_vars )

        # Optimizer の設定
        self._G_optimizer = nnOptimizer._optimizer
        self._D_optimizer = nnOptimizer._optimizer
        
        # トレーニングステップの設定
        self._G_train_step = nnOptimizer.train_step( self._G_loss_op )
        self._D_train_step = nnOptimizer.train_step( self._D_loss_op )

        # tf.control_dependencies(...) : sess.run で実行する際のトレーニングステップの依存関係（順序）を定義
        with tf.control_dependencies( [self._G_train_step, self._D_train_step] ):
            # tf.no_op(...) : 何もしない Operator を返す。（トレーニングの依存関係を定義するのに使用）
            self._train_step = tf.no_op( name = 'train' )
        

        return self._train_step


    def fit( self, X_train, y_train ):
        """
        指定されたトレーニングデータで、モデルの fitting 処理を行う。
        [Input]
            X_train : np.ndarray ( shape = [n_samples, n_features] )
                トレーニングデータ（特徴行列）
            
            y_train : np.ndarray ( shape = [n_samples] )
                トレーニングデータ用のクラスラベル（教師データ）のリスト
        [Output]
            self : 自身のオブジェクト
        """
        """
        # 学習経過表示用の途中生成画像
        sample_images = []
        sample_image_data = np.random.rand( 8, z_dim ) * 2.0 - 1.0
        """

        # 入力データの shape にチェンネルデータがない場合
        # shape = [image_height, image_width]
        if( X_train.ndim == 3 ):
            # shape を [image_height, image_width] → [image_height, image_width, n_channel=1] に reshape
            X_train = np.expand_dims( X_train, axis = 3 )

        #----------------------------
        # 学習開始処理
        #----------------------------
        # Variable の初期化オペレーター
        self._init_var_op = tf.global_variables_initializer()

        # Session の run（初期化オペレーター）
        self._session.run( self._init_var_op )

        #--------------------------------------------------------
        # 学習処理
        #--------------------------------------------------------
        # for ループでエポック数分トレーニング
        for epoch in range( self._epochs ):
            # ミニバッチ学習処理のためランダムサンプリング
            idx_shuffled = np.random.choice( len(X_train), size = self._batch_size )
            X_train_shuffled = X_train[ idx_shuffled ]
            y_train_shuffled = y_train[ idx_shuffled ]
            #print( "X_train_shuffled.shape", X_train_shuffled.shape )  # shape = [32, 28, 28, 1]

            # 設定された最適化アルゴリズム Optimizer でトレーニング処理を run
            self._session.run(
                self._train_step,
                feed_dict = {
                    self._image_holder: X_train_shuffled
                    #self._t_holder: y_train_shuffled
                }
            )
            
            # 評価処理を行う loop か否か
            # % : 割り算の余りが 0 で判断
            if ( ( (epoch+1) % self._eval_step ) == 0 ):
                # 損失関数値の算出
                loss_total, loss_G, loss_D = \
                self._session.run(
                    [ self._loss_op, self._G_loss_op, self._D_loss_op ],
                    feed_dict = {
                        self._image_holder: X_train_shuffled
                        #self._t_holder: y_train_shuffled
                    }
                )

                self._losses_train.append( loss_total )
                self._losses_G_train.append( loss_G )
                self._losses_D_train.append( loss_D )
                print( "epoch %d / loss_total = %0.1f / loss_G = %0.1f / loss_D = %0.1f" % ( epoch + 1, loss_total, loss_G, loss_D ) )

                """
                # 途中生成画像の保存
                image_eval = self.generate_samples()
                output_file = "output_image/temp_output_image{}.jpg".format( epoch + 1 )
                scipy.misc.imsave( output_file, image_eval )
                """

        return self._y_out_op


    def generate_samples( self, n_samples ):
        """
        トレーニング経過可視化のための、サンプル画像データを生成する。

        """

        return