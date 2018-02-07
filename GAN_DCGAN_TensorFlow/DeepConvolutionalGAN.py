# -*- coding:utf-8 -*-
# Anaconda 5.1.0 環境 (TensorFlow インストール済み)

"""
    更新情報
    [18/02/07] : 新規作成
    [18/xx/xx] : 
               : 
"""

import numpy

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops

# NN 関連自作クラス
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


class DeepConvolutionalGAN( object ):
    """
    DCGAN [Deep Convolutional GAN] を表すクラス
    ------------------------------------------------------------------------------------------------
    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の前頭にアンダースコア _ を付ける.
        _session : tf.Session()
            自身の Session
        _init_var_op : tf.global_variables_initializer()
            全 Variable の初期化オペレーター

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
        _D_y_out_op : Operator
            Descriminator のモデルの出力のオペレーター
            

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
            n_labels = 10
        ):
        """
        コンストラクタ（厳密にはイニシャライザ）
        """
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
        self._D_y_out_op = None

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
        self._X_holder = tf.placeholder( 
                             tf.float32, 
                             shape = [ None, image_height, image_width, n_channels ]
                         )

        self._t_holder = tf.placeholder( 
                             tf.int32, 
                             shape = [ None, n_labels ]
                         )

        # evaluate 関連の初期化
        self._losses_train = []

        return


    def print( self, str ):
        print( "-----------------------------------------" )
        print( "DeepConvolutionalGAN" )
        print( self )
        print( str )

        print( "_session : \n", self._session )
        print( "_init_var_op : \n", self._init_var_op )
        
        print( "_G_loss_op : \n", self._G_loss_op )
        print( "_G_optimizer : \n", self._G_optimizer )
        print( "_G_train_step : \n", self._G_train_step )
        print( "_G_y_out_op : \n", self._G_y_out_op )

        print( "_D_loss_op : \n", self._D_loss_op )
        print( "_D_optimizer : \n", self._D_optimizer )
        print( "_D_train_step : \n", self._D_train_step )
        print( "_D_y_out_op : \n", self._D_y_out_op )

        print( "_epoches : ", self._epochs )
        print( "_batch_size : ", self._batch_size )
        print( "_eval_step : ", self._eval_step )

        print( "_image_height : " , self._image_height )
        print( "_image_width : " , self._image_width )
        print( "_n_channels : " , self._n_channels )

        print( "_n_G_deconv_featuresMap : " , self._n_G_deconv_featuresMap )
        print( "_n_D_conv_featuresMap : " , self._n_D_conv_featuresMap )

        print( "_n_labels : " , self._n_labels )

        print( "_X_holder : ", self._X_holder )
        print( "_t_holder : ", self._t_holder )

        print( "_weights : \n", self._weights )
        if( (self._session != None) and (self._init_var_op != None) ):
            print( self._session.run( self._weights ) )

        if( (self._session != None) and (self._init_var_op != None) ):
            print( self._session.run( self._biases ) )

        print( "-----------------------------------------" )

        return

    def generator( self ):
        """
        GAN の generator のモデルを構築する。
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

        # 
        depths = self._n_G_deconv_featuresMap
        f_size = self._image_height / 2**(len(depths)-1)


        return

    def discriminator( self ):

        return


    def model( self ):
        """
        モデルの定義を行い、
        最終的なモデルの出力のオペレーターを設定する。

        [Output]
            self._y_out_op : Operator
                モデルの出力のオペレーター
        """
        self.generator()
        self.discriminator()

        return self._D_y_out_op
