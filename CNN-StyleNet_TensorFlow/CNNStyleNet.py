# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境 (TensorFlow インストール済み)

"""
    更新情報
    [17/11/21] : 新規作成
    [17/xx/xx] : 
               : 
"""
# I/O 関連
import os

import scipy.misc
import scipy.io

import numpy as np

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops

# 自作クラス
import NNActivation
from NNActivation import NNActivation               # ニューラルネットワークの活性化関数を表すクラス
from NNActivation import Sigmoid
from NNActivation import Relu
from NNActivation import Softmax

import NNLoss                                       # ニューラルネットワークの損失関数を表すクラス
from NNLoss import L1Norm
from NNLoss import L2Norm
from NNLoss import BinaryCrossEntropy
from NNLoss import CrossEntropy
from NNLoss import SoftmaxCrossEntropy
from NNLoss import SparseSoftmaxCrossEntropy

import NNOptimizer                                  # ニューラルネットワークの最適化アルゴリズム Optimizer を表すクラス
from NNOptimizer import GradientDecent
from NNOptimizer import Momentum
from NNOptimizer import NesterovMomentum
from NNOptimizer import Adagrad
from NNOptimizer import Adadelta


class CNNStyleNet( object ):
    """
    CNN-StyleNet / NeuralStyle（ニューラルスタイル）を表すクラスを表すクラス.
    TensorFlow での StyleNet の処理をクラスでラッピング。
    ------------------------------------------------------------------------------------------------
    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
        _session : tf.Session()
            自身の Session
        _init_var_op : tf.global_variables_initializer()
            全 Variable の初期化オペレーター
        _loss_op : Operator
            損失関数を表すオペレーター
        _optimizer : Optimizer
            モデルの最適化アルゴリズム
        _train_step : 
            トレーニングステップ
        _y_out_op : Operator
            モデルの出力のオペレーター

        _epochs : int
            エポック数（トレーニング回数）
        _batch_size : int
            ミニバッチ学習でのバッチサイズ
        _eval_step : int
            学習処理時に評価指数の算出処理を行う step 間隔

        _image_content_path : str
            内容画像ファイルのパス
        _image_style_path : str
            スタイル画像ファイルのパス
        _image_content :　ndarray
            内容画像の配列　The array obtained by reading the image.
        _image_style :　ndarray
            スタイル画像の配列　The array obtained by reading the image.

        _weight_image_content : float
        _weight_image_style : float
        _weight_regularization : int
        
        _vgg_layers : list <str>
            モデルの層を記述したリスト
            先頭文字が、
            "c" : 畳み込み層
            "r" : Rulu
            "p" : プーリング層

        _n_conv_strides : int
            CNN の畳み込み処理（特徴マップ生成）でストライドさせる pixel 数

        _n_pool_wndsize : int
            プーリング処理用のウィンドウサイズ
        _n_pool_strides : int
            プーリング処理時のストライドさせる pixel 数

        _losses_train : list <float32>
            トレーニングデータでの損失関数の値の list

        _image_holder : placeholder
            画像データを供給するための placeholder

    [protedted] protedted な使用法を想定 


    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）


    """
    def __init__( 
            self,
            session = tf.Session( config = tf.ConfigProto(log_device_placement=True) ),
            epochs = 1000,
            batch_size = 1,
            image_content_path = "",
            image_style_path = "",
            weight_image_content = 5.0,
            weight_image_style = 500.0,
            weight_regularization = 100,
            output_generations = 250
        ):

        tf.set_random_seed(12)
        
        # メンバ変数の初期化
        self._session = session
        self._init_var_op = None

        self._loss_op = None
        self._optimizer = None
        self._train_step = None
        self._y_out_op = None

        # 各パラメータの初期化
        self._epochs = epochs
        self._batch_size = batch_size
        self._eval_step = eval_step

        self._image_content_path = image_content_path
        self._image_style_path = image_style_path
        self._image_content = None
        self._image_style = None

        self._weight_image_content = weight_image_content
        self._weight_image_style = weight_image_style
        self._weight_regularization = weight_regularization

        # evaluate 関連の初期化
        self._losses_train = []

        # 画像の読み込み
        self.load_image_contant_style( image_content_path, image_style_path )

        # モデルの定義
        self._vgg_layers = [
            "conv1_1", "relu1_1",
            'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1',
            'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1',
            'conv3_2', 'relu3_2',
            'conv3_3', 'relu3_3',
            'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1',
            'conv4_2', 'relu4_2',
            'conv4_3', 'relu4_3',
            'conv4_4', 'relu4_4', 'pool4',
            'conv5_1', 'relu5_1',
            'conv5_2', 'relu5_2',
            'conv5_3', 'relu5_3',
            'conv5_4', 'relu5_4'
        ]
        
        return

    def print( self, str ):
        print( "----------------------------------" )
        print( self )
        print( str )

        print( "_session : \n", self._session )
        print( "_init_var_op : \n", self._init_var_op )
        print( "_loss_op : \n", self._loss_op )
        print( "_y_out_op : \n", self._y_out_op )

        print( "_epoches : ", self._epochs )
        print( "_batch_size : ", self._batch_size )
        print( "_eval_step : ", self._eval_step )

        print( "_image_content_path : ", self._image_content_path )
        print( "_image_style_path : ", self._image_style_path )
        print( "_image_content : \n", self._image_content )
        print( "_image_style : \n", self._image_style )

        print( "_weight_image_content : ", self._weight_image_content )
        print( "_weight_image_style : ", self._weight_image_style )
        print( "_weight_regularization : ", self._weight_regularization )

        print( "_vgg_layers : \n", self._vgg_layers )

        print( "----------------------------------" )

        return


    def load_image_contant_style( self, image_content_path, image_style_path ):
        """
        内容画像とスタイル画像を読み込む。
        """
        self._image_content_path = image_content_path
        self._image_style_path = image_style_path
        
        # 指定された path の画像を読み込む
        # scipy.misc.imread(...) : Read an image from a file as an array.
        # Returns : The array obtained by reading the image.
        self._image_content = scipy.misc.imread( image_content_path )
        self._image_style = scipy.misc.imread( image_style_path )

        # ２つの画像の合成するので、
        # スタイル画像の shape を内容画像の shape と合わせておく。
        # shape[1] : 
        new_shape = ( self._image_content.shape[1] / self._image_style.shape[1] )
        print( "_image_content.shape[1]", self._image_content.shape[1] )
        print( "_image_style.shape[1]", self._image_style.shape[1] )
        pint( "new_shape", new_shape )
        self._image_style = scipy.misc.imsize( self._image_style, new_shape )

        return

    def load_model_info( mat_file_path ):
        """
        学習済みの StyleNet モデル用のデータである mat データから、パラメータを読み込み。
        imagenet-vgg-verydee-19.mat : この mat データは、MATLAB オブジェクトが含まれている

        [Input]
            mat_file_path : str
                imagenet-vgg-verydee-19.mat ファイルの path
        
        [Output]

        """
        vgg_data = scipy.io.loadmat( mat_file_path )
        print( "vgg_data :\n", vgg_data )

        # ?
        normalization_matrix = vgg_data[ "normalization" ][0][0][0]
        print( "normalization_matrix :\n", normalization_matrix )

        # ? matrix の row , colum に対しての平均値をとった matrix
        matrix_mean = np.mean( normalization_matrix, axis = (0,1) )

        # モデルの重み
        network_weight = vgg_data[ "layers" ][0]
        print( "network_weight :\n", network_weight )

        return ( matrix_mean, network_weight )


    def model( self ):
        """
        モデルの定義を行い、
        最終的なモデルの出力のオペレーター self._y_out_op を設定する。
        [Output]
            self._y_out_op : Operator
                モデルの出力のオペレーター
        """
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
        self._loss_op = nnLoss.loss( t_holder = self._t_holder, y_out_op = self._y_out_op )
        
        return self._loss_op


    def optimizer( self, nnOptimizer ):
        """
        モデルの最適化アルゴリズムの設定を行う。
        [Input]
            nnOptimizer : NNOptimizer のクラスのオブジェクト
        [Output]
            optimizer の train_step
        """
        self._optimizer = nnOptimizer._optimizer
        self._train_step = nnOptimizer.train_step( self._loss_op )
        
        return self._train_step
