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
            （画像内容層の損失関数＋画像スタイル層の損失関数＋全変動ノイズの損失関数）からなる
        _loss_content_op : Operator
        _loss_style_op : Operator
        _loss_total_var_op : Operator
            全変動損失

        _optimizer : Optimizer
            モデルの最適化アルゴリズム
        _train_step : 
            トレーニングステップ
        _y_out_op : Operator
            モデルの出力のオペレーター

        _weights : list <Variable>
            モデルの各層の重みの Variable からなる list
        _biases : list <float>
            モデルの各層のバイアス項からなる list

        _epochs : int
            エポック数（トレーニング回数）
        _eval_step : int
            学習処理時に評価指数の算出処理＆途中生成画像の出力を行う step 間隔

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
        _content_layer : str
            内容層の構成。"relu4_2"
        _style_layers : list <str>
            スタイル層の構成。reluX_1 層の組み合わせ
        _vgg_network : list <Tensor>
            _vgg_layers で定義した各層の中身（Tensor）

        _features_content : list <>
            内容層の特徴量
        _features_style : list <>
            画像層の特徴量

        _n_conv_strides : int
            CNN の畳み込み処理（特徴マップ生成）でストライドさせる pixel 数
        _n_pool_wndsize : int
            プーリング処理用のウィンドウサイズ
        _n_pool_strides : int
            プーリング処理時のストライドさせる pixel 数

        _losses_train : list <float32>
            トレーニングデータでの損失関数の値の list

        _image_content_holder : placeholder
            内容層に画像データを供給するための placeholder
        _image_style_holder : placeholder
            スタイル層に画像データを供給するための placeholder

    [protedted] protedted な使用法を想定 


    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）


    """
    def __init__( 
            self,
            image_content_path,
            image_style_path,
            session = tf.Session( config = tf.ConfigProto(log_device_placement=True) ),
            epochs = 5000,
            eval_step = 50,
            weight_image_content = 5.0,
            weight_image_style = 500.0,
            weight_regularization = 100,
            n_strides = 1,
            n_pool_wndsize = 2,
            n_pool_strides = 2
        ):

        tf.set_random_seed(12)
        
        # メンバ変数の初期化
        self._session = session
        self._init_var_op = None

        self._loss_op = None
        self._loss_content_op = None
        self._loss_style_op = None
        self._loss_total_var_op = None
        self._optimizer = None
        self._train_step = None
        self._y_out_op = None

        self._weights = []
        self._biases = []

        # 各パラメータの初期化
        self._epochs = epochs
        self._eval_step = eval_step
        
        self._image_content_path = image_content_path
        self._image_style_path = image_style_path
        self._image_content = None
        self._image_style = None
        self._features_content = {}
        self._features_style = {}

        self._weight_image_content = weight_image_content
        self._weight_image_style = weight_image_style
        self._weight_regularization = weight_regularization
        
        self._n_strides = n_strides
        self._n_pool_wndsize = n_pool_wndsize
        self._n_pool_strides = n_pool_strides

        # evaluate 関連の初期化
        self._losses_train = []
        self._losses_content_train = []
        self._losses_style_train = []
        self._losses_total_var_train = []

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
        
        self._content_layer = "relu4_2"
        self._style_layers = [ "relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1" ]

        self._vgg_network = {}

        # place holder の設定
        self._image_content_holder = \
            tf.placeholder( 
                "float", 
                shape = ( (1,) + self._image_content.shape )    # ? : 4 つの次元を持つように画像の行列の形状を reshape 
            )

        self._image_style_holder = \
            tf.placeholder( 
                "float", 
                shape = ( (1,) + self._image_style.shape )      # ? : 4 つの次元を持つように画像の行列の形状を reshape
            )

        # 
        self._noize_image_var = tf.Variable(
                            tf.random_normal( shape = (1,) + self._image_content.shape ) * 0.256
                        )

        self._norm_mean_matrix = None

        return

    def print( self, str ):
        print( "----------------------------------" )
        print( self )
        print( str )

        print( "_session : ", self._session )
        print( "_init_var_op : ", self._init_var_op )

        print( "_loss_op : ", self._loss_op )
        print( "_optimizer : ", self._optimizer )
        print( "_train_step : ", self._train_step )

        print( "_y_out_op : ", self._y_out_op )

        print( "_epoches : ", self._epochs )
        print( "_eval_step : ", self._eval_step )

        print( "_image_content_path : ", self._image_content_path )
        print( "_image_style_path : ", self._image_style_path )

        print( "_weight_image_content : ", self._weight_image_content )
        print( "_weight_image_style : ", self._weight_image_style )
        print( "_weight_regularization : ", self._weight_regularization )

        print( "_vgg_layers : \n", self._vgg_layers )
        print( "_content_layer : ", self._content_layer )        
        print( "_style_layers : ", self._style_layers )
        print( "_vgg_network : \n", self._vgg_network )

        print( "_image_content.shape : \n", self._image_content.shape )
        print( "_image_style.shape : \n", self._image_style.shape )
        
        print( "_image_content_holder : \n", self._image_content_holder )
        print( "_image_style_holder : \n", self._image_style_holder )

        print( "_features_content :", self._features_content )
        print( "_features_style :", self._features_style )

        print( "_noize_image_var :", self._noize_image_var )

        print( "_n_strides : " , self._n_strides )
        print( "_n_pool_wndsize : " , self._n_pool_wndsize )
        print( "_n_pool_strides : " , self._n_pool_strides )

        print( "_weights : \n", self._weights )
        if( (self._session != None) and (self._init_var_op != None) ):
            print( self._session.run( self._weights ) )

        print( "_biases : \n", self._biases )
        #if( (self._session != None) and (self._init_var_op != None) ):
            #print( self._session.run( self._biases ) )

        print( "----------------------------------" )

        return

    def set_style_layers( style_layers = [ "relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1" ] ):
        """
        スタイル層の構成を指定する。
        
        [Input]
            style_layers : list <str>
                スタイル層の構成
                reluX_1 層の組み合わせが設定可能

        """
        self._style_layers = style_layers
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
        # new_shape : 合わせるための高さ、幅の倍率
        new_shape = ( self._image_content.shape[1] / self._image_style.shape[1] )
        #print( "_image_content.shape[1]", self._image_content.shape[1] )
        #print( "_image_style.shape[1]", self._image_style.shape[1] )
        #print( "new_shape", new_shape )
        self._image_style = scipy.misc.imresize( self._image_style, new_shape )

        return

    def load_model_info( self, mat_file_path ):
        """
        学習済みの StyleNet モデル用のデータである mat データから、パラメータを読み込み。
        imagenet-vgg-verydee-19.mat : この mat データは、MATLAB オブジェクトが含まれている

        [Input]
            mat_file_path : str
                imagenet-vgg-verydee-19.mat ファイルの path
        
        [Output]
            matrix_mean :
                画像を正規化するための正規化行列
                [ 123.68   116.779  103.939]

            network_weight :
                学習済み CNN モデルの重み、バイアス項等を含んだ MATLAB データ
                {'__header__': b'MATLAB 5.0 MAT-file Platform: posix, Created on: Sat Sep 19 12:27:40 2015',
                '__version__': '1.0', '__globals__': [], 
                'layers': array([[ array([[ (array([[ array([[[[ 0.39416704, -0.08419707, -0.03631314, ..., -0.10720515, ...
                ...
                array([[ 0.,  0.,  0.,  0.]],
                array(['conv'], dtype='<U4'), 
                array(['fc8'], dtype='<U3'),
                array([[ 1.,  1.]]))]],dtype=[('weights', 'O'), ('pad', 'O'), ('type', 'O'), ('name', 'O'), ('stride', 'O')])
                array([[ (array(['softmax'],dtype='<U7'),
                array(['prob'],...
        """
        vgg_data = scipy.io.loadmat( mat_file_path )
        #print( "vgg_data :\n", vgg_data )

        # ?
        normalization_matrix = vgg_data[ "normalization" ][0][0][0]
        #print( "normalization_matrix :\n", normalization_matrix )

        # ? matrix の row , colum に対しての平均値をとった matrix
        matrix_mean = np.mean( normalization_matrix, axis = (0,1) )

        # モデルの重み
        network_weight = vgg_data[ "layers" ][0]
        #print( "network_weight :\n", network_weight )

        return ( matrix_mean, network_weight )


    def model( self ):
        """
        モデルの定義を行い、
        最終的なモデルの出力のオペレーター self._y_out_op を設定する。
        [Output]
            self._y_out_op : Operator
                モデルの出力のオペレーター
        """
        #------------------------------------------------------
        # 学習済み StyleNet 用 CNN モデルのパラメータを読み込む
        #------------------------------------------------------
        vgg_file_path = "C:\Data\MachineLearning_DataSet\CNN-StyleNet\imagenet-vgg-verydeep-19.mat"

        # 学習済み CNN モデルの重み＋バイアス項を含んだ network_weights と
        # 画像を正規化するための正規化行列を取り出す。
        self._norm_mean_matrix, network_weights = self.load_model_info( mat_file_path = vgg_file_path )
        #print( "norm_mean_matrix :\n", self._norm_mean_matrix )
        #print( "network_weights :\n", network_weights )

        #------------------------------------
        # 内容画像層の構築
        #------------------------------------
        image_content_tsr = self._image_content_holder  # 入力 Tensor, 一時保存用 Tensor の設定
        network_content = {}                            # 内容画像層のモデル構造（Tensor型の list）
        self._features_content = {}                     # 内容画像層の特徴量

        # _vgg_layers を構成する layer から layer を取り出し、
        # 種類に応じて、モデルを具体的に構築していく。
        for ( i, layer ) in enumerate( self._vgg_layers ):
            # layer "convx_x" の先頭の文字列が畳み込み層を表す "c" の場合 
            if ( layer[0] == "c" ):
                # network_weights から weights とバイアス項に対応するデータを抽出
                weights, bias = network_weights[i][0][0][0][0]
                #print( "weights :\n", weights )
                #print( "bias :\n", bias )
                
                # StyleNet モデルに対応するように reshape
                weights = np.transpose( weights, (1,0,2,3) )
                bias = bias.reshape(-1)

                # 畳み込み層を構築
                conv_layer_op = \
                    tf.nn.conv2d(
                        input = image_content_tsr,
                        filter = tf.constant( weights ),                       # 畳込み処理で input で指定した Tensor との積和に使用する filter 行列（カーネル）
                        strides = [ 1, self._n_strides, self._n_strides, 1 ],  # strides[0] = strides[3] = 1. とする必要がある]
                        padding = "SAME"                                       # ゼロパディングを利用する場合は SAME を指定
                    )

                image_content_tsr = tf.nn.bias_add( conv_layer_op, bias )
                
                # リストに追加しておく
                self._weights.append( tf.constant( weights ) )
                self._biases.append( bias )

            # layer "relux_x" の先頭の文字列が Relu を表す "r" の場合 
            elif ( layer[0] == "r" ):
                image_content_tsr = tf.nn.relu( image_content_tsr )

            # layer "pool_x" の先頭の文字列がプーリング層を表す "p" の場合 
            else:
                image_content_tsr = \
                    tf.nn.max_pool(
                        value = image_content_tsr,
                        ksize = [ 1, self._n_pool_wndsize, self._n_pool_wndsize, 1 ],    # プーリングする範囲（ウィンドウ）のサイズ
                        strides = [ 1, self._n_pool_strides, self._n_pool_strides, 1 ],  # ストライドサイズ strides[0] = strides[3] = 1. とする必要がある
                        padding = "SAME"                                                 # ゼロパディングを利用する場合は SAME を指定
                    )
            
            network_content[ layer ] = image_content_tsr
            print( "image_content_tsr :\n", image_content_tsr )
        #
        print( "network_content :\n", network_content )

        # 内容画像の行列を正規化
        content_minus_mean_matrix = self._image_content - self._norm_mean_matrix
        content_norm_matrix = np.array( [content_minus_mean_matrix] )

        print( "content_minus_mean_matrix :\n", content_minus_mean_matrix.shape )
        print( "content_norm_matrix.shape :\n", content_norm_matrix.shape )

        # 構築した 内容画像層のモデルを session.run(...) し、
        # 学習済み CNN モデルから、内容層の特徴量（画像の内容、形状）を抽出する。
        self._features_content[ self._content_layer ] =\
            self._session.run( 
                network_content[ self._content_layer ], 
                feed_dict = { self._image_content_holder : content_norm_matrix } 
            )

        #------------------------------------
        # スタイル画像層の構築
        #------------------------------------
        image_style_tsr = self._image_style_holder
        network_style = {}          # スタイル画像層のモデル構造
        self._features_style = {}   # スタイル画像層の特徴量

        # _vgg_layers を構成する layer から layer を取り出し、
        # 種類に応じて、モデルを具体的に構築していく。
        for ( i, layer ) in enumerate( self._vgg_layers ):
            # layer "convx_x" の先頭の文字列が畳み込み層を表す "c" の場合 
            if ( layer[0] == "c" ):
                # network_weights から weights とバイアス項に対応するデータを抽出
                weights, bias = network_weights[i][0][0][0][0]

                # StyleNet モデルに対応するように reshape
                weights = np.transpose( weights, (1,0,2,3) )
                bias = bias.reshape(-1)

                # 畳み込み層を構築
                conv_layer_op = \
                    tf.nn.conv2d(
                        input = image_style_tsr,
                        filter = tf.constant( weights ),                       # 畳込み処理で input で指定した Tensor との積和に使用する filter 行列（カーネル）
                        strides = [ 1, self._n_strides, self._n_strides, 1 ],  # strides[0] = strides[3] = 1. とする必要がある]
                        padding = "SAME"                                       # ゼロパディングを利用する場合は SAME を指定
                    )

                image_style_tsr = tf.nn.bias_add( conv_layer_op, bias )

                # リストに追加しておく
                self._weights.append( tf.constant( weights ) )
                self._biases.append( bias )

            # layer "relux_x" の先頭の文字列が Relu を表す "r" の場合 
            elif ( layer[0] == "r" ):
                image_style_tsr = tf.nn.relu( image_style_tsr )

            # layer "pool_x" の先頭の文字列がプーリング層を表す "p" の場合 
            else:
                image_style_tsr = \
                    tf.nn.max_pool(
                        value = image_style_tsr,
                        ksize = [ 1, self._n_pool_wndsize, self._n_pool_wndsize, 1 ],    # プーリングする範囲（ウィンドウ）のサイズ
                        strides = [ 1, self._n_pool_strides, self._n_pool_strides, 1 ],  # ストライドサイズ strides[0] = strides[3] = 1. とする必要がある
                        padding = "SAME"                                                 # ゼロパディングを利用する場合は SAME を指定
                    )

            
            network_style[ layer ] = image_style_tsr
            print( "image_style_tsr :\n", image_style_tsr )
        #
        print( "network_style :\n", network_style )

        # スタイル画像の行列を正規化
        style_minus_mean_matrix = self._image_style - self._norm_mean_matrix
        style_norm_matrix = np.array( [style_minus_mean_matrix] )

        print( "style_minus_mean_matrix :\n", style_minus_mean_matrix.shape )
        print( "style_norm_matrix.shape :\n", style_norm_matrix.shape )

        # 構築した スタイル画像層のモデルを session.run(...) し、
        # 学習済み CNN モデルから、内容層の特徴量（画像の内容、形状）を抽出する。
        for layer in self._style_layers:
            layer_output =\
                self._session.run( 
                    network_style[ layer ], 
                    feed_dict = { self._image_style_holder : style_norm_matrix } 
                )

            # ?
            layer_output = np.reshape( layer_output, ( -1, layer_output.shape[3] ) )

            # ? グラム行列 A^T * A
            style_gram_matrix = np.matmul( layer_output.T, layer_output ) / layer_output.size

            # 特徴量のリストに格納
            self._features_style[ layer ] = style_gram_matrix

        #--------------------------------------------------------------------
        # 内容画像とスタイル画像を組み合わせる処理のモデルを構築
        # ここで構築したモデル（Variable）が、StyleNet のトレーニング対象となる
        # この処理は、ランダムノイズを適用した Variable に対する vgg_net
        #--------------------------------------------------------------------
        self._noize_image_var = tf.Variable(
                            tf.random_normal( shape = (1,) + self._image_content.shape ) * 0.256
                        )
        noize_image_tsr = self._noize_image_var

        # _vgg_layers を構成する layer から layer を取り出し、
        # 種類に応じて、モデルを具体的に構築していく。
        for ( i, layer ) in enumerate( self._vgg_layers ):
            # layer "convx_x" の先頭の文字列が畳み込み層を表す "c" の場合 
            if ( layer[0] == "c" ):
                # network_weights から weights とバイアス項に対応するデータを抽出
                weights, bias = network_weights[i][0][0][0][0]

                # StyleNet モデルに対応するように reshape
                weights = np.transpose( weights, (1,0,2,3) )
                bias = bias.reshape(-1)

                # 畳み込み層を構築
                conv_layer_op = \
                    tf.nn.conv2d(
                        input = noize_image_tsr,
                        filter = tf.constant( weights ),                       # 畳込み処理で input で指定した Tensor との積和に使用する filter 行列（カーネル）
                        strides = [ 1, self._n_strides, self._n_strides, 1 ],  # strides[0] = strides[3] = 1. とする必要がある]
                        padding = "SAME"                                       # ゼロパディングを利用する場合は SAME を指定
                    )

                noize_image_tsr = tf.nn.bias_add( conv_layer_op, bias )

                # リストに追加しておく
                self._weights.append( tf.constant( weights ) )
                self._biases.append( bias )

            # layer "relux_x" の先頭の文字列が Relu を表す "r" の場合 
            elif ( layer[0] == "r" ):
                noize_image_tsr = tf.nn.relu( noize_image_tsr )

            # layer "pool_x" の先頭の文字列がプーリング層を表す "p" の場合 
            else:
                noize_image_tsr = \
                    tf.nn.max_pool(
                        value = noize_image_tsr,
                        ksize = [ 1, self._n_pool_wndsize, self._n_pool_wndsize, 1 ],    # プーリングする範囲（ウィンドウ）のサイズ
                        strides = [ 1, self._n_pool_strides, self._n_pool_strides, 1 ],  # ストライドサイズ strides[0] = strides[3] = 1. とする必要がある
                        padding = "SAME"                                                 # ゼロパディングを利用する場合は SAME を指定
                    )

            
            self._vgg_network[ layer ] = noize_image_tsr
            print( "noize_image_tsr :\n", noize_image_tsr )
        #
        print( "_vgg_network :\n", self._vgg_network )
        #self._y_out_op = self._vgg_network

        return self._y_out_op


    def loss( self ):
        """
        損失関数の定義を行う。
                    
        [Output]
            self._loss_op : Operator
                損失関数を表すオペレーター
        """
        #-------------------------------------------------------
        # 内容画像層の損失値
        #-------------------------------------------------------
        self._loss_content_op = \
            self._weight_image_content * \
            ( 2 * tf.nn.l2_loss( 
                      self._vgg_network[self._content_layer] - self._features_content[self._content_layer] 
                  ) / self._features_content[ self._content_layer ].size
            )

        #-------------------------------------------------------
        # 内容画像層の損失値
        #-------------------------------------------------------
        loss_style = 0
        style_losses = []
        for style_layer in self._style_layers:
            # スタイル層の style_layer 番目のモデルの内容を抽出
            layer = self._vgg_network[ style_layer ]

            #
            feats, height, width, channels = [x.value for x in layer.get_shape()]
            size = height * width * channels
            features = tf.reshape( layer, (-1, channels) )

            style_gram_matrix = tf.matmul( tf.transpose(features), features ) / size
            style_expected = self._features_style[ style_layer ]
            #style_temp_loss = sess.run(2 * tf.nn.l2_loss(style_gram_matrix - style_expected) / style_expected.size)
            #print('Layer: {}, Loss: {}'.format(style_layer, style_temp_loss))

            style_losses.append(
                2 * tf.nn.l2_loss( style_gram_matrix - style_expected ) / style_expected.size
            )

        self._loss_style_op = 0
        self._loss_style_op += self._weight_image_style  * tf.reduce_sum( style_losses )

        #-------------------------------------------------------
        # 内容層とスタイル層のノイズ付き合成加工に応じた、全変動損失関数
        # 滑らかな結果を得るためことを目的としている
        #-------------------------------------------------------
        # ?
        # tf.reduce_prod(...) : 積の操作で縮約
        total_var_x = self._session.run( 
            tf.reduce_prod( self._noize_image_var[ :, 1:, :, : ].get_shape() )   #  
        )
        total_var_y = self._session.run(
            tf.reduce_prod( self._noize_image_var[ :, :, 1:, : ].get_shape() )
        )

        # ?
        first_term = self._weight_regularization  * 2
        second_term_numerator = tf.nn.l2_loss(
                                    self._noize_image_var[ :, 1:, :, : ] 
                                    - self._noize_image_var[ :, :( (1,) + self._image_content.shape )[1] - 1, :, : ]
                                )
        second_term = second_term_numerator / total_var_y
        third_term = ( 
                         tf.nn.l2_loss( 
                             self._noize_image_var[ :, :, 1:, : ] 
                             - self._noize_image_var[ :, :, :( (1,) + self._image_content.shape )[2] - 1, : ] 
                         ) / total_var_x 
                     )
        self._loss_total_var_op = first_term * ( second_term + third_term )

        #-------------------------------------------------------
        # 最終的な損失関数の Operator
        #-------------------------------------------------------
        self._loss_op = self._loss_content_op + self._loss_style_op + self._loss_total_var_op
        
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


    def run( self ):
        """
        StyleNet を駆動し、損失値と途中生成画像を生成し、一時保存する。
        """
        #----------------------------
        # 画像生成開始処理
        #----------------------------
        # Variable の初期化オペレーター
        self._init_var_op = tf.global_variables_initializer()

        # Session の run（初期化オペレーター）
        self._session.run( self._init_var_op )

        # 合成画像保存用ディレクトリの作成
        if ( os.path.isdir( "output_image" ) == False):
            os.makedirs( "output_image" )

        #-----------------------------------------
        # 画像生成処理
        #-----------------------------------------
        for epoch in range( self._epochs ):
            # 設定された最適化アルゴリズム Optimizer で
            # トレーニング処理（内容画像とスタイル画像に対するがノイズ付き合成）を run
            self._session.run( self._train_step )

            # 評価処理ステップの場合
            if ( (epoch + 1) % self._eval_step == 0 ):
                # 損失関数値の算出
                loss = self._session.run( self._loss_op )
                loss_content = self._session.run( self._loss_content_op )
                loss_style = self._session.run( self._loss_style_op )
                loss_total_var = self._session.run( self._loss_total_var_op )

                self._losses_train.append( loss )
                self._losses_content_train.append( loss_content )
                self._losses_content_train.append( loss_style )
                self._losses_content_train.append( loss_total_var )

                print( "epoch %d / loss = %0.1f / loss_content = %0.1f / loss_style = %0.1f / loss_total_var = %0.1f" % 
                      ( epoch + 1, loss, loss_content, loss_style, loss_total_var ) )
                
                # 途中生成画像の保存
                image_eval = self._session.run( self._noize_image_var )
                image_eval = image_eval.reshape( self._image_content.shape )
                image_eval_add_mean = image_eval + self._norm_mean_matrix

                output_file = "output_image/temp_output_image{}.jpg".format( epoch + 1 )
                output_add_mean_file = "output_image/temp_output_add_mean_image{}.jpg".format( epoch + 1 )
                scipy.misc.imsave( output_file, image_eval )
                scipy.misc.imsave( output_add_mean_file, image_eval_add_mean )

        # 最終生成画像の保存
        image_eval = self._session.run( self._noize_image_var )
        image_eval = image_eval.reshape(
                         self._image_content.shape
                     )

        output_file = "output_image/output_image.jpg"
        scipy.misc.imsave( output_file, image_eval )

        return


    def show_output_image( self ):
        """
        合成出力した画像を plot する。
        """
        return


    def save_output_image( self, file_dir, file_name ):
        """
        合成出力した画像を保存する。

        [Input]
            file_dir : str
                保存するディレクトリ名
            file_name : str
                保存するファイル名
        """

        return

    def save_output_image_gif( file_dir, file_name ):
        """
        合成出力した画像を生成過程を含めて保存する。
        """
        
        return