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

        _losses_train : list <float32>
            トレーニングデータでの損失関数の値の list

    [protedted] protedted な使用法を想定 

        image_height : int
            入力画像データの高さ（ピクセル単位）
        image_width : int
            入力画像データの幅（ピクセル単位）
        n_channels : int
            入力画像データのチャンネル数
            1 : グレースケール画像
        n_labels : int
            出力ラベル数（全結合層の出力側のノード数）

        X_holder : placeholder
            入力層に入力データを供給するための placeholder
        t_holder : placeholder
            出力層に教師データを供給するための placeholder

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
        fc6_op : Operator
        fc7_op : Operator
        fc8_op : Operator

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）


    """
    def __init__(
            self, 
            session = tf.Session(),
            epochs = 20,
            batch_size = 50,
            eval_step = 1,
            save_step = 100,
            image_height = 32,
            image_width = 32,
            n_channels = 1,
            n_labels = 10
        ):

        super().__init__( session )

        # 各パラメータの初期化
        self._epochs = epochs
        self._batch_size = batch_size
        self._eval_step = eval_step
        self._save_step = save_step

        self.image_height = image_height
        self.image_width = image_width
        self.n_channels = n_channels
        self.n_labels = n_labels

        # evaluate 関連の初期化
        self._losses_train = []

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
        self.fc6_op = None
        self.fc7_op = None
        self.fc8_op = None

        # placeholder の初期化
        # shape の列（横方向）は、各層の次元（ユニット数）に対応させる。
        # shape の行は、None にして汎用性 batch_size を確保
        self.X_holder = tf.placeholder( 
                            tf.float32, 
                            shape = [ None, self.image_height, self.image_width, self.n_channels ],     # [ None, image_height, image_width, n_channels ]
                            name = "X_holder"
                        )

        self.t_holder = tf.placeholder( 
                            tf.float32, 
                            shape = [ None, self.n_labels ],  # [None, n_labels]
                            name = "t_holder"
                        )

        return


    def print( self, str ):
        print( "----------------------------------" )
        print( str )
        print( self )

        print( "_session :", self._session )
        print( "_init_var_op : \n", self._init_var_op )
        print( "_y_out_op :", self._y_out_op )

        print( "_epoches : ", self._epochs )
        print( "_batch_size : ", self._batch_size )
        print( "_eval_step : ", self._eval_step )
        print( "_save_step : ", self._save_step )

        print( "image_height : " , self.image_height )
        print( "image_width : " , self.image_width )
        print( "n_channels : " , self.n_channels )
        print( "n_labels : " , self.n_labels )

        print( "_loss_op :", self._loss_op )
        print( "_optimizer :", self._optimizer )
        print( "_train_step :", self._train_step )

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
        
        print( "_losses_train :\n", self._losses_train )

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


    def convolution_layer( 
            self, 
            input_tsr, 
            filter_height, filter_width, n_input_channels, n_output_channels, 
            name = "conv", reuse = False
        ):
        """
        畳み込み層を構築する。

        [Input]

        [Output]

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


    def fully_conntected_layer( 
            self, 
            input_tsr, 
            n_output_units,
            nnActivation, 
            name = "fc", 
            reuse = False 
        ):
        """
        VGG16 の全結合層を構築する。
        """
        # Variable の名前空間（スコープ定義）
        with tf.variable_scope( name, reuse = reuse ):
            # input_tsr の shape を取り出し、list 構造に変換する。（ 0 番目の要素であるバッチ数は除外）
            shape = input_tsr.get_shape().as_list()[1:]
            #print( "input_tsr / shape", shape )

            # input_tsr の shape の積が入力 Tensor の shape となる。
            n_input_units = np.prod(shape)
            #print( "n_input_units", n_input_units )

            # 入力 Tensor の shape を２次元化
            if( n_input_units > 1 ):
                input_tsr_reshaped = tf.reshape( input_tsr, shape = [-1, n_input_units] )

            #
            weights = self.init_bias_variable( input_shape = [ n_input_units, n_output_units ] )
            biases = self.init_bias_variable( input_shape = [n_output_units] )

            # 全結合層の入力と出力
            fc_in_op = tf.nn.bias_add( tf.matmul(input_tsr_reshaped, weights), biases )
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

        self.conv4_3_op = self.convolution_layer( 
                              input_tsr = self.conv4_2_op, 
                              filter_height = 3, filter_width = 3, n_input_channels = 512, n_output_channels = 512,
                              name = "conv4_3",
                              reuse = False
                          )

        self.pool4_op = self.pooling_layer( input_tsr = self.conv4_3_op, name = "pool4", reuse = False )

        #-----------------------------------------------------------------------------
        # layer 5
        #-----------------------------------------------------------------------------
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

        #-----------------------------------------------------------------------------
        # fc layers
        #-----------------------------------------------------------------------------
        # fc6
        self.fc6_op = self.fully_conntected_layer( 
                          input_tsr = self.pool5_op,
                          n_output_units = 512,
                          nnActivation = Relu(), 
                          name = "fc6", 
                          reuse = False
                      )

        """
        self.fc6_op = self.fully_conntected_layer( 
                          input_tsr = self.pool5_op,
                          n_output_units = 4096,
                          nnActivation = Relu(), 
                          name = "fc6", 
                          reuse = False
                      )
        """

        # fc7
        """
        self.fc7_op = self.fully_conntected_layer( 
                          input_tsr = self.fc6_op,
                          n_output_units = 1000,
                          nnActivation = Relu(), 
                          name = "fc7", 
                          reuse = False
                      )
        """

        # fc8
        """
        self.fc8_op = self.fully_conntected_layer( 
                          input_tsr = self.fc7_op,
                          n_output_units = 1000,
                          nnActivation = Softmax(), 
                          name = "fc8", 
                          reuse = False
                      )
        """

        #-----------------------------------------------------------------------------
        # params for converting to answer-label-size (for CIFAR-10 dataset)
        #-----------------------------------------------------------------------------
        weight = tf.Variable( tf.truncated_normal([512, self.n_labels], 0.0, 1.0) * 0.01, name='w_last' )
        bias = tf.Variable( tf.truncated_normal([self.n_labels], 0.0, 1.0) * 0.01, name='b_last' )

        #-----------------------------------------------------------------------------
        # model output
        #-----------------------------------------------------------------------------
        y_in_op = tf.add( tf.matmul(self.fc6_op, weight), bias )

        self._y_out_op = tf.nn.softmax( y_in_op )
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

        # 余分な次元を削除
        #t_holder_ = tf.squeeze( tf.cast(self.t_holder, tf.float32) )

        # 指定された loss 関数を設定
        self._loss_op = nnLoss.loss( t_holder = self.t_holder, y_out_op = self._y_out_op )

        return self._loss_op


    def optimizer( self, nnOptimizer, reuse = False ):
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
        def generate_minibatch( X, y, batch_size, bSuffle = True, random_seed = 12 ):
            """
            指定された（トレーニング）データから、ミニバッチ毎のデータを生成する。
            （各 Epoch 処理毎に呼び出されることを想定している。）

            """
            # 各 Epoch 度に shuffle し直す。
            if( bSuffle == True ):
                idxes = np.arange( y.shape[0] )   # 0 ~ y.shape[0] の連番 idxes を生成

                # メルセンヌツイスターというアルゴリズムを使った擬似乱数生成器。
                # コンストラクタに乱数の種(シード)を設定。
                random_state = np.random.RandomState( random_seed )
                random_state.shuffle( idxes )
                
                # shuffle された連番 idxes 値のデータに置き換える。
                X = X[idxes]
                y = y[idxes]

            # 0 ~ 行数まで batch_size 間隔でループ
            for i in range( 0, X.shape[0], batch_size ):
                # yield 文で逐次データを return（関数の処理を一旦停止し、値を返す）
                # メモリ効率向上のための処理
                yield ( np.squeeze( X[i:i+batch_size,:] ), np.squeeze( y[i:i+batch_size] ) )


        #----------------------------
        # 学習開始処理
        #----------------------------
        # Variable の初期化オペレーター
        self._init_var_op = tf.global_variables_initializer()

        # Session の run（初期化オペレーター）
        self._session.run( self._init_var_op )

        # ミニバッチの繰り返し回数
        n_batches = X_train.shape[0] // self._batch_size     # バッチ処理の回数
        n_minibatch_iterations = self._epochs * n_batches    # ミニバッチの総繰り返し回数
        n_minibatch_iteration = 0                            # ミニバッチの現在の繰り返し回数
        
        print( "n_batches :", n_batches )
        print( "n_minibatch_iterations :", n_minibatch_iterations )

        # （学習済みモデルの）チェックポイントファイルの作成
        self.save_model()

        #-------------------
        # 学習処理
        #-------------------
        # for ループでエポック数分トレーニング
        for epoch in range( 1, self._epochs+1 ):
            # ミニバッチサイズ単位で for ループ
            # エポック毎に shuffle し直す。
            gen_minibatch = generate_minibatch( X = X_train, y = y_train , batch_size = self._batch_size, bSuffle = True, random_seed = 12 )

            # n_batches = X_train.shape[0] // self._batch_size 回のループ
            for i ,(batch_x, batch_y) in enumerate( gen_minibatch, 1 ):
                n_minibatch_iteration += 1

                # 設定された最適化アルゴリズム Optimizer でトレーニング処理を run
                """
                self._session.run(
                    self._train_step,
                    feed_dict = {
                        self.X_holder: batch_x,
                        self.t_holder: batch_y
                    }
                )
                """

                loss, _, = self._session.run(
                               [ self._loss_op, self._train_step ],
                               feed_dict = {
                                   self.X_holder: batch_x,
                                   self.t_holder: batch_y
                               }
                           )

                self._losses_train.append( loss )

                print( "Epoch: %d/%d | minibatch iteration: %d/%d | loss = %0.5f |" % 
                      ( epoch, self._epochs, n_minibatch_iteration, n_minibatch_iterations, loss ) )

                # モデルの保存処理を行う loop か否か
                # % : 割り算の余りが 0 で判断
                if ( ( (n_minibatch_iteration) % self._save_step ) == 0 ):
                    self.save_model()

        # fitting 処理終了後、モデルのパラメータを保存しておく。
        self.save_model()

        return self._y_out_op


    def predict( self, X_test ):
        """
        fitting 処理したモデルで、推定を行い、予想クラスラベル値を返す。
        [Input]
            X_test : numpy.ndarry ( shape = [n_samples, n_features] )
                予想したい特徴行列
        [Output]
            results : numpy.ndarry ( shape = [n_samples] )
                予想結果（分類モデルの場合は、クラスラベル）
        """
        probs = self._session.run(
                   self._y_out_op,
                   feed_dict = { self.X_holder: X_test }
               )
        
        #print( "probs :", probs )

        # numpy.argmax(...) : 多次元配列の中の最大値の要素を持つインデックスを返す
        # axis : 最大値を読み取る軸の方向 (1 : 行方向)
        predict = np.argmax( probs, axis = 1 )
        print( "predict :", predict )

        return predict

        return


    def predict_proba( self, X_test ):
        """
        fitting 処理したモデルで、推定を行い、クラスの所属確率の予想値を返す。
        proba : probability
        [Input]
            X_test : numpy.ndarry ( shape = [n_samples, n_features] )
                予想したい特徴行列
        """
        probs = self._session.run(
                   self._y_out_op,
                   feed_dict = { self.X_holder: X_test }
               )
        
        #print( "probs :", probs )

        return probs


    def accuracy( self, X_test, y_test ):
        """
        指定したデータでの正解率 [accuracy] を計算する。
        """
        # 予想ラベルを算出する。
        predict = self.predict( X_test )

        # 正解数
        n_corrects = np.sum( np.equal( predict, y_test ) )
        #print( "np.equal( predict, y_test ) :", np.equal( predict, y_test ) )
        #print( "n_correct :", n_correct )

        # 正解率 = 正解数 / データ数
        accuracy = n_corrects / X_test.shape[0]

        return accuracy


    def accuracy_labels( self, X_test, y_test ):
        """
        指定したデータでのラベル毎の正解率 [acuuracy] を算出する。
        """
        # 予想ラベルを算出する。
        predict = self.predict( X_test )

        # ラベル毎の正解率のリスト
        n_labels = len( np.unique( y_test ) )    # ユニークな要素数
        accuracys = []

        for label in range(n_labels):
            # label 値に対応する正解数
            # where(...) : 条件を満たす要素番号を抽出
            n_correct = len( np.where( predict == label )[0] )
            """
            n_correct = np.sum( 
                            np.equal( 
                                np.where( predict == label )[0], 
                                np.where( y_test == label )[0]
                            ) 
                        )
            """

            # サンプル数
            n_sample = len( np.where( y_test == label )[0] )

            accuracy = n_correct / n_sample
            accuracys.append( accuracy )
            
            #print( "np.where( predict == label ) :", np.where( predict == label ) )
            #print( "np.where( predict == label )[0] :", np.where( predict == label )[0] )
            print( " %d / n_correct = %d / n_sample = %d" % ( label, n_correct, n_sample ) )

        return accuracys

