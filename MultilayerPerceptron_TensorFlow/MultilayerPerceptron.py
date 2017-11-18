# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境 (TensorFlow インストール済み)

"""
    更新情報
    [17/10/14] : 新規作成
    [17/xx/xx] : 
               : 
"""
import numpy

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops

# scikit-learn ライブラリ
from sklearn.utils import shuffle

# 自作クラス
#from NeuralNetworkBase import NeuralNetworkBase    # 親クラス

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


class MultilayerPerceptron( object ):
    """
    多層パーセプトロンを表すクラス
    TensorFlow での多層パーセプトロンの処理をクラス（任意の層に DNN 化可能な柔軟なクラス）でラッピングし、
    scikit-learn ライブラリの classifier, estimator とインターフェイスを共通化することで、
    scikit-learn ライブラリとの互換性のある自作クラス
    ----------------------------------------------------------------------------------------------------
    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
        _n_inputLayer : int
            入力層のノード数
        _n_hiddenLayers : shape = [h1,h2,h3,...] 
            h1 : 1 つ目の隠れ層のユニット数、h2 : 2 つ目の隠れ層のユニット数、...
        _n_outputLayer : int
            出力層のノード数

        _weights : list <Variable>
            モデルの各層の重みの Variable からなる list
        _biases : list <Variable>
            モデルの各層のバイアス項の  Variable からなる list

        _epochs : int
            エポック数（トレーニング回数）
        _batch_size : int
            ミニバッチ学習でのバッチサイズ

        _losses_train : list <float32>
            トレーニングデータでの損失関数の値の list

        _session : tf.Session()
            自身の Session
        _init_var_op : tf.global_variables_initializer()
            全 Variable の初期化オペレーター

        _activate_hiddenLayer : NNActivatation クラス
            隠れ層からの活性化関数の種類
        _activate_outputLayer : NNActivatation クラス
            出力層からの活性化関数

        _loss_op : Operator
            損失関数を表すオペレーター
        _optimizer : Optimizer
            モデルの最適化アルゴリズム
        _train_step : 
            トレーニングステップ
        _y_out_op : Operator
            モデルの出力のオペレーター
            
        _X_holder : placeholder
            入力層にデータを供給するための placeholder
        _t_holder : placeholder
            出力層に教師データを供給するための placeholder
        _keep_prob_holder : placeholder
            ドロップアウトしない確率 (1-p) にデータを供給するための placeholder
        
    [protedted] protedted な使用法を想定 

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """

    def __init__( 
            self, 
            session = tf.Session(), 
            n_inputLayer = 1, n_hiddenLayers = [1,1,1], n_outputLayer = 1, 
            activate_hiddenLayer = NNActivation(),
            activate_outputLayer = NNActivation(),
            epochs = 1000,
            batch_size = 1 
        ):
        """
        コンストラクタ（厳密にはイニシャライザ）
        """
        tf.set_random_seed(12)
        
        # 引数で指定された Session を設定
        self._session = session

        # 各パラメータの初期化
        self._n_inputLayer = n_inputLayer
        self._n_hiddenLayers = n_hiddenLayers
        self._n_outputLayer = n_outputLayer

        self._weights = []
        self._biases = []

        self._activate_hiddenLayer = activate_hiddenLayer
        self._activate_outputLayer = activate_outputLayer

        self._epochs = epochs
        self._batch_size = batch_size

        # evaluate 関連の初期化
        self._losses_train = []

        # オペレーターの初期化
        self._init_var_op = None
        self._loss_op = None
        self._optimizer = None
        self._train_step = None
        self._y_out_op = None

        # placeholder の初期化
        # shape の列（横方向）は、各層の次元（ユニット数）に対応させる。
        # shape の行は、None にして汎用性を確保
        self._X_holder = tf.placeholder( tf.float32, shape = [None, self._n_inputLayer] )
        self._t_holder = tf.placeholder( tf.float32, shape = [None, self._n_outputLayer] )
        self._keep_prob_holder = tf.placeholder( tf.float32 )

        return
    
    def print( self, str ):
        print( "----------------------------------" )
        print( "MultilayerPerceptron" )
        print( self )
        print( str )

        print( "_session : ", self._session )
        print( "_init_var_op :\n", self._init_var_op )

        print( "_n_inputLayer : ", self._n_inputLayer )
        print( "_n_hiddenLayers : ", self._n_hiddenLayers )
        print( "_n_outputLayer : ", self._n_outputLayer )

        print( "_weights : \n", self._weights )
        print( self._session.run( self._weights ) )

        print( "_biases : \n", self._biases )
        print( self._session.run( self._biases ) )

        print( "_activate_hiddenLayer :", self._activate_hiddenLayer )
        print( "_activate_outputLayer :", self._activate_outputLayer )

        print( "_epoches :", self._epochs )
        print( "_batch_size :", self._batch_size )

        print( "_loss_op :", self._loss_op )
        print( "_optimizer :", self._optimizer )
        print( "_train_step :", self._train_step )
        print( "_y_out_op :", self._y_out_op )

        print( "_X_holder : ", self._X_holder )
        print( "_t_holder : ", self._t_holder )
        print( "_keep_prob_holder : ", self._keep_prob_holder )

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
        weight_var = tf.Variable( init_tsr )
        
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

        #init_tsr = tf.zeros( shape = input_shape )
        init_tsr = tf.random_normal( shape = input_shape )

        # バイアス項の Variable
        bias_var = tf.Variable( init_tsr )

        return bias_var


    def model( self ):
        """
        モデルの定義（計算グラフの構築）を行い、
        最終的なモデルの出力のオペレーターを設定する。

        [Output]
            self._y_out_op : Operator
                モデルの出力のオペレーター
        """
        # 計算グラフの構築
        #print( "len( _n_hiddenLayers ) : ", len( self._n_hiddenLayers ) )
        #print( "len( [_n_hiddenLayers] ) : ", len( [self._n_hiddenLayers] ) )
        #print( "_n_hiddenLayers", self._n_hiddenLayers.shape )
        #print( "_n_hiddenLayers", self._n_hiddenLayers.shape[0] )

        #--------------------------------------------------------------
        # 隠れ層が１つのみの場合
        #--------------------------------------------------------------
        if ( len( self._n_hiddenLayers ) == 1 ):
            # 入力層 ~ 隠れ層
            self._weights.append( self.init_weight_variable( input_shape = [self._n_inputLayer, self._n_hiddenLayers[0] ] ) )
            self._biases.append( self.init_bias_variable( input_shape = [self._n_hiddenLayers[0]] ) )

            # 隠れ層への入力 : h_in = W*x + b
            h_in_op = tf.matmul( self._X_holder, self._weights[0] ) + self._biases[0]
            
            # 隠れ層からの出力
            h_out_op = self._activate_hiddenLayer.activate( h_in_op )
            #h_out_op = tf.nn.sigmoid( h_in_op )
            #print( "activate function [hidden layer] = sigmoid" )

            # 隠れ層 ~ 出力層
            self._weights.append( self.init_weight_variable( input_shape = [self._n_hiddenLayers[0], self._n_outputLayer] ) )
            self._biases.append( self.init_bias_variable( input_shape = [self._n_outputLayer] ) )
        
        #--------------------------------------------------------------
        # 隠れ層が複数個ある場合
        #--------------------------------------------------------------
        else:
            # i=0 : 入力層 ~ 隠れ層
            # i=1,2... : 隠れ層 ~ 隠れ層
            for (i, n_hidden) in enumerate( self._n_hiddenLayers ):
                # 入力層 ~ 隠れ層
                if (i==0):
                    input_dim = self._n_inputLayer
                    input_holder = self._X_holder

                # 隠れ層 ~ 隠れ層
                else:
                    input_dim = self._n_hiddenLayers[i-1]
                    input_holder = h_out_op

                # 重みの Variable の list に、入力層 ~ 隠れ層 or 隠れ層 ~ 隠れ層の重みを追加
                self._weights.append( self.init_weight_variable( input_shape = [input_dim, n_hidden] ) )

                # バイアス項の Variable の list に、入力層 ~ 隠れ層 or 隠れ層 ~ 隠れ層のバイアス項を追加
                self._biases.append( self.init_bias_variable( input_shape = [n_hidden] ) )

                # 隠れ層への入力 : h_in = W*x + b
                h_in_op = tf.matmul( input_holder, self._weights[-1] ) + self._biases[-1]

                # 隠れ層からの出力
                h_out_op = self._activate_hiddenLayer.activate( h_in_op )
                #h_out_op = tf.nn.sigmoid( h_in_op )
                #print( "activate function [hidden layer] = sigmoid" )
                #h_out_op = tf.nn.relu( h_in_op )
                #print( "activate function [hidden layer] = Relu" )

                # ドロップアウト処理
                #output_holder = tf.nn.dropout( h_out_op, self._keep_prob_holder )
                #output_holder = h_out_op
        
            # 隠れ層 ~ 出力層
            self._weights.append( self.init_weight_variable( input_shape = [self._n_hiddenLayers[-1], self._n_outputLayer] ) )
            self._biases.append( self.init_bias_variable( input_shape = [self._n_outputLayer] ) )


        #--------------------------------------------------------------
        # 出力層への入力
        #--------------------------------------------------------------
        y_in_op = tf.matmul( h_out_op, self._weights[-1] ) + self._biases[-1]

        #--------------------------------------------------------------
        # モデルの出力
        #--------------------------------------------------------------
        self._y_out_op = self._activate_outputLayer.activate( y_in_op )
        
        # ２分類問題の場合
        # sigmoid
        #self._y_out_op = tf.nn.sigmoid( y_in_op )
        #print( "activate function [output layer] = sigmoid" )
        
        # Relu
        #self._y_out_op = tf.nn.relu( y_in_op )
        #print( "activate function [output layer] = Relu" )

        # 多分類問題の場合
        # softmax
        #self._y_out_op = tf.nn.softmax( y_in_op )
        #print( "activate function [output layer] = softmax" )

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
        # TensorFlow 用にデータを reshape
        #y_train.reshape( [len(y_train), 1] )

        #----------------------------
        # 学習開始処理
        #----------------------------
        # Variable の初期化オペレーター
        self._init_var_op = tf.global_variables_initializer()

        # Session の run（初期化オペレーター）
        self._session.run( self._init_var_op )
        
        #print( "init_weights", self._session.run( self._weights ) )

        #-------------------
        # 学習処理
        #-------------------
        n_batches = len( X_train ) // self._batch_size     # バッチ処理の回数

        # for ループでエポック数分トレーニング
        for epoch in range( self._epochs ):
            # ミニバッチ学習処理のためランダムサンプリング
            X_train_shuffled, y_train_shuffled = shuffle( X_train, y_train )
            
            # ２クラス分類の場合
            if (self._n_outputLayer == 1):
                # shape を placeholder の形状に合わせるためにするため [...] で囲み、transpose() する。
                # shape を (n_samples, → (n_samples,1) に reshape
                y_train_shuffled = numpy.transpose( [ y_train_shuffled ] )
            
            for i in range( n_batches ):
                it_start = i * self._batch_size
                it_end = it_start + self._batch_size

                self._session.run(
                    self._train_step,
                    feed_dict = {
                        self._X_holder: X_train_shuffled[it_start:it_end],
                        self._t_holder: y_train_shuffled[it_start:it_end]
                    }
                )

            # 損失関数の値をストック
            # ２クラス分類の場合
            if (self._n_outputLayer == 1):
                # shape を (n_samples, → (n_samples,1) に reshape
                loss = self._loss_op.eval(
                           session = self._session,
                           feed_dict = {
                               self._X_holder: X_train,
                               self._t_holder: numpy.transpose( [ y_train ] )
                           }
                       )
            # 多クラス分類の場合
            else:
                loss = self._loss_op.eval(
                           session = self._session,
                           feed_dict = {
                               self._X_holder: X_train,
                               self._t_holder: y_train
                           }
                       )

            self._losses_train.append( loss )

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
        # 出力層の活性化関数が sigmoid のとき（２クラスの識別）
        if ( self._activate_outputLayer._node_name == "Activate_Sigmoid_op" ):
            predict_op = tf.to_int64( tf.greater( self._y_out_op, 0.5 ) )
        # 出力層の活性化関数が softmax のとき（多クラスの識別）
        elif ( self._activate_outputLayer._node_name == "Activate_Softmax_op" ):
            predict_op = tf.arg_max( input = self._y_out_op, dimension = 1 )
        else:
            predict_op = tf.to_int64( tf.greater( self._y_out_op, 0.5 ) )

        predict = predict_op.eval( 
                   session = self._session,
                   feed_dict = {
                       self._X_holder: X_test
                   }
               )
        
        
        return predict


    def predict_proba( self, X_test ):
        """
        fitting 処理したモデルで、推定を行い、クラスの所属確率の予想値を返す。
        proba : probability

        [Input]
            X_test : numpy.ndarry ( shape = [n_samples, n_features] )
                予想したい特徴行列
        """
        prob = self._y_out_op.eval(
                   session = self._session,
                   feed_dict = {
                       self._X_holder: X_test 
                   }
               )

        # X_test のデータ数、特徴数に応じて reshape
        #prob = prob.reshape( (len[X_test], len[X_test[0]]) )

        return prob


    def accuracy( self, X_test, y_test):
        """
        指定したデータでの正解率 [accuracy] を計算する。
        """
        # 出力層の活性化関数が sigmoid のとき（２クラスの識別）
        if ( self._activate_outputLayer._node_name == "Activate_Sigmoid_op" ):
            correct_predict_op = tf.equal( 
                                     tf.to_float( tf.greater( self._y_out_op, 0.5 ) ), 
                                     self._t_holder 
                                 )
        # 出力層の活性化関数が softmax のとき（多クラスの識別）
        elif ( self._activate_outputLayer._node_name == "Activate_Softmax_op" ):
            correct_predict_op = tf.equal(
                                     tf.arg_max( self._y_out_op, dimension = 1 ),
                                     tf.arg_max( self._t_holder, dimension = 1 )
                                 )
        else:
            correct_predict_op = tf.equal( 
                                     tf.to_float( tf.greater( self._y_out_op, 0.5 ) ), 
                                     self._t_holder 
                                 )

        # correct_predict_op は、feed_dict で与えるデータ分（全データ）の結果（合っていた数）を返すので、
        # tf.reduce_mean(..) でその平均値を計算すれば、合っていた数 / 全データ数 = 正解率　が求まる。
        accuracy_op = tf.reduce_mean( tf.cast( correct_predict_op, tf.float32 ) )
        
        # ２クラス分類の場合
        if (self._n_outputLayer == 1):
            # shape を (n_samples, → (n_samples,1) に reshape
            y_test = numpy.transpose( [ y_test ] )

        accuracy = accuracy_op.eval(
                       session = self._session,
                       feed_dict = {
                           self._X_holder: X_test,
                           self._t_holder: y_test
                       }                       
                   )

        return accuracy
