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

# 親クラス（自作クラス）
from NeuralNetworkBase import NeuralNetworkBase


class MultilayerPerceptron( object ):
    """
    多層パーセプトロンを表すクラス（自作クラス）
    ニューラルネットワークの基底クラス NeuralNetworkBase （自作の基底クラス）を継承している。
    
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

        _losses_train : list <float32>
            トレーニングデータでの損失関数の値の list


        _session : tf.Session()
            自身の Session

        _init_var_op : tf.global_variables_initializer()
            全 Variable の初期化オペレーター
    
        _loss_op : Operator
            損失関数を表すオペレーター

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

    def __init__( self, n_inputLayer = 1, n_hiddenLayers = [1,1,1], n_outputLayer = 1 ):
        """
        コンストラクタ（厳密にはイニシャライザ）
        """
        # メンバ変数の初期化
        self._n_inputLayer = n_inputLayer
        self._n_hiddenLayers = n_hiddenLayers
        self._n_outputLayer = n_outputLayer

        self._weights = []
        self._biases = []

        #
        self._session = tf.Session()
        self._init_var_op = None
        #self._session = None
        #self._init_var_op = None

        self._loss_op = None
        self._y_out_op = None

        # shape の列（横方向）は、各層の次元（ユニット数）に対応させる。
        # shape の行は、None にして汎用性を確保
        self._X_holder = tf.placeholder( tf.float32, shape = [None, self._n_inputLayer] )
        self._t_holder = tf.placeholder( tf.float32, shape = [None, self._n_outputLayer] )
        self._keep_prob_holder = tf.placeholder( tf.float32 )
        
        #
        self._losses_train = []

        return
    
    def print( self, str ):
        print( "----------------------------------" )
        print( "MultilayerPerceptron" )
        print( self )
        print( str )

        print( "_n_inputLayer : ", self._n_inputLayer )
        print( "_n_hiddenLayers : ", self._n_hiddenLayers )
        print( "_n_outputLayer : ", self._n_outputLayer )

        print( "_session : ", self._session )
        print( "_init_var_op :\n", self._init_var_op )
        print( "_loss_op :", self._loss_op )
        print( "_y_out_op :", self._y_out_op )

        print( "_X_holder : ", self._X_holder )
        print( "_t_holder : ", self._t_holder )
        print( "_keep_prob_holder : ", self._keep_prob_holder )

        print( "_weights : \n", self._weights )
        print( self._session.run( self._weights ) )

        print( "_biases : \n", self._biases )
        print( self._session.run( self._biases ) )

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
        init_tsr = tf.zeros( shape = input_shape )

        # バイアス項の Variable
        bias_var = tf.Variable( init_tsr )

        return bias_var


    def models( self ):
        """
        モデルの定義（計算グラフの構築）を行い、
        最終的なモデルの出力のオペレーター self._y_out_op を設定する。

        [Output]
            self._y_out_op : Operator
                モデルの出力のオペレーター
        """
        # 計算グラフの構築
        # i=0 : 入力層 ~ 隠れ層
        # i=1,2... : 隠れ層 ~ 隠れ層
        for (i, n_hidden) in enumerate( self._n_hiddenLayers ):
            print("i=", i)
            print("n_hidden=", n_hidden)

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
            #h_out_op = tf.nn.relu( h_in_op )
            h_out_op = tf.nn.sigmoid( h_in_op )
            #output_holder = tf.nn.dropout( h_out_op, self._keep_prob_holder )
            #output_holder = h_out_op

        # 隠れ層 ~ 出力層
        self._weights.append( self.init_weight_variable( input_shape = [self._n_hiddenLayers[-1], self._n_outputLayer] ) )
        self._biases.append( self.init_bias_variable( input_shape = [self._n_outputLayer] ) )

        # 出力層への入力
        y_in_op = tf.matmul( h_out_op, self._weights[-1] ) + self._biases[-1]

        # モデルの出力
        # ２分類問題の場合
        if (self._n_inputLayer == 2 ):
            # sigmoid
            self._y_out_op = tf.nn.sigmoid( y_in_op )

            # Relu
            #self._y_out_op = tf.nn.relu( y_in_op )

        # 多分類問題の場合
        else:
            # softmax
            self._y_out_op = tf.nn.softmax( y_in_op )

        return self._y_out_op

    
    def loss( self ):
        """
        損失関数の定義を行う。
        
        [Output]
            self._loss_op : Operator
                損失関数を表すオペレーター
        """

        # クロス・エントロピー
        # tf.clip_by_value(...) : 下限値、上限値を設定
        """
        self._loss_op = tf.reduce_mean(                     # ミニバッチ度に平均値を計算
                            -tf.reduce_sum( 
                                self._t_holder * tf.log( tf.clip_by_value(self._y_out_op, 1e-10, 1.0) ), 
                                reduction_indices = [1]     # 行列の方向
                            )
                        )
        """
        # ２値問題の場合
        """
        self._loss_op = -tf.reduce_sum( 
                            self._t_holder * tf.log(self._y_out_op) + (1-self._t_holder)*tf.log(1-self._y_out_op)
                        )
        """
        self._loss_op = tf.reduce_mean(
                            tf.square( self._t_holder - self._y_out_op )
                        )

        return self._loss_op

    
    def optimizer( self ):
        """
        モデルの最適化アルゴリズムの設定を行う。

        [Output]
            optimizer の train_step
        """
        optimizer = tf.train.GradientDescentOptimizer( learning_rate = 0.01 )
        train_step = optimizer.minimize( self._loss_op )

        return train_step


    def fit( self, X_train, y_train ):
        """
        指定されたトレーニングデータで、モデルの fitting 処理を行う。

        [Input]
            X_train : numpy.ndarray ( shape = [n_samples, n_features] )
                トレーニングデータ（特徴行列）
            
            y_train : numpy.ndarray ( shape = [n_samples] )
                レーニングデータ用のクラスラベル（教師データ）のリスト

        [Output]
            self : 自身のオブジェクト
        """
        # TensorFlow 用にデータを reshape
        #y_train.reshape( [len(y_train), 1] )

        #----------------------------
        # モデルの設定＆学習開始処理
        #----------------------------
        # モデルの設定
        #self.models()

        # 損失関数の設定
        #self.loss()

        # 最適化アルゴリズムの設定
        train_step = self.optimizer()
        print( "train_step", train_step )

        # Variable の初期化オペレーター
        self._init_var_op = tf.global_variables_initializer()

        # Session の run（初期化オペレーター）
        #self._session = tf.Session()
        self._session.run( self._init_var_op )
        
        print( "init_weights", self._session.run( self._weights ) )

        #-------------------
        # 学習処理
        #-------------------
        epochs = len( X_train )
        batch_size = 10

        # for ループでエポック数分トレーニング
        for epoch in range( epochs ):
            # ミニバッチ学習処理のためランダムサンプリング
            X_train_shuffled, y_train_shuffled = shuffle( X_train, y_train )
            
            # shape を placeholder の形状に合わせるためにするため [...] で囲み、transpose() する。
            y_train_shuffled = numpy.transpose( [ y_train_shuffled ] )

            #print( "X_train_shuffled", X_train_shuffled )
            #print( "y_train_shuffled", y_train_shuffled )

            for i in range( batch_size ):
                it_start = i * batch_size
                it_end = it_start + batch_size

                self._session.run(
                    train_step,
                    feed_dict = {
                        self._X_holder: X_train_shuffled[it_start:it_end],
                        self._t_holder: y_train_shuffled[it_start:it_end]
                    }
                )

            # 損失関数の値をストック
            loss = self._loss_op.eval(
                       session = self._session,
                       feed_dict = {
                           self._X_holder: X_train_shuffled,
                           #self._t_holder: numpy.transpose( [ y_train ] )
                           self._t_holder: y_train_shuffled
                       }
                   )
            #print( "loss", loss )
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
        """
        predict_op = tf.equal( 
                        tf.to_float( tf.greater( self._y_out_op, 0.5 ) ), 
                        self._t_holder
                     )

        """
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
        """

        return
