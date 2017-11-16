# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境 (TensorFlow インストール済み)

"""
    更新情報
    [17/11/04] : 新規作成
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
from NNActivation import NNActivation               # ニューラルネットワークの活性化関数を表すクラス


class ConvolutionalNN(object):
    """
    畳み込みニューラルネットワーク [CNN : Convolutional Neural Network] を表すクラス.
    TensorFlow での CNN の処理をクラス（任意の層に DNN 化可能な柔軟なクラス）でラッピングし、
    scikit-learn ライブラリの classifier, estimator とインターフェイスを共通化することで、
    scikit-learn ライブラリとの互換性のある自作クラス
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

        _weights : list <Variable>
            モデルの各層の重みの Variable からなる list
        _biases : list <Variable>
            モデルの各層のバイアス項の  Variable からなる list

        _learning_rate : float
            学習率
        _epochs : int
            エポック数（トレーニング回数）
        _batch_size : int
            ミニバッチ学習でのバッチサイズ

        _image_width : int
            入力画像データの幅（ピクセル単位）
        _image_height : int
            入力画像データの高さ（ピクセル単位）
        _n_ConvLayer_features : list <int>
            畳み込み層の特徴量の数
        _n_channels : int
            入力画像データのチャンネル数
            1 : グレースケール画像

        _n_strides : int
            CNN の畳み込み処理でストライドさせる pixel 数
        _n_fullyLayers : int
            全結合層の入力側のノード数
        _n_labels : int
            出力ラベル数（全結合層の出力側のノード数）

        _losses_train : list <float32>
            トレーニングデータでの損失関数の値の list

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
            session = tf.Session( config = tf.ConfigProto(log_device_placement=True) ),
            learning_rate = 0.01, 
            epochs = 1000,
            batch_size = 1,
            image_width = 28,
            image_height = 28,
            n_ConvLayer_features = [25, 50],
            n_channels = 1,
            n_strides = 1,
            n_fullyLayers = 100,
            n_labels = 10
        ) :
        """
        コンストラクタ（厳密にはイニシャライザ）
        """
        tf.set_random_seed(12)

        # このクラスの処理で使われる Session の設定
        self._session = session

        # 各種 Operator の初期化
        self._init_var_op = None
        self._loss_op = None
        self._optimizer = None
        self._train_step = None
        self._y_out_op = None

        # 各パラメータの初期化
        self._weights = []
        self._biases = []

        self._learning_rate = learning_rate
        self._epochs = epochs
        self._batch_size = batch_size
        
        self._image_width = image_width
        self._image_height = image_height
        self._n_ConvLayer_features = n_ConvLayer_features
        self._n_channels = n_channels
        self._n_strides = n_strides
        self._n_fullyLayers = n_fullyLayers
        self._n_labels = n_labels

        # placeholder の初期化
        # shape の列（横方向）は、各層の次元（ユニット数）に対応させる。
        # shape の行は、None にして汎用性を確保
        self._X_holder = tf.placeholder( 
                             tf.float32, 
                             shape = [ None, image_width, image_height, n_channels ]
                         )

        self._t_holder = tf.placeholder( 
                             tf.int32, 
                             shape = [ None ]
                         )

        self._keep_prob_holder = tf.placeholder( tf.float32 )

        # evaluate 関連の初期化
        self._losses_train = []

        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "CNN" )
        print( self )
        print( str )

        print( "_session : ", self._session )

        print( "_weights : \n", self._weights )
        print( self._session.run( self._weights ) )

        print( "_biases : \n", self._biases )
        print( self._session.run( self._biases ) )

        print( "_learning_rate : ", self._learning_rate )
        print( "_epoches : ", self._epochs )
        print( "_batch_size : ", self._batch_size )

        print( "_image_width : " , self._image_width )
        print( "_image_height : " , self._image_height )
        print( "_n_ConvLayer_features :\n" , self._n_ConvLayer_features )
        print( "_n_channels : " , self._n_channels )
        print( "_n_strides : " , self._n_strides )
        print( "_n_fullyLayers : " , self._n_fullyLayers )
        print( "_n_labels : " , self._n_labels )

        print( "_init_var_op :\n", self._init_var_op )
        print( "_loss_op : ", self._loss_op )
        print( "_optimizer : ", self._optimizer )
        print( "_train_step : ", self._train_step )
        print( "_y_out_op : ", self._y_out_op )

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
        #----------------------------------------------------------------------
        # 畳み込み層 ~ 活性化関数 ~ プーリング層 ~
        #----------------------------------------------------------------------
        # 重みの Variable の list に、１つ目の畳み込み層の重み（フィルタ行列）を追加
        # この重みは、畳み込み処理の画像データに対するフィルタ処理に使う Tensor のことである。
        self._weights.append( 
            self.init_weight_variable( 
                input_shape = [4, 4, self._n_channels, self._n_ConvLayer_features[0] ]  # 4, 4 : フィルタ処理後の出力 pixcel サイズ（幅、高さ） 
            ) 
        )
        
        # バイアス項の Variable の list に、畳み込み層のバイアス項を追加
        self._biases.append( self.init_bias_variable( input_shape = [ self._n_ConvLayer_features[0] ] ) )

        # 畳み込み層のオペレーター
        conv_op1 = tf.nn.conv2d(
                       input = self._X_holder,
                       filter = self._weights[0],   # 畳込み処理で input で指定した Tensor との積和に使用する filter 行列 (Tensor)
                       strides = [ 1, self._n_strides, self._n_strides, 1 ], # strides[0] = strides[3] = 1. とする必要がある
                       padding = "SAME"     # ゼロパディングを利用する場合はSAMEを指定
                   )

        # 畳み込み層からの出力（活性化関数）オペレーター
        # バイアス項を加算したものを活性化関数に通す
        conv_out_op1 = NNActivation( activate_type = "relu" ).activate( 
                           tf.nn.bias_add( conv_op1, self._biases[0] ) 
                       )
        
        # プーリング層のオペレーター
        pool_op1 = tf.nn.max_pool(
                       value = conv_out_op1,
                       ksize = [ 1, 2, 2, 1 ],  # プーリングする範囲のサイズ
                       strides = [ 1, 2, 2, 1 ], # strides[0] = strides[3] = 1. とする必要がある
                       padding = "SAME"     # ゼロパディングを利用する場合はSAMEを指定
                   )

        # ２つ目の畳み込み層
        self._weights.append( 
            self.init_weight_variable( 
                input_shape = [4, 4, self._n_ConvLayer_features[0], self._n_ConvLayer_features[1] ]  # 4, 4 : フィルタ処理後の出力 pixcel サイズ（幅、高さ） 
            ) 
        )
        self._biases.append( self.init_bias_variable( input_shape = [ self._n_ConvLayer_features[1] ] ) )

        conv_op2 = tf.nn.conv2d(
                       input = pool_op1,
                       filter = self._weights[1],   # 畳込み処理で input で指定した Tensor との積和に使用する filter 行列 (Tensor)
                       strides = [ 1, self._n_strides, self._n_strides, 1 ], # strides[0] = strides[3] = 1. とする必要がある
                       padding = "SAME"     # ゼロパディングを利用する場合はSAMEを指定
                   )
        conv_out_op2 = NNActivation( activate_type = "relu" ).activate( 
                           tf.nn.bias_add( conv_op2, self._biases[1] ) 
                       )
        pool_op2 = tf.nn.max_pool(
                       value = conv_out_op2,
                       ksize = [ 1, 2, 2, 1 ],  # プーリングする範囲のサイズ
                       strides = [ 1, 2, 2, 1 ], # strides[0] = strides[3] = 1. とする必要がある
                       padding = "SAME"     # ゼロパディングを利用する場合はSAMEを指定
                   )
        #----------------------------------------------------------------------
        # ~ 全結合層
        #----------------------------------------------------------------------
        # 全結合層の入力側
        # 重み & バイアス項の Variable の list に、全結合層の入力側に対応する値を追加
        fullyLayers_width = self._image_width // (2*2)    # ? (2 * 2 : pooling 処理の範囲)
        fullyLayers_height = self._image_height // (2*2)  # ?
        fullyLayers_input_size = fullyLayers_width * fullyLayers_height * self._n_ConvLayer_features[-1] # ?
        print( "fullyLayers_input_size : ", fullyLayers_input_size )

        self._weights.append( 
            self.init_weight_variable( 
                input_shape = [ fullyLayers_input_size, self._n_fullyLayers ] 
            )
        )
        self._biases.append( self.init_bias_variable( input_shape = [ self._n_fullyLayers ] ) )

        # 全結合層の出力側
        # 重み & バイアス項のの Variable の list に、全結合層の出力側に対応する値を追加
        self._weights.append( 
            self.init_weight_variable( 
                input_shape = [ self._n_fullyLayers, self._n_labels ] 
            )
        )
        self._biases.append( self.init_bias_variable( input_shape = [ self._n_labels ] ) )

        # 全結合層への入力
        # 1 * N のユニットに対応するように reshape
        pool_op_shape = pool_op2.get_shape().as_list()      # ? [batch_size, 7, 7, _n_ConvLayer_features[-1] ]
        print( "pool_op2.get_shape().as_list() :\n", pool_op_shape )
        fullyLayers_shape = pool_op_shape[1] * pool_op_shape[2] * pool_op_shape[3]
        flatted_input = tf.reshape( pool_op2, [ -1, fullyLayers_shape ] )    # 1 * N に平坦化 (reshape) された値
        #flatted_input = numpy.reshape( pool_op2, (None, fullyLayers_shape) )
        print( "flatted_input :", flatted_input )

        # 全結合層の入力側へのオペレーター
        fullyLayers_in_op = NNActivation( activate_type = "relu" ).activate(
                                tf.add( tf.matmul( flatted_input, self._weights[-2] ), self._biases[-2] )
                            )

        # 全結合層の出力側へのオペレーター
        fullyLayers_out_op = tf.add( tf.matmul( fullyLayers_in_op, self._weights[-1] ), self._biases[-1] )

        self._y_out_op = fullyLayers_out_op

        return self._y_out_op


    def loss( self, type = "l2-norm", original_loss_op = None ):
        """
        損失関数の定義を行う。
        
        [Input]
            type : str
                損失関数の種類
                "original" : 独自の損失関数
                "l1-norm" : L1 損失関数（L1ノルム）
                "l2-norm" : L2 損失関数（L2ノルム）
                "binary-cross-entropy" : クロス・エントロピー交差関数（２クラスの分類問題）
                "cross-entropy" : クロス・エントロピー交差関数（多クラスの分類問題）
                "softmax-cross-entrpy" : ソフトマックス クロス・エントロピー損失関数
                "sparse-softmax-cross-entrpy" : 疎なソフトマックス クロス・エントロピー損失関数
                "sigmoid-cross-entropy" : シグモイド・クロス・エントロピー損失関数
                "weighted-cross-entropy" : 重み付きクロス・エントロピー損失関数
                "sparse-softmax-cross-entrpy" : 疎なソフトマックスクロス・エントロピー損失関数

            original_loss_op : Operator
                type = "original" で独自の損失関数とした場合の損失関数を表すオペレーター

        [Output]
            self._loss_op : Operator
                損失関数を表すオペレーター
        """
        # 独自の損失関数
        if ( type == "original" ):
            self._loss_op = original_loss_op

        # L1 損失関数
        elif ( type == "l1-norm" ):
            # 回帰問題の場合
            self._loss_op = tf.reduce_mean(
                                tf.abs( self._t_holder - self._y_out_op )
                            )
        # L2 損失関数
        elif ( type == "l2-norm" ):
            # 回帰問題の場合
            self._loss_op = tf.reduce_mean(
                                tf.square( self._t_holder - self._y_out_op )
                            )

        # クロス・エントロピー（２クラスの分類問題）
        elif ( type == "binary-cross-entropy" ):
            # クロス・エントロピー
            # ２クラスの分類問題の場合
            self._loss_op = -tf.reduce_sum( 
                                self._t_holder * tf.log( self._y_out_op ) + 
                                ( 1 - self._t_holder ) * tf.log( 1 - self._y_out_op )
                            )

        # クロス・エントロピー（多クラスの分類問題）
        elif ( type == "cross-entropy" ):
            # 多クラスの分類問題の場合
            # softmax で正規化済みの場合
            # tf.clip_by_value(...) : 下限値、上限値を設定
            self._loss_op = tf.reduce_mean(                     # ミニバッチ度に平均値を計算
                                -tf.reduce_sum( 
                                    self._t_holder * tf.log( tf.clip_by_value(self._y_out_op, 1e-10, 1.0) ), 
                                    reduction_indices = [1]     # sum をとる行列の方向 ( 1:row 方向 )
                                )
                            )

        # ソフトマックス　クロス・エントロピー（多クラスの分類問題）
        elif ( type == "softmax-cross-entropy" ):
            # softmax で正規化済みでない場合
            self._loss_op = tf.reduce_mean(
                                tf.nn.softmax_cross_entropy_with_logits(
                                    labels = self._t_holder,
                                    logits = self._y_out_op
                                )
                            )
        # 疎なマックス　クロス・エントロピー
        elif( type == "sparse-softmax-cross-entropy" ):
            #
            self._loss_op = tf.reduce_mean(
                                tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    logits = self._y_out_op,
                                    labels = self._t_holder
                                )
                            )
        # その他（デフォルト）
        else:
            self._loss_op = -tf.reduce_sum( 
                                self._t_holder * tf.log(self._y_out_op) +
                                ( 1 - self._t_holder ) * tf.log( 1 - self._y_out_op )
                            )
        

        return self._loss_op


    def optimizer( self, type = "gradient-descent", original_opt = None ):
        """
        モデルの最適化アルゴリズムの設定を行う。
        [Input]
            type : str
                最適化アルゴリズムの種類
                "original" : 独自の最適化アルゴリズム
                "gradient-descent" : 最急降下法 tf.train.GradientDescentOptimizer(...)
                "momentum" : モメンタム tf.train.MomentumOptimizer( ..., use_nesterov = False )
                "momentum-nesterov" : Nesterov モメンタム  tf.train.MomentumOptimizer( ..., use_nesterov = True )
                "ada-grad" : Adagrad tf.train.AdagradOptimizer(...)
                "ada-delta" : Adadelta tf.train.AdadeletaOptimizer(...)

            original_opt : Optimizer
                独自の最適化アルゴリズム

        [Output]
            optimizer の train_step
        """
        if ( type == "original" ):
            self._optimizer = original_opt

        elif ( type == "gradient-descent" ):
            self._optimizer = tf.train.GradientDescentOptimizer( learning_rate = self._learning_rate )
        
        elif ( type == "momentum" ):
            self._optimizer = tf.train.MomentumOptimizer( 
                                  learning_rate = self._learning_rate, 
                                  momentum = 0.9,
                                  use_nesterov = False
                              )
        
        elif ( type == "momentum-nesterov" ):
            self._optimizer = tf.train.MomentumOptimizer( 
                                  learning_rate = self._learning_rate, 
                                  momentum = 0.9,
                                  use_nesterov = True
                              )

        elif ( type == "ada-grad" ):
            self._optimizer = tf.train.AdagradOptimizer( learning_rate = self._learning_rate )

        elif ( type == "ada-delta" ):
            self._optimizer = tf.train.AdadeltaOptimizer( learning_rate = self._learning_rate, rho = 0.95 )

        else:
            self._optimizeroptimizer = tf.train.GradientDescentOptimizer( learning_rate = self._learning_rate )


        self._train_step = self._optimizer.minimize( self._loss_op )
        
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
        #----------------------------
        # 学習開始処理
        #----------------------------
        """
        X_eval_holder = tf.placeholder( 
                            tf.float32, 
                            shape = [ X_train.shape[0], self._image_width, self._image_height, self._n_channels ]
                         )
        t_eval_holder = tf.placeholder( tf.int32, shape = X_train.shape[0] )
        """

        # Variable の初期化オペレーター
        self._init_var_op = tf.global_variables_initializer()

        # Session の run（初期化オペレーター）
        self._session.run( self._init_var_op )

        #-------------------
        # 学習処理
        #-------------------
        n_batches = len( X_train ) // self._batch_size     # バッチ処理の回数
        print( "len( X_train ) :", len( X_train ) )
        print( "n_batches :", n_batches )

        # for ループでエポック数分トレーニング
        for epoch in range( self._epochs ):
            # ミニバッチ学習処理のためランダムサンプリング
            #X_train_shuffled, y_train_shuffled = shuffle( X_train, y_train )
            idx_shuffled = numpy.random.choice( len(X_train), size = self._batch_size )
            X_train_shuffled = X_train[ idx_shuffled ]
            y_train_shuffled = y_train[ idx_shuffled ]

            # shape を (batchsize, image_width, image_height) → (batchsize, image_width, image_height, 1) に reshape
            X_train_shuffled = numpy.expand_dims( X_train_shuffled, 3 )

            #print( "X_train_shuffled", X_train_shuffled )
            #print( "y_train_shuffled", y_train_shuffled )

            # 
            self._session.run(
                self._train_step,
                feed_dict = {
                    self._X_holder: X_train_shuffled,
                    self._t_holder: y_train_shuffled
                }
            )
            
            #
            loss = self._loss_op.eval(
                       session = self._session,
                       feed_dict = {
                           self._X_holder: X_train_shuffled,
                           self._t_holder: y_train_shuffled
                       }
                   )

            self._losses_train.append( loss )

            print( "loss = ", loss )

            """
            #
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
            """
            """
            # 損失関数の値をストック
            # shape を (batchsize, image_width, image_height) → (batchsize, image_width, image_height, 1) に reshape
            X_train = numpy.expand_dims( X_train, 3 )

            loss = self._loss_op.eval(
                       session = self._session,
                       feed_dict = {
                           X_eval_holder: X_train,
                           t_eval_holder: y_train
                       }
                   )

            self._losses_train.append( loss )
            """

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
        predict_op = numpy.argmax( self._y_out_op, axis = 1 )

        predict = predict_op.eval( 
                   session = self._session,
                   feed_dict = {
                       self._X_holder: X_test
                   }
               )
        """
        # shape を (n_samples, image_width, image_height) → (n_samples, image_width, image_height, 1) に reshape
        X_test_reshaped = numpy.expand_dims( X_test, 3 )

        predicts = self._session.run(
                      self._y_out_op,
                      feed_dict = { self._X_holder: X_test_reshaped }
                  )
        
        print( "predicts :", predicts )

        # numpy.argmax(...) : 多次元配列の中の最大値の要素を持つインデックスを返す
        # axis : 最大値を読み取る軸の方向 (1 : 行方向)
        predict = numpy.argmax( predicts, axis = 1 )
        print( "predict :", predict )

        return predict


    def predict_proba( self, X_test ):
        """
        fitting 処理したモデルで、推定を行い、クラスの所属確率の予想値を返す。
        proba : probability

        [Input]
            X_test : numpy.ndarry ( shape = [n_samples, n_features] )
                予想したい特徴行列
        """
        # shape を (n_samples, image_width, image_height) → (n_samples, image_width, image_height, 1) に reshape
        X_test_reshaped = numpy.expand_dims( X_test, 3 )

        prob = self._y_out_op.eval(
                   session = self._session,
                   feed_dict = {
                       self._X_holder: X_test_reshaped 
                   }
               )

        # X_test のデータ数、特徴数に応じて reshape
        #prob = prob.reshape( (len[X_test], len[X_test[0]]) )

        return prob


    def accuracy( self, X_test, y_test ):
        """
        指定したデータでの正解率 [accuracy] を計算する。
        """
        # 予想ラベル
        predict = self.predict( X_test )

        # 正解数
        n_correct = numpy.sum( numpy.equal( predict, y_test ) )
        print( "numpy.equal( predict, y_test ) :", numpy.equal( predict, y_test ) )
        print( "n_correct :", n_correct )

        # 正解率 = 正解数 / データ数
        accuracy = n_correct / X_test.shape[0]

        return accuracy
