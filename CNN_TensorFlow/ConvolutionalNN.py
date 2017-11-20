# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境 (TensorFlow インストール済み)

"""
    更新情報
    [17/11/04] : 新規作成
    [17/11/19] : NeuralNetworkBase クラスの子クラスになるように修正
               : 畳み込み層でのカーネルのサイズ、プーリング層でのウィンドウサイズ、ストライドサイズの値をコンストラクタで設定出来るように修正
               : 
"""

import numpy

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops

# scikit-learn ライブラリ
from sklearn.utils import shuffle

# 自作クラス
from NeuralNetworkBase import NeuralNetworkBase    # 親クラス

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


class ConvolutionalNN( NeuralNetworkBase ):
    """
    畳み込みニューラルネットワーク [CNN : Convolutional Neural Network] を表すクラス.
    TensorFlow での CNN の処理をクラス（任意の層に DNN 化可能な柔軟なクラス）でラッピングし、
    scikit-learn ライブラリの classifier, estimator とインターフェイスを共通化することで、
    scikit-learn ライブラリとの互換性のある自作クラス
    ------------------------------------------------------------------------------------------------
    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
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

        _n_ConvLayer_featuresMap : list <int>
            畳み込み層で変換される特徴マップの枚数
            conv1 : _n_ConvLayer_featuresMap[0]
            conv2 : _n_ConvLayer_featuresMap[1]
            ...
        _n_ConvLayer_kernels : list <int>
            CNN の畳み込み処理時のカーネルのサイズ
            conv1 : _n_ConvLayer_kernels[0] * _n_ConvLayer_kernels[0]
            conv2 : _n_ConvLayer_kernels[1] * _n_ConvLayer_kernels[1]
            ...
        _n_strides : int
            CNN の畳み込み処理（特徴マップ生成）でストライドさせる pixel 数

        _n_pool_wndsize : int
            プーリング処理用のウィンドウサイズ
        _n_pool_strides : int
            プーリング処理時のストライドさせる pixel 数

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
            epochs = 1000,
            batch_size = 1,
            eval_step = 1,
            image_height = 28,
            image_width = 28,
            n_channels = 1,
            n_ConvLayer_featuresMap = [25, 50],
            n_ConvLayer_kernels = [4, 4],
            n_strides = 1,
            n_pool_wndsize = 2,
            n_pool_strides = 2,
            n_fullyLayers = 100,
            n_labels = 10
        ) :
        """
        コンストラクタ（厳密にはイニシャライザ）
        """
        super().__init__( session )

        tf.set_random_seed(12)

        # 各パラメータの初期化
        self._weights = []
        self._biases = []

        self._epochs = epochs
        self._batch_size = batch_size
        self._eval_step = eval_step
        
        self._image_height = image_height
        self._image_width = image_width
        self._n_channels = n_channels

        self._n_ConvLayer_featuresMap = n_ConvLayer_featuresMap
        self._n_ConvLayer_kernels = n_ConvLayer_kernels
        self._n_strides = n_strides

        self._n_pool_wndsize = n_pool_wndsize
        self._n_pool_strides = n_pool_strides

        self._n_fullyLayers = n_fullyLayers
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

        self._keep_prob_holder = tf.placeholder( tf.float32 )

        # evaluate 関連の初期化
        self._losses_train = []

        return

    def print( self, str ):
        print( "----------------------------------" )
        print( str )
        print( self )

        print( "_session : ", self._session )
        print( "_init_var_op :\n", self._init_var_op )

        print( "_loss_op : ", self._loss_op )
        print( "_optimizer : ", self._optimizer )
        print( "_train_step : ", self._train_step )
        print( "_y_out_op : ", self._y_out_op )

        print( "_epoches : ", self._epochs )
        print( "_batch_size : ", self._batch_size )
        print( "_eval_step : ", self._eval_step )

        print( "_image_height : " , self._image_height )
        print( "_image_width : " , self._image_width )
        print( "_n_channels : " , self._n_channels )
        print( "_n_ConvLayer_featuresMap :" , self._n_ConvLayer_featuresMap )
        print( "_n_ConvLayer_kernels :" , self._n_ConvLayer_kernels )
        print( "_n_strides : " , self._n_strides )
        print( "_n_pool_wndsize : " , self._n_pool_wndsize )
        print( "_n_pool_strides : " , self._n_pool_strides )
        print( "_n_fullyLayers : " , self._n_fullyLayers )
        print( "_n_labels : " , self._n_labels )

        print( "_X_holder : ", self._X_holder )
        print( "_t_holder : ", self._t_holder )
        print( "_keep_prob_holder : ", self._keep_prob_holder )

        print( "_weights : \n", self._weights )
        if( (self._session != None) and (self._init_var_op != None) ):
            print( self._session.run( self._weights ) )

        if( (self._session != None) and (self._init_var_op != None) ):
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
        # １つ目の畳み込み層 ~ 活性化関数 ~ プーリング層 ~
        #----------------------------------------------------------------------
        # 重みの Variable の list に、１つ目の畳み込み層の重み（カーネル）を追加
        # この重みは、畳み込み処理の画像データに対するフィルタ処理（特徴マップ生成）に使うカーネルを表す Tensor のことである。
        self._weights.append( 
            self.init_weight_variable( 
                input_shape = [ 
                    self._n_ConvLayer_kernels[0], self._n_ConvLayer_kernels[0], 
                    self._n_channels, 
                    self._n_ConvLayer_featuresMap[0] 
                ]
            ) 
        )
        
        # バイアス項の Variable の list に、畳み込み層のバイアス項を追加
        self._biases.append( 
            self.init_bias_variable( input_shape = [ self._n_ConvLayer_featuresMap[0] ] ) 
        )

        # 畳み込み層のオペレーター
        conv_op1 = tf.nn.conv2d(
                       input = self._X_holder,
                       filter = self._weights[0],   # 畳込み処理で input で指定した Tensor との積和に使用する filter 行列（カーネル）
                       strides = [ 1, self._n_strides, self._n_strides, 1 ], # strides[0] = strides[3] = 1. とする必要がある
                       padding = "SAME"     # ゼロパディングを利用する場合はSAMEを指定
                   )

        # 畳み込み層からの出力（活性化関数）オペレーター
        # バイアス項を加算したものを活性化関数に通す
        conv_out_op1 = Relu().activate( tf.nn.bias_add( conv_op1, self._biases[0] ) )
                
        # プーリング層のオペレーター
        pool_op1 = tf.nn.max_pool(
                       value = conv_out_op1,
                       ksize = [ 1, self._n_pool_wndsize, self._n_pool_wndsize, 1 ],    # プーリングする範囲（ウィンドウ）のサイズ
                       strides = [ 1, self._n_pool_strides, self._n_pool_strides, 1 ],  # ストライドサイズ strides[0] = strides[3] = 1. とする必要がある
                       padding = "SAME"                                                 # ゼロパディングを利用する場合はSAMEを指定
                   )


        #----------------------------------------------------------------------
        # ２つ目以降の畳み込み層 ~ 活性化関数 ~ プーリング層 ~
        #----------------------------------------------------------------------
        # 畳み込み層のカーネル
        self._weights.append( 
            self.init_weight_variable( 
                input_shape = [ 
                    self._n_ConvLayer_kernels[1], self._n_ConvLayer_kernels[1], 
                    self._n_ConvLayer_featuresMap[0], self._n_ConvLayer_featuresMap[1] 
                ]
            ) 
        )

        self._biases.append( 
            self.init_bias_variable( input_shape = [ self._n_ConvLayer_featuresMap[1] ] ) 
        )

        # 畳み込み層のオペレーター
        conv_op2 = tf.nn.conv2d(
                       input = pool_op1,
                       filter = self._weights[1],   # 畳込み処理で input で指定した Tensor との積和に使用する filter 行列 (Tensor)
                       strides = [ 1, self._n_strides, self._n_strides, 1 ], # strides[0] = strides[3] = 1. とする必要がある
                       padding = "SAME"     # ゼロパディングを利用する場合はSAMEを指定
                   )

        conv_out_op2 = Relu().activate( tf.nn.bias_add( conv_op2, self._biases[1] ) )
        
        # プーリング層のオペレーター
        pool_op2 = tf.nn.max_pool(
                       value = conv_out_op2,
                       ksize = [ 1, self._n_pool_wndsize, self._n_pool_wndsize, 1 ],    # プーリングする範囲（ウィンドウ）のサイズ
                       strides = [ 1, self._n_pool_strides, self._n_pool_strides, 1 ],  # ストライドサイズ strides[0] = strides[3] = 1. とする必要がある
                       padding = "SAME"                                                 # ゼロパディングを利用する場合はSAMEを指定
                   )

        #----------------------------------------------------------------------
        # ~ 全結合層 ~ 出力層
        #----------------------------------------------------------------------
        # 全結合層の入力側
        # 重み & バイアス項の Variable の list に、全結合層の入力側に対応する値を追加
        fullyLayers_width = self._image_width // (2*2)    # ? (2 * 2 : pooling 処理の範囲)
        fullyLayers_height = self._image_height // (2*2)  # ?
        fullyLayers_input_size = fullyLayers_width * fullyLayers_height * self._n_ConvLayer_featuresMap[-1] # ?
        print( "fullyLayers_input_size : ", fullyLayers_input_size )

        self._weights.append( 
            self.init_weight_variable( 
                input_shape = [ fullyLayers_input_size, self._n_fullyLayers ] 
            )
        )
        self._biases.append( self.init_bias_variable( input_shape = [ self._n_fullyLayers ] ) )

        # 全結合層への入力
        # 1 * N のユニットに対応するように reshape
        pool_op_shape = pool_op2.get_shape().as_list()      # ? [batch_size, 7, 7, _n_ConvLayer_features[-1] ]
        print( "pool_op2.get_shape().as_list() :\n", pool_op_shape )
        fullyLayers_shape = pool_op_shape[1] * pool_op_shape[2] * pool_op_shape[3]
        flatted_input = tf.reshape( pool_op2, [ -1, fullyLayers_shape ] )    # 1 * N に平坦化 (reshape) された値
        #flatted_input = numpy.reshape( pool_op2, (None, fullyLayers_shape) )
        print( "flatted_input :", flatted_input )

        # 全結合層の入力側へのオペレーター
        fullyLayers_in_op = Relu().activate( tf.add( tf.matmul( flatted_input, self._weights[-1] ), self._biases[-1] ) )


        # 全結合層の出力側
        # 重み & バイアス項のの Variable の list に、全結合層の出力側に対応する値を追加
        self._weights.append( 
            self.init_weight_variable( 
                input_shape = [ self._n_fullyLayers, self._n_labels ] 
            )
        )
        self._biases.append( self.init_bias_variable( input_shape = [ self._n_labels ] ) )
        
        # 全結合層の出力側へのオペレーター
        fullyLayers_out_op = tf.add( tf.matmul( fullyLayers_in_op, self._weights[-1] ), self._biases[-1] )
        self._y_out_op = fullyLayers_out_op

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
        # 入力データの shape にチェンネルデータがない場合
        # shape = [image_height, image_width]
        if( X_train.ndim == 3 ):
            # shape を [image_height, image_width] → [image_height, image_width, n_channel=1] に reshape
            X_train = numpy.expand_dims( X_train, axis = 3 )

        #----------------------------
        # 学習開始処理
        #----------------------------
        # Variable の初期化オペレーター
        self._init_var_op = tf.global_variables_initializer()

        # Session の run（初期化オペレーター）
        self._session.run( self._init_var_op )

        #-------------------
        # 学習処理
        #-------------------
        n_batches = len( X_train ) // self._batch_size     # バッチ処理の回数
        #print( "len( X_train ) :", len( X_train ) )
        #print( "n_batches :", n_batches )

        # for ループでエポック数分トレーニング
        for epoch in range( self._epochs ):
            # ミニバッチ学習処理のためランダムサンプリング
            idx_shuffled = numpy.random.choice( len(X_train), size = self._batch_size )
            X_train_shuffled = X_train[ idx_shuffled ]
            y_train_shuffled = y_train[ idx_shuffled ]
            #print( "X_train_shuffled.shape", X_train_shuffled.shape )

            # 設定された最適化アルゴリズム Optimizer でトレーニング処理を run
            self._session.run(
                self._train_step,
                feed_dict = {
                    self._X_holder: X_train_shuffled,
                    self._t_holder: y_train_shuffled
                }
            )
            
            # 評価処理を行う loop か否か
            # % : 割り算の余りが 0 で判断
            if ( ( (epoch+1) % self._eval_step ) == 0 ):
                # 損失関数値の算出
                loss = self._loss_op.eval(
                       session = self._session,
                       feed_dict = {
                           self._X_holder: X_train_shuffled,
                           self._t_holder: y_train_shuffled
                       }
                   )

                self._losses_train.append( loss )
                print( "epoch %d / loss = %f" % ( epoch, loss ) )

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
        # 入力データの shape にチェンネルデータがない場合
        # shape = [image_height, image_width]
        if( X_test.ndim == 3 ):
            # shape を [image_height, image_width] → [image_height, image_width, n_channel=1] に reshape
            X_test = numpy.expand_dims( X_test, axis = 3 )

        prob = self._session.run(
                   self._y_out_op,
                   feed_dict = { self._X_holder: X_test }
               )
        
        #print( "predicts :", predicts )

        # numpy.argmax(...) : 多次元配列の中の最大値の要素を持つインデックスを返す
        # axis : 最大値を読み取る軸の方向 (1 : 行方向)
        predict = numpy.argmax( prob, axis = 1 )
        #print( "predict :", predict )

        return predict


    def predict_proba( self, X_test ):
        """
        fitting 処理したモデルで、推定を行い、クラスの所属確率の予想値を返す。
        proba : probability

        [Input]
            X_test : numpy.ndarry ( shape = [n_samples, n_features] )
                予想したい特徴行列
        """
        # 入力データの shape にチェンネルデータがない場合
        # shape = [image_height, image_width]
        if( X_test.ndim == 3 ):
            # shape を [image_height, image_width] → [image_height, image_width, n_channel=1] に reshape
            X_test = numpy.expand_dims( X_test, axis = 3 )

        prob = self._y_out_op.eval(
                   session = self._session,
                   feed_dict = {
                       self._X_holder: X_test 
                   }
               )
        
        return prob


    def accuracy( self, X_test, y_test ):
        """
        指定したデータでの正解率 [accuracy] を計算する。
        """
        # 予想ラベルを算出する。
        predict = self.predict( X_test )

        # 正解数
        n_correct = numpy.sum( numpy.equal( predict, y_test ) )
        #print( "numpy.equal( predict, y_test ) :", numpy.equal( predict, y_test ) )
        #print( "n_correct :", n_correct )

        # 正解率 = 正解数 / データ数
        accuracy = n_correct / X_test.shape[0]

        return accuracy

    def accuracy_labels( self, X_test, y_test ):
        """
        指定したデータでのラベル毎の正解率 [acuuracy] を算出する。
        """
        # 予想ラベルを算出する。
        predict = self.predict( X_test )

        # ラベル毎の正解率のリスト
        n_labels = len( numpy.unique( y_test ) )    # ユニークな要素数
        accuracys = []

        for label in range(n_labels):
            # label 値に対応する正解数
            # where(...) : 条件を満たす要素番号を抽出
            n_correct = len( numpy.where( predict == label )[0] )
            """
            n_correct = numpy.sum( 
                            numpy.equal( 
                                numpy.where( predict == label )[0], 
                                numpy.where( y_test == label )[0]
                            ) 
                        )
            """

            # サンプル数
            n_sample = len( numpy.where( y_test == label )[0] )

            accuracy = n_correct / n_sample
            accuracys.append( accuracy )
            
            #print( "numpy.where( predict == label ) :", numpy.where( predict == label ) )
            #print( "numpy.where( predict == label )[0] :", numpy.where( predict == label )[0] )
            print( " %d / n_correct = %d / n_sample = %d" % ( label, n_correct, n_sample ) )

        return accuracys
