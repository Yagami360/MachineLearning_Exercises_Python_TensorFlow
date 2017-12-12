# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境 (TensorFlow インストール済み)

"""
    更新情報
    [17/12/12] : 新規作成
    [xx/xx/xx] : 

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

import NNOptimizer                                  # ニューラルネットワークの最適化アルゴリズム Optimizer を表すクラス


class RecurrectNNEncoderDecoderLSTM( NeuralNetworkBase ):
    """
    LSTM による RNN Encoder-Decoder を表すクラス.
    TensorFlow での RNN Encoder-Decoder の処理をクラス（任意の層に DNN 化可能な柔軟なクラス）でラッピングし、
    scikit-learn ライブラリの classifier, estimator とインターフェイスを共通化することで、
    scikit-learn ライブラリとの互換性のある自作クラス
    ------------------------------------------------------------------------------------------------
    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
        _n_inputLayer : int
            入力層のノード数
        _n_hiddenLayer : int
            隠れ層のノード数
        _n_outputLayer : int
            出力層のノード数

        _n_in_sequence_encoder : int
            時系列データを区切った Encoder の各シークエンスの長さ（サイズ）
        _n_in_sequence_decoder : int
            時系列データを区切った Decoder の各シークエンスの長さ（サイズ）

        _rnn_cells_encoder : list <LSTMCell クラスのオブジェクト> <tf.Tensor 'RNN-LSTM/RNN-LSTM/lstm_cell>
            RNN Encoder-Decoder の Encoder 構造を提供する cell のリスト
            この `cell` は、内部（プロパティ）で state（隠れ層の状態）を保持しており、
            これを次の時間の隠れ層に順々に渡していくことで、時間軸の逆伝搬を実現する。
        _rnn_states_encoder : list <Tensor>
            Encoder の cell の状態のリスト
        _rnn_cells_decoder : lsit <LSTMCell クラスのオブジェクト> <tf.Tensor 'RNN-LSTM/RNN-LSTM/lstm_cell>
            RNN Encoder-Decoder の Decoder 構造を提供する cell のリスト
        _rnn_states_decoder : list <Tensor>
            Decoder の cell の状態のリスト

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

        _losses_train : list <float32>
            トレーニングデータでの損失関数の値の list

        _X_holder : placeholder
            入力層にデータを供給するための placeholder
        _t_holder : placeholder
            出力層に教師データを供給するための placeholder
        _dropout_holder : placeholder
            ドロップアウトしない確率 (1-p) にデータを供給するための placeholder
        _batch_size_holder : placeholder
            バッチサイズ _batch_size にデータを供給するための placeholder
            cell.zero_state(...) でバッチサイズを指定する必要があり、可変長に対応するために必要
        _bTraining_holder : placeholder
            トレーニング処理中か否かを表す placeholder

    [protedted] protedted な使用法を想定 

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( 
            self,
            session = tf.Session( config = tf.ConfigProto(log_device_placement=True) ),
            n_inputLayer = 1, n_hiddenLayer = 1, n_outputLayer = 1, 
            n_in_sequence_encoder = 25,
            n_in_sequence_decoder = 25,
            epochs = 1000,
            batch_size = 10,
            eval_step = 1
        ):
        """
        コンストラクタ（厳密にはイニシャライザ）
        """
        super().__init__( session )

        tf.set_random_seed(12)

        # 各パラメータの初期化
        self._n_inputLayer = n_inputLayer
        self._n_hiddenLayer = n_hiddenLayer
        self._n_outputLayer = n_outputLayer

        self._n_in_sequence_encoder = n_in_sequence_encoder
        self._n_in_sequence_decoder = n_in_sequence_decoder

        self._rnn_cells_encoder = []
        self._rnn_states_encoder = []
        self._rnn_cells_decoder = []
        self._rnn_states_decoder = []

        self._weights = []
        self._biases = []

        self._epochs = epochs
        self._batch_size = batch_size
        self._eval_step = eval_step        

        # evaluate 関連の初期化
        self._losses_train = []

        # placeholder の初期化
        # shape の列（横方向）は、各層の次元（ユニット数）に対応させる。
        # shape の行は、None にして汎用性を確保
        self._X_holder = tf.placeholder( 
                             tf.float32, 
                             shape = [ None, self._n_in_sequence_encoder, self._n_inputLayer ],
                             name = "X_holder"
                         )

        self._t_holder = tf.placeholder( 
                             tf.float32, 
                             shape = [ None, self._n_in_sequence_decoder, self._n_outputLayer ],
                             name = "t_holder"
                         )

        self._dropout_holder = tf.placeholder( tf.float32, name = "dropout_holder" )
        self._batch_size_holder = tf.placeholder( tf.int32, shape=[], name = "batch_size_holder" )
        self._bTraining_holder = tf.placeholder( tf.bool, name = "bTraining_holder" )

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

        print( "_n_inputLayer : ", self._n_inputLayer )
        print( "_n_hiddenLayer : ", self._n_hiddenLayer )
        print( "_n_outputLayer : ", self._n_outputLayer )

        print( "_n_in_sequence_encoder :", self._n_in_sequence_encoder )
        print( "_n_in_sequence_decoder :", self._n_in_sequence_decoder )

        print( "_epoches : ", self._epochs )
        print( "_batch_size : ", self._batch_size )
        print( "_eval_step : ", self._eval_step )

        print( "_X_holder :", self._X_holder )
        print( "_t_holder :", self._t_holder )
        print( "_dropout_holder :", self._dropout_holder )
        print( "_batch_size_holder :", self._batch_size_holder )
        print( "_bTraing_holder :", self._bTraining_holder )

        print( "_rnn_cells_encoder : \n", self._rnn_cells_encoder )
        print( "_rnn_states_encoder : \n", self._rnn_states_encoder )
        print( "_rnn_cells_decoder : \n", self._rnn_cells_decoder )
        print( "_rnn_states_decoder : \n", self._rnn_states_decoder )

        print( "_weights : \n", self._weights )
        if( (self._session != None) and (self._init_var_op != None) ):
            print( self._session.run( self._weights ) )

        print( "_biases : \n", self._biases )
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

        #init_tsr = tf.zeros( shape = input_shape )
        init_tsr = tf.random_normal( shape = input_shape )

        # バイアス項の Variable
        bias_var = tf.Variable( init_tsr, name = "init_bias_var" )

        return bias_var


    def model( self ):
        """
        モデルの定義（計算グラフの構築）を行い、
        最終的なモデルの出力のオペレーターを設定する。

        [Output]
            self._y_out_op : Operator
                モデルの出力のオペレーター
        """
        #--------------------------------------------------------------
        # 入力層 ~ 隠れ層
        #--------------------------------------------------------------
        #--------------------------------------------------------------
        # Encoder
        #--------------------------------------------------------------
        # tf.contrib.rnn.LSTMCell(...) : 時系列に沿った RNN 構造を提供するクラス `LSTMCell` のオブジェクト cell を返す。
        # この cell は、内部（プロパティ）で state（隠れ層の状態）を保持しており、
        # これを次の時間の隠れ層に順々に渡していくことで、時間軸の逆伝搬を実現する。
        cell_encoder = tf.contrib.rnn.LSTMCell( 
                           num_units = self._n_hiddenLayer,     # int, The number of units in the RNN cell.
                           forget_bias = 1.0                    # 忘却ゲートのバイアス項 / Default : 1.0  in order to reduce the scale of forgetting at the beginning of the training.
                       )
        #self._rnn_cells_encoder.append( cell_encoder ) # 後述の処理で同様の処理が入るので不要

        # 最初の時間 t0 では、過去の隠れ層がないので、
        # cell.zero_state(...) でゼロの状態を初期設定する。
        initial_state_encoder_tsr = cell_encoder.zero_state( self._batch_size_holder, tf.float32 )
        self._rnn_states_encoder.append( initial_state_encoder_tsr )

        # Encoder の過去の隠れ層の再帰処理
        with tf.variable_scope('Encoder'):
            for t in range( self._n_in_sequence_encoder ):
                if (t > 0):
                    # tf.get_variable_scope() : 名前空間を設定した Variable にアクセス
                    # reuse_variables() : reuse フラグを True にすることで、再利用できるようになる。
                    tf.get_variable_scope().reuse_variables()

                # LSTMCellクラスの `__call__(...)` を順次呼び出し、
                # 各時刻 t における出力 cell_output, 及び状態 state を算出
                cell_encoder_output, state_encoder_tsr = cell_encoder( inputs = self._X_holder[:, t, :], state = self._rnn_states_encoder[-1] )

                # 過去の隠れ層の出力をリストに追加
                self._rnn_cells_encoder.append( cell_encoder_output )
                self._rnn_states_encoder.append( state_encoder_tsr )

        # 最終的な Encoder の出力
        #output_encoder = self._rnn_cells_encoder[-1]
        
        #--------------------------------------------------------------
        # Decoder
        #--------------------------------------------------------------
        cell_decoder = tf.contrib.rnn.LSTMCell( 
                           num_units = self._n_hiddenLayer,     # int, The number of units in the RNN cell.
                           forget_bias = 1.0                    # 忘却ゲートのバイアス項 / Default : 1.0  in order to reduce the scale of forgetting at the beginning of the training.
                       )

        # Decoder の初期状態は Encoder の最終出力
        self._rnn_cells_decoder.append( self._rnn_cells_encoder[-1] )

        # Decoder の初期状態は Encoder の最終出力
        initial_state_decoder_tsr = self._rnn_states_encoder[-1]
        self._rnn_states_decoder.append( initial_state_decoder_tsr )

        # 隠れ層 ~ 出力層の重みを事前に設定
        self._weights.append( self.init_weight_variable( input_shape = [self._n_hiddenLayer, self._n_outputLayer] ) )
        self._biases.append( self.init_bias_variable( input_shape = [self._n_outputLayer] ) )

        # ?
        eval_outputs = []

        # Decoder の過去の隠れ層の再帰処理
        with tf.variable_scope('Decoder'):
            # t = 1 ~ self._n_in_sequence_decoder 間のループ処理 (t != 0)
            # t = 0 を含まないのは、Decoder の t = 0 の初期状態は、Encoder の最終出力で処理済みのため
            for t in range( 1, self._n_in_sequence_decoder ):
                if (t > 1):
                    # tf.get_variable_scope() : 名前空間を設定した Variable にアクセス
                    # reuse_variables() : reuse フラグを True にすることで、再利用できるようになる。
                    tf.get_variable_scope().reuse_variables()

                # トレーニング処理中の場合のルート
                if ( self._bTraining_holder == True ):
                    with tf.name_scope( "Traning_root" ):
                        # LSTMCellクラスの `__call__(...)` を順次呼び出し、
                        # 各時刻 t における出力 cell_output, 及び状態 state を算出
                        cell_decoder_output, state_decoder_tsr = cell_decoder( inputs = self._t_holder[:, t-1, :], state = self._rnn_states_decoder[-1] )
                
                # loss 値などの評価用の値の計算時のルート
                # デコーダーの次の step における出力計算時、self._t_holder[:, t-1, :] という正解データ（教師データ）を使用しないようにルート分岐させる。
                else:
                    with tf.name_scope( "Eval_root" ):
                        # matmul 計算時、直前の出力 self._rnn_cells_decoder[-1] を入力に用いる
                        cell_decoder_output = tf.matmul( self._rnn_cells_decoder[-1], self._weights[-1] ) + self._biases[-1]
                        cell_decoder_output = tf.nn.softmax( cell_decoder_output )

                        # ?
                        eval_outputs.append( cell_decoder_output )

                        # ?
                        cell_decoder_output = tf.one_hot( tf.argmax(cell_decoder_output, -1), depth = self._n_in_sequence_decoder)
                    
                        # ?
                        cell_decoder_output, state_decoder_tsr = cell_decoder( cell_decoder_output, self._rnn_states_decoder[-1] )

                # 過去の隠れ層の出力をリストに追加
                self._rnn_cells_decoder.append( cell_decoder_output )
                self._rnn_states_decoder.append( state_decoder_tsr )

        # トレーニング処理中の場合のルート
        if ( self._bTraining_holder == True ):
            with tf.name_scope( "Traning_root" ):
                #--------------------------------------------------------------
                # 出力層への入力
                #--------------------------------------------------------------
                # self._rnn_cells_decoder の形状を shape = ( データ数, デコーダーのシーケンス長, 隠れ層のノード数 ) に reshape 
                # tf.concat(...) : Tensorを結合する。引数 axis で結合する dimension を決定
                output = tf.reshape( 
                             tf.concat( self._rnn_cells_decoder, axis = 1 ),
                             shape = [ -1, self._n_in_sequence_decoder, self._n_hiddenLayer ]
                        )
        
                #y_in_op = tf.matmul( output, self._weights[-1] ) + self._biases[-1]

                # 3 階の Tensorとの積を取る（２階なら行列なので matmul でよかった）
                # Σ_{ijk} の j 成分を残して、matmul する
                # tf.einsum(...) : Tensor の積の アインシュタインの縮約表現
                # equation : the equation is obtained from the more familiar element-wise （要素毎の）equation by
                # 1. removing variable names, brackets, and commas, 
                # 2. replacing "*" with ",", 
                # 3. dropping summation signs, 
                # and 4. moving the output to the right, and replacing "=" with "->".
                y_in_op = tf.einsum( "ijk,kl->ijl", output, self._weights[-1] ) + self._biases[-1]
        
                #--------------------------------------------------------------
                # モデルの出力
                #--------------------------------------------------------------
                # softmax
                self._y_out_op = tf.nn.softmax( y_in_op )

        # loss 値などの評価用の値の計算時のルート
        else:
            with tf.name_scope( "Eval_root" ):
                #--------------------------------------------------------------
                # 出力層への入力
                #--------------------------------------------------------------
                y_in_op = tf.matmul( self._rnn_cells_decoder[-1], self._weights[-1] ) + self._biases[-1]

                #--------------------------------------------------------------
                # モデルの出力
                #--------------------------------------------------------------
                # softmax
                self._y_out_op = tf.nn.softmax( y_in_op )

                # ?
                eval_outputs.append( self._y_out_op )

                # ?
                # self._y_out_op の形状を shape = ( データ数, デコーダーのシーケンス長, 出力層ののノード数 ) に reshape 
                # tf.concat(...) : Tensorを結合する。引数 axis で結合する dimension を決定
                self._y_out_op = tf.reshape(
                                     tf.concat( eval_outputs, axis = 1 ),
                                     [-1, self._n_in_sequence_decoder, self._n_outputLayer ]
                                 )

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
        # for ループでエポック数分トレーニング
        for epoch in range( self._epochs ):
            # ミニバッチ学習処理のためランダムサンプリング
            idx_shuffled = numpy.random.choice( len(X_train), size = self._batch_size )
            X_train_shuffled = X_train[ idx_shuffled ]
            y_train_shuffled = y_train[ idx_shuffled ]
            
            # 設定された最適化アルゴリズム Optimizer でトレーニング処理を run
            self._session.run(
                self._train_step,
                feed_dict = {
                    self._X_holder: X_train_shuffled,
                    self._t_holder: y_train_shuffled,
                    self._batch_size_holder: self._batch_size,
                    self._bTraining_holder: True
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
                           self._t_holder: y_train_shuffled,
                           self._batch_size_holder: self._batch_size,
                           self._bTraining_holder: False
                       }
                   )

                self._losses_train.append( loss )
                print( "epoch %d / loss = %f" % ( epoch, loss ) )

        return self._y_out_op


    def predict( self, X_test ):
        """
        fitting 処理したモデルで、推定を行い、時系列データの予想値を返す。

        [Input]
            X_test : numpy.ndarry ( shape = [n_samples, n_features(=n_in_sequence), dim] )
                予想したい特徴行列（時系列データの行列）
                n_samples : シーケンスに分割した時系列データのサンプル数
                n_features(=n_in_sequence) : １つのシーケンスのサイズ
                dim : 各シーケンスの要素の次元数

        [Output]
            predicts : numpy.ndarry ( shape = [n_samples] )
                予想結果（分類モデルの場合は、クラスラベル）
        """
        prob = self._session.run(
                   self._y_out_op,
                   feed_dict = { 
                       self._X_holder: X_test,
                       self._batch_size_holder: 1,
                       self._bTraining_holder: False
                   }
               )
        
        # numpy.argmax(...) : 多次元配列の中の最大値の要素を持つインデックスを返す
        # axis : 最大値を読み取る軸の方向 (1 : 行方向)
        predicts = numpy.argmax( prob, axis = -1 )

        return predicts


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
                       self._X_holder: X_test,
                       self._batch_size_holder: 1,
                       self._bTraining_holder: False
                   }
               )
        
        return prob


    def accuracy( self, X_test, y_test ):
        """
        指定したデータでの正解率 [accuracy] を計算する。
        """
        # 予想ラベルを算出する。
        predicts = self.predict( X_test )

        # 正解数
        n_correct = numpy.sum( numpy.equal( predicts, y_test ) )
        #print( "numpy.equal( predict, y_test ) :", numpy.equal( predict, y_test ) )
        #print( "n_correct :", n_correct )

        # 正解率 = 正解数 / データ数
        accuracy = n_correct / X_test.shape[0]

        return accuracy

