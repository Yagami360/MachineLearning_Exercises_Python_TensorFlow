# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境 (TensorFlow インストール済み)

"""
    更新情報
    [18/04/19] : 新規作成
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


class Many2OneMultiRNNLSTM( NeuralNetworkBase ):
    """description of class
    LSTM による Many-to-one な 多層 RNN を表すクラス.
    TensorFlow での多層 RNN の処理をクラス（任意の層に DNN 化可能な柔軟なクラス）でラッピングし、
    scikit-learn ライブラリの classifier, estimator とインターフェイスを共通化することで、
    scikit-learn ライブラリとの互換性のある自作クラス

    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
        _n_hiddenLayers : int
            １つの隠れ層のノードに集約されている LSTM の数
        _n_MultiRNN : int
            多層 RNN の層数
        _n_in_sequence_encoder : int
            Encoder の各シークエンスの長さ（サイズ）

        _epochs : int
            エポック数（トレーニング回数）
        _batch_size : int
            ミニバッチ学習でのバッチサイズ
        _eval_step : int
            学習処理時に評価指数の算出処理を行う step 間隔
        _losses_train : list <float32>
            トレーニングデータでの損失関数の値の list
        _encoder_input_holder : placeholder
            Encoder の入力層にデータを供給するための placeholder
        _t_holder : placeholder
            Decoder の出力層に教師データを供給するための placeholder
        _dropout_holder : placeholder
            ドロップアウトしない確率 (1-p) にデータを供給するための placeholder

        _n_vocab : int
            vocabulary size （埋め込み行列の行数）
        _embedding_matrix_var : Variable
            埋め込み行列を表す Variable
        _embedding_lookup_op : Operator
            埋め込み検索演算を表す Operator
        
        _rnn_cells : list<BasicRNNCell クラスのオブジェクト> <tensorflow.python.ops.rnn_cell_impl.BasicRNNCell>
            RNN 構造を提供する cell のリスト <tf.nn.rnn_cell.BasicLSTMCell(...)>
            この `cell` は、内部（プロパティ）で state（隠れ層の状態）を保持しており、
            これを次の時間の隠れ層に順々に渡していくことで、時間軸の逆伝搬を実現する。
        _rnn_states : list<Tensor>
            cell の状態

    [protedted] protedted な使用法を想定 

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）


    """
    def __init__( 
            self,
            session = tf.Session( config = tf.ConfigProto(log_device_placement=True) ),
            n_hiddenLayer = 128,
            n_MultiRNN = 1,
            n_in_sequence_encoder = 25,
            n_vocab = 100,
            epochs = 1000,
            batch_size = 10,
            eval_step = 1,
            save_step = 100
        ):

        super().__init__( session )
        
        tf.set_random_seed(12)

        # 各パラメータの初期化
        self._n_hiddenLayer = n_hiddenLayer
        self._n_MultiRNN = n_MultiRNN
        self._n_in_sequence_encoder = n_in_sequence_encoder

        self._epochs = epochs
        self._batch_size = batch_size
        self._eval_step = eval_step   
        
        self._n_vocab = n_vocab
        self._save_step = save_step

        # evaluate 関連の初期化
        self._losses_train = []

        # 埋め込み関連の初期化
        self._embedding_matrix_var = None
        self._embedding_lookup_op = None

        # RNN Cell の初期化
        self._rnn_cells = []
        self._rnn_states = []

        # placeholder の初期化
        # shape の列（横方向）は、各層の次元（ユニット数）に対応させる。
        # shape の行は、None にして汎用性を確保
        self._encoder_input_holder = tf.placeholder( 
                             tf.int32, 
                             shape = [ self._batch_size, self._n_in_sequence_encoder ],
                             name = "encoder_input_holder"
                         )
        
        self._t_holder = tf.placeholder( 
                             tf.float32, 
                             shape = [ self._batch_size ],
                             name = "t_holder"
                         )
        
        self._dropout_holder = tf.placeholder( tf.float32, name = "dropout_holder" )

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

        print( "_n_hiddenLayer : ", self._n_hiddenLayer )
        print( "_n_MultiRNN : ", self._n_MultiRNN )
        print( "_n_in_sequence_encoder : ", self._n_in_sequence_encoder )
        print( "_n_vocab : ", self._n_vocab )

        print( "_epoches : ", self._epochs )
        print( "_batch_size : ", self._batch_size )
        print( "_eval_step : ", self._eval_step )

        print( "_encoder_input_holder : ", self._encoder_input_holder )
        print( "_t_holder : ", self._t_holder )
        print( "_dropout_holder : ", self._dropout_holder )

        print( "_rnn_cells : \n", self._rnn_cells )
        #if( (self._session != None) and (self._init_var_op != None) ):
            #print( self._session.run( self._rnn_cells ) )

        print( "_rnn_states : \n", self._rnn_states )
        #if( (self._session != None) and (self._init_var_op != None) ):
            #print( self._session.run( self._rnn_states ) )

        print( "----------------------------------" )
        return


    def model( self ):
        """
        モデルの定義（計算グラフの構築）を行い、
        最終的なモデルの出力のオペレーターを設定する。
        [Output]
            self._y_out_op : Operator
                モデルの出力のオペレーター
        """
        #--------------------------------------------------------------
        # Encoder 側の埋め込み層
        #--------------------------------------------------------------
        embed_size = 256
        with tf.name_scope( 'EmbeddingLayer' ):
            # 埋め込み行列を表す Variable
            self._embedding_matrix_var = tf.Variable( 
                                             #tf.random_uniform( [self._n_vocab, self._n_in_sequence_encoder], -1.0, 1.0 ),
                                             tf.random_uniform( [self._n_vocab, embed_size], -1.0, 1.0 ),
                                             name = "embedding_matrix_var"
                                         )

            # tf.nn.embedding_lookup(...) : バッチ内の各ソース単語について、ベクトルをルックアップ（検索）
            self._embedding_lookup_op = tf.nn.embedding_lookup( self._embedding_matrix_var, self._encoder_input_holder )
            print( "self._embedding_lookup_op :", self._embedding_lookup_op )


        #--------------------------------------------------------------
        # many-to-one の RNN
        #--------------------------------------------------------------        
        # 時系列に沿った RNN 構造を提供するクラス BasicLSTMCell の cell を取得する。
        # この cell は、内部（プロパティ）で state（隠れ層の状態）を保持しており、
        # これを次の時間の隠れ層に順々に渡していくことで、時間軸の逆伝搬を実現する。
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell( 
                        self._n_hiddenLayer,    # int, The number of units in the RNN(LSTM) cell.
                        forget_bias=0.0,        # 忘却ゲート
                        state_is_tuple=True 
                    )

        # cell に Dropout を適用する。（＝中間層に dropout 機能を追加）
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper( lstm_cell, output_keep_prob=self._dropout_holder )

        # 総数に対応した cell のリストを Multi RNN 化
        cells = tf.nn.rnn_cell.MultiRNNCell( [lstm_cell] * self._n_MultiRNN, state_is_tuple=True )
        
        """
        cells = tf.nn.rnn_cell.MultiRNNCell(
                    [ tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.BasicLSTMCell( 
                            self._n_hiddenLayer,          # int, The number of units in the RNN(LSTM) cell.
                            forget_bias=0.0,              # 忘却ゲート
                            state_is_tuple=True 
                        ),
                        output_keep_prob = self._dropout_holder
                    ) 
                    for i in range(self._n_MultiRNN) ]
                )
        """
        print( "cells : ", cells )
        
        # cell 初期状態を定義
        # 最初の時間 t0 では、過去の隠れ層がないので、
        # cell.zero_state(...) でゼロの状態を初期設定する。
        init_state_tsr = cells.zero_state( batch_size=self._batch_size, dtype=tf.float32 )
        self._rnn_states.append( init_state_tsr )
        print( "init_state_tsr :", init_state_tsr )
        
        # tf.nn.dynamic_rnn(...) を用いて、シーケンス長が可変長な RNN シーケンスを作成する。
        # outputs_tsr: The RNN output Tensor
        # state_tsr : The final state
        # lstm_outputs / shape = [batch_size, max_time, cells.output_size]
        outputs_tsr, final_state_tsr = tf.nn.dynamic_rnn(
                                           cells,
                                           self._embedding_lookup_op,     
                                           initial_state = init_state_tsr
                                       )
        self._rnn_cells.append( outputs_tsr )
        self._rnn_states.append( final_state_tsr )
        print( "outputs_tsr :", outputs_tsr )                   # outputs_tsr : Tensor("rnn/transpose:0", shape=(100, 200, 256), dtype=float32)
        print( "outputs_tsr[:,-1] :", outputs_tsr[:,-1] )       # outputs_tsr[:,-1]
        print( "self._rnn_cells[-1] :", self._rnn_cells[-1] )
        print( "final_state_tsr :", final_state_tsr )

        #---------------------------------------------
        # fully connected layer
        #---------------------------------------------
        # 出力層への入力
        # This layer implements the operation: outputs = activation(inputs.kernel + bias)
        # Where activation is the activation function passed as the activation argument (if not None)
        y_in_op = tf.layers.dense(
                      inputs = outputs_tsr[:,-1],    # RNN Cell の最終的な Output
                      units = 1,                     # Integer or Long, dimensionality of the output space. / one-hot encoding していないので出力ノード数は 1
                      activation = None,
                      name = "logits"
                  )

        print( "y_in_op :", y_in_op )              # y_in_op : Tensor("logits/BiasAdd:0", shape=(100, 1), dtype=float32)

        # tf.squeeze(...) : size が 1 の次元を削除し次元数を減らす
        y_in_op = tf.squeeze( y_in_op )
        print( "y_in_op :", y_in_op )              # y_in_op : Tensor("Squeeze:0", shape=(100,), dtype=float32)
        
        #--------------------------------------------------------------
        # モデルの出力
        #--------------------------------------------------------------
        # sigmoid で活性化して最終出力
        self._y_out_op = tf.nn.sigmoid( y_in_op )
        print( "_y_out_op :", self._y_out_op )
        
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
        def generate_minibatch( x, y, batch_size ):
            n_batches = len( x ) // batch_size
            x = x[:n_batches*batch_size]
            y = y[:n_batches*batch_size]

            # batch_size 間隔で for loop
            for i in range( 0, len(x), batch_size ):
                # yield 文で逐次データを return（関数の処理を一旦停止し、値を返す）
                # メモリ効率向上のための処理
                yield x[i:i+batch_size], y[i:i+batch_size]
        
        #----------------------------
        # 学習開始処理
        #----------------------------
        # Variable の初期化オペレーター
        self._init_var_op = tf.global_variables_initializer()

        # Session の run（初期化オペレーター）
        self._session.run( self._init_var_op )

        # ミニバッチの繰り返し回数
        minibatch_iteration = 1
        
        #-------------------
        # 学習処理
        #-------------------
        # for ループでエポック数分トレーニング
        for epoch in range( self._epochs ):
            # 各エポックの最初に、RNN Cell の状態を初期状態にリセット
            # このプロセスを繰り返すことにより、エポックを通じての現在の状態の更新を実現する。
            rnn_cell_state = self._session.run( self._rnn_states[0] )

            # ミニバッチサイズ単位で for ループ
            for batch_x, batch_y in generate_minibatch( X_train, y_train , self._batch_size ):
                # 設定された最適化アルゴリズム Optimizer でトレーニング処理を run
                _, rnn_cell_state = self._session.run(
                                        [ self._train_step, self._rnn_states[-1] ],
                                        feed_dict = {
                                            self._encoder_input_holder: batch_x,
                                            self._t_holder: batch_y,
                                            self._dropout_holder: 0.5,
                                            self._rnn_states[0]: rnn_cell_state
                                        }
                                    )

                # RNN cell の最終状態に現在の状態を feed し、最終状態を更新する。
                # このプロセスを繰り返すことにより、エポックを通じての現在の状態の更新を実現する。
                """
                rnn_cell_state = self._session.run(
                                     self._rnn_states[-1],
                                     feed_dict = {
                                         self._rnn_states[0] : rnn_cell_state
                                     }
                                 )
                """

                minibatch_iteration += 1
            
                # 評価処理を行う loop か否か
                # % : 割り算の余りが 0 で判断
                if ( ( (epoch+1) % self._eval_step ) == 0 ):
                    # 損失関数値の算出
                    loss = self._loss_op.eval(
                               session = self._session,
                               feed_dict = {
                                   self._encoder_input_holder: batch_x,
                                   self._t_holder: batch_y,
                                   self._dropout_holder: 0.5
                               }
                           )

                    self._losses_train.append( loss )
                    print( "Epoch: %d/%d, minibatch iteration: %d / loss = %0.5f" % ( (epoch+1), self._epochs, minibatch_iteration, loss ) )


        return self._y_out_op



    def predict( self, X_test ):
        """
        fitting 処理したモデルで、推定を行い、予想値を返す。

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
        def generate_minibatch( x, batch_size ):
            n_batches = len( x ) // batch_size
            x = x[:n_batches*batch_size]

            # batch_size 間隔で for loop
            for i in range( 0, len(x), batch_size ):
                # yield 文で逐次データを return（関数の処理を一旦停止し、値を返す）
                # メモリ効率向上のための処理
                yield x[i:i+batch_size]

        preds = []
        test_state = self._session.run( self._rnn_states[0] )   # RNN Cell の初期状態で Session run()

        for (i, batch_x) in enumerate( generate_minibatch(X_test, self._batch_size), 1 ):
            pred, test_state = self._session.run(
                                   [ tf.cast( tf.round(self._y_out_op), tf.int32 ), self._rnn_states[-1] ],
                                   feed_dict = {
                                       self._encoder_input_holder: batch_x,
                                       self._dropout_holder: 0.5,
                                       self._rnn_states[0]: test_state
                                   }
                               )

            preds.append( pred )

            #print( "pred :", pred )
            #print( "test_state :", test_state )

        # numpy.concatenate(...) :
        return numpy.concatenate( preds )


    def predict_proba( self, X_test ):
        """
        fitting 処理したモデルで、推定を行い、クラスの所属確率の予想値を返す。
        proba : probability
        [Input]
            X_test : numpy.ndarry ( shape = [n_samples, n_features] )
                予想したい特徴行列
        """
        def generate_minibatch( x, batch_size ):
            n_batches = len( x ) // batch_size
            x = x[:n_batches*batch_size]

            # batch_size 間隔で for loop
            for i in range( 0, len(x), batch_size ):
                # yield 文で逐次データを return（関数の処理を一旦停止し、値を返す）
                # メモリ効率向上のための処理
                yield x[i:i+batch_size]

        preds = []
        test_state = self._session.run( self._rnn_states[0] )   # RNN Cell の初期状態で Session run()

        for (i, batch_x) in enumerate( generate_minibatch(X_test, self._batch_size), 1 ):
            pred, test_state = self._session.run(
                                   #[ "probabilities:0", self._rnn_states[-1] ],     # ?
                                   [ self._y_out_op, self._rnn_states[-1] ],
                                   feed_dict = {
                                       self._encoder_input_holder: batch_x,
                                       self._dropout_holder: 0.5,
                                       self._rnn_states[0]: test_state
                                   }
                               )

            preds.append( pred )

            #print( "pred :", pred )
            #print( "test_state :", test_state )

        # numpy.concatenate(...) :
        return numpy.concatenate( preds )


    def accuracy( self, X_test, y_test ):
        """
        指定したデータでの正解率 [accuracy] を計算する。
        """
        # 予想ラベルを算出する。
        predict = self.predict( X_test )

        # 正解数
        n_correct = numpy.sum( numpy.equal( predict, y_test[0:predict.shape[0]] ) )
        #print( "numpy.equal( predict, y_test ) :", numpy.equal( predict, y_test ) )    # ValueError: operands could not be broadcast together with shapes (25000,) (25001,) 
        #print( "n_correct :", n_correct )

        # 正解率 = 正解数 / データ数
        accuracy = n_correct / X_test.shape[0]

        return accuracy

