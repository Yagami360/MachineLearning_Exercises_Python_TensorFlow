# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境 (TensorFlow インストール済み)

"""
    更新情報
    [17/12/14] : 新規作成
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
from RecurrectNNEncoderDecoderLSTM import RecurrectNNEncoderDecoderLSTM

import NNActivation
from NNActivation import NNActivation               # ニューラルネットワークの活性化関数を表すクラス
from NNActivation import Sigmoid
from NNActivation import Relu
from NNActivation import Softmax

import NNLoss                                       # ニューラルネットワークの損失関数を表すクラス

import NNOptimizer                                  # ニューラルネットワークの最適化アルゴリズム Optimizer を表すクラス

class RecurrectNNEncoderDecoderEmbeddingLSTM( RecurrectNNEncoderDecoderLSTM ):
    """
    LSTM による RNN Encoder-Decoder を表すクラス.
    TensorFlow での RNN Encoder-Decoder の処理をクラス（任意の層に DNN 化可能な柔軟なクラス）でラッピングし、
    scikit-learn ライブラリの classifier, estimator とインターフェイスを共通化することで、
    scikit-learn ライブラリとの互換性のある自作クラス
    ------------------------------------------------------------------------------------------------
    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
        _n_vocab : int
            vocabulary size （埋め込み行列の行数）

        _embedding_matrix_var : Variable
            埋め込み行列を表す Variable
        _embedding_lookup_op : Operator
            埋め込み検索演算を表す Operator

    [protedted] protedted な使用法を想定 

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( 
            self,
            session = tf.Session( config = tf.ConfigProto(log_device_placement=True) ),
            n_inputLayer = 1, n_hiddenLayer = 1, n_outputLayer = 1, 
            n_in_sequence_encoder = 25,
            n_in_sequence_decoder = 25,
            n_vocab = 500,
            epochs = 1000,
            batch_size = 10,
            eval_step = 1,
            save_step = 100
        ):
        """
        コンストラクタ（厳密にはイニシャライザ）
        """
        super().__init__( 
            session,
            n_inputLayer, n_hiddenLayer, n_outputLayer,
            n_in_sequence_encoder, n_in_sequence_decoder,
            epochs, batch_size, eval_step
        )

        self._n_vocab = n_vocab
        self._save_step = save_step

        self._embedding_matrix_var = None
        self._embedding_lookup_op = None

        # placeholder の初期化
        # shape の列（横方向）は、各層の次元（ユニット数）に対応させる。
        # shape の行は、None にして汎用性を確保
        self._X_holder = tf.placeholder( 
                             tf.int32, 
                             shape = [ None, self._n_in_sequence_encoder ],
                             name = "X_holder"
                         )

        self._t_holder = tf.placeholder( 
                             tf.int32, 
                             shape = [ None, self._n_in_sequence_decoder ],
                             name = "t_holder"
                         )

        self._dropout_holder = tf.placeholder( tf.float32, name = "dropout_holder" )
        self._batch_size_holder = tf.placeholder( tf.int32, shape=[], name = "batch_size_holder" )
        self._bTraining_holder = tf.placeholder( tf.bool, name = "bTraining_holder" )

        return

    def print( self, str):
        print( "---------------------------" )
        print( self )     
        super().print( str )

        print( "self._n_vocab : {}".format( self._n_vocab ) )

        print( "self._save_step : {}".format( self._save_step ) )

        print( "self._embedding_matrix_var : {}".format( self._embedding_matrix_var ) )
        print( "self._embedding_lookup_op : {}".format( self._embedding_lookup_op ) )

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
        # 埋め込み行列（単語ベクトルの集合）と埋め込み探索演算を作成
        #--------------------------------------------------------------
        # 埋め込み行列を表す Variable
        self._embedding_matrix_var = tf.Variable( 
                                         tf.random_uniform( [self._n_vocab, self._n_in_sequence_encoder], -1.0, 1.0 ),
                                         name = "embedding_matrix_var"
                                     )

        # tf.nn.embedding_lookup(...) : バッチ内の各ソース単語について、ベクトルをルックアップ（検索）
        self._embedding_lookup_op = tf.nn.embedding_lookup( self._embedding_matrix_var, self._X_holder )
        #embedding_expanded_op = tf.expand_dims( self._embedding_lookup_op, -1 )

        #--------------------------------------------------------------
        # Encoder
        #--------------------------------------------------------------

        # tf.contrib.rnn.BasicRNNCell(...) : 時系列に沿った RNN 構造を提供するクラス `BasicRNNCell` のオブジェクト cell を返す。
        # この cell は、内部（プロパティ）で state（隠れ層の状態）を保持しており、
        # これを次の時間の隠れ層に順々に渡していくことで、時間軸の逆伝搬を実現する。
        cell_encoder = tf.contrib.rnn.BasicRNNCell( 
                           num_units = self._n_hiddenLayer     # int, The number of units in the RNN cell.
                       )

        # 最初の時間 t0 では、過去の隠れ層がないので、
        # cell.zero_state(...) でゼロの状態を初期設定する。
        init_state_encoder_tsr = cell_encoder.zero_state( self._batch_size_holder, tf.float32 )

        self._rnn_cells_encoder.append( cell_encoder )
        self._rnn_states_encoder.append( init_state_encoder_tsr )

        # 過去の隠れ層の再帰処理
        with tf.variable_scope('Encoder'):
            # 動的に動作する RNN シーケンス を作成
            # outputs_tsr: The RNN output Tensor
            # state_tsr : The final state
            outputs_tsr, state_tsr = tf.nn.dynamic_rnn(  
                                         cell_encoder, 
                                         self._embedding_lookup_op, 
                                         dtype=tf.float32 
                                     )
        
            self._rnn_states_encoder.append( state_tsr )

        # 隠れ層 ~ 出力層への重み
        self._weights.append( self.init_weight_variable( input_shape = [self._n_hiddenLayer, self._n_vocab] ) )
        self._biases.append( self.init_bias_variable( input_shape = [self._n_vocab] ) )

        #--------------------------------------------------------------
        # Decoder
        #--------------------------------------------------------------
        # tf.contrib.legacy_seq2seq.rnn_decoder : RNN decoder for the sequence-to-sequence model.
        # [Args:]
        #   decoder_inputs : A list of 2D Tensors [batch_size x input_size].
        #   initial_state: 2D Tensor with shape [batch_size x cell.state_size].
        #   cell: rnn_cell.RNNCell defining the cell function and size.
        #   loop_function:
        #       If not None, this function will be applied to the i-th output in order to generate the i+1-st input, 
        #       and decoder_inputs will be ignored, except for the first element ("GO" symbol). 
        #       This can be used for decoding, but also for training to emulate http://arxiv.org/abs/1506.03099. Signature -- loop_function(prev, i) = next
        #
        #decoder = tf.contrib.legacy_seq2seq.rnn_decoder
        
        
        
        #--------------------------------------------------------------
        # 
        #--------------------------------------------------------------
        return self._y_out_op

