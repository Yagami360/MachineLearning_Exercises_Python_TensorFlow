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
        # with tf.name_scope( "embedding_layer" ):

        # 埋め込み行列を表す Variable
        self._embedding_matrix_var = tf.Variable( 
                                         tf.random_uniform( [self._n_vocab, self._n_in_sequence_encoder], -1.0, 1.0 ),
                                         name = "embedding_matrix_var"
                                     )

        # tf.nn.embedding_lookup(...) : バッチ内の各ソース単語について、ベクトルをルックアップ（検索）
        # self._embedding_lookup_op / shape = [None, self._n_in_sequence_encoder, self._n_in_sequence_encoder]
        self._embedding_lookup_op = tf.nn.embedding_lookup( self._embedding_matrix_var, self._X_holder )
        #embedding_expanded_op = tf.expand_dims( self._embedding_lookup_op, -1 )
        print( "self._embedding_lookup_op :", self._embedding_lookup_op )

        # 埋め込みを 1 次元 Tensor 状に reshape
        # tf.split(...) : Tensorを指定した次元方向に分割
        # shape = [None, self._n_in_sequence_encoder, self._n_in_sequence_encoder] → shape = [self._n_in_sequence_encoder]
        # 各 Tensor の shape は、rnn_inputs[i] / shape = [None, 1, self._n_in_sequence_encoder] 
        rnn_inputs = tf.split( 
                         value = self._embedding_lookup_op,                 # 分割する Tensor
                         num_or_size_splits = self._n_in_sequence_encoder,  # 分割する数
                         axis=1                                             # 分割する次元方向
                     )
        print( "rnn_inputs :", rnn_inputs )
        
        # shape = 1 の次元を trimmed（トリミング）
        # tf.squeeze(...) : 指定された size 数に該当する次元を削除する。
        # shape = [None, 1, self._n_in_sequence_encoder] → shape = [None, self._n_in_sequence_encoder]
        # 各 Tensor の shape は、rnn_inputs_trimmed[i] / shape = [None, self._n_in_sequence_encoder] 
        rnn_inputs_trimmed = [ tf.squeeze( tsr, [1] ) for tsr in rnn_inputs ]
        print( "rnn_inputs_trimmed :", rnn_inputs_trimmed )

        #--------------------------------------------------------------
        # Encoder
        # 埋め込み層を使用しているので Encoder は不要？
        #--------------------------------------------------------------
        # 忘却ゲートなしの LSTM
        cell_encoder = tf.contrib.rnn.BasicLSTMCell( self._n_hiddenLayer )
        init_state_encoder = cell_encoder.zero_state( self._batch_size_holder, tf.float32 )
        self._rnn_cells_encoder.append( cell_encoder )
        self._rnn_states_encoder.append( init_state_encoder )
        
        #--------------------------------------------------------------
        # Decoder
        #--------------------------------------------------------------
        # 隠れ層 → 出力層への重み
        self._weights.append( self.init_weight_variable( input_shape = [self._n_hiddenLayer, self._n_vocab] ) )
        self._biases.append( self.init_bias_variable( input_shape = [self._n_vocab] ) )

        # ? 損失関数等の評価指数の計算時の RNN の再帰処理（ループ処理）を行う関数
        # i 番目の出力から、i+1 番目の入力（埋め込みベクトル）を取得する。
        # 後の、tf.contrib.legacy_seq2seq.rnn_decoder の call 関数の引数 loop_function で指定
        def eval_rnn_loop( prev_output_op, count ):
            # prev_output_op : i 番目の出力
            # matmul 計算時、直前の出力 self._rnn_cells_decoder[-1] を入力に用いる
            prev_trans_output_op = tf.matmul( prev_output_op, self._weights[-1] ) + self._biases[-1]

            # ? 出力インデックスを取得
            # tf.stop_gradient(...) : https://stackoverflow.com/questions/33727935/how-to-use-stop-gradient-in-tensorflow
            #   an operation that acts as the identity function in the forward direction,
            #   but stops the accumulated gradient from flowing through that operator in the backward direction. 
            #    It does not prevent backpropagation altogether, but instead prevents an individual tensor from contributing to the gradients that are computed for an expression. 
            prev_op_symbol = tf.stop_gradient( tf.arg_max( prev_trans_output_op, 1 ) )

            # 埋め込みベクトルを取得
            output = tf.nn.embedding_lookup( self._embedding_matrix_var, prev_op_symbol )
            
            return output

        # ? Decoder からの再帰処理に基づく、出力リストと最終的な state を取得する。
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
        #  f.contrib.legacy_seq2seq.rnn_decoder : <function rnn_decoder at 0x00000206908E0158>
        decoder = tf.contrib.legacy_seq2seq.rnn_decoder
        print( "tf.contrib.legacy_seq2seq.rnn_decoder :", decoder ) 
        
        """
        # トレーニング処理中の場合のルート
        if ( self._bTraining_holder == True ):
            # tf.contrib.legacy_seq2seq.rnn_decoder の __call__() 関数を呼び出し、
            # Decoder からの再帰処理に基づく、出力リストと最終的な state を取得する。
            # shape = [self._n_in_sequence_decoder] 個の Tensor 配列 / 各 Tensor の shape = [None, self._n_hidden_Layer]
            outputs_decoder, last_state_decoder = decoder(
                                                      decoder_inputs = rnn_inputs_trimmed,  # 埋め込み層からの入力
                                                      initial_state = self._rnn_states_encoder[-1],
                                                      cell = self._rnn_cells_encoder[-1],
                                                      loop_function = None
                                                  )

        # loss 値などの評価用の値の計算時のルート
        else:
            # tf.contrib.legacy_seq2seq.rnn_decoder の __call__() 関数を呼び出し、
            # Decoder からの再帰処理に基づく、出力リストと最終的な state を取得する。
            outputs_decoder, last_state_decoder = decoder(
                                                      decoder_inputs = rnn_inputs_trimmed,  # 埋め込み層からの入力
                                                      initial_state = self._rnn_states_encoder[-1],
                                                      cell = self._rnn_cells_encoder[-1],
                                                      loop_function = eval_rnn_loop
                                                  )

        """

        # tf.contrib.legacy_seq2seq.rnn_decoder の __call__() 関数を呼び出し、
        # Decoder からの再帰処理に基づく、出力リストと最終的な state を取得する。
        # shape = [self._n_in_sequence_decoder] 個の Tensor 配列 / 各 Tensor の shape = [None, self._n_hidden_Layer]
        outputs_decoder, last_state_decoder = decoder(
                                                  decoder_inputs = rnn_inputs_trimmed,  # 埋め込み層からの入力
                                                  initial_state = self._rnn_states_encoder[-1],
                                                  cell = self._rnn_cells_encoder[-1],
                                                  loop_function = None
                                              )

        print( "outputs_decoder :\n", outputs_decoder )
        
        # Decoder の出力を `tf.concat(...)` で結合し、`tf.reshape(...)` で適切な形状に reshape する。 
        # self.outputs_decoder の形状を shape = ( データ数, 隠れ層のノード数 ) に reshape 
        # tf.concat(...) : Tensorを結合する。引数 axis で結合する dimension を決定
        output = tf.reshape( 
                     tf.concat( outputs_decoder, axis = 1 ),
                     shape = [ -1, self._n_hiddenLayer ]
                 )
        
        print( "output :\n", output )

        #--------------------------------------------------------------
        # 出力層
        #--------------------------------------------------------------
        # 出力層への入力
        # shape = [None, self._n_vocab]
        y_in_op = tf.matmul( output, self._weights[-1] ) + self._biases[-1]
        
        # 最終的な出力
        # shape = [None, self._n_vocab]
        self._y_out_op = tf.nn.softmax( y_in_op )

        return self._y_out_op


    def loss( self ):

        # ?
        # tf.contrib.legacy_seq2seq.sequence_loss_by_example : Weighted cross-entropy loss for a sequence of logits (per example).
        loss_func = tf.contrib.legacy_seq2seq.sequence_loss_by_example

        return self._loss_op