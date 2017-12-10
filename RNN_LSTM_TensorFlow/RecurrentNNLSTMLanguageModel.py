# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境 (TensorFlow インストール済み)

"""
    更新情報
    [17/12/09] : 新規作成
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
from RecurrectNNLanguageModel import RecurrectNNLanguageModel

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
from NNOptimizer import Adam

class RecurrentNNLSTMLanguageModel( RecurrectNNLanguageModel ):
    """
    リカレントニューラルネットワーク [RNN : Recurrent Neural Network] の 
    [LSTM : Long short-term memory]（長短期記憶モデル）を表すクラス。
    TensorFlow での RNN の処理をクラス（任意の層に DNN 化可能な柔軟なクラス）でラッピングし、
    scikit-learn ライブラリの classifier, estimator とインターフェイスを共通化することで、
    scikit-learn ライブラリとの互換性のある自作クラス
    ------------------------------------------------------------------------------------------------
    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
        _rnn_cells : list<LSTMCell クラスのオブジェクト> <tf.Tensor 'RNN-LSTM/RNN-LSTM/lstm_cell>
            RNN 構造を提供する cell のリスト
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
            n_inputLayer = 1, n_hiddenLayer = 1, n_outputLayer = 1, 
            n_in_sequence = 25,
            n_vocab = 100,
            n_in_embedding_vec = 50,
            epochs = 1000,
            batch_size = 10,
            eval_step = 1,
            save_step = 500
        ):
        """
        コンストラクタ（厳密にはイニシャライザ）
        """
        # 親クラスである ReccurentRNN クラスのコンストラクタ呼び出し
        super().__init__( 
            session, 
            n_inputLayer, n_hiddenLayer, n_outputLayer, 
            n_in_sequence, n_vocab, n_in_embedding_vec, 
            epochs, batch_size, eval_step 
        )
        self._save_step = save_step


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
        self._embedding_matrix_var = tf.Variable( 
                                         tf.random_uniform( [self._n_vocab, self._n_in_embedding_vec], -1.0, 1.0 ),
                                         name = "embedding_matrix_var"
                                     )

        # tf.nn.embedding_lookup(...) : バッチ内の各ソース単語について、ベクトルをルックアップ（検索）
        self._embedding_lookup_op = tf.nn.embedding_lookup( self._embedding_matrix_var, self._X_holder )
        #embedding_expanded_op = tf.expand_dims( self._embedding_lookup_op, -1 )

        #--------------------------------------------------------------
        # 入力層 ~ 隠れ層
        #--------------------------------------------------------------
        # tf.contrib.rnn.BasicRNNCell(...) : 時系列に沿った RNN 構造を提供するクラス `BasicRNNCell` のオブジェクト cell を返す。
        # この cell は、内部（プロパティ）で state（隠れ層の状態）を保持しており、
        # これを次の時間の隠れ層に順々に渡していくことで、時間軸の逆伝搬を実現する。
        cell = tf.contrib.rnn.LSTMCell( 
                   num_units = self._n_hiddenLayer     # int, The number of units in the RNN cell.
                   #activation = "tanh"                  # Nonlinearity to use. Default: tanh
               )
        self._rnn_cells.append( cell )

        #-----------------------------------------------------------------
        # 過去の隠れ層の再帰処理
        #-----------------------------------------------------------------
        with tf.variable_scope('RNN-LSTM-LM'):
            # 最初の時間 t0 では、過去の隠れ層がないので、
            # cell.zero_state(...) でゼロの状態を初期設定する。
            #initial_state_tsr = cell.zero_state( self._batch_size_holder, tf.float32 )
            #self._rnn_states.append( initial_state_tsr )

            # 動的に動作する RNN シーケンス を作成
            # outputs_tsr: The RNN output Tensor
            # state_tsr : The final state
            outputs_tsr, state_tsr = tf.nn.dynamic_rnn(  
                                    cell, 
                                    self._embedding_lookup_op, 
                                    dtype=tf.float32 
                                )
        
            self._rnn_states.append( state_tsr )
            print( "outputs_tsr :", outputs_tsr )   # outputs_tsr : Tensor("rnn/transpose:0", shape=(?, 25, 10), dtype=float32)
            print( "state_tsr :", state_tsr )       # state_tsr : Tensor("rnn/while/Exit_2:0", shape=(?, 10), dtype=float32)
        
            # ドロップアウト処理を施す
            output = tf.nn.dropout( outputs_tsr, self._keep_prob_holder )
            print( "output :", output )             # output : Tensor("dropout/mul:0", shape=(?, 25, 10), dtype=float32)

            # 予想値を取得するため、RNN を並び替えて、最後の出力を取り出す
            output = tf.transpose( output, [1, 0, 2] )
            print( "output :", output )             # output : Tensor("transpose_1:0", shape=(25, ?, 10), dtype=float32)

            # 最終的な隠れ層の出力
            # tf.gather(...) : axis で指定した階でスライスして，indeices で指定したインデックスのテンソルだけ取り出す。
            h_out_op = tf.gather( output, int(output.get_shape()[0]) - 1 )
            print( "h_out_op :", h_out_op )         # h_out_op : Tensor("Gather:0", shape=(?, 10), dtype=float32)

        # 隠れ層 ~ 出力層
        self._weights.append( self.init_weight_variable( input_shape = [self._n_hiddenLayer, self._n_outputLayer] ) )
        self._biases.append( self.init_bias_variable( input_shape = [self._n_outputLayer] ) )

        #--------------------------------------------------------------
        # 出力層への入力
        #--------------------------------------------------------------
        y_in_op = tf.matmul( h_out_op, self._weights[-1] ) + self._biases[-1]

        #--------------------------------------------------------------
        # モデルの出力
        #--------------------------------------------------------------
        # softmax
        self._y_out_op = Softmax().activate( y_in_op )

        return self._y_out_op
