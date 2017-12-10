# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境 (TensorFlow インストール済み)

"""
    更新情報
    [17/12/01] : 新規作成
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
from RecurrentNN import RecurrentNN

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


class RecurrentNNLSTM( RecurrentNN ):
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
            epochs = 1000,
            batch_size = 10,
            eval_step = 1
        ):
        """
        コンストラクタ（厳密にはイニシャライザ）
        """
        # 親クラスである ReccurentRNN クラスのコンストラクタ呼び出し
        super().__init__( session, n_inputLayer, n_hiddenLayer, n_outputLayer, n_in_sequence, epochs, batch_size, eval_step )

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
        # 入力層 ~ 隠れ層
        #--------------------------------------------------------------
        # tf.contrib.rnn.LSTMCell(...) : 時系列に沿った RNN 構造を提供するクラス `LSTMCell` のオブジェクト cell を返す。
        # この cell は、内部（プロパティ）で state（隠れ層の状態）を保持しており、
        # これを次の時間の隠れ層に順々に渡していくことで、時間軸の逆伝搬を実現する。
        cell = tf.contrib.rnn.LSTMCell( 
                   num_units = self._n_hiddenLayer,     # int, The number of units in the RNN cell.
                   forget_bias = 1.0                    # 忘却ゲートのバイアス項 / Default : 1.0  in order to reduce the scale of forgetting at the beginning of the training.
                   #activation = "tanh"                  # Nonlinearity to use. Default: tanh
               )
        #print( "cell :", cell )

        # 最初の時間 t0 では、過去の隠れ層がないので、
        # cell.zero_state(...) でゼロの状態を初期設定する。
        initial_state_tsr = cell.zero_state( self._batch_size_holder, tf.float32 )
        #print( "initial_state_tsr :", initial_state_tsr )

        #-----------------------------------------------------------------
        # 過去の隠れ層の再帰処理
        #-----------------------------------------------------------------
        self._rnn_states.append( initial_state_tsr )

        with tf.variable_scope('RNN-LSTM'):
            for t in range( self._n_in_sequence ):
                if (t > 0):
                    # tf.get_variable_scope() : 名前空間を設定した Variable にアクセス
                    # reuse_variables() : reuse フラグを True にすることで、再利用できるようになる。
                    tf.get_variable_scope().reuse_variables()

                # LSTMCellクラスの `__call__(...)` を順次呼び出し、
                # 各時刻 t における出力 cell_output, 及び状態 state を算出
                cell_output, state_tsr = cell( inputs = self._X_holder[:, t, :], state = self._rnn_states[-1] )

                # 過去の隠れ層の出力をリストに追加
                self._rnn_cells.append( cell_output )
                self._rnn_states.append( state_tsr )

        # 最終的な隠れ層の出力
        output = self._rnn_cells[-1]

        # 隠れ層 ~ 出力層
        self._weights.append( self.init_weight_variable( input_shape = [self._n_hiddenLayer, self._n_outputLayer] ) )
        self._biases.append( self.init_bias_variable( input_shape = [self._n_outputLayer] ) )

        #--------------------------------------------------------------
        # 出力層への入力
        #--------------------------------------------------------------
        y_in_op = tf.matmul( output, self._weights[-1] ) + self._biases[-1]

        #--------------------------------------------------------------
        # モデルの出力
        #--------------------------------------------------------------
        # 線形活性
        self._y_out_op = y_in_op

        return self._y_out_op
