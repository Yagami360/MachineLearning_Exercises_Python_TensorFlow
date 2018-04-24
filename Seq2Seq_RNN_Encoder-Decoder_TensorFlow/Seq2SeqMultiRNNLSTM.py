# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境 (TensorFlow インストール済み)

"""
    更新情報
    [18/04/24] : 新規作成
    [xx/xx/xx] :

"""

import numpy as np

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


class Seq2SeqMultiRNNLSTM( NeuralNetworkBase ):
    """description of class
    LSTM による Seq2Seq での (many-to-many) な 多層 RNN を表すクラス.
    TensorFlow での多層 RNN の処理をクラス（任意の層に DNN 化可能な柔軟なクラス）でラッピングし、
    scikit-learn ライブラリの classifier, estimator とインターフェイスを共通化することで、
    scikit-learn ライブラリとの互換性のある自作クラス

    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
        _n_classes : int
            テキストコーパスの文字の総数
        _n_steps : int
            ミニバッチの分割ステップ数（入出力の placeholder の横 size になる）

        _n_hiddenLayers : int
            １つの隠れ層のノードに集約されている LSTM の数
        _n_MultiRNN : int
            多層 RNN の層数

        _epochs : int
            エポック数（学習対象のデータセットが完全に通過した回数）
        _batch_size : int
            ミニバッチ学習でのバッチサイズ
        _eval_step : int
            学習処理時に評価指数の算出処理を行う step 間隔

        _losses_train : list <float32>
            トレーニングデータでの損失関数の値の list

        _input_holder : placeholder
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
            n_classes = 100,
            n_steps = 10,
            n_hiddenLayer = 128,
            n_MultiRNN = 1,
            n_vocab = 100,
            epochs = 1000,
            batch_size = 10,
            eval_step = 1,
            save_step = 100
        ):

        super().__init__( session )
        
        tf.set_random_seed(12)

        # 各パラメータの初期化
        self._n_classes = n_classes
        self._n_steps = n_steps

        self._n_hiddenLayer = n_hiddenLayer
        self._n_MultiRNN = n_MultiRNN

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
        self._input_holder = tf.placeholder( 
                                 tf.int32, 
                                 shape = [ self._batch_size, self._n_steps ],
                                 name = "input_holder"
                             )
        
        self._t_holder = tf.placeholder( 
                             tf.float32, 
                             shape = [ self._batch_size, self._n_steps ],
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

        print( "_n_classes : ", self._n_classes )
        print( "_n_steps : ", self._n_steps )
        print( "_n_hiddenLayer : ", self._n_hiddenLayer )
        print( "_n_MultiRNN : ", self._n_MultiRNN )
        print( "_n_vocab : ", self._n_vocab )

        print( "_epoches : ", self._epochs )
        print( "_batch_size : ", self._batch_size )
        print( "_eval_step : ", self._eval_step )

        print( "_input_holder : ", self._input_holder )
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


