# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境 (TensorFlow インストール済み)

"""
    更新情報
    [18/04/24] : 新規作成
    [xx/xx/xx] :

"""

import os
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
            テキストコーパスの文字の種類の総数
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

        _bSamplingMode : bool
            サンプリングモード（非トレーニング処理中）か否かを表す bool 値

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
            n_classes = 500,
            n_steps = 10,
            n_hiddenLayer = 128,
            n_MultiRNN = 1,
            epochs = 100,
            batch_size = 50,
            eval_step = 1,
            save_step = 100,
            bSamplingMode = False
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
        self._save_step = save_step
        self._bSamplingMode = bSamplingMode

        # evaluate 関連の初期化
        self._losses_train = []

        # RNN Cell の初期化
        self._rnn_cells = []
        self._rnn_states = []

        # サンプリングモードに応じた処理の切り替え
        if ( self._bSamplingMode == False ):
            self._batch_size = batch_size
            self._n_steps = n_steps
        else:
            self._batch_size = 1
            self._n_steps = 1


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

        print( "_epoches : ", self._epochs )
        print( "_batch_size : ", self._batch_size )
        print( "_eval_step : ", self._eval_step )
        print( "_save_step : ", self._save_step )
        print( "_bSamplingMode : ", self._bSamplingMode )

        print( "_input_holder : ", self._input_holder )
        print( "_t_holder : ", self._t_holder )
        print( "_dropout_holder : ", self._dropout_holder )

        print( "_rnn_cells : \n", self._rnn_cells )
        print( "_rnn_states : \n", self._rnn_states )
        
        print( "----------------------------------" )
        return


    def model( self, reuse = False ):
        """
        モデルの定義（計算グラフの構築）を行い、
        最終的なモデルの出力のオペレーターを設定する。
        [Output]
            self._y_out_op : Operator
                モデルの出力のオペレーター
        """
        with tf.variable_scope( tf.get_variable_scope(), reuse = reuse ):
            #--------------------------------------------------------------
            # 学習時と推定時で処理の切り替えを行う。
            #--------------------------------------------------------------
            # placeholder の初期化        
            self._input_holder = tf.placeholder( 
                                     tf.int32, 
                                     shape = [ self._batch_size, self._n_steps ],
                                     name = "input_holder"
                                 )
        
            self._t_holder = tf.placeholder( 
                                 tf.int32, 
                                 shape = [ self._batch_size, self._n_steps ],
                                 name = "t_holder"
                             )

            self._dropout_holder = tf.placeholder( tf.float32, name = "dropout_holder" )

            #--------------------------------------------------------------
            # Encoder 側の埋め込み層
            #--------------------------------------------------------------
            # ? 不要。 one-hot encoding するため？

    
            #--------------------------------------------------------------
            # one-hot encoding
            #--------------------------------------------------------------
            x_onehot = tf.one_hot( self._input_holder, depth = self._n_classes )    # 入力データを one-hot encoding

            #--------------------------------------------------------------
            # many-to-many の RNN
            #--------------------------------------------------------------        
            # 時系列に沿った RNN 構造を提供するクラス BasicLSTMCell の cell を取得する。
            # この cell は、内部（プロパティ）で state（隠れ層の状態）を保持しており、
            # これを次の時間の隠れ層に順々に渡していくことで、時間軸の逆伝搬を実現する。
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell( 
                            self._n_hiddenLayer,    # int, The number of units in the RNN(LSTM) cell.
                            forget_bias = 0.0,        # 忘却ゲート
                            state_is_tuple = True 
                        )

            # cell に Dropout を適用する。（＝中間層に dropout 機能を追加）
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper( lstm_cell, output_keep_prob=self._dropout_holder )

            # 総数に対応した cell のリストを Multi RNN 化
            cells = tf.nn.rnn_cell.MultiRNNCell( [lstm_cell] * self._n_MultiRNN, state_is_tuple=True )
            #print( "cells : ", cells )
        
            # cell 初期状態を定義
            # 最初の時間 t0 では、過去の隠れ層がないので、
            # cell.zero_state(...) でゼロの状態を初期設定する。
            init_state_tsr = cells.zero_state( batch_size = self._batch_size, dtype=tf.float32 )
            self._rnn_states.append( init_state_tsr )
            #print( "init_state_tsr :", init_state_tsr )
        
            # tf.nn.dynamic_rnn(...) を用いて、シーケンス長が可変長な RNN シーケンスを作成する。
            # outputs_tsr: The RNN output Tensor
            # state_tsr : The final state
            # lstm_outputs / shape = [batch_size, max_time, cells.output_size]
            outputs_tsr, final_state_tsr = tf.nn.dynamic_rnn(
                                               cells,
                                               inputs = x_onehot,               # 入力として、one-hot encoding された入力データ
                                               initial_state = init_state_tsr
                                           )

            self._rnn_cells.append( outputs_tsr )
            self._rnn_states.append( final_state_tsr )
            #print( "outputs_tsr :", outputs_tsr )                   # outputs_tsr : Tensor("rnn/transpose:0", shape=(100, 200, 256), dtype=float32)
            #print( "outputs_tsr[:,-1] :", outputs_tsr[:,-1] )       # outputs_tsr[:,-1]
            #print( "self._rnn_cells[-1] :", self._rnn_cells[-1] )
            #print( "final_state_tsr :", final_state_tsr )

            #---------------------------------------------
            # fully connected layer
            #---------------------------------------------
            # ２次元 Tensor に reshape
            outputs_reshaped_tsr = tf.reshape( outputs_tsr, shape = [-1, self._n_hiddenLayer ] )

            # 出力層への入力
            # This layer implements the operation: outputs = activation(inputs.kernel + bias)
            # Where activation is the activation function passed as the activation argument (if not None)
            y_in_op = tf.layers.dense(
                          inputs = outputs_reshaped_tsr,    # RNN Cell の最終的な Output
                          units = self._n_classes,          # Integer or Long, dimensionality of the output space. / one-hot なので、特徴量の数に対応させる。
                          activation = None
                      )

            #print( "y_in_op :", y_in_op )              #
        
            #--------------------------------------------------------------
            # モデルの出力
            #--------------------------------------------------------------
            # softmax 出力（出力ノードが複数個存在するため）
            self._y_out_op = Softmax().activate( y_in_op )
            #print( "_y_out_op :", self._y_out_op )
        
        return self._y_out_op

    
    def loss( self, nnLoss, reuse = False ):
        """
        損失関数の定義を行う。
        
        [Input]
            nnLoss : NNLoss クラスのオブジェクト
            
        [Output]
            self._loss_op : Operator
                損失関数を表すオペレーター
        """
        with tf.variable_scope( tf.get_variable_scope(), reuse = reuse ):
            t_onehot = tf.one_hot( self._t_holder, depth = self._n_classes )               # 出力データを one-hot encoding
            t_reshaped_holder = tf.reshape( t_onehot, shape = [-1, self._n_classes ] )     # loss 値の計算時の y_out_op との形状の整合性のため reshape
        
            self._loss_op = nnLoss.loss( t_holder = t_reshaped_holder, y_out_op = self._y_out_op )
        
        return self._loss_op


    def optimizer( self, nnOptimizer, reuse = False ):
        """
        モデルの最適化アルゴリズムの設定を行う。

        [Input]
            nnOptimizer : NNOptimizer のクラスのオブジェクト

        [Output]
            optimizer の train_step
        """
        with tf.variable_scope( tf.get_variable_scope(), reuse = reuse ):
            #------------------------------------------------
            # 勾配損失問題を回避するための勾配刈り込み処理
            #------------------------------------------------
            # 後で訓練可能変数のみを集めるための処理（ trainable=Trueの変数 ）
            trainable_vars = tf.trainable_variables()
        
            # 全体のノルムの大きさを抑える。
            grads, _ = tf.clip_by_global_norm(
                           tf.gradients( self._loss_op, trainable_vars ),   #
                           clip_norm = 5
                       )

            #------------------------------------------------
            # Optimizer, train step の設定
            #------------------------------------------------
            #self._optimizer = nnOptimizer._optimizer
            self._optimizer = tf.train.AdamOptimizer( 0.001 )

            #self._train_step = nnOptimizer.train_step( self._loss_op )

            # grad として勾配を取り出す
            self._train_step = self._optimizer.apply_gradients( zip(grads, trainable_vars) )

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
        def generate_minibatch( x, y, batch_size, n_steps ):
            n_batches = int( x.shape[1] / n_steps )

            # batch_size 間隔で for loop
            for i in range( n_batches ):
                # yield 文で逐次データを return（関数の処理を一旦停止し、値を返す）
                # メモリ効率向上のための処理
                yield x[:, i*n_steps:(i+1)*n_steps], y[:, i*n_steps:(i+1)*n_steps]

        #----------------------------
        # 学習開始処理
        #----------------------------
        # Variable の初期化オペレーター
        self._init_var_op = tf.global_variables_initializer()

        # Session の run（初期化オペレーター）
        self._session.run( self._init_var_op )

        # ミニバッチの繰り返し回数
        n_batches = int( X_train.shape[1] / self._n_steps )
        n_minibatch_iterations = n_batches * self._epochs
        print( "n_batches :", n_batches )
        print( "n_minibatch_iterations :", n_minibatch_iterations )

        # （学習済みモデルの）チェックポイントファイルの作成
        self.save_model()
        
        #-------------------
        # 学習処理
        #-------------------
        # for ループでエポック数分トレーニング
        for epoch in range( self._epochs ):
            # 各エポックの最初に、RNN Cell の状態を初期状態にリセット
            # このプロセスを繰り返すことにより、エポックを通じての現在の状態の更新を実現する。
            rnn_cell_state = self._session.run( self._rnn_states[0] )

            # ミニバッチサイズ単位で for ループ
            gen_minibatch = generate_minibatch( X_train, y_train , self._batch_size, self._n_steps )    # 関数アドレス

            for i ,(batch_x, batch_y) in enumerate( gen_minibatch, 1 ):
                minibatch_iteration = epoch*n_batches + i

                # 設定された最適化アルゴリズム Optimizer でトレーニング処理を run
                loss, _, rnn_cell_state = self._session.run(
                                              [ self._loss_op, self._train_step, self._rnn_states[-1] ],
                                              feed_dict = {
                                                  self._input_holder: batch_x,
                                                  self._t_holder: batch_y,
                                                  self._dropout_holder: 0.5,
                                                  self._rnn_states[0]: rnn_cell_state
                                              }
                                          )

                self._losses_train.append( loss )
                print( "Epoch: %d/%d, minibatch iteration: %d / loss = %0.5f" % ( (epoch+1), self._epochs, minibatch_iteration, loss ) )
                
                # モデルの保存処理を行う loop か否か
                # % : 割り算の余りが 0 で判断
                if ( ( (minibatch_iteration+1) % self._save_step ) == 0 ):
                    self.save_model()


        return self._y_out_op


    def sampling( self, output_length, text2int_dir, int2text_dir, start_seq = "the " ):
        """
        学習済みモデルで、次の文書の確率算出し、それに基づいた文字を生成する。（サンプリング）

        [Input]

        [Output]

        """
        def sampling_char_with_predprob( probs, n_vocab, n_top = 5 ):
            """
            予測確率に従って、実際の文字（単語）をサンプリング
            貪欲なサンプリングではない手法で、
            "for the for the for the ..." のような同じフレーズの繰り返しを防ぐ。

            """
            # 余計な次元削除
            p = np.squeeze( probs )

            # 0 番目の要素 "unknown" を除外
            p = p[1:]

            # 上位 n_top 個にソートするために、上位以外を 0 で埋める。
            idxs_zero = np.argsort(p)[:-n_top]
            p[ idxs_zero ] = 0.0

            # 全体の確率を正規化
            p = p / np.sum( p )

            # 予想確率 p に基づき、idx をランダムサンプリング
            char_idxs = np.random.choice( n_vocab - 1, 1, p = p )
            char_idx = char_idxs[0]

            return char_idx


        if( self._bSamplingMode == False ):
            print( "this mesod is invalid error : not sampling mode" )
            return

        # 学習済みモデルを読み込み
        self.load_model()
        
        #------------------------------------------------------------
        # 引数で指定された開始シーケンス start_seq を起点にモデルを実行
        #------------------------------------------------------------
        # 学習済みモデルで RNN Cell の状態を初期状態にリセット
        rnn_cell_state = self._session.run( self._rnn_states[0] )

        # 引数で指定された開始シーケンスを単語単位に分割
        start_seqs = start_seq.split()

        # 予想シーケンスを初期化（別オブジェクトとして初期化）
        pred_seq = [str for str in start_seqs]

        for str in start_seqs:
            str = str.lower()       # 大文字 → 小文字に変換
            x = np.zeros( (1,1) )   # shape = [1,1]
            x[0,0] = text2int_dir[str]

            probs, rnn_cell_state = self._session.run(
                                        [ self._y_out_op, self._rnn_states[-1] ],
                                        feed_dict = {
                                            self._input_holder: x,
                                            self._dropout_holder: 1.0,
                                            self._rnn_states[0]: rnn_cell_state
                                        }
                                    )

            print( "probs :", probs )

            char_id = sampling_char_with_predprob( probs, n_vocab = self._n_classes, n_top = 5 )
            #p = np.squeeze( probs )         # 余計な次元削除
            #char_id = np.argmax( p[1:] )    # 最大の確率を与えるインデックス。0 番目の "unknown" は除外

            pred_seq.append( int2text_dir[char_id] )

        #------------------------------------------------
        # 初回予想シーケンス pred_seq を起点にモデルを実行
        #------------------------------------------------
        for i in range( output_length ):
            x[0,0] = char_id

            probs, rnn_cell_state = self._session.run(
                                        [ self._y_out_op, self._rnn_states[-1] ],
                                        feed_dict = {
                                            self._input_holder: x,
                                            self._dropout_holder: 1.0,
                                            self._rnn_states[0]: rnn_cell_state
                                        }
                                    )

            char_id = sampling_char_with_predprob( probs, n_vocab = self._n_classes, n_top = 5 )
            #p = np.squeeze( probs )         # 余計な次元削除
            #char_id = np.argmax( p[1:] )    # 最大の確率を与えるインデックス。0 番目の "unknown" は除外

            pred_seq.append( int2text_dir[char_id] )

        #------------------------------------------------
        # 予想文を出力
        #------------------------------------------------
        print( "sampling text :\n", " ".join( pred_seq ) )

        return