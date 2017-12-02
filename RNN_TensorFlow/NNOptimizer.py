# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境 (TensorFlow インストール済み)

"""
    更新情報
    [17/11/18] : 新規作成
    [17/11/20] : 最急降下法で学習率が幾何学的に減衰していく最適化アルゴリズム GradentDecentDecay 追加
    [17/12/03] : RMSProp アルゴリズムを追加
    [xx/xx/xx]
"""

import numpy

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops


class NNOptimzer( object ):
    """
    ニューラルネットワークの最適化アルゴリズム Optimizer を表すクラス
    実際の最適化アルゴリズムを表すクラスの実装は、このクラスを継承し、オーバーライドするを想定している。
    ------------------------------------------------------------------------------------------------
    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
        _optimizer : Optimizer
            最適化アルゴリズム
        _train_step : 
            トレーニングステップ
        _node_name : str
            この Optimizer ノードの名前

        _learning_rate : float
            学習率 (0.0~1.0)
            
    [protedted] protedted な使用法を想定 
        
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( self, learning_rate = 0.001, node_name = "Optimizer" ):
        self._optimizer = None
        self._train_step = None
        self._node_name = node_name
        self._learning_rate = learning_rate

        return

    def print( str ):
        print( "NNOptimizer" )
        print( self )
        print( str )
        print( "_optimizer :", self._optimizer )
        print( "_train_step :", self._train_step )
        print( "_node_name :", self._node_name )
        print( "_learning_rate :", self._learning_rate )

        return

    def optimizer( self ):
        """
        最適化アルゴリズム Optimizer の設定を行う。

        [Output]
            optimizer
        """
        return self._optimizer


    def train_step( self, loss_op ):
        """
        トレーニングステップの設定を行う。

        [Input]
            loss_op : Operation
                損失関数のオペレーター

        [Output]
            optimizer のトレーニングステップ
        """
        return self._train_step


class GradientDecent( NNOptimzer ):
    """
    最急降下法を表すクラス
    NNOptimizer クラスの子クラスとして定義
    """
    def __init__( self, learning_rate = 0.001, node_name = "GradientDecent_Optimizer" ):
        self._learning_rate = learning_rate
        self._node_name = node_name
        self._optimizer = self.optimizer()
        self._train_step = None

        return
    
    def optimizer( self ):
        self._optimizer = tf.train.GradientDescentOptimizer( learning_rate = self._learning_rate )
        return self._optimizer

    def train_step( self, loss_op ):
        self._train_step = self._optimizer.minimize( loss_op )
        return self._train_step


class GradientDecentDecay( NNOptimzer ):
    """
    最急降下法（学習率が減衰）を表すクラス
    NNOptimizer クラスの子クラスとして定義

    減衰
    learning_rate * ( 1 - learning_rate)^(n_generation/n_gen_to_wait)
    [public]
        _n_generation : int
            学習率を幾何学的に減衰させるためのパラメータ
        _n_gen_to_wait : int
            学習率を幾何学的に減衰させるためのパラメータ
            学習率を減衰されるステップ間隔
        _lr_recay : float
            学習率を幾何学的に減衰させるためのパラメータ

        _recay_learning_rate : 
            tf.train.exponential_decay(...) の戻り値
    """
    def __init__( 
        self, learning_rate = 0.001, 
        n_generation = 500, n_gen_to_wait = 5, 
        lr_recay = 0.1, 
        node_name = "GradientDecentDecay_Optimizer" 
    ):
        self._learning_rate = learning_rate
        self._n_generation = n_generation
        self._n_gen_to_wait = n_gen_to_wait
        self._lr_recay = lr_recay

        self._recay_learning_rate = tf.train.exponential_decay( 
                                        learning_rate = learning_rate, 
                                        global_step = n_generation, 
                                        decay_steps = n_gen_to_wait, 
                                        decay_rate = lr_recay, 
                                        staircase = True 
                                    )

        self._node_name = node_name
        self._optimizer = self.optimizer()
        self._train_step = None

        return
    
    def optimizer( self ):
        self._optimizer = tf.train.GradientDescentOptimizer( learning_rate = self._recay_learning_rate )
        return self._optimizer

    def train_step( self, loss_op ):
        self._train_step = self._optimizer.minimize( loss_op )
        return self._train_step


class Momentum( NNOptimzer ):
    """
    モメンタム アルゴリズムを表すクラス
    NNOptimizer クラスの子クラスとして定義
    """
    def __init__( self, learning_rate = 0.001, momentum = 0.9, node_name = "Momentum_Optimizer" ):
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._node_name = node_name
        self._optimizer = self.optimizer()
        self._train_step = None

        return
    
    def optimizer( self ):
        self._optimizer = tf.train.MomentumOptimizer( 
                              learning_rate = self._learning_rate, 
                              momentum = self._momentum,
                              use_nesterov = False
                          )

        return self._optimizer

    def train_step( self, loss_op ):
        self._train_step = self._optimizer.minimize( loss_op )
        return self._train_step


class NesterovMomentum( NNOptimzer ):
    """
    Nesterov モメンタム アルゴリズムを表すクラス
    NNOptimizer クラスの子クラスとして定義
    """
    def __init__( self, learning_rate = 0.001, momentum = 0.9, node_name = "NesterovMomentum_Optimizer" ):
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._node_name = node_name
        self._optimizer = self.optimizer()
        self._train_step = None

        return
    
    def optimizer( self ):
        self._optimizer = tf.train.MomentumOptimizer( 
                              learning_rate = self._learning_rate, 
                              momentum = self._momentum,
                              use_nesterov = True
                          )

        return self._optimizer

    def train_step( self, loss_op ):
        self._train_step = self._optimizer.minimize( loss_op )
        return self._train_step


class Adagrad( NNOptimzer ):
    """
    Adagrad アルゴリズムを表すクラス
    NNOptimizer クラスの子クラスとして定義
    """
    def __init__( self, learning_rate = 0.001, node_name = "Adagrad_Optimizer" ):
        self._learning_rate = learning_rate
        self._node_name = node_name
        self._optimizer = self.optimizer()
        self._train_step = None

        return
    
    def optimizer( self ):
        self._optimizer = tf.train.AdagradOptimizer( learning_rate = self._learning_rate )
        return self._optimizer

    def train_step( self, loss_op ):
        self._train_step = self._optimizer.minimize( loss_op )
        return self._train_step


class Adadelta( NNOptimzer ):
    """
    Adadelta アルゴリズムを表すクラス
    NNOptimizer クラスの子クラスとして定義
    """
    def __init__( self, learning_rate = 0.001, rho = 0.95, node_name = "Adadelta_Optimizer" ):
        self._learning_rate = learning_rate
        self._rho = rho
        self._node_name = node_name
        self._optimizer = self.optimizer()
        self._train_step = None

        return
    
    def optimizer( self ):
        self._optimizer = tf.train.AdadeltaOptimizer( learning_rate = self._learning_rate, rho = self._rho )
        return self._optimizer

    def train_step( self, loss_op ):
        self._train_step = self._optimizer.minimize( loss_op )
        return self._train_step


class Adam( NNOptimzer ):
    """
    Adam アルゴリズムを表すクラス
    NNOptimizer クラスの子クラスとして定義
    """
    def __init__( self, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.99, node_name = "Adam_Optimizer" ):
        self._learning_rate = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._node_name = node_name
        self._optimizer = self.optimizer()
        self._train_step = None

        return
    
    def optimizer( self ):
        self._optimizer = tf.train.AdamOptimizer( 
                              learning_rate = self._learning_rate, 
                              beta1 = self._beta1,
                              beta2 = self._beta2
                          )

        return self._optimizer

    def train_step( self, loss_op ):
        self._train_step = self._optimizer.minimize( loss_op )
        return self._train_step


class RMSProp( NNOptimzer ):
    """
    RMSProp アルゴリズムを表すクラス
    NNOptimizer クラスの子クラスとして定義
    """
    def __init__( self, learning_rate = 0.001, decay = 0.9, momentum = 0.0, node_name = "RMSProp_Optimizer" ):
        self._learning_rate = learning_rate
        self._decay = decay
        self._momentum = momentum
        self._node_name = node_name
        self._optimizer = self.optimizer()
        self._train_step = None

        return
    
    def optimizer( self ):
        self._optimizer = tf.train.RMSPropOptimizer( 
                              learning_rate = self._learning_rate,
                              decay = self._decay,
                              momentum = self._momentum
                          )

        return self._optimizer

    def train_step( self, loss_op ):
        self._train_step = self._optimizer.minimize( loss_op )
        return self._train_step