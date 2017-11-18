# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境 (TensorFlow インストール済み)

"""
    更新情報
    [17/11/18] : 新規作成
    [17/xx/xx] : 
               : 
"""

import numpy

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops

class NNLoss( object ):
    """
    ニューラルネットワークの損失関数（評価関数、誤差関数）を表すクラス
    実際の損失関数を表すクラスの実装は、このクラスを継承し、オーバーライドするを想定している。
    ------------------------------------------------------------------------------------------------
    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
        _loss_op : 損失関数の Operater
        _node_name : str
            この Operator ノードの名前
            
    [protedted] protedted な使用法を想定 
        
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( self, node_name = "Loss_op" ):
        self._loss_op = loss( t_holder, y_out_op, node_name )
        self._node_name = node_name

        return

    def print( str ):
        print( "NNLoss" )
        print( self )
        print( str )
        print( "_loss_op :", self._loss_op )
        print( "_node_name :", self._node_name )
                
        return


    def loss( self, t_holder, y_out_op ):
        """
        損失関数のオペレーターを返す。

        [Input]
            _t_holder : Placeholder
                教師データの Placeholder
            _y_out_op : Operator
                出力の Operator

        [Output]
            損失関数のオペレーター
        """
        self._loss_op = None

        return self._loss_op


class L1Norm( NNLoss ):
    """
    L1 ノルムの損失関数
    NNLoss クラスの子クラスとして定義
    """
    def __init__( self, node_name = "Loss_L1Norm_op" ):
        self._loss_op = None
        self._node_name = node_name
        
        return

    def loss( self, t_holder, y_out_op, node_name = "Loss_L1Norm_op" ):
        # tf.reduce_mean(...) : 
        self._loss_op = tf.reduce_mean(
                            tf.abs( t_holder - y_out_op )
                        )

        return self._loss_op


class L2Norm( NNLoss ):
    """
    L2 ノルムの損失関数
    NNLoss クラスの子クラスとして定義
    """
    def __init__( self, node_name = "Loss_L2Norm_op" ):
        self._loss_op = None
        self._node_name = node_name
        
        return

    def loss( self, t_holder, y_out_op, node_name = "Loss_L2Norm_op" ):
        self._loss_op = tf.reduce_mean(
                            tf.square( t_holder - y_out_op )
                        )
        
        return self._loss_op


class BinaryCrossEntropy( NNLoss ):
    """
    ２値のクロス・エントロピーの損失関数
    NNLoss クラスの子クラスとして定義
    """
    def __init__( self, node_name = "Loss_BinaryCrossEntropy_op" ):
        self._loss_op = None
        self._node_name = node_name
        
        return

    def loss( self, t_holder, y_out_op ):
        self._loss_op = -tf.reduce_sum( 
                            t_holder * tf.log( y_out_op ) + 
                            ( 1 - t_holder ) * tf.log( 1 - y_out_op )
                        )
        
        return self._loss_op


class CrossEntropy( NNLoss ):
    """
    クロス・エントロピーの損失関数
    NNLoss クラスの子クラスとして定義
    """
    def __init__( self, node_name = "Loss_CrossEntropy_op" ):
        self._loss_op = None
        self._node_name = node_name

        return

    def loss( self, t_holder, y_out_op, node_name = "Loss_CrossEntropy_op" ):
        # softmax で正規化済みの場合
        # tf.clip_by_value(...) : 下限値、上限値を設定
        self._loss_op = tf.reduce_mean(                     # ミニバッチ度に平均値を計算
                            -tf.reduce_sum( 
                                t_holder * tf.log( tf.clip_by_value(y_out_op, 1e-10, 1.0) ), 
                                reduction_indices = [1]     # sum をとる行列の方向 ( 1:row 方向 )
                            )
                        )
        
        return self._loss_op


class SoftmaxCrossEntropy( NNLoss ):
    """
    ソフトマックス・クロス・エントロピーの損失関数
    NNLoss クラスの子クラスとして定義
    """
    def __init__( self, node_name = "Loss_SoftmaxCrossEntropy_op" ):
        self._loss_op = None
        self._node_name = node_name

        return

    def loss( self, t_holder, y_out_op ):
        # softmax で正規化済みでない場合
        self._loss_op = tf.reduce_mean(
                            tf.nn.softmax_cross_entropy_with_logits(
                                labels = t_holder,
                                logits = y_out_op
                            )
                        )
        
        return self._loss_op


class SparseSoftmaxCrossEntropy( NNLoss ):
    """
    疎なソフトマックス・クロス・エントロピーの損失関数
    NNLoss クラスの子クラスとして定義
    """
    def __init__( self, node_name = "Loss_SparseSoftmaxCrossEntropy_op" ):
        self._loss_op = None
        self._node_name = node_name
        
        return

    def loss( self, t_holder, y_out_op ):
        self._loss_op = tf.reduce_mean(
                            tf.nn.sparse_softmax_cross_entropy_with_logits(
                                logits = y_out_op,
                                labels = t_holder
                            )
                        )
        
        return self._loss_op


