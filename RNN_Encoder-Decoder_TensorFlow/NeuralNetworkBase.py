# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境 (TensorFlow インストール済み)

"""
    更新情報
    [17/10/14] : 新規作成
    [17/11/19] : scikit-learn ライブラリの推定器 estimator の基本クラス `BaseEstimator`, `ClassifierMixin` を継承しているように変更
               : 各種抽象メソッド追加
    [17/12/01] :  TensorBoard に計算グラフを表示するためのファイルを書き込むための関数 write_tensorboard_graph(...) 追加
    [xx/xx/xx] :

"""

from abc import ABCMeta, abstractmethod             # 抽象クラスを作成するための ABC クラス

import numpy

# scikit-learn ライブラリ関連
from sklearn.base import BaseEstimator              # 推定器 Estimator の上位クラス. get_params(), set_params() 関数が定義されている.
from sklearn.base import ClassifierMixin            # 推定器 Estimator の上位クラス. score() 関数が定義されている.

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops

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


class NeuralNetworkBase( BaseEstimator, ClassifierMixin ):
    """
    ニューラルネットワークの基底クラス（自作クラス）
    scikit-learn ライブラリの推定器 estimator の基本クラス BaseEstimator, ClassifierMixin を継承している.
    TensorFlow ライブラリを使用
    ニューラルネットワークの基本的なフレームワークを想定した仮想メソッドからなる抽象クラス。
    実際のニューラルネットワークを表すクラスの実装は、このクラスを継承し、オーバーライドするを想定している。
    
    ----------------------------------------------------------------------------------------------------
    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
        _session : tf.Session()
            自身の Session
        _init_var_op : tf.global_variables_initializer()
            全 Variable の初期化オペレーター

        _loss_op : Operator
            損失関数を表すオペレーター
        _optimizer : Optimizer
            モデルの最適化アルゴリズム
        _train_step : 
            トレーニングステップ
        _y_out_op : Operator
            モデルの出力のオペレーター

    [protedted] protedted な使用法を想定 

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """

    def __init__( self, session = tf.Session() ):
        """
        コンストラクタ（厳密にはイニシャライザ）
        """
        # メンバ変数の初期化
        self._session = session
        self._init_var_op = None

        self._loss_op = None
        self._optimizer = None
        self._train_step = None
        self._y_out_op = None

        return


    def print( self, str ):
        print( "NeuralNetworkBase" )
        print( self )
        print( str )

        print( "_session : \n", self._session )
        print( "_init_var_op : \n", self._init_var_op )
        print( "_loss_op : \n", self._loss_op )
        print( "_y_out_op : \n", self._y_out_op )

        return


    @abstractmethod
    def model( self ):
        """
        モデルの定義を行い、
        最終的なモデルの出力のオペレーター self._y_out_op を設定する。
        （抽象メソッド）

        [Output]
            self._y_out_op : Operator
                モデルの出力のオペレーター
        """
        return self._y_out_op

    
    @abstractmethod
    def loss( self, nnLoss ):
        """
        損失関数（誤差関数、コスト関数）の定義を行う。
        （抽象メソッド）

        [Input]
            nnLoss : NNLoss クラスのオブジェクト
            
        [Output]
            self._loss_op : Operator
                損失関数を表すオペレーター
        """
        return self._loss_op

    
    @abstractmethod
    def optimizer( self, nnOptimizer ):
        """
        モデルの最適化アルゴリズムの設定を行う。（抽象メソッド）

        [Input]
            nnOptimizer : NNOptimizer のクラスのオブジェクト

        [Output]
            optimizer の train_step

        """
        return self._train_step

    
    @abstractmethod
    def fit( self, X_train, y_train ):
        """
        指定されたトレーニングデータで、モデルの fitting 処理を行う。（抽象メソッド）

        [Input]
            X_train : numpy.ndarray ( shape = [n_samples, n_features] )
                トレーニングデータ（特徴行列）
            
            y_train : numpy.ndarray ( shape = [n_samples] )
                レーニングデータ用のクラスラベル（教師データ）のリスト

        [Output]
            self : 自身のオブジェクト
        """
        return self


    @abstractmethod
    def predict( self, X_features ):
        """
        fitting 処理したモデルで、推定を行い、予想値を返す。（抽象メソッド）

        [Input]
            X_features : numpy.ndarry ( shape = [n_samples, n_features] )
                予想したい特徴行列

        [Output]
            results : numpy.ndaary ( shape = [n_samples] )
                予想結果（分類モデルの場合は、クラスラベル）
        """
        return


    @abstractmethod
    def predict_proba( self, X_test ):
        """
        fitting 処理したモデルで、推定を行い、クラスの所属確率の予想値を返す。（抽象メソッド）
        proba : probability

        [Input]
            X_test : numpy.ndarry ( shape = [n_samples, n_features] )
                予想したい特徴行列
        """
        return

    @abstractmethod
    def accuracy( self, X_test, y_test ):
        """
        指定したデータでの正解率 [accuracy] を計算する。
        """
        return

    @abstractmethod
    def accuracy_labels( self, X_test, y_test ):
        """
        指定したデータでのラベル毎の正解率 [acuuracy] を算出する。
        """
        return


    def write_tensorboard_graph( self, dir = "./TensorBoard" ):
        """
        TensorBoard に計算グラフを表示するためのファイルを書き込む。
        [Input]
            dir : str
                TensorBoard 用のファイルを作成するディレクトリのパス
        """
        # TensorBoard 用のファイル（フォルダ）を作成
        merged = tf.summary.merge_all() # Add summaries to tensorboard
        summary_writer = tf.summary.FileWriter( dir, graph = self._session.graph )    # tensorboard --logdir=${PWD}
        
        return