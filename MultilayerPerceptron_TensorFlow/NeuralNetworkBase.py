# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境 (TensorFlow インストール済み)

"""
    更新情報
    [17/10/14] : 新規作成
    [17/xx/xx] : 
               : 
"""

from abc import ABCMeta, abstractmethod     # 抽象クラスを作成するための ABC クラス

import numpy

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops


class NeuralNetworkBase( metaclass=ABCMeta ):
    """
    ニューラルネットワークの基底クラス（自作クラス）
    TensorFlow ライブラリを使用
    
    ニューラルネットワークの基本的なフレームワークを想定した仮想メソッドからなる抽象クラス。
    実際のニューラルネットワークを表すクラスの実装は、このクラスを継承することで行うことを想定している。
    
    ----------------------------------------------------------------------------------------------------

    [protedted] protedted な使用法を想定 
        _n_inputLayer : int
            入力層のノード数
        _n_outputLayer : int
            出力層のノード数

    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """

    def __init__( self ):
        """
        コンストラクタ（厳密にはイニシャライザ）
        """
        # メンバ変数の初期化
        self._n_inputLayer = 1
        self._n_outputLayer = 1

        return

    
    def print( self, str ):
        print( "NeuralNetworkBase" )
        print( self )
        print( str )

        return


    @classmethod
    @abstractmethod
    def models( self ):
        """
        モデルの定義を行う。（抽象メソッド）
        """
        return self

    
    @classmethod
    @abstractmethod
    def loss( self ):
        """
        損失関数（誤差関数、コスト関数）の定義を行う。（抽象メソッド）
        """
        return self

    
    @classmethod
    @abstractmethod
    def optimizer( self ):
        """
        モデルの最適化アルゴリズムの設定を行う。（抽象メソッド）
        """
        return self

    
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
