# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境 (TensorFlow インストール済み)

"""
    更新情報
    [17/10/14] : 新規作成
    [17/xx/xx] : 
               : 
"""

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops

import numpy

# 親クラス（自作クラス）
from NeuralNetworkBase import NeuralNetworkBase


class MultilayerPerceptron( NeuralNetworkBase ):
    """
    多層パーセプトロンを表すクラス（自作クラス）
    ニューラルネットワークの基底クラス NeuralNetworkBase （自作の基底クラス）を継承している。
    
    ----------------------------------------------------------------------------------------------------

    [protedted] protedted な使用法を想定 
        _n_inputLayer : int
            入力層のノード数
        _n_hideenLayers : shape = [h1,h2,h3,...] 
            h1 : 1 つ目の隠れ層のユニット数、h2 : 2 つ目の隠れ層のユニット数、...
        _n_outputLayer : int
            出力層のノード数

    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
        _X_holder : placeholder
            入力層にデータを供給するための placeholder

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """

    def __init__( self, n_inputLayer = 1, n_hideenLayers = [1,1,1], n_outputLayer = 1 ):
        """
        コンストラクタ（厳密にはイニシャライザ）
        """
        # メンバ変数の初期化
        self._n_inputLayer = n_inputLayer
        self._n_hideenLayers = n_hideenLayers
        self._n_outputLayer = n_outputLayer

        # shape は、入力層の次元 self._n_inputLayer に対応させる。
        self._X_holder = tf.placeholder( tf.float32, shape = [None, self._n_inputLayer] )
        
        return
    
    def print( self, str ):
        print( "----------------------------------" )
        print( "MultilayerPerceptron" )
        print( self )
        print( str )
        print( "_n_inputLayer : ", self._n_inputLayer )
        print( "_n_hideenLayers : ", self._n_hideenLayers )
        print( "_n_outputLayer : ", self._n_outputLayer )
        print( "----------------------------------" )
        return

    @classmethod
    def models( self ):
        """
        モデルの定義を行う。
        """

        return self

    @classmethod
    def loss( self ):
        """
        損失関数の定義を行う。
        """

        return self

    @classmethod
    def optimizer( self ):
        """
        モデルの最適化アルゴリズムの設定を行う。
        """

        return self

    def fit( self, X_train, y_train ):
        """
        指定されたトレーニングデータで、モデルの fitting 処理を行う。

        [Input]
            X_train : numpy.ndarray ( shape = [n_samples, n_features] )
                トレーニングデータ（特徴行列）
            
            y_train : numpy.ndarray ( shape = [n_samples] )
                レーニングデータ用のクラスラベル（教師データ）のリスト

        [Output]
            self : 自身のオブジェクト
        """
        return self


    def predict( self, X_features ):
        """
        fitting 処理したモデルで、推定を行い、予想値を返す。（抽象メソッド）

        [Input]
            X_features : numpy.ndarry ( shape = [n_samples, n_features] )
                予想したい特徴行列

        [Output]
            results : numpy.ndarry ( shape = [n_samples] )
                予想結果（分類モデルの場合は、クラスラベル）
        """

        return

