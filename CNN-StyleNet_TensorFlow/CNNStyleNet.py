# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境 (TensorFlow インストール済み)

"""
    更新情報
    [17/11/21] : 新規作成
    [17/xx/xx] : 
               : 
"""
# I/O 関連
import os

import scipy.misc
import scipy.io

import numpy as np

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops

# 自作クラス
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


class ConvolutionalNN( NeuralNetworkBase ):
    """
    CNN-StyleNet / NeuralStyle（ニューラルスタイル）を表すクラスを表すクラス.
    TensorFlow での StyleNet の処理をクラスでラッピング。
    ------------------------------------------------------------------------------------------------
    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
        _weights : list <Variable>
            モデルの各層の重みの Variable からなる list
        _biases : list <Variable>
            モデルの各層のバイアス項の  Variable からなる list

        _epochs : int
            エポック数（トレーニング回数）
        _batch_size : int
            ミニバッチ学習でのバッチサイズ
        _eval_step : int
            学習処理時に評価指数の算出処理を行う step 間隔

        _image_height : int
            入力画像データの高さ（ピクセル単位）
        _image_width : int
            入力画像データの幅（ピクセル単位）
        _n_channels : int
            入力画像データのチャンネル数
            1 : グレースケール画像

        _n_ConvLayer_featuresMap : list <int>
            畳み込み層で変換される特徴マップの枚数
            conv1 : _n_ConvLayer_featuresMap[0]
            conv2 : _n_ConvLayer_featuresMap[1]
            ...
        _n_ConvLayer_kernels : list <int>
            CNN の畳み込み処理時のカーネルのサイズ
            conv1 : _n_ConvLayer_kernels[0] * _n_ConvLayer_kernels[0]
            conv2 : _n_ConvLayer_kernels[1] * _n_ConvLayer_kernels[1]
            ...
        _n_strides : int
            CNN の畳み込み処理（特徴マップ生成）でストライドさせる pixel 数

        _n_pool_wndsize : int
            プーリング処理用のウィンドウサイズ
        _n_pool_strides : int
            プーリング処理時のストライドさせる pixel 数

        _n_fullyLayers : int
            全結合層の入力側のノード数
        _n_labels : int
            出力ラベル数（全結合層の出力側のノード数）

        _losses_train : list <float32>
            トレーニングデータでの損失関数の値の list

        _X_holder : placeholder
            入力層にデータを供給するための placeholder
        _t_holder : placeholder
            出力層に教師データを供給するための placeholder
        _keep_prob_holder : placeholder
            ドロップアウトしない確率 (1-p) にデータを供給するための placeholder

    [protedted] protedted な使用法を想定 


    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）


    """