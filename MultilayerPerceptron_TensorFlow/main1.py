# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境 (TensorFlow インストール済み)
#     <Anaconda Prompt>
#     conda create -n tensorflow python=3.5
#     activate tensorflow
#     pip install --ignore-installed --upgrade tensorflow
#     pip install --ignore-installed --upgrade tensorflow-gpu

import numpy
import pandas
import matplotlib.pyplot as plt

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops


# 自作クラス
from MLPlot import MLPlot
from MLPreProcess import MLPreProcess
from MultilayerPerceptron import MultilayerPerceptron


def main():
    """
    多層パーセプトロンを用いた、アヤメデータの識別
    """
    print("Enter main()")

    #======================================================================
    # データセットを読み込み or 生成
    # Import or generate data.
    #======================================================================
    X_features, y_labels = MLPreProcess.generateCirclesDataSet( input_n_samples = 100 )
    
    #======================================================================
    # データを変換、正規化
    # Transform and normalize data.
    # ex) data = tf.nn.batch_norm_with_global_normalization(...)
    #======================================================================
    #y_labels.reshape( [100, 1] )
    
    #======================================================================
    # データセットをトレーニングデータ、テストデータ、検証データセットに分割
    #======================================================================
    X_train, X_test, y_train, y_test \
    = MLPreProcess.dataTrainTestSplit( X_input = X_features, y_input = y_labels, ratio_test = 0.3, input_random_state = 1 )

    #======================================================================
    # アルゴリズム（モデル）のパラメータを設定
    # Set algorithm parameters.
    # ex) learning_rate = 0.01  iterations = 1000
    #======================================================================
    # 多層パーセプトロンクラスのオブジェクト生成
    mlp1 = MultilayerPerceptron( n_inputLayer = len(X_features[0]), n_hiddenLayers = [3,3], n_outputLayer = 1 )
    mlp1.print( "mlp1" )

    #======================================================================
    # 変数とプレースホルダを設定
    # Initialize variables and placeholders.
    # TensorFlow は, 損失関数を最小化するための最適化において,
    # 変数と重みベクトルを変更 or 調整する。
    # この変更や調整を実現するためには, 
    # "プレースホルダ [placeholder]" を通じてデータを供給（フィード）する必要がある。
    # そして, これらの変数とプレースホルダと型について初期化する必要がある。
    # ex) a_var = tf.constant(42)
    #     x_input_holder = tf.placeholder(tf.float32, [None, input_size])
    #     y_input_holder = tf.placeholder(tf.fload32, [None, num_classes])
    #======================================================================


    #======================================================================
    # モデルの構造を定義する。
    # Define the model structure.
    # ex) add_op = tf.add(tf.mul(x_input_holder, weight_matrix), b_matrix)
    #======================================================================
    mlp1.models()
    mlp1.print( "after models()" )

    #======================================================================
    # 損失関数を設定する。
    # Declare the loss functions.
    #======================================================================
    mlp1.loss()
    mlp1.print( "after loss()" )

    #======================================================================
    # モデルの初期化と学習（トレーニング）
    # ここまでの準備で, 実際に, 計算グラフ（有向グラフ）のオブジェクトを作成し,
    # プレースホルダを通じて, データを計算グラフ（有向グラフ）に供給する。
    # Initialize and train the model.
    #
    # ex) 計算グラフを初期化する方法の１つの例
    #     with tf.Session( graph = graph ) as session:
    #         ...
    #         session.run(...)
    #         ...
    #     session = tf.Session( graph = graph )  
    #     session.run(…)
    #======================================================================
    mlp1.fit( X_train, y_train )
    mlp1.print( "after fit()" )

    #======================================================================
    # モデルの評価
    # (Optional) Evaluate the model.
    #======================================================================
    # plot
    MLPlot.drawDiscriminantRegions( X_features, y_labels, classifier = mlp1 )

    #======================================================================
    # ハイパーパラメータのチューニング (Optional)
    #======================================================================


    #======================================================================
    # デプロイと新しい成果指標の予想 (Optional)
    #======================================================================


    print("Finish main()")
    return
    

if __name__ == '__main__':
     main()
