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
from MultilayerPerceptron import MultilayerPerceptron   #
from NNActivation import NNActivation                   # ニューラルネットワークの活性化関数を表すクラス


def main():
    """
    多層パーセプトロンを用いた、MIST データでの手書き数字文字のパターン認識処理
    """
    print("Enter main()")

    #======================================================================
    # データセットを読み込み or 生成
    # Import or generate data.
    #======================================================================
    # MSIT データが格納されているフォルダへのパス
    mist_path = "D:\Data\MachineLearning_DataSet\MIST"

    X_train, y_train = MLPreProcess.load_mist( mist_path, "train" )
    X_test, y_test = MLPreProcess.load_mist( mist_path, "t10k" )

    print( "X_train.shape[0]", X_train.shape[0] )
    print( "X_train.shape[1]", X_train.shape[1] )
    print( "X_test.shape[0]", X_test.shape[0] )
    print( "X_test.shape[1]", X_test.shape[1] )

    #---------------------------------------------------------------------
    # MIST 画像を plot
    #---------------------------------------------------------------------
    # 先頭の 0~9 のラベルの画像データを plot
    """
    # plt.subplots(...) から,
    # Figure クラスのオブジェクト、Axis クラスのオブジェクト作成
    figure, axis = plt.subplots( 
                       nrows = 2, ncols = 5,
                       sharex = True, sharey = True     # x,y 軸をシャアする
                   )

    # 2 × 5 配列を１次元に変換
    axis = axis.flatten()

    # 数字の 0~9 の plot 用の for ループ
    for i in range(10):
        image = X_train[y_train == i][0]    #
        image = image.reshape(28,28)        # １次元配列を shape = [28 ,28] に reshape
        axis[i].imshow(
            image,
            cmap = "Greys",
            interpolation = "nearest"   # 補間方法
        )

    axis[0].set_xticks( [] )
    axis[0].set_yticks( [] )
    plt.tight_layout()
    MLPlot.saveFigure( fileName = "MultilayerPerceptron_3-1.png" )
    plt.show()
    """

    # 特定のラベルの 25 枚の画像データを plot
    figure, axis = plt.subplots( nrows = 5, ncols = 5, sharex = True, sharey = True )
    axis = axis.flatten()
    for i in range(25):
        image = X_train[y_train == 7][i].reshape(28,28)    
        axis[i].imshow( image, cmap = "Greys", interpolation = "nearest" )
    
    axis[0].set_xticks( [] )
    axis[0].set_yticks( [] )
    plt.tight_layout()
    MLPlot.saveFigure( fileName = "MultilayerPerceptron_3-2.png" )
    plt.show()

    #======================================================================
    # データセットをトレーニングデータ、テストデータ、検証データセットに分割
    #======================================================================
    
    #======================================================================
    # データを変換、正規化
    # Transform and normalize data.
    # ex) data = tf.nn.batch_norm_with_global_normalization(...)
    #======================================================================   
    

    #======================================================================
    # アルゴリズム（モデル）のパラメータを設定
    # Set algorithm parameters.
    # ex) learning_rate = 0.01  iterations = 1000
    #======================================================================
    # 多層パーセプトロンクラスのオブジェクト生成
    mlp1 = MultilayerPerceptron(
               session = tf.Session(),
               n_inputLayer = 3, 
               n_hiddenLayers = [5,5],
               n_outputLayer = 3,
               activate_hiddenLayer = NNActivation( "relu" ),
               activate_outputLayer = NNActivation( "softmax" ),
               learning_rate = 0.05,
               epochs = 500,
               batch_size = 50
           )
    
    
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
    

    #======================================================================
    # 損失関数を設定する。
    # Declare the loss functions.
    #======================================================================
    
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
    
    #======================================================================
    # モデルの評価
    # (Optional) Evaluate the model.
    #======================================================================
    
    
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
