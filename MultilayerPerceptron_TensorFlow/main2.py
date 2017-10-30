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
from NNActivation import NNActivation     # ニューラルネットワークの活性化関数を表すクラス


def main():
    """
    多層パーセプトロンを用いた、アヤメデータの識別（３クラスの分類問題）
    """
    print("Enter main()")

    #======================================================================
    # データセットを読み込み or 生成
    # Import or generate data.
    #======================================================================
    X_features, y_labels = MLPreProcess.load_iris()
    
    # 3,4 列目の特徴量を抽出し、X_features に保管
    X_features = X_features[:, [2,3]]

    print( "X_features :\n", X_features )

    #======================================================================
    # データセットをトレーニングデータ、テストデータ、検証データセットに分割
    #======================================================================
    X_train, X_test, y_train, y_test \
    = MLPreProcess.dataTrainTestSplit( X_input = X_features, y_input = y_labels, ratio_test = 0.2, input_random_state = 1 )

    #======================================================================
    # データを変換、正規化
    # Transform and normalize data.
    # ex) data = tf.nn.batch_norm_with_global_normalization(...)
    #======================================================================   
    # 標準化
    X_train_std, X_test_std = MLPreProcess.standardizeTrainTest( X_train, X_test )
    print( "X_train_std :\n", X_train_std )
    print( "y_train :\n", y_train )

    # 分割したデータを行方向に結合（後で plot データ等で使用する）
    X_combined_std = numpy.vstack( (X_train_std, X_test_std) )  # list:(X_train_std, X_test_std) で指定
    y_combined     = numpy.hstack( (y_train, y_test) )

    print( "X_combined_std.shape :\n", X_combined_std.shape )
    print( "X_combined_std :\n", X_combined_std )

    # One-hot encode
    session = tf.Session()
    encode_holder = tf.placeholder(tf.int64, [None])
    y_oneHot_enoded_op = tf.one_hot( encode_holder, depth=3, dtype=tf.float32 ) # depth が 出力層のノード数に対応
    session.run( tf.global_variables_initializer() )
    y_train_encoded = session.run( y_oneHot_enoded_op, feed_dict = { encode_holder: y_train } )
    y_test_encoded = session.run( y_oneHot_enoded_op, feed_dict = { encode_holder: y_test } )
    
    # [depth=3, n_sample=150] → [n_sample, depth]
    #y_train_encoded = numpy.transpose( y_train_encoded )
    #y_test_encoded = numpy.transpose( y_test )

    print( "y_train_encoded :\n" , y_train_encoded )
    print( "y_test_encoded :\n" , y_test_encoded )
    print( "y_train_encoded.shape :\n" , y_train_encoded.shape )
    print( "y_test_encoded.shape :\n" , y_test_encoded.shape )

    #======================================================================
    # アルゴリズム（モデル）のパラメータを設定
    # Set algorithm parameters.
    # ex) learning_rate = 0.01  iterations = 1000
    #======================================================================
    # 多層パーセプトロンクラスのオブジェクト生成
    mlp1 = MultilayerPerceptron(
               session = tf.Session(),
               n_inputLayer = len(X_features[0]), 
               n_hiddenLayers = [5],
               n_outputLayer = 3,
               activate_hiddenLayer = NNActivation( "sigmoid" ),
               activate_outputLayer = NNActivation( "softmax" ),
               learning_rate = 0.05,
               epochs = 500,
               batch_size = 50
           )

    mlp2 = MultilayerPerceptron(
               session = tf.Session(),
               n_inputLayer = len(X_features[0]), 
               n_hiddenLayers = [5,5],
               n_outputLayer = 3,
               activate_hiddenLayer = NNActivation( "relu" ),       # Relu
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
    mlp1.model()
    mlp2.model()

    #======================================================================
    # 損失関数を設定する。
    # Declare the loss functions.
    #======================================================================
    mlp1.loss( type = "cross-entropy" )
    mlp2.loss( type = "cross-entropy" )

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
    # モデルの最適化アルゴリズムを設定
    mlp1.optimizer( type = "gradient-descent" )
    mlp2.optimizer( type = "gradient-descent" )

    # トレーニングデータで fitting 処理
    mlp1.fit( X_train_std, y_train_encoded )
    mlp2.fit( X_train_std, y_train_encoded )

    mlp1.print( "after fit()" )
    mlp2.print( "after fit()" )
    #print( mlp1._session.run( mlp1._weights[0] ) )

    #======================================================================
    # モデルの評価
    # (Optional) Evaluate the model.
    #======================================================================
    predict1 = mlp1.predict( X_test_std )
    print( "predict1 :\n", predict1 )

    prob1 = mlp1.predict_proba( X_test_std )
    print( "prob1 :\n", prob1 )

    # テストデータでの正解率
    accuracy1 = mlp1.accuracy( X_test_std, y_test_encoded )
    accuracy2 = mlp2.accuracy( X_test_std, y_test_encoded )

    print( "accuracy1 [test data] : ", accuracy1 )
    print( "accuracy2 [test data] : ", accuracy2 )
    
    # トレーニング回数に対する loss 値の plot
    plt.clf()
    #plt.subplot( 1, 2, 1 )
    plt.plot(
        range( 0, mlp1._epochs ), mlp1._losses_train,
        label = 'train data : MLP = 2-5-3 (sigmoid-softmax)',
        linestyle = '-',
        #linewidth = 2,
        color = 'red'
    )
    #plt.title( "loss" )
    #plt.legend( loc = 'best' )
    #plt.ylim( [0, 1.05] )
    #plt.xlabel( "Epocs" )
    #plt.tight_layout()

    #plt.subplot( 1, 2, 2 )
    plt.plot(
        range( 0, mlp2._epochs ), mlp2._losses_train,
        label = 'train data : MLP = 2-5-5-3 (relu-relu-softmax)',
        linestyle = '--',
        #linewidth = 2,
        color = 'blue'
    )
    plt.title( "loss" )
    plt.legend( loc = 'best' )
    #plt.ylim( [0, 1.05] )
    plt.xlabel( "Epocs" )
    plt.tight_layout()
    
    MLPlot.saveFigure( fileName = "MultilayerPerceptron_2-1.png" )
    plt.show()
    

    # 識別結果＆境界の plot
    plt.clf()
    plt.subplot( 1, 2, 1 )
    MLPlot.drawDiscriminantRegions( X_combined_std, y_combined, classifier = mlp1 )
    plt.title( "Mulutiplelayer Perceptron : 2-5-3 \n activation : sigmoid-softmax" )
    plt.legend( loc = 'best' )

    plt.subplot( 1, 2, 2 )
    MLPlot.drawDiscriminantRegions( X_combined_std, y_combined, classifier = mlp2 )
    plt.title( "Mulutiplelayer Perceptron : 2-5-5-3 \n activation : relu-relu-softmax" )
    plt.legend( loc = 'best' )

    MLPlot.saveFigure( fileName = "MultilayerPerceptron_2-2.png" )
    plt.show()

    
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
