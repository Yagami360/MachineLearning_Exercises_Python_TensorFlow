# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境 (TensorFlow インストール済み)
#     [Anaconda Prompt]
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

# scikit-learn ライブラリ関連


# 自作クラス
from MLPlot import MLPlot

def main():
    """
    TensorFlow での誤差逆伝播法の実装
    """
    print("Enter main()")

    #======================================================================
    # ① 回帰問題での誤差逆伝播法
    #======================================================================
    # Reset graph
    ops.reset_default_graph()

    # Session の設定
    session = tf.Session()

    #----------------------------------------------------------------------
    # データセットを読み込み or 生成
    # Import or generate data.
    #----------------------------------------------------------------------
    num_train = 100             # トレーニング回数

    # 正規分布の従う乱数生成 : x = N(1.0, 0.1)
    x_rnorms = numpy.random.normal(    
                   loc = 1.0,          # 平均値 
                   scale = 0.1,        # 分散値
                   size = num_train    # 乱数の数
               )   
    print( "x_rnorms :", x_rnorms )

    # 教師データ（目的値） : y=10
    y_targets = numpy.repeat( 10., num_train )
    print( "y_targets :", y_targets )

    #----------------------------------------------------------------------
    # データを変換、正規化
    # Transform and normalize data.
    # ex) data = tf.nn.batch_norm_with_global_normalization(...)
    #----------------------------------------------------------------------

    #----------------------------------------------------------------------
    # アルゴリズム（モデル）のパラメータを設定
    # Set algorithm parameters.
    # ex) learning_rate = 0.01  iterations = 1000
    #----------------------------------------------------------------------
    input_learning_rate = 0.01    # 最急降下法での学習率

    #----------------------------------------------------------------------
    # 変数とプレースホルダを設定
    # Initialize variables and placeholders.
    # TensorFlow は, 損失関数を最小化するための最適化において,
    # 変数と重みベクトルを変更 or 調整する。
    # この変更や調整を実現するためには, 
    # "プレースホルダ [placeholder]" を通じてデータを供給（フィード）する必要がある。
    # そして, これらの変数とプレースホルダと型について初期化する必要がある。
    # ex) a_var = tf.constant(42)
    #     x_input = tf.placeholder(tf.float32, [None, input_size])
    #     y_input = tf.placeholder(tf.fload32, [None, num_classes])
    #----------------------------------------------------------------------
    # x_rnorms に乗算するための変数 a ( a * x_rnorms )
    # モデルのパラメータ（ハイパーパラメータ）で、最適値は 10 になる
    a_var = tf.Variable( tf.random_normal( shape = [1] ) )
    print( "a_var : \n", a_var )

    # オペレーター mul_op での x_rnorm にデータを供給する placeholder
    x_rnorms_holder = tf.placeholder( shape = [1], dtype = tf.float32 )

    # L2 損失関数（オペレーター）loss = y_targets - x_rnorms = 10 - N(1.0, 0.1) での
    # y_targets に教師データを供給する placeholder
    y_targets_holder = tf.placeholder( shape = [1], dtype = tf.float32 )
    
    #----------------------------------------------------------------------
    # モデルの構造（計算グラフ）を定義する。
    # Define the model structure.
    # ex) y_pred = tf.add(tf.mul(x_input, weight_matrix), b_matrix)
    #----------------------------------------------------------------------
    # 計算グラフに乗算追加
    mul_op = tf.multiply( x_rnorms_holder, a_var )
    
    #----------------------------------------------------------------------
    # 損失関数を設定
    # Declare the loss functions.
    #----------------------------------------------------------------------
    # L2 正則化の損失関数のオペレーション
    # 教師データと正規分布からの乱数に変数を乗算した結果の出力の差からなる L2 ノルムの損失関数
    # （変数 a が 10 のとき誤差関数の値が 0 になるようなモデルにする。）
    l2_loss_op = tf.square( mul_op - y_targets_holder )

    #----------------------------------------------------------------------
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
    #----------------------------------------------------------------------
    # Variable の初期化
    init_op = tf.global_variables_initializer()
    session.run( init_op )

    # 最急降下法によるパラメータの最適化を行う。
    optGD_op = tf.train.GradientDescentOptimizer( learning_rate = input_learning_rate )

    # 学習プロセスは、損失関数の最小化
    train_step = optGD_op.minimize( l2_loss_op )
    
    print( "tf.train.GradientDescentOptimizer(...) : \n", optGD_op )
    print( "optGD_op.minimize( l2_loss_op ) : \n", train_step )

    a_var_list = []
    loss_list = []
    # for ループでオンライン学習
    for i in range( num_train ):
        # RNorm のイテレータ : ランダムサンプリング
        it = numpy.random.choice( num_train )

        x_rnorm = [ x_rnorms[ it ] ]    # shape を [1] にするため [...] で囲む
        y_target = [ y_targets[ it ] ]  # ↑

        session.run( 
            train_step,                     # 学習プロセス（オペレーター）
            feed_dict = { x_rnorms_holder: x_rnorm, y_targets_holder: y_target } 
        )

        a_var_list.append( session.run( a_var ) )
        loss_list.append( 
            session.run( l2_loss_op, feed_dict = { x_rnorms_holder: x_rnorm, y_targets_holder: y_target } )
        )
        
        #print( "epoc :", i )
        #print( "parameter-a :", a )
        #print( "loss function :", loss )

    #print( "a_var_list", a_var_list )
    #print( "loss_list", loss_list )

    #----------------------------------------------------------------------
    # モデルの評価
    # (Optional) Evaluate the model.
    #----------------------------------------------------------------------
    plt.clf()

    # plot Figure
    plt.subplot(1, 2, 1)
    plt.plot(
        range( 0, num_train), a_var_list,
        label = 'a ( model parameter )',
        linestyle = '-',
        #linewidth = 2,
        color = 'red'
    )
    
    plt.title( "a ( model parameter )" )
    plt.legend( loc = 'best' )
    plt.ylim( [0, 10.5] )
    plt.xlabel( "Epocs ( Number of training )" )
    plt.tight_layout()


    plt.subplot(1, 2, 2)
    plt.plot(
        range( 0, num_train), loss_list,
        label = 'loss function ( L2-norm )',
        linestyle = '-',
        #linewidth = 2,
        color = 'blue'
    )
    plt.title( "loss function ( L2-norm )" )
    plt.legend( loc = 'best' )
    plt.xlabel( "Epocs ( Number of training )" )
    #plt.ylim( [0, 100] )
    plt.tight_layout()

    MLPlot.saveFigure( fileName = "ProcessingForMachineLearning_TensorFlow_3-1.png" )
    #plt.show()

    #======================================================================
    # 分類問題での誤差逆伝播法
    #======================================================================
    # Reset graph
    ops.reset_default_graph()

    # Session の設定
    session = tf.Session()

    #----------------------------------------------------------------------
    # データセットを読み込み or 生成
    # Import or generate data.
    #----------------------------------------------------------------------
    # 2 つの正規分布 N(-1, 1) , N(3, 1) の正規分布に従う乱数をそれぞれ 50 個、合計 100 個生成
    # numpy.concatenate(...) : ２個以上の配列を軸指定して結合する。軸指定オプションの axis はデフォルトが 0
    x_rnorms = numpy.concatenate(
                   ( numpy.random.normal(-1,1,50), numpy.random.normal(3,1,50) )
               )
    
    # 生成した 2 つの正規分布 N(-1, 1) , N(3, 1) のそれぞれのクラスラベルを {0,1} とする。
    y_targets = numpy.concatenate( ( numpy.repeat(0,50), numpy.repeat(1,50) ) )

    print( "x_rnorms", x_rnorms )
    print( "y_targets", y_targets )

    #----------------------------------------------------------------------
    # データを変換、正規化
    # Transform and normalize data.
    # ex) data = tf.nn.batch_norm_with_global_normalization(...)
    #----------------------------------------------------------------------

    #----------------------------------------------------------------------
    # アルゴリズム（モデル）のパラメータを設定
    # Set algorithm parameters.
    # ex) learning_rate = 0.01  iterations = 1000
    #----------------------------------------------------------------------

    #----------------------------------------------------------------------
    # 変数とプレースホルダを設定
    # Initialize variables and placeholders.
    # TensorFlow は, 損失関数を最小化するための最適化において,
    # 変数と重みベクトルを変更 or 調整する。
    # この変更や調整を実現するためには, 
    # "プレースホルダ [placeholder]" を通じてデータを供給（フィード）する必要がある。
    # そして, これらの変数とプレースホルダと型について初期化する必要がある。
    # ex) a_var = tf.constant(42)
    #     x_input = tf.placeholder(tf.float32, [None, input_size])
    #     y_input = tf.placeholder(tf.fload32, [None, num_classes])
    #----------------------------------------------------------------------
    # x_rnorms にデータを供給する placeholder
    x_rnorms_holder = tf.placeholder( shape = [1], dtype = tf.float32 )

    # y_targets にデータを供給する placeholder
    y_targets_holder = tf.placeholder( shape = [1], dtype = tf.float32 )
    
    # Variable (model parameter)
    # このモデルのパラメータとなる Variable として、シグモイド関数の平行移動分 sigmoid( x + a_var ) となる変数
    # この平行移動したシグモイド関数が、クラスラベル {0,1} を識別するモデルとなる。
    # そして、最適化アルゴリズムの実行過程で、この Variable の値が 
    # TensorFlow によって、適切に変更されていることを確認するのが、このサンプルコードでの目的の１つである。
    a_var = tf.Variable( tf.random_normal(mean=10, shape=[1]) )
    
    #----------------------------------------------------------------------
    # モデルの構造（計算グラフ）を定義する。
    # Define the model structure.
    # ex) y_pred = tf.add(tf.mul(x_input, weight_matrix), b_matrix)
    #----------------------------------------------------------------------
    # シグモイド関数の平行移動演算（オペレーター）を計算グラフに追加する。
    # この際、加算演算 `tf.add(...)` に加えてシグモイド関数 `tf.sigmoid(...)` をラップする必要があるように思えるが、
    # 後に定義する損失関数が、シグモイド・クロス・エントロピー関数 tf.nn.sigmoid_cross_entropy_with_logit(...)` であるため、
    # このシグモイド関数の演算が自動的に行われるために、ここでは必要ないことに注意。
    add_op = tf.add( x_rnorms_holder, a_var )
    
    # 後に定義する損失関数が、シグモイド・クロス・エントロピー関数 : tf.nn.sigmoid_cross_entropy_with_logit(...)
    # が、追加の次元（バッチ数が関連付けられているバッチデータ）を必要とするため、
    # tf.expand_dims(...) を用いて、オペレーター、placeholder に次元を追加する。
    add_expaned_op = tf.expand_dims(add_op, 0)
    y_expaned_targets_holder = tf.expand_dims(y_targets_holder, 0)

    print( "add_op" , add_op)
    print( "add_expaned_op" , add_expaned_op)
    print( "y_targets_holder" , y_targets_holder)
    print( "y_expaned_targets_holder" , y_expaned_targets_holder)

    #----------------------------------------------------------------------
    # 損失関数を設定
    # Declare the loss functions.
    #----------------------------------------------------------------------
    # 損失関数として、正規分布乱数 `x_rnorms` をシグモイド関数で変換し、
    # クロス・エントロピーをとる、シグモイド・クロス・エントロピー関数 : `tf.nn.sigmoid_cross_entropy_with_logit(...)` を使用する。
    loss_op = tf.nn.sigmoid_cross_entropy_with_logits(
                  logits = add_expaned_op,              # 最終的な推計値。sigmoid 変換する必要ない
                  labels = y_expaned_targets_holder     # 教師データ
              )

    #----------------------------------------------------------------------
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
    #----------------------------------------------------------------------
    # Variable の初期化
    init_op = tf.global_variables_initializer()
    session.run( init_op )

    # 最急降下法によるパラメータの最適化を行う。
    optGD_op = tf.train.GradientDescentOptimizer( learning_rate = 0.05 )

    # 学習プロセスは、損失関数の最小化
    train_step = optGD_op.minimize( loss_op )
    
    print( "tf.train.GradientDescentOptimizer(...) : \n", optGD_op )
    print( "optGD_op.minimize( loss_op ) : \n", train_step )

    a_var_list = []
    loss_list = []
    # for ループでオンライン学習
    for i in range( 1500 ):
        # RNorm のイテレータ : ランダムサンプリング
        it = numpy.random.choice( num_train )

        x_rnorm = [ x_rnorms[ it ] ]    # shape を [1] にするため [...] で囲む
        y_target = [ y_targets[ it ] ]  # ↑

        session.run( 
            train_step,                     # 学習プロセス（オペレーター）
            feed_dict = { x_rnorms_holder: x_rnorm, y_targets_holder: y_target } 
        )

        a_var_list.append( session.run( a_var ) )
        
        loss = session.run( loss_op, feed_dict = { x_rnorms_holder: x_rnorm, y_targets_holder: y_target } )
        loss_reshaped = loss[0, 0]  # shape = (1,1) [[ xxx ]] → xxx に変換
        loss_list.append( loss_reshaped )

        
        #print( "epoc :", i )
        #print( "parameter-a :", a )
        #print( "loss :", loss )
        #print( "loss_reshaped :", loss_reshaped )

    #print( "a_var_list", a_var_list )
    #print( "loss_list", loss_list )

    #----------------------------------------------------------------------
    # モデルの評価
    # (Optional) Evaluate the model.
    #----------------------------------------------------------------------
    plt.clf()

    # plot Figure
    plt.subplot(1, 2, 1)
    plt.plot(
        range( 0, 1500), a_var_list,
        label = 'a ( model parameter )',
        linestyle = '-',
        #linewidth = 2,
        color = 'red'
    )
    
    plt.title( "a ( model parameter )" )
    plt.legend( loc = 'best' )
    plt.ylim( [-1.5, 10.5] )
    plt.xlabel( "Epocs ( Number of training )" )
    plt.tight_layout()


    plt.subplot(1, 2, 2)
    plt.plot(
        range( 0, 1500, 1), loss_list,
        label = 'loss function ( sigmoid-cross-entropy )',
        linestyle = '-',
        #linewidth = 2,
        color = 'blue'
    )
    plt.title( "loss function ( sigmoid-cross-entropy )" )
    plt.legend( loc = 'best' )
    plt.xlabel( "Epocs ( Number of training )" )
    #plt.ylim( [0, 100] )
    plt.tight_layout()

    MLPlot.saveFigure( fileName = "ProcessingForMachineLearning_TensorFlow_3-2.png" )
    plt.show()

    print("Finish main()")
    return
    

if __name__ == '__main__':
     main()

