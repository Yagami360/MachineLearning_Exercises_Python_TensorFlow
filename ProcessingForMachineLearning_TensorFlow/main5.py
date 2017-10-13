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
    TensorFlow でのモデルの評価の実装
    """
    print("Enter main()")

    #=========================================
    # 回帰問題でのモデルの評価
    #=========================================
    # Reset graph
    ops.reset_default_graph()

    # Session の設定
    session = tf.Session()

    #----------------------------------------------------------------------
    # データセットを読み込み or 生成
    # Import or generate data.
    #----------------------------------------------------------------------
    # 正規分布 N(1, 0.1) に従う乱数の list
    x_rnorms = numpy.random.normal(1, 0.1, 100)

    # 教師データ（目的値）の list
    y_targets = numpy.repeat(10.0, 100)

    #----------------------------------------------------------------------
    # データを変換、正規化
    # Transform and normalize data.
    # ex) data = tf.nn.batch_norm_with_global_normalization(...)
    #----------------------------------------------------------------------
    # 正規分布 N(1, 0.1) から生成した 100 個乱数の list `x_rnorms` を、
    # TensorFlow を用いてトレーニングデータとテストデータに分割する。

    # トレーニングデータのインデックス範囲
    train_indices = numpy.random.choice(                # ランダムサンプリング
                        len( x_rnorms ),                # 100
                        round( len(x_rnorms)*0.8 ),     # 80% : 100*0.8 = 80
                        replace = False                 # True:重複あり、False:重複なし
                    )

    # テストデータのインデックス範囲
    test_indices = numpy.array( 
                       list( set( range(len(x_rnorms)) ) - set( train_indices ) )   #  set 型（集合型） {...} の list を ndarry 化
                   )

    # トレーニングデータ、テストデータに分割
    x_train = x_rnorms[ train_indices ]
    x_test = x_rnorms[ test_indices ]
    y_train = y_targets[ train_indices ]
    y_test = y_targets[ test_indices ]

    print( "train_indices", train_indices )
    print( "test_indices", test_indices )
    print( "set( range(len(x_rnorms))", set( range(len(x_rnorms)) ) )
    print( "x_train", x_train )
    print( "x_test", x_test )
    print( "y_train", y_train )
    print( "y_test", y_test )

    #----------------------------------------------------------------------
    # アルゴリズム（モデル）のパラメータを設定
    # Set algorithm parameters.
    # ex) learning_rate = 0.01  iterations = 1000
    #----------------------------------------------------------------------
    # ミニバッチ学習でのバッチサイズを指定
    # これは、計算グラフに対し、一度に供給するトレーニングデータの数となる。
    batch_size = 20

    #----------------------------------------------------------------------
    # 変数とプレースホルダを設定
    # Initialize variables and placeholders.
    # TensorFlow は, 損失関数を最小化するための最適化において,
    # 変数と重みベクトルを変更 or 調整する。
    # この変更や調整を実現するためには, 
    # "プレースホルダ [placeholder]" を通じてデータを供給（フィード）する必要がある。
    # そして, これらの変数とプレースホルダと型について初期化する必要がある。
    # ex) a_tsr = tf.constant(42)
    #     x_input_holder = tf.placeholder(tf.float32, [None, input_size])
    #     y_input_holder = tf.placeholder(tf.fload32, [None, num_classes])
    #----------------------------------------------------------------------
    # placeholder の形状 shape は、先のコードとは異なり、
    # 1 次元目を `None`, 2 次元目をバッチサイズとする。
    # 1 次元目を明示的に `20` としても良いが、`None` とすることでより汎用的になる。
    # 後のバッチ学習では, shape = [None, 1] → shape = [20, 1] と解釈される。
    # オンライン学習では、shape = [None, 1] → shape = [1, 1] と解釈される。
    x_rnorms_holder = tf.placeholder( shape = [None, 1], dtype = tf.float32 )   # x_rnorms にデータを供給する placeholder
    y_targets_holder = tf.placeholder( shape = [None, 1], dtype = tf.float32 )  # y_targets にデータを供給する placeholder

    # このモデルのパラメータとなる Variable として、そして、この正規分布乱数値の list である x_rnorms に対し、
    # 行列での乗算演算を行うための変数 A_var を設定する。
    # 最適化アルゴリズムの実行過程で、この Variable の値が TensorFlow によって、
    # 適切に変更されていることを確認するのが、このサンプルコードでの目的の１つである。
    # placeholder の shape の変更に合わせて、shape = [1,1] としている。
    A_var = tf.Variable( tf.random_normal(shape=[1,1]) )
    
    #----------------------------------------------------------------------
    # モデルの構造（計算グラフ）を定義する。
    # Define the model structure.
    # ex) add_op = tf.add(tf.mul(x_input_holder, weight_matrix), b_matrix)
    #----------------------------------------------------------------------
    # 行列での乗算演算
    matmul_op = tf.matmul( x_rnorms_holder, A_var )

    #----------------------------------------------------------------------
    # 損失関数を設定
    # Declare the loss functions.
    #----------------------------------------------------------------------
    # 損失関数として、この教師データと正規分布からの乱数に Variable を乗算した結果の出力の差からなる L2 ノルムの損失関数を定義するが、
    # バッチのデータ点ごとに、すべての L2 ノルムの損失の平均を求める必要があるので、
    # L2 ノルムの損失関数を `reduce_mean(...)` という関数（平均を算出する関数）でラップする。
    loss_op = tf.reduce_mean( tf.square( matmul_op - y_targets_holder ) )

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
    # 最適化アルゴリズム : Optimizer として、最急降下法（勾配降下法）を設定し、パラメータの最適化を行う。
    # （学習プロセスは、損失関数の最小化）
    GD_opt = tf.train.GradientDescentOptimizer( learning_rate = 0.02 )
    train_step = GD_opt.minimize( loss_op )

    # Variable の初期化
    init_op = tf.global_variables_initializer()
    session.run( init_op )

    #---------------------------------
    # ミニバッチ学習
    #---------------------------------
    A_var_list_batch = []
    loss_list_batch = []

    # for ループで各エポックに対し、ミニバッチ学習を行い、パラメータを最適化していく。
    for i in range( 100 ):
        # RNorm のイテレータ : ランダムサンプリング
        it = numpy.random.choice( len(x_train), size = batch_size )  # ミニバッチ処理

        x_rnorm = numpy.transpose( [ x_train[ it ] ] )    # shape を [1] にするため [...] で囲む
        y_target = numpy.transpose( [ y_train[ it ] ] )  # ↑

        session.run( 
            train_step,                     # 学習プロセス（オペレーター）
            feed_dict = { x_rnorms_holder: x_rnorm, y_targets_holder: y_target } 
        )

        A_batch = session.run( A_var )
        loss_batch = session.run( loss_op, feed_dict = { x_rnorms_holder: x_rnorm, y_targets_holder: y_target } )

        A_batch_reshaped = A_batch[0, 0]        # shape = (1,1) [[ xxx ]] → xxx に変換
        #loss_batch_reshaped = loss_batch[0, 0]  # shape = (1,1) [[ xxx ]] → xxx に変換

        A_var_list_batch.append( A_batch_reshaped )
        loss_list_batch.append( loss_batch )

    print( "A_var_list_batch", A_var_list_batch )
    print( "loss_list_batch", loss_list_batch )

    #----------------------------------------------------------------------
    # モデルの評価
    # (Optional) Evaluate the model.
    #----------------------------------------------------------------------
    # MSE の算出
    # 尚、この MSE 値は、トレーニングデータでのトレーニング終了後の、ランダムサンプリングされた 1 つのデータに対する値となる。
    loss_train = session.run( 
                    loss_op, 
                    feed_dict = { 
                        x_rnorms_holder: numpy.transpose( [x_train] ), 
                        y_targets_holder: numpy.transpose( [y_train] )
                    } 
                 )

    loss_test = session.run( 
                    loss_op, 
                    feed_dict = { 
                        x_rnorms_holder: numpy.transpose( [x_test] ), 
                        y_targets_holder: numpy.transpose( [y_test] )
                    } 
                 )

    mse_train = numpy.round( loss_train, 2 )    # ２乗
    mse_test = numpy.round( loss_test, 2 )

    print( "MSE (train data) : ", mse_train )
    print( "MSE (test data) : ", mse_test )

    #print( "numpy.transpose( [x_train] )", numpy.transpose( [x_train] ) )

    '''
    # 全データに対する MSE の算出
    mse_list_train = []
    mse_list_test = []

    for i in range( x_train ):
        loss_train = session.run( 
                        loss_op, 
                        feed_dict = { 
                            x_rnorms_holder: numpy.transpose( [x_train] ), 
                            y_targets_holder: numpy.transpose( [y_train] )
                        } 
                     )

        mse_train = numpy.round( loss_train, 2 )    # ２乗
        mse_list_train.append( mse_train )
    
    print( "mse_list_train", mse_list_train )
    '''
    #--------------------
    # plot figure
    #--------------------
    '''
    plt.clf()
    
    plt.subplot(1, 2, 1)
    plt.plot(
        range( 0, 100 ), A_var_list_batch,
        label = 'batch training (batch size = 20)',
        linestyle = '-',
        #linewidth = 2,
        color = 'red'
    )
    plt.title( "A_var ( model parameter )" )
    plt.legend( loc = 'best' )
    plt.ylim( [0, 10.5] )
    plt.xlabel( "Epocs ( Number of training )" )
    plt.tight_layout()


    plt.subplot(1, 2, 2)
    plt.plot(
        range( 0, 100 ), loss_list_batch,
        label = 'batch training (batch size = 20)',
        linestyle = '-',
        #linewidth = 2,
        color = 'red'
    )
    plt.title( "loss function ( L2-norm )" )
    plt.legend( loc = 'best' )
    plt.xlabel( "Epocs ( Number of training )" )
    #plt.ylim( [0, 100] )
    plt.tight_layout()

    MLPlot.saveFigure( fileName = "ProcessingForMachineLearning_TensorFlow_5-1.png" )
    plt.show()
    '''


    print("Finish main()")
    return

if __name__ == '__main__':
     main()
