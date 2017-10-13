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

    #print( "train_indices", train_indices )
    #print( "test_indices", test_indices )
    #print( "set( range(len(x_rnorms))", set( range(len(x_rnorms)) ) )
    #print( "x_train", x_train )
    #print( "x_test", x_test )
    #print( "y_train", y_train )
    #rint( "y_test", y_test )

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

    #=========================================
    # 分類問題でのモデルの評価
    #=========================================
    # Reset graph
    ops.reset_default_graph()

    # Session の設定
    session = tf.Session()

    #----------------------------------------------------------------------
    # データセットを読み込み or 生成
    # Import or generate data.
    #----------------------------------------------------------------------
    # 2 つの正規分布 N(-1, 1) , N(2, 1) の正規分布に従う乱数をそれぞれ 50 個、合計 100 個生成
    # numpy.concatenate(...) : ２個以上の配列を軸指定して結合する。軸指定オプションの axis はデフォルトが 0
    x_rnorms = numpy.concatenate(
                   ( numpy.random.normal(-1,1,50), numpy.random.normal(2,1,50) )
               )
    
    # 生成した 2 つの正規分布 N(-1, 1) , N(2, 1) のそれぞれのクラスラベルを {0,1} とする。
    y_targets = numpy.concatenate( ( numpy.repeat(0., 50), numpy.repeat(1., 50) ) )

    print( "x_rnorms", x_rnorms )
    print( "y_targets", y_targets )

    #----------------------------------------------------------------------
    # データを変換、正規化
    # Transform and normalize data.
    # ex) data = tf.nn.batch_norm_with_global_normalization(...)
    #----------------------------------------------------------------------
    # 2 つの正規分布 N(-1, 1) , N(3, 1) の正規分布に従う乱数をそれぞれ 50 個、合計 100 個、
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

    #----------------------------------------------------------------------
    # アルゴリズム（モデル）のパラメータを設定
    # Set algorithm parameters.
    # ex) learning_rate = 0.01  iterations = 1000
    #----------------------------------------------------------------------
    # ミニバッチ学習でのバッチサイズを指定
    # これは、計算グラフに対し、一度に供給するトレーニングデータの数となる。
    batch_size = 25

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
    x_rnorms_holder = tf.placeholder( shape = [1, None], dtype = tf.float32 )

    # y_targets にデータを供給する placeholder
    y_targets_holder = tf.placeholder( shape = [1, None], dtype = tf.float32 )
    
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
    
    #----------------------------------------------------------------------
    # 損失関数を設定
    # Declare the loss functions.
    #----------------------------------------------------------------------
    # 損失関数として、正規分布乱数 `x_rnorms` をシグモイド関数で変換し、
    # クロス・エントロピーをとる、シグモイド・クロス・エントロピー関数 : `tf.nn.sigmoid_cross_entropy_with_logit(...)` を使用する。
    loss_op = tf.nn.sigmoid_cross_entropy_with_logits(
                  logits = add_op,              # 最終的な推計値。sigmoid 変換する必要ない
                  labels = y_targets_holder     # 教師データ
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
    # 最急降下法によるパラメータの最適化を行う。
    optGD_op = tf.train.GradientDescentOptimizer( learning_rate = 0.05 )

    # 学習プロセスは、損失関数の最小化
    train_step = optGD_op.minimize( loss_op )
    
    print( "tf.train.GradientDescentOptimizer(...) : \n", optGD_op )
    print( "optGD_op.minimize( loss_op ) : \n", train_step )

    # Variable の初期化
    init_op = tf.global_variables_initializer()
    session.run( init_op )

    #------------------------------------------
    # 指定した最適化アルゴリズムでの学習過程
    #------------------------------------------
    a_var_list = []
    loss_list = []

    # for ループで各エポックに対し、ミニバッチ学習を行い、パラメータを最適化していく。
    for i in range( 1800 ):
        # RNorm のイテレータ : ランダムサンプリング
        it = numpy.random.choice( len(x_train), size = batch_size )  # ミニバッチ処理

        x_rnorm = [ x_train[ it ] ]    # shape を [1] にするため [...] で囲む
        y_target = [ y_train[ it ] ]  # ↑

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
    # このコードでの目的である分類モデルの評価指数として、TensorFlow を用いて正解率を算出する。
    # そのためにはまず、モデルと同じ演算を実施し、予測値を算出する。
    y_pred_op = tf.squeeze( 
                 tf.round(
                    tf.nn.sigmoid( tf.add(x_rnorms_holder, a_var) )
                 ) 
             )
    
    print( "y_pred_op", y_pred_op )

    # 続いて、`tf.eequal(...)` を用いて、予測値 `y_pred` と目的値 `y_target` が等価であるか否かを確認する。
    correct_pred_op = tf.equal( y_pred_op, y_targets_holder )
    print( "correct_pred_op", correct_pred_op )
    
    # これにより、bool 型の Tensor `Tensor("Equal:0", dtype=bool)` が `correct_pred` に代入されるので、
    # この値を `float32` 型に `tf.cast(...)` で明示的にキャストした後、平均値を `tf.reduce_mean(...)` で求める。
    accuracy_op = tf.reduce_mean( tf.cast(correct_pred_op, tf.float32) )
    print( "accuracy_op", accuracy_op )

    accuracy_train = session.run( 
        accuracy_op, 
        feed_dict = {
            x_rnorms_holder: [x_train],     # shape を合わせるため,[...] で囲む
            y_targets_holder: [y_train]
        }
    )

    accuracy_test = session.run( 
        accuracy_op, 
        feed_dict = {
            x_rnorms_holder: [x_test], 
            y_targets_holder: [y_test]
        }
    )

    print( "Accuracy (train data) : ", accuracy_train )
    print( "Accuracy (test data) : ", accuracy_test )
    
    '''
    plt.clf()

    # plot Figure
    plt.subplot(1, 2, 1)
    plt.plot(
        range( 0, 1800), a_var_list,
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
        range( 0, 1800), loss_list,
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
    '''



    #MLPlot.saveFigure( fileName = "ProcessingForMachineLearning_TensorFlow_5-1.png" )
    #plt.show()



    print("Finish main()")
    return

if __name__ == '__main__':
     main()
