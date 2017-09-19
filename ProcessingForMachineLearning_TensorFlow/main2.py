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
    損失関数の実装
    """
    print("Enter main()")

    #======================================================================
    # 損失関数の実装
    #======================================================================
    #-------------------------------------------------------------------
    # 回帰問題の為の損失関数
    #-------------------------------------------------------------------
    # Reset graph
    ops.reset_default_graph()

    # Session の設定
    session = tf.Session()

    # 一連の予想値
    x_predicts_tsr = tf.linspace( -1. , 1. , 500 )     # -1 ~ +1 の範囲の 500 個のシーケンス Tensor
    print( "x_predicts_tsr :\n", x_predicts_tsr )

    # 目的値
    target_tsr = tf.constant( 0. )   #  定数 Tensor で値（目的値）を 0 としている。
    print( "target_tsr :\n", target_tsr )
    
    # 回帰問題の為の損失関数のオペレーション作成
    l2_loss_op = tf.square( target_tsr - x_predicts_tsr )   # L2 正則化の損失関数のオペレーション
    l1_loss_op = tf.abs( target_tsr - x_predicts_tsr )      # L1 正則化の損失関数のオペレーション
    
    print( "l2_loss_op = tf.square( target - x_predicts_tsr ) :\n", l2_loss_op )
    print( "l1_loss_op = tf.abs( target - x_predicts_tsr ) :\n", l1_loss_op )

    # Session を run してオペレーションを実行
    axis_x_list = session.run( x_predicts_tsr )     # 損失関数のグラフ化のためのグラフ化のための x 軸の値のリスト 
    output_l2_loss = session.run( l2_loss_op )      # L2 正則化の損失関数の値 （グラフの y 軸値の list）
    output_l1_loss = session.run( l1_loss_op )      # L1 正則化の損失関数の値 （グラフの y 軸値の list）
    
    # TensorBoard 用のファイル（フォルダ）を作成
    # Add summaries to tensorboard
    #merged = tf.summary.merge_all( key='summaries' )
    # tensorboard --logdir=${PWD}
    #summary_writer = tf.summary.FileWriter( "./TensorBoard", graph = session.graph )

    session.close()
    
    #---------------------------------------
    # plot loss functions
    #---------------------------------------
    plt.clf()
 
    # plot L2 loss function
    plt.plot( 
        axis_x_list, output_l2_loss, 
        label='L2 loss ( target = 0 )',
        linestyle = ':',
        #linewidth = 2,
        color = 'red'
    )
    # plot L1 loss function
    plt.plot( 
        axis_x_list, output_l1_loss, 
        label='L1 loss ( target = 0 )',
        linestyle = '--',
        #linewidth = 2,
        color = 'blue'
    )
    plt.title( "loss functions ( for regression )" )
    plt.legend( loc = 'best' )
    plt.ylim( [-0.05, 1.0] )
    plt.tight_layout()

    
    MLPlot.saveFigure( fileName = "ProcessingForMachineLearning_TensorFlow_2-1.png" )
    #plt.show()

    #-------------------------------------------------------------------
    # 分類問題の為の損失関数
    #-------------------------------------------------------------------
    # Reset graph
    ops.reset_default_graph()

    # Session の設定
    session = tf.Session()

    # 一連の予想値
    x_predicts_tsr = tf.linspace( -3. , 5. , 500 )     # -3 ~ +5 の範囲の 500 個のシーケンス Tensor
    print( "x_predicts_tsr :\n", x_predicts_tsr )

    # 目的値
    target_tsr = tf.constant( 1. )          #  定数 Tensor で値（目的値）を 1.0 としている。
    targets_tsr = tf.fill( [500,], 1. )     #  定数 Tensor で値（目的値）を 1.0 としている。
    
    
    # 回帰問題の為の損失関数のオペレーション作成
    hinge_loss_op = tf.maximum( 0., 1. - tf.multiply( target_tsr, x_predicts_tsr ) )   # ヒンジ損失関数のオペレーション
    cross_entopy_loss_op = - tf.multiply( target_tsr, tf.log( x_predicts_tsr ) ) \
                           - tf.multiply( 1 - target_tsr, tf.log( 1 - x_predicts_tsr ) ) # L1 正則化の損失関数のオペレーション
    

    # シグモイド・クロス・エントロピー関数
    # dim で指定された次元(Rank)分を input に追加
    x_expanded_predicts_tsr = tf.expand_dims( input = x_predicts_tsr, dim = 1 )
    expaned_targets_tsr = tf.expand_dims( input = targets_tsr, dim = 1 )
    
    #print( "session.run( x_predicts_tsr ) : \n", session.run( x_predicts_tsr ) )
    #print( "session.run( x_expanded_predicts_tsr ) : \n", session.run( x_expanded_predicts_tsr ) )
    #print( "session.run( targets_tsr ) :\n", session.run( targets_tsr ) )    
    #print( "session.run( expaned_targets_tsr ) : \n", session.run( expaned_targets_tsr ) )

    # シグモイド・クロス・エントロピー関数のオペレーション
    # x = logits, z = labels. 
    # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
    # tf.nn.softmax_cross_entropy_with_logits(...) : 推計結果のsoftmax値を計算して、cross-entropyを計算します。
    sigmoid_cross_entropy_loss_op = tf.nn.softmax_cross_entropy_with_logits( 
                                        logits = x_expanded_predicts_tsr,   # 最終的な推計値。softmax はする必要ない
                                        labels = expaned_targets_tsr        # 教師データ
                                    )

    # 重み付けクロス・エントロピー損失関数
    # loss = targets * -log(sigmoid(logits)) * pos_weight + (1 - targets) * -log(1 - sigmoid(logits))
    weight_tsr = tf.constant( 0.5 )      # 重み付けの値の定数 Tensor
    weighted_cross_entropy_loss_op = tf.nn.weighted_cross_entropy_with_logits(                                        
                                         logits = x_predicts_tsr, 
                                         targets = targets_tsr,
                                         pos_weight = weight_tsr
                                     )

    print( "hinge_loss_op :\n", hinge_loss_op )
    print( "cross_entopy_loss_op :\n", cross_entopy_loss_op )
    print( "sigmoid_cross_entropy_loss_op :\n", sigmoid_cross_entropy_loss_op )
    print( "weighted_cross_entropy_loss_op :\n", weighted_cross_entropy_loss_op )

    # ソフトマックスクロス・エントロピー損失関数 [softmax cross-entrpy loss function] 
    # L = -actual * (log(softmax(pred))) - (1-actual)(log(1-softmax(pred)))
    unscaled_logits = tf.constant( [[1., -3., 10.]] )   # 正規化されていない予測値
    target_dist = tf.constant( [[0.1, 0.02, 0.88]] )    # 目的値の確率分布
    softmax_entropy_op = tf.nn.softmax_cross_entropy_with_logits(
                            logits = unscaled_logits,   # 最終的な推計値。softmax はする必要ない
                            labels = target_dist        # 教師データ（ここでは、目的値の確率分布）
                         )

    # 疎なソフトマックスクロス・エントロピー損失関数 [ sparse softmax cross-entrpy loss function] 
    unscaled_logits = tf.constant( [[1., -3., 10.]] )   # 正規化されていない予測値
    sparse_target_dist = tf.constant( [2] )             # 分類クラス {-1 or 1} が真となるクラスのインデックス
    sparse_entropy_op = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=unscaled_logits,     # 最終的な推計値。softmax はする必要ない
                            labels=sparse_target_dist   # 教師データ（ここでは、クラスラベル）
                        )
    
    # Session を run してオペレーションを実行
    axis_x_list = session.run( x_predicts_tsr )                                         # 損失関数のグラフ化のためのグラフ化のための x 軸の値のリスト 
    output_hinge_loss = session.run( hinge_loss_op )                                    # ヒンジ損失関数の値 （グラフの y 軸値の list）
    output_cross_entropy_loss = session.run( cross_entopy_loss_op )                     # クロスエントロピー損失関数の値 （グラフの y 軸値の list）
    output_sigmoid_cross_entropy_loss = session.run( sigmoid_cross_entropy_loss_op )    # シグモイド・クロス・エントロピー損失関数の値 （グラフの y 軸値の list）
    output_weighted_cross_entropy_loss = session.run( weighted_cross_entropy_loss_op )  # 重み付けクロス・エントロピー損失関数の値 （グラフの y 軸値の list）
    
    output_softmax_entropy_loss = session.run( softmax_entropy_op )                     # ソフトマックスクロス・エントロピー損失関数の値 （グラフの y 軸値の list）
    print( "output_softmax_entropy_loss : ", output_softmax_entropy_loss )
    
    output_sparse_entropy_loss = session.run( sparse_entropy_op )                       # 疎なソフトマックスクロス・エントロピー損失関数の値 （グラフの y 軸値の list）
    print( "output_sparse_entropy_loss : ", output_sparse_entropy_loss )

    #---------------------------------------
    # plot loss functions
    #---------------------------------------
    plt.clf()

    # plot hinge function
    plt.plot( 
        axis_x_list, output_hinge_loss, 
        label = 'hinge loss ( target = 1 )',
        linestyle = ':',
        #linewidth = 2,
        color = 'red'
    )

    # plot cross-entropy
    plt.plot( 
        axis_x_list, output_cross_entropy_loss, 
        label = 'cross-entropy loss ( target = 1 )',
        linestyle = '--',
        #linewidth = 2,
        color = 'blue'
    )

    # plot sigmoid cross-entropy
    plt.plot( 
        axis_x_list, output_sigmoid_cross_entropy_loss, 
        label = 'sigmoid cross-entropy loss ( target = 1 )',
        linestyle = '-.',
        linewidth = 1,
        color = 'lightgreen'
    )

    # plot weigted cross-entropy
    plt.plot( 
        axis_x_list, output_weighted_cross_entropy_loss, 
        label = 'weigted cross-entropy loss ( target = 1 )',
        linestyle = '-.',
        #linewidth = 2,
        color = 'mediumpurple'
    )

    plt.title( "loss functions ( for classification )" )
    plt.legend( loc = 'best' )
    plt.ylim( [-1.5, 3.0] )
    plt.tight_layout()

    MLPlot.saveFigure( fileName = "ProcessingForMachineLearning_TensorFlow_2-2.png" )
    plt.show()


    print("Finish main()")
    return
    

if __name__ == '__main__':
     main()