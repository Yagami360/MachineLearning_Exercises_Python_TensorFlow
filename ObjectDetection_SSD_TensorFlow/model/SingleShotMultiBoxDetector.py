# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境 (TensorFlow 1.4.0 インストール済み)

"""
    更新情報
    [18/05/14] : 新規作成
    [xx/xx/xx] : 

"""

import numpy as np

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops

# 自作モジュール
from model.NeuralNetworkBase import NeuralNetworkBase
from model.BaseNetwork import BaseNetwork
from model.BaseNetwork import BaseNetworkVGG16
from model.BaseNetwork import BaseNetworkResNet

from model.DefaultBox import DefaultBox
from model.DefaultBox import DefaultBoxes
from model.BoundingBox import BoundingBox


class SingleShotMultiBoxDetector( NeuralNetworkBase ):
    """
    SSD [Single Shot muitibox Detector] を表すクラス。

    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
        _default_boxes : list<DefaultBoxes>
            一連のデフォルトボックスを表すクラス DefaultBoxes のオブジェクト
            
    [protedted] protedted な使用法を想定 
        image_height : int
            入力画像データの高さ（ピクセル単位）
        image_width : int
            入力画像データの幅（ピクセル単位）
        n_channels : int
            入力画像データのチャンネル数
            1 : グレースケール画像

        n_classes : int
            識別クラス数
        n_boxes : list<int>
            ? the number of boxes per feature map

        base_vgg16 : BaseNetworkVGG16
            SSD のベースネットワークとしての VGG16 を表す BaseNetworkVGG16 クラスのオブジェクト

        conv6_op : Operator
        pool6_op : Operator
        conv7_op : Operator
        conv8_1_op : Operator
        conv8_2_op : Operator
        conv9_1_op : Operator
        conv9_2_op : Operator
        conv10_1_op : Operator
        conv10_2_op : Operator
        conv11_1_op : Operator
        conv11_2_op : Operator

        f_maps : list<Tensor>
            特徴マップのリスト
        pred_confidences : list<int>
            ? 予想される特徴マップのクラス所属の確信度
        pred_locations : list<int>
            ? 予想される特徴マップの位置（形状のオフセット）


    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( 
            self,
            session = tf.Session(),
            image_height = 32,
            image_width = 32,
            n_channels = 1,
            n_classes = 21,
            n_boxes = [ 4, 6, 6, 6, 6, 6 ]
        ):
        
        super().__init__( session )

        # 各パラメータの初期化
        self.image_height = image_height
        self.image_width = image_width
        self.n_channels = n_channels

        self.n_classes = n_classes
        self.n_boxes = n_boxes

        # SDD モデルの各オペレーター
        self.base_vgg16 = BaseNetworkVGG16( 
                              session = self._session,
                              image_height = self.image_height,
                              image_width = self.image_width,
                              n_channels = self.n_channels
                          )

        self.conv6_op = None
        self.pool6_op = None
        self.conv7_op = None
        self.conv8_1_op = None
        self.conv8_2_op = None
        self.conv9_1_op = None
        self.conv9_2_op = None
        self.conv10_1_op = None
        self.conv10_2_op = None
        self.conv11_1_op = None
        self.conv11_2_op = None

        #
        self.fmaps = []
        self.pred_confidences = []
        self.pred_locations = []

        #
        self._default_boxes = None

        return


    def print( self, str ):
        print( "----------------------------------" )
        print( str )
        print( self )

        print( "_session :", self._session )
        print( "_init_var_op : \n", self._init_var_op )
        print( "_y_out_op :", self._y_out_op )

        print( "_loss_op :", self._loss_op )
        print( "_optimizer :", self._optimizer )
        print( "_train_step :", self._train_step )

        print( "image_height : " , self.image_height )
        print( "image_width : " , self.image_width )
        print( "n_channels : " , self.n_channels )
        
        print( "n_classes", self.n_classes )
        print( "n_boxes", self.n_boxes )

        self.base_vgg16.print( "base network" )
        print( "conv6_op :", self.conv6_op )
        print( "pool6_op :", self.pool6_op )
        print( "conv7_op :", self.conv7_op )
        print( "conv8_1_op :", self.conv8_1_op )
        print( "conv8_2_op :", self.conv8_2_op )
        print( "conv9_1_op :", self.conv9_1_op )
        print( "conv9_2_op :", self.conv9_2_op )
        print( "conv10_1_op :", self.conv10_1_op )
        print( "conv10_2_op :", self.conv10_2_op )
        print( "conv11_1_op :", self.conv11_1_op )
        print( "conv11_2_op :", self.conv11_2_op )

        if( self.fmaps == [] ):
            print( "fmaps :", self.fmaps )
        else:
            print( "fmaps :" )
            for i, fmap in enumerate( self.fmaps ):
                print( "    fmaps[%d] : %s" % (i, fmap) )
        
        print( "pred_confidences :", self.pred_confidences )
        print( "pred_locations :", self.pred_locations )

        print( "_default_boxes :", self._default_boxes )
        #if( self._default_boxes != None ):
            #self._default_boxes.print()

        print( "----------------------------------" )

        return


    def init_weight_variable( self, input_shape, name = "init_weight_var" ):
        """
        重みの初期化を行う。
        重みは TensorFlow の Variable で定義することで、
        学習過程（最適化アルゴリズム Optimizer の session.run(...)）で自動的に TensorFlow により、変更される値となる。
        [Input]
            input_shape : [int,int]
                重みの Variable を初期化するための Tensor の形状
        [Output]
            正規分布に基づく乱数で初期化された重みの Variable 
            session.run(...) はされていない状態。
        """

        # ゼロで初期化すると、うまく重みの更新が出来ないので、正規分布に基づく乱数で初期化
        # tf.truncated_normal(...) : Tensor を正規分布なランダム値で初期化する
        init_tsr = tf.truncated_normal( shape = input_shape, stddev = 0.01 )

        # 重みの Variable
        weight_var = tf.Variable( init_tsr, name = name )
        
        return weight_var


    def init_bias_variable( self, input_shape, name = "init_bias_var" ):
        """
        バイアス項 b の初期化を行う。
        バイアス項は TensorFlow の Variable で定義することで、
        学習過程（最適化アルゴリズム Optimizer の session.run(...)）で自動的に TensorFlow により、変更される値となる。
        [Input]
            input_shape : [int,int]
                バイアス項の Variable を初期化するための Tensor の形状
        [Output]
            ゼロ初期化された重みの Variable 
            session.run(...) はされていない状態。
        """

        init_tsr = tf.random_normal( shape = input_shape )

        # バイアス項の Variable
        bias_var = tf.Variable( init_tsr, name = name )

        return bias_var


    def convolution_layer( 
            self, 
            input_tsr, 
            filter_height, filter_width,
            n_strides,
            n_output_channels, 
            name = "conv", reuse = False
        ):
        """
        畳み込み層を構築する。
        
        [Input]
            input_tsr : Tensor / Placeholder
                畳み込み層への入力 Tensor
            filter_height : int
                フィルターの高さ（カーネル行列の行数）
            filter_width : int
                フィルターの幅（カーネル行列の列数）
            n_strides : int
                畳み込み処理（特徴マップ生成）でストライドさせる pixel 数
            n_output_channels : int
                畳み込み処理後のデータのチャンネル数

        [Output]
            out_op : Operator
                畳み込み処理後の出力オペレーター
        """
        #print('@'+name+' layer')
        #print('Current input size in convolution layer is: '+str(input_tsr.get_shape().as_list()))

        # Variable の名前空間（スコープ定義）
        with tf.variable_scope( name, reuse = reuse ):
            # 入力データ（画像）のチャンネル数取得
            input_shape = input_tsr.get_shape().as_list()
            n_input_channels = input_shape[-1]

            # 畳み込み層の重み（カーネル）を追加
            # この重みは、畳み込み処理の画像データに対するフィルタ処理（特徴マップ生成）に使うカーネルを表す Tensor のことである。
            # kernel_shape : [ [(filterの高さ) , (filterの幅) , (入力チャネル数) , (出力チャネル数) ]
            kernel = self.init_weight_variable( input_shape = [filter_height, filter_width, n_input_channels, n_output_channels] )
            bias = self.init_bias_variable( input_shape = [n_output_channels] )

            # 畳み込み演算
            conv_op = tf.nn.conv2d(
                          input = input_tsr,
                          filter = kernel,
                          strides = [1, n_strides, n_strides, 1],   # strides[0] = strides[3] = 1. とする必要がある
                          padding = "SAME",
                          name = name
                      )
            
            # 活性化関数として Relu で出力
            out_op = tf.nn.relu( tf.add(conv_op,bias) )

        #print('    ===> output size is: '+str(out_op.get_shape().as_list()))

        return out_op


    def pooling_layer( self, input_tsr, name = "pool", reuse = False ):
        """
        VGG16 のプーリング層を構築する。

        [Input]
            input_tsr : Tensor / Placeholder
                畳み込み層への入力 Tensor
        [Output]
            pool_op : Operator
                プーリング処理後の出力オペレーター
        """
        # Variable の名前空間（スコープ定義）
        with tf.variable_scope( name, reuse = reuse ):
            # Max Pooling 演算
            pool_op = tf.nn.max_pool(
                          value = input_tsr,
                          ksize = [1, 2, 2, 1],
                          strides = [1, 2, 2, 1],
                          padding = "SAME",
                          name = name
                      )

        return pool_op


    def model( self ):
        """
        モデルの定義（計算グラフの構築）を行い、
        最終的なモデルの出力のオペレーターを設定する。
        [Output]
            self._y_out_op : Operator
                モデルの出力のオペレーター
        """   
        #-----------------------------------------------------------------------------
        # ベースネットワーク
        #-----------------------------------------------------------------------------
        self.base_vgg16.model()

        #-----------------------------------------------------------------------------
        # layer 6
        #-----------------------------------------------------------------------------
        self.conv6_op = self.convolution_layer( 
                            input_tsr = self.base_vgg16._y_out_op, 
                            filter_height = 3, filter_width = 3,
                            n_strides = 1,
                            n_output_channels = 1024,
                            name = "conv6", 
                            reuse = False
                        )

        self.pool6_op = self.pooling_layer( input_tsr = self.conv6_op, name = "pool6", reuse = False )

        #-----------------------------------------------------------------------------
        # layer 7
        #-----------------------------------------------------------------------------
        self.conv7_op = self.convolution_layer( 
                            input_tsr = self.pool6_op, 
                            filter_height = 1, filter_width = 1,
                            n_strides = 1,
                            n_output_channels = 1024,
                            name = "conv7", 
                            reuse = False
                        )

        #-----------------------------------------------------------------------------
        # layer 8
        #-----------------------------------------------------------------------------
        self.conv8_1_op = self.convolution_layer( 
                              input_tsr = self.conv7_op, 
                              filter_height = 1, filter_width = 1,
                              n_strides = 1,
                              n_output_channels = 256,
                              name = "conv8_1", 
                              reuse = False
                          )

        self.conv8_2_op = self.convolution_layer( 
                              input_tsr = self.conv8_1_op, 
                              filter_height = 3, filter_width = 3,
                              n_strides = 2,
                              n_output_channels = 512,
                              name = "conv8_2", 
                              reuse = False
                          )

        #-----------------------------------------------------------------------------
        # layer 9
        #-----------------------------------------------------------------------------
        self.conv9_1_op = self.convolution_layer( 
                              input_tsr = self.conv8_2_op, 
                              filter_height = 1, filter_width = 1,
                              n_strides = 1,
                              n_output_channels = 128,
                              name = "conv9_1", 
                              reuse = False
                          )

        self.conv9_2_op = self.convolution_layer( 
                              input_tsr = self.conv9_1_op, 
                              filter_height = 3, filter_width = 3,
                              n_strides = 2,
                              n_output_channels = 256,
                              name = "conv9_2", 
                              reuse = False
                          )

        #-----------------------------------------------------------------------------
        # layer 10
        #-----------------------------------------------------------------------------
        self.conv10_1_op = self.convolution_layer( 
                               input_tsr = self.conv9_2_op, 
                               filter_height = 1, filter_width = 1,
                               n_strides = 1,
                               n_output_channels = 128,
                               name = "conv10_1", 
                               reuse = False
                           )

        self.conv10_2_op = self.convolution_layer( 
                              input_tsr = self.conv10_1_op, 
                              filter_height = 3, filter_width = 3,
                              n_strides = 2,
                              n_output_channels = 256,
                              name = "conv10_2", 
                              reuse = False
                          )

        #-----------------------------------------------------------------------------
        # layer 11
        #-----------------------------------------------------------------------------
        self.conv11_1_op = self.convolution_layer( 
                               input_tsr = self.conv10_2_op, 
                               filter_height = 1, filter_width = 1,
                               n_strides = 1,
                               n_output_channels = 128,
                               name = "conv11_1", 
                               reuse = False
                           )

        self.conv11_2_op = self.convolution_layer( 
                              input_tsr = self.conv11_1_op, 
                              filter_height = 3, filter_width = 3,
                              n_strides = 3,
                              n_output_channels = 256,
                              name = "conv11_2", 
                              reuse = False
                          )

        #-----------------------------------------------------------------------------
        # Extra Feature Maps （アーキテクチャ図の青線部分＜各層 → Detections per Classes＞）
        #-----------------------------------------------------------------------------
        self.fmaps = []

        # extra feature map 1
        self.fmaps.append( 
            self.convolution_layer(
                input_tsr = self.base_vgg16._y_out_op,
                filter_height = 3, filter_width = 3,
                n_strides = 1,
                n_output_channels = self.n_boxes[0] * ( self.n_classes + 4 ),
                name = "fmap1", 
                reuse = False
            )
        )

        # extra feature map 2
        self.fmaps.append( 
            self.convolution_layer(
                input_tsr = self.conv7_op,
                filter_height = 3, filter_width = 3,
                n_strides = 1,
                n_output_channels = self.n_boxes[1] * ( self.n_classes + 4 ),
                name = "fmap2", 
                reuse = False
            )
        )

        # extra feature map 3
        self.fmaps.append( 
            self.convolution_layer(
                input_tsr = self.conv8_2_op,
                filter_height = 3, filter_width = 3,
                n_strides = 1,
                n_output_channels = self.n_boxes[2] * ( self.n_classes + 4 ),
                name = "fmap3", 
                reuse = False
            )
        )

        # extra feature map 4
        self.fmaps.append( 
            self.convolution_layer(
                input_tsr = self.conv9_2_op,
                filter_height = 3, filter_width = 3,
                n_strides = 1,
                n_output_channels = self.n_boxes[3] * ( self.n_classes + 4 ),
                name = "fmap4", 
                reuse = False
            )
        )

        # extra feature map 5
        self.fmaps.append( 
            self.convolution_layer(
                input_tsr = self.conv10_2_op,
                filter_height = 3, filter_width = 3,
                n_strides = 1,
                n_output_channels = self.n_boxes[4] * ( self.n_classes + 4 ),
                name = "fmap5", 
                reuse = False
            )
        )

        # extra feature map 6
        self.fmaps.append( 
            self.convolution_layer(
                input_tsr = self.conv11_2_op,
                filter_height = 1, filter_width = 1,
                n_strides = 1,
                n_output_channels = self.n_boxes[5] * ( self.n_classes + 4 ),
                name = "fmap6", 
                reuse = False
            )
        )

        #-----------------------------------------------------------------------------
        # extra feature maps による物体の所属クラスとスコア値の算出
        #-----------------------------------------------------------------------------
        fmaps_reshaped = []
        for i, fmap in zip( range(len(self.fmaps)), self.fmaps ):
            # [batch_size=None, image_height, image_width, n_channles]
            output_shape = fmap.get_shape().as_list()
            
            # extra feature map の高さ、幅
            fmap_height = output_shape[1]
            fmap_width = output_shape[2]
            
            # [batch_size=None, image_height, image_width, n_channles] → [batch_size=None, xxx, self.n_classes + 4 ] に　reshape
            fmap_reshaped = tf.reshape( fmap, [-1, fmap_width * fmap_height * self.n_boxes[i], self.n_classes + 4] )
            print( "fmap_reshaped[%d] : %s" % ( i, fmap_reshaped ) )

            #
            fmaps_reshaped.append( fmap_reshaped )

        # reshape した fmap を結合
        fmap_concatenated = tf.concat( fmaps_reshaped, axis=1 )
        print( "fmap_concatenated :", fmap_concatenated )     # Tensor("concat:0", shape=(?, 8752, 25), dtype=float32)

        # 特徴マップが含む物体の確信度と予想位置（形状のオフセット）
        self.pred_confidences = fmap_concatenated[ :, :, :self.n_classes ]
        self.pred_locations = fmap_concatenated[ :, :, self.n_classes: ]
        #print( 'confidences: ' + str( self.pred_confidences.get_shape().as_list() ) )   # [None, 8752, 21]
        #print( 'locations: ' + str( self.pred_locations.get_shape().as_list() ) )       # [None, 8752, 4]


        #-----------------------------------------------------------------------------
        # model output
        #-----------------------------------------------------------------------------
        self._y_out_op = self.conv11_2_op

        #return self.fmaps, self.pred_confidences, self.pred_locations
        return self._y_out_op


    def generate_default_boxes_in_fmaps( self ):
        """
        各 extra feature map に対応したデフォルトボックスを生成する。

        [Output]
            self._default_boxes : DefaultBoxes
                生成した 一連のデフォルトボックス群を表すクラス DefaultBoxes のオブジェクト
        """
        fmap_shapes = [ fmap.get_shape().as_list() for fmap in self.fmaps ]
        print( 'fmap shapes is ' + str(fmap_shapes) )

        # 一連のデフォルトボックス群を表すクラス DefaultBoxes のオブジェクトを生成
        self._default_boxes = DefaultBoxes(
                                  n_fmaps = len( self.fmaps ), fmap_shapes = fmap_shapes,
                                  scale_min = 0.2, scale_max = 0.9
                              )
        
        return self._default_boxes


    def loss( self, nnLoss ):
        """
        損失関数（誤差関数、コスト関数）の定義を行う。
        SSD の損失関数は、位置特定誤差（loc）と確信度誤差（conf）の重み付き和であり、
        （SSD の学習は、複数の物体カテゴリーを扱うことを考慮して行われるため２つの線形和をとる。）以下の式で与えられる。
        Loss = (Loss_conf + a*Loss_loc) / N

        [Input]
            nnLoss : NNLoss クラスのオブジェクト
            
        [Output]
            self._loss_op : Operator
                損失関数を表すオペレーター
        """
        def smooth_L1( self, x ):
            """
            smooth L1 loss func

            Args: the list of x range
            Returns: result tensor
            """
            sml1 = tf.multiply( 0.5, tf.pow(x, 2.0) )
            sml2 = tf.subtract( tf.abs(x), 0.5 )
            cond = tf.less( tf.abs(x), 1.0 )

            return tf.where( cond, sml1, sml2 )


        # 生成したデフォルトボックスの総数
        total_boxes = len( self._default_boxes._default_boxes )
        print( "total_boxes", total_boxes )     # 9700

        #---------------------------------------------------------------------------
        # 各種 Placeholder の生成
        #---------------------------------------------------------------------------
        # ground truth label （正解ボックスの所属クラス）の placeholder
        gt_labels_holder = tf.placeholder( shape = [None, total_boxes], dtype = tf.int32 )

        # ground truth boxes （正解ボックス）の placeholder
        gt_boxes_holder = tf.placeholder( shape = [None, total_boxes, 4], dtype = tf.float32 )

        # positive (デフォルトボックスと正解ボックスのマッチングが正) list の placeholder
        # negative (デフォルトボックスと正解ボックスのマッチングが負) list の placeholder
        pos_holder = tf.placeholder( shape = [None, total_boxes], dtype = tf.float32 )
        neg_holder = tf.placeholder( shape = [None, total_boxes], dtype = tf.float32 )

        #---------------------------------------------------------------------------
        # 位置特定誤差 L_loc
        #---------------------------------------------------------------------------

        #---------------------------------------------------------------------------
        # 確信度誤差 L_conf
        #---------------------------------------------------------------------------

        #---------------------------------------------------------------------------
        # 合計誤差 L
        #---------------------------------------------------------------------------

        # required placeholder for loss
        #self._loss_op, loss_conf, loss_loc, self.pos, self.neg, self.gt_labels, self.gt_boxes = self.ssd.loss(len(self.dboxes))

        #self._loss_op = nnLoss.loss( t_holder = self._t_holder, y_out_op = self._y_out_op )

        return self._loss_op


    def optimizer(self, nnOptimizer):
        """
        モデルの最適化アルゴリズムの設定を行う。（抽象メソッド）

        [Input]
            nnOptimizer : NNOptimizer のクラスのオブジェクト

        [Output]
            optimizer の train_step
        """
        self._optimizer = nnOptimizer._optimizer
        self._train_step = nnOptimizer.train_step( self._loss_op )
        
        return self._train_step



