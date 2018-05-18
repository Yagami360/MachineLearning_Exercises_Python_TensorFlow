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
from model.DefaultBox import DefaultBoxSet
from model.BoundingBox import BoundingBox
from model.BBoxMatcher import BBoxMatcher


class SingleShotMultiBoxDetector( NeuralNetworkBase ):
    """
    SSD [Single Shot muitibox Detector] を表すクラス。

    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
        _default_box_set : list<DefaultBoxes>
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
            epochs = 20,
            batch_size = 50,
            eval_step = 1,
            save_step = 100,
            image_height = 32,
            image_width = 32,
            n_channels = 1,
            n_classes = 21,
            n_boxes = [ 4, 6, 6, 6, 6, 6 ]
        ):
        
        super().__init__( session )

        # 各パラメータの初期化
        self._epochs = epochs
        self._batch_size = batch_size
        self._eval_step = eval_step
        self._save_step = save_step

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
        self._default_box_set = None
        self._matcher = None

        #
        self._losses_train = []

        # 各種 Placeholder
        # ground truth label （正解ボックスの所属クラス）の placeholder
        self.gt_labels_holder = None

        # ground truth boxes （正解ボックス）の placeholder
        self.gt_boxes_holder = None

        # positive (デフォルトボックスと正解ボックスのマッチングが正) list の placeholder
        # negative (デフォルトボックスと正解ボックスのマッチングが負) list の placeholder
        self.pos_holder = None
        self.neg_holder = None

        return


    def print( self, str ):
        print( "----------------------------------" )
        print( str )
        print( self )

        print( "_session :", self._session )
        print( "_init_var_op : \n", self._init_var_op )
        print( "_y_out_op :", self._y_out_op )

        print( "_epoches : ", self._epochs )
        print( "_batch_size : ", self._batch_size )
        print( "_eval_step : ", self._eval_step )
        print( "_save_step : ", self._save_step )

        print( "image_height : " , self.image_height )
        print( "image_width : " , self.image_width )
        print( "n_channels : " , self.n_channels )
        
        print( "n_classes", self.n_classes )
        print( "n_boxes", self.n_boxes )

        print( "_loss_op :", self._loss_op )
        print( "_optimizer :", self._optimizer )
        print( "_train_step :", self._train_step )

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

        print( "_default_box_set :", self._default_box_set )
        if( self._default_box_set != None ):
            print( "_default_box_set._n_fmaps :", self._default_box_set._n_fmaps )
            print( "total boxes :", len( self._default_box_set._default_boxes ) )
            #self._default_box_set.print()

        print( "gt_labels_holder :", self.gt_labels_holder )
        print( "gt_boxes_holder :", self.gt_boxes_holder )
        print( "pos_holder :", self.pos_holder )
        print( "neg_holder :", self.neg_holder )

        print( "_losses_train", self._losses_train )
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
        # Tensor("concat:0", shape=(?, 8752, 25), dtype=float32)
        # 25 = 21(クラス数) + 4( (xmin, ymin, xmax, ymax) の 4 次元の情報で物体を囲む矩形の位置 )
        fmap_concatenated = tf.concat( fmaps_reshaped, axis = 1 )
        print( "fmap_concatenated :", fmap_concatenated )

        # 特徴マップが含む物体の確信度と予想位置（形状のオフセット）
        # pred_confidences.shape = [None, 8752, 21] | 21: クラス数
        # pred_locations.shape = [None, 8752, 4]  | 4 : (xmin, ymin, xmax, ymax) の 4 次元の情報で物体を囲む矩形の位置
        self.pred_confidences = fmap_concatenated[ :, :, :self.n_classes ]
        self.pred_locations = fmap_concatenated[ :, :, self.n_classes: ]
        #print( 'confidences: ' + str( self.pred_confidences.get_shape().as_list() ) )
        #print( 'locations: ' + str( self.pred_locations.get_shape().as_list() ) )

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
            self._default_box_set : DefaultBoxSet
                生成した 一連のデフォルトボックス群を表すクラス DefaultBoxSet のオブジェクト
        """
        # extra feature map の形状（ピクセル単位）
        fmap_shapes = [ fmap.get_shape().as_list() for fmap in self.fmaps ]
        print( 'fmap shapes is ' + str(fmap_shapes) )

        # 各 extra feature maps に対応した、各デフォルトボックスのアスペクト比
        #aspects = [ 1.0, 2.0, 3.0, 1.0/2.0, 1.0/3.0 ]
        aspect_set = [
                         [1.0, 1.0, 2.0, 1.0/2.0],                 # extra fmap 1
                         [1.0, 1.0, 2.0, 1.0/2.0, 3.0, 1.0/3.0],   # extra fmap 2
                         [1.0, 1.0, 2.0, 1.0/2.0, 3.0, 1.0/3.0],   #
                         [1.0, 1.0, 2.0, 1.0/2.0, 3.0, 1.0/3.0],
                         [1.0, 1.0, 2.0, 1.0/2.0, 3.0, 1.0/3.0],
                         [1.0, 1.0, 2.0, 1.0/2.0, 3.0, 1.0/3.0],
                     ]

        # 一連のデフォルトボックス群を表すクラス DefaultBoxSet のオブジェクトを生成
        self._default_box_set = DefaultBoxSet( scale_min = 0.2, scale_max = 0.9 )
        
        # 一連のデフォルトボックス群を生成
        self._default_box_set.generate_boxes( fmaps_shapes = fmap_shapes, aspect_set = aspect_set )

        return self._default_box_set


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
        def smooth_L1( x ):
            """
            smooth L1 loss func

            smoothL1 = 0.5 * x^2 ( if |x| < 1 )
                     = |x| -0.5 (otherwise)
            """
            # 0.5 * x^2
            sml1 = tf.multiply( 0.5, tf.pow(x, 2.0) )

            # |x| - 0.5
            sml2 = tf.subtract( tf.abs(x), 0.5 )
            
            # 条件 : |x| < 1
            cond = tf.less( tf.abs(x), 1.0 )

            return tf.where( cond, sml1, sml2 )

        # 生成したデフォルトボックスの総数
        total_boxes = len( self._default_box_set._default_boxes )
        #print( "total_boxes", total_boxes )     # 8752

        #---------------------------------------------------------------------------
        # 各種 Placeholder の生成
        #---------------------------------------------------------------------------
        # ground truth label （正解ボックスの所属クラス）の placeholder
        self.gt_labels_holder = tf.placeholder( shape = [None, total_boxes], dtype = tf.int32, name = "gt_labels_holder" )

        # ground truth boxes （正解ボックス）の placeholder
        self.gt_boxes_holder = tf.placeholder( shape = [None, total_boxes, 4], dtype = tf.float32, name = "gt_boxes_holder"  )

        # positive (デフォルトボックスと正解ボックスのマッチングが正) list の placeholder
        # negative (デフォルトボックスと正解ボックスのマッチングが負) list の placeholder
        self.pos_holder = tf.placeholder( shape = [None, total_boxes], dtype = tf.float32, name = "pos_holder"  )
        self.neg_holder = tf.placeholder( shape = [None, total_boxes], dtype = tf.float32, name = "neg_holder"  )

        #---------------------------------------------------------------------------
        # 位置特定誤差 L_loc
        # L_loc = Σ_(i∈pos) Σ_(m) { x_ij^k * smoothL1( predbox_i^m - gtbox_j^m ) }
        #---------------------------------------------------------------------------
        smoothL1_op = smooth_L1( x = ( self.gt_boxes_holder - self.pred_locations ) )
        # ?
        loss_loc_op = tf.reduce_sum( smoothL1_op, reduction_indices = 2 ) * self.pos_holder
        
        # ?
        loss_loc_op = tf.reduce_sum( loss_loc_op, reduction_indices = 1 ) / ( 1e-5 + tf.reduce_sum( self.pos_holder, reduction_indices = 1 ) )
        
        #---------------------------------------------------------------------------
        # 確信度誤差 L_conf
        # L_conf = Σ_(i∈pos) { x_ij^k * log( softmax(c) ) }, c = カテゴリ、ラベル
        #---------------------------------------------------------------------------
        # ?
        loss_conf_op = tf.nn.sparse_softmax_cross_entropy_with_logits( 
                           logits = self.pred_confidences, 
                           labels = self.gt_labels_holder 
                       )

        loss_conf_op = loss_conf_op * ( self.pos_holder + self.neg_holder )
        
        # ?
        loss_conf_op = tf.reduce_sum( loss_conf_op, reduction_indices = 1 ) / ( 1e-5 + tf.reduce_sum( ( self.pos_holder + self.neg_holder ), reduction_indices = 1) )

        #---------------------------------------------------------------------------
        # 合計誤差 L
        #---------------------------------------------------------------------------
        self._loss_op = tf.reduce_sum( loss_conf_op + loss_loc_op )

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


    def fit( self, X_train, y_train ):
        """
        指定されたトレーニングデータで、モデルの fitting 処理を行う。
        [Input]
            X_train : list ( shape = [n_samples, (image,h,w,c)] )
                トレーニングデータ（特徴行列）
            
            y_train : numpy.ndarray ( shape = [n_samples] )
                トレーニングデータ用のクラスラベル（教師データ）のリスト
        [Output]
            self : 自身のオブジェクト
        """
        def generate_minibatch( X, y, batch_size, bSuffle = True, random_seed = 12 ):
            """
            指定された（トレーニング）データから、ミニバッチ毎のデータを生成する。
            （各 Epoch 処理毎に呼び出されることを想定している。）
            """
            # 各 Epoch 度に shuffle し直す。
            if( bSuffle == True ):
                idxes = np.arange( len(y) )   # 0 ~ y.shape[0] の連番 idxes を生成

                # メルセンヌツイスターというアルゴリズムを使った擬似乱数生成器。
                # コンストラクタに乱数の種(シード)を設定。
                random_state = np.random.RandomState( random_seed )
                random_state.shuffle( idxes )
                
                # shuffle された連番 idxes 値のデータに置き換える。
                X_ = [] 
                y_ = []
                for idx in idxes:
                    X_.append( X[idx] )
                    y_.append( y[idx] )

            # 0 ~ 行数まで batch_size 間隔でループ
            for i in range( 0, len(X_), batch_size ):
                # mini batch data
                batch_X_ = X_[i:i+batch_size]
                batch_y_ = y_[i:i+batch_size]

                # yield 文で逐次データを return（関数の処理を一旦停止し、値を返す）
                # メモリ効率向上のための処理
                yield ( batch_X_, batch_y_ )

        #----------------------------------------------------------
        # 学習開始処理
        #----------------------------------------------------------
        # Variable の初期化オペレーター
        self._init_var_op = tf.global_variables_initializer()

        # Session の run（初期化オペレーター）
        self._session.run( self._init_var_op )

        # ミニバッチの繰り返し回数
        n_batches = len( X_train ) // self._batch_size       # バッチ処理の回数
        n_minibatch_iterations = self._epochs * n_batches    # ミニバッチの総繰り返し回数
        n_minibatch_iteration = 0                            # ミニバッチの現在の繰り返し回数
        
        print( "n_batches :", n_batches )
        print( "n_minibatch_iterations :", n_minibatch_iterations )

        # （学習済みモデルの）チェックポイントファイルの作成
        #self.save_model()

        #----------------------------------------------------------
        # eval 項目の設定
        #----------------------------------------------------------
        self._matcher = BBoxMatcher( n_classes = self.n_classes, default_box_set = self._default_box_set )

        positives = []      # self.pos_holder に供給するデータ : 正解ボックスとデフォルトボックスの一致
        negatives = []      # self.neg_holder に供給するデータ : 正解ボックスとデフォルトボックスの不一致
        ex_gt_labels = []   # self.gt_labels_holder に供給するデータ : 正解ボックスの所属クラスのラベル
        ex_gt_boxes = []    # self.gt_boxes_holder に供給するデータ : 正解ボックス

        #----------------------------------------------------------
        # 学習処理
        #----------------------------------------------------------
        # for ループでエポック数分トレーニング
        for epoch in range( 1, self._epochs+1 ):
            # ミニバッチサイズ単位で for ループ
            # エポック毎に shuffle し直す。
            gen_minibatch = generate_minibatch( X = X_train, y = y_train , batch_size = self._batch_size, bSuffle = True, random_seed = 12 )

            # n_batches = X_train.shape[0] // self._batch_size 回のループ
            for i ,(batch_x, batch_y) in enumerate( gen_minibatch, 1 ):
                n_minibatch_iteration += 1

                #-------------------------------------------------------------------------------------
                # 特徴マップに含まれる物体のクラス所属の確信度、長方形位置を取得
                #-------------------------------------------------------------------------------------
                f_maps, pred_confs, pred_locs = self._session.run(
                                                    [ self.fmaps, self.pred_confidences, self.pred_locations ], 
                                                    feed_dict = { self.base_vgg16.X_holder: batch_x }
                                                )

                #print( "fmaps :", f_maps )
                #print( "pred_confs :", pred_confs )
                #print( "pred_locs :", pred_locs )

                # batch_size 文のループ
                for i in range( len(batch_x) ):
                    actual_labels = []
                    actual_loc_rects = []

                    #-------------------------------------------------------------------------------------
                    # 教師データの物体のクラス所属の確信度、長方形位置のフォーマットを変換
                    #-------------------------------------------------------------------------------------                    
                    # 教師データから物体のクラス所属の確信度、長方形位置情報を取り出し
                    # 画像に存在する物体の数分ループ処理
                    for obj in batch_y[i]:
                        # 長方形の位置情報を取り出し
                        loc_rect = obj[:4]

                        # 所属クラス情報を取り出し＆ argmax でクラス推定
                        label = np.argmax( obj[4:] )

                        # 位置情報のフォーマットをコンバート
                        # [ top_left_x, top_left_y, bottom_right_x, bottom_right_y ] → [ top_left_x, top_left_y, width, height ]
                        # [ top_left_x, top_left_y, width, height ] → [ center_x, center_y, width, height ]
                        loc_rect = np.array( [ loc_rect[0], loc_rect[1], loc_rect[2]-loc_rect[0], loc_rect[3]-loc_rect[1] ] )
                        loc_rect = np.array( [ loc_rect[0] - loc_rect[2] * 0.5, loc_rect[1] - loc_rect[3] * 0.5, abs(loc_rect[2]), abs(loc_rect[3]) ] )

                        #
                        actual_loc_rects.append( loc_rect )
                        actual_labels.append( label )

                    #-------------------------------------------------------------------------------------
                    # デフォルトボックスと正解ボックスのマッチング処理（マッチング戦略）
                    #-------------------------------------------------------------------------------------
                    pos_list, neg_list, expanded_gt_labels, expanded_gt_locs = self._matcher.match( 
                                                                                   pred_confs, pred_locs, actual_labels, actual_loc_rects
                                                                               )

                    # マッチング結果を追加
                    positives.append( pos_list )
                    negatives.append( neg_list )
                    ex_gt_labels.append( expanded_gt_labels )
                    ex_gt_boxes.append( expanded_gt_locs )

                #-------------------------------------------------------------------------------------
                # 設定された最適化アルゴリズム Optimizer でトレーニング処理を run
                #-------------------------------------------------------------------------------------
                loss, _, = self._session.run(
                               [ self._loss_op, self._train_step ],
                               feed_dict = {
                                   self.base_vgg16.X_holder: batch_x,
                                   self.pos_holder: positives,
                                   self.neg_holder: negatives,
                                   self.gt_labels_holder: ex_gt_labels,
                                   self.gt_boxes_holder: ex_gt_boxes

                               }
                           )

                self._losses_train.append( loss )

                print( "Epoch: %d/%d | minibatch iteration: %d/%d | loss = %0.5f |" % 
                      ( epoch, self._epochs, n_minibatch_iteration, n_minibatch_iterations, loss ) )

                # モデルの保存処理を行う loop か否か
                # % : 割り算の余りが 0 で判断
                if ( ( (n_minibatch_iteration) % self._save_step ) == 0 ):
                    self.save_model()

        # fitting 処理終了後、モデルのパラメータを保存しておく。
        self.save_model()

        return self._y_out_op

