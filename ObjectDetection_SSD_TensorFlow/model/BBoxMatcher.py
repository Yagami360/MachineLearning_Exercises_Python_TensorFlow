# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境 
# + TensorFlow 1.4.0 インストール済み
# + OpenCV 3.3.1 インストール済み

"""
    更新情報
    [18/05/18] : 新規作成
    [xx/xx/xx] : 

"""

import numpy as np

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops

from model.DefaultBox import DefaultBoxSet
from model.BoundingBox import BoundingBox 


class BBoxMatcher( object ):
    """
    デフォルトボックスと正解ボックスのマッチング戦略を表すクラス。

    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
        _n_claases : int
            判別するクラス数（ラベル数）

        _default_box_set : list <DefaultBoxSet>
            一連のデフォルトボックスのリスト

    [protedted] protedted な使用法を想定 


    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( 
            self, 
            n_classes,
            default_box_set 
        ):
        self._n_classes = n_classes
        self._default_box_set = default_box_set

        return



    def print( self, str = None ):
        print( "----------------------------------" )
        print( str )
        print( self )

        print( "_n_classes :", self._n_classes )
        print( "_default_box_set :", self._default_box_set )
        print( "----------------------------------" )

        return

    def calc_jaccard( self, rect1, rect2 ):
        """
        指定した２つの長方形 rect の jaccard overlap 値を計算する

        Jaccard index.
        Jaccard index is defined as #(A∧B) / #(A∨B)
        """

        def intersection(rect1, rect2):
            """
            intersecton of units
            compute boarder line top, left, right and bottom.
            rect is defined as [ top_left_x, top_left_y, width, height ]
            """
            top = max(rect1[1], rect2[1])
            left = max(rect1[0], rect2[0])
            right = min(rect1[0] + rect1[2], rect2[0] + rect2[2])
            bottom = min(rect1[1] + rect1[3], rect2[1] + rect2[3])

            if bottom > top and right > left:
                return (bottom-top)*(right-left)

            return 0

        rect1_ = [x if x >= 0 else 0 for x in rect1]
        rect2_ = [x if x >= 0 else 0 for x in rect2]
        s = rect1_[2] * rect1_[3] + rect2_[2] * rect2_[3]

        # rect1 and rect2 => A∧B
        intersect = intersection( rect1_, rect2_ )

        # rect1 or rect2 => A∨B
        union = s - intersect

        # A∧B / A∨B
        return intersect / union


    def extract_highest_indicies( self, pred_confs, max_length ):
        """
        extract specific indicies, that is, have most high loss_confs.

        Args:
            pred_confs: predicated confidences
            max_length: max length of extracted indicies (in here, pos*3)
        Returns:
            extracted indicies of boxes (confidences).
        """
        loss_confs = []

        for pred_conf in pred_confs:
            # クラスの確信度 pred_conf から、? pred を計算
            pred = np.exp( pred_conf ) / ( np.sum(np.exp(pred_conf)) + 1e-5 )
            
            # ?
            loss_confs.append( np.amax(pred) )

        # ?
        size = min( len(loss_confs), max_length )
        
        # ?
        indicies = np.argpartition( loss_confs, -size )[-size:]

        return indicies


    def match( self, pred_confs, pred_locs, actual_labels, actual_locs ):
        """
       （デフォルトボックスと正解ボックスの）マッチング戦略を選択する。
        具体的には、訓練では、どのデフォルトボックスが正解ボックスとなるのか決定する必要があり、その結果を元にネットワークを学習させるが、
        各正解ボックスは、座標位置、アスペクト比、スケール値が異なる幾つかのデフォルトボックスから選択するが、
        これらデフォルトボックスに対して、jaccard overlap （下図）の最良値（最もエリアが重複している）で、
        各正解ボックスのマッチ度（エリアの重複度）を算出することになる。

        provides matching method

        match default boxes and bouding boxes.
        matching computes pos and neg count for the computation of loss.
        now, the most noting point is that it is not important that whether class label is correctly predicted.
        class label loss is evaled by loss_conf

        matches variable have some Box instance and most of None.
        if jaccard >= 0.5, that matches box has Box(gt_loc, gt_label).
        then, sort by pred_confs loss and extract 3*pos boxes, which they have Box([], classes) => background.

        when compute losses, we need transformed ground truth labels and locations
        because each box has self confidence and location.
        so, we should prepare expanded labels and locations whose size is as same as len(matches).

        [Input]
            pred_confs: list<ndarry>
                predicated confidences
            pred_locs: list<ndarry>
                predicated locations
            actual_labels: list<int>
                answer class labels
            actual_locs: list<nadarry>
                answer box locations
                 
        [Output]
            postive_list: if pos -> 1 else -> 0
            negative_list: if neg and label is not classes(not unknown class) 1 else 0
            expanded_gt_labels: gt_label if pos else classes
            expanded_gt_locs: gt_locs if pos else [0, 0, 0, 0]
        """
        n_pos = 0                   # 正解とマッチする BBOX 数
        n_neg = 0                   # 該当する所属クラスなしの BBOX 数
        pos_list = []               #
        neg_list = []               #
        expanded_gt_labels = []     #
        expanded_gt_locs = []       #
        bboxes_matched = []         # マッチングした bounding box のリスト
        bboxes_label_matched = []   # マッチングした bounding box のラベルのリスト

        # バウンディングボックスのリストを初期化（サイズは、一連のデフォルトボックスの合計数）
        for i in range( len(self._default_box_set._default_boxes) ):
            bboxes_matched.append( None )

        #-------------------------------------------------------------------
        # 各デフォルトボックスに対して、jaccard overlap 値を算出し、
        # 正解と判定される場合に、そのバウンディングボックス作成
        #-------------------------------------------------------------------
        for gt_label, gt_box in zip( actual_labels, actual_locs ):

            for i in range( len(bboxes_matched) ):
                dbox_rect = [ 
                                self._default_box_set._default_boxes[i]._center_x, 
                                self._default_box_set._default_boxes[i]._center_y,
                                self._default_box_set._default_boxes[i]._height,
                                self._default_box_set._default_boxes[i]._width
                            ]

                jacc = self.calc_jaccard( gt_box,  dbox_rect )
                
                # jaccard overlap が 0.5 の値よりも大きいデフォルトボックスを正解ボックスと判定させ、学習させる。
                # これにより、正解ボックスに複数に重なり合っているデフォルトボックスについて、高いスコア予想が可能になる。
                if( jacc >= 0.5 ):
                    # （正解ボックスとマッチする）バウンディングボックスを生成
                    bboxes_matched[i] = BoundingBox( label = gt_label, rect_loc = gt_box )

                    # マッチ数加算
                    n_pos += 1

                    # マッチしたラベル追加
                    bboxes_label_matched.append( gt_label )

        
        #-------------------------------------------------------------------
        # 各デフォルトボックスに対して、非該当
        # 非該当クラスなしのバウンディングボックス作成
        #-------------------------------------------------------------------
        # ?
        neg_pos = 5

        # ? クラスの確信度が上位のインデックスを取得
        indicies = self.extract_highest_indicies( pred_confs, n_pos * neg_pos )

        for i in indicies:
            if( n_neg > n_pos * neg_pos ):
                    break

            # 該当する所属クラスなしの場合
            if( bboxes_matched[i] is None and self._n_classes-1 != np.argmax(pred_confs[i]) ):
                # label = n_class - 1 の BBOX 作成
                bboxes_matched[i] = BoundingBox( label = self._n_classes - 1, rect_loc= [] )

                # 非該当数増加
                n_neg += 1

        #-------------------------------------------------------------------
        # 生成した各バウンディングボックスに対し、
        #-------------------------------------------------------------------
        # バウンディングボックス数分のループ
        for box in bboxes_matched:
            # if box is None
            # => Neither positive nor negative
            if box is None:
                pos_list.append(0)
                neg_list.append(0)
                expanded_gt_labels.append( self._n_classes - 1 )
                expanded_gt_locs.append( [0] * 4 )

            # if box's loc is empty
            # => Negative
            elif 0 == len( box._rect_loc ):
                pos_list.append(0)
                neg_list.append(1)
                expanded_gt_labels.append( self._n_classes - 1 )
                expanded_gt_locs.append( [0] * 4 )

            # if box's loc is specified
            # => Positive
            else:
                pos_list.append(1)
                neg_list.append(0)
                expanded_gt_labels.append( box._label)
                expanded_gt_locs.append( box._rect_loc )


        return pos_list, neg_list, expanded_gt_labels, expanded_gt_locs



