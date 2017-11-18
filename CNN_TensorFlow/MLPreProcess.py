# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境

"""
    更新情報
    [17/08/16] : 検証用のサンプルデータセット生成関数を追加
    [17/08/31] : クラス名を DataPreProcess → MLDreProcess に改名
    [17/10/21] : アヤメデータの読み込み関数 load_iris(...) を追加
    [17/10/28] : MNIST データの読み込み関数 load_mnist(...) を追加
    [17/11/17] : CIFAR-10 データの読み込み関数 load_cifar10(...) を追加
               : CIFAR-10 データのランダムに加工した読み込み関数 load_cifar10_with_trasform(...) を追加
"""

import os
import sys

import struct
import numpy

# Data Frame & IO 関連
import pandas
from io import StringIO

# scikit-learn ライブラリ関連
from sklearn import datasets                            # scikit-learn ライブラリのデータセット群
from sklearn.datasets import make_moons                 # 半月状のデータセット生成
from sklearn.datasets import make_circles               # 同心円状のデータセット生成

#from sklearn.cross_validation import train_test_split  # scikit-learn の train_test_split関数の old-version
from sklearn.model_selection import train_test_split    # scikit-learn の train_test_split関数の new-version
from sklearn.metrics import accuracy_score              # 正解率、誤識別率の計算用に使用

from sklearn.preprocessing import Imputer               # データ（欠損値）の保管用に使用
from sklearn.preprocessing import LabelEncoder          # 
from sklearn.preprocessing import OneHotEncoder         # One-hot encoding 用に使用
from sklearn.preprocessing import MinMaxScaler          # scikit-learn の preprocessing モジュールの MinMaxScaler クラス
from sklearn.preprocessing import StandardScaler        # scikit-learn の preprocessing モジュールの StandardScaler クラス

from sklearn.pipeline import Pipeline                   # パイプライン

# TensorFlow ライブラリ関連
import tensorflow as tf


class MLPreProcess( object ):
    """
    機械学習用のデータの前処理を行うクラス
    データフレームとして, pandas DataFrame のオブジェクトを持つ。（コンポジション：集約）
    sklearn.preprocessing モジュールのラッパークラス
    
    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
        df_ : pandas DataFrame のオブジェクト（データフレーム）
    
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """

    def __init__( self, dataFrame = pandas.DataFrame() ):
        """
        コンストラクタ（厳密にはイニシャライザ）

        [Input]
            dataFrame : pandas DataFrame オブジェクト
        """
        self.df_ = dataFrame
        return

    def print( self, str = '' ):
        print("\n")
        print("-------------------------------------------------------------------")
        print( str )
        print("\n")
        print("<pandas DataFrame> \n")
        #print( "rows, colums :", self.df_.shape )
        print( self.df_ )
        print( self )
        print("-------------------------------------------------------------------")
        return

    def setDataFrameFromList( self, list ):
        """
        [Input]
            list : list

        """
        self.df_ = pandas.DataFrame( list )

        return self

    def setDataFrameFromDataFrame( self, dataFrame ):
        """
        [Input]
            dataFrame : pandas DataFrame のオブジェクト

        """
        self.df_ = dataFrame

        return self

    def setDataFrameFromCsvData( self, csv_data ):
        """
        csv フォーマットのデータを pandas DataFrame オブジェクトに変換して読み込む.

        [Input]
            csv_data : csv フォーマットのデータ
        """
        # read_csv() 関数を用いて, csv フォーマットのデータを pandas DataFrame オブジェクトに変換して読み込む.
        self.df_ = pandas.read_csv( StringIO( csv_data ) )
        return self

    def setDataFrameFromCsvFile( self, csv_fileName ):
        """
        csv ファイルからデータフレームを構築する

        [Input]
            csv_fileName : string
                csvファイルパス＋ファイル名
        """
        self.df_ = pandas.read_csv( csv_fileName, header = None )
        return self

    def getNumpyArray( self ):
        """
        pandas Data Frame オブジェクトを Numpy 配列にして返す
        """
        values = self.df_.values     # pandas DataFrame の value 属性
        return values

    #---------------------------------------------------------
    # 検証用サンプルデータセットを出力する関数群
    #---------------------------------------------------------
    @staticmethod
    def generateMoonsDataSet( input_n_samples = 100, input_noize = 0.1, input_random_state = 123 ):
        """
        半月形のデータセットを生成する。

        [Input]
            input_n_samples : int

            input_random_state : int
                
        [Output]
            dat_X : numpy.ndarray
                2次元 Numpy 配列
            dat_y : numpy.ndarray
                クラスラベル 0 or 1 (1次元 Numpy 配列)
        """

        dat_X, dat_y = make_moons( n_samples = input_n_samples, random_state = input_random_state, noise = input_noize )

        # 戻り値のオブジェクトの型確認
        #print( isinstance(dat_X, list) )
        #print( isinstance(dat_y, list) )
        #print( isinstance(dat_X, pandas.DataFrame) )
        #print( isinstance(dat_X, numpy.ndarray) )

        return dat_X, dat_y

    @staticmethod
    def generateCirclesDataSet( input_n_samples = 1000, input_random_state = 123, input_noize = 0.1, input_factor = 0.2 ):
        """
        同心円形のデータセットを生成する。

        [Input]
            input_n_samples : int

            input_random_state : int
                seed used by the random number generator
            input_noize : float
                Gaussian noise
            input_factor : float
                Scale factor between inner and outer circle.
        [Output]
            dat_X : numpy.ndarray
                2次元 Numpy 配列
            dat_y : numpy.ndarray
                クラスラベル 0 or 1 (1次元 Numpy 配列)

        """
        dat_X, dat_y = make_circles( 
                           n_samples = input_n_samples, random_state = input_random_state, 
                           noise = input_noize, 
                           factor = input_factor
                       )

        return dat_X, dat_y

    #---------------------------------------------------------
    # 検証用サンプルデータセットを読み込むする関数群
    #---------------------------------------------------------
    @staticmethod
    def load_iris():
        """
        アヤメデータを読み込む。

        [Output]
            X_features : numpy.ndarray
                特徴行列（2次元 Numpy 配列）
                150 × 4
            y_labels : numpy.ndarray
                Iris の品種 [] を表すクラスラベル 0 or 1 or 2 (1次元 Numpy 配列)
                0: 'setosa', 1: 'versicolor', 2: 'virginica'

        -------------------------------------------------------------------------------------
        pandas.DataFrame(iris.data, columns=iris.feature_names)
             sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
        0                  5.1               3.5                1.4               0.2
        1                  4.9               3.0                1.4               0.2
        2                  4.7               3.2                1.3               0.2
        3                  4.6               3.1                1.5               0.2
        4                  5.0               3.6                1.4               0.2
        5                  5.4               3.9                1.7               0.4
        6                  4.6               3.4                1.4               0.3
        7                  5.0               3.4                1.5               0.2
        8                  4.4               2.9                1.4               0.2
        9                  4.9               3.1                1.5               0.1
        10                 5.4               3.7                1.5               0.2
        ... (中略)
        149                5.9               3.0                5.1               1.8

        iris.target
        array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        """
        iris = datasets.load_iris()

        X_features = iris.data
        y_labels = iris.target

        return X_features, y_labels


    @staticmethod
    def load_mnist( path, kind = "train" ):
        """
        検証データ用の MNIST データを読み込む。
        [Input]
            path : str
                MNIST データセットが格納されているフォルダへのパス
            kind : str
                読み込みたいデータの種類（トレーニング用データ or テスト用データ）
                "train" : トレーニング用データ
                "t10k" : テスト用データ
        [Output]
            images : [n_samples = 60,000, n_features = 28*28 = 784]
                トレーニングデータ用の画像データ
            labels : [n_samples = 60,000,]
                トレーニングデータ用のラベルデータ（教師データ）
                0~9 の数字ラベル
        """
        # path の文字列を結合して、MIST データへのパスを作成
        # (kind = train) %s → train-images.idx3-ubyte, train-labels.idx1-ubyte
        # (kind = t10k)  %s → t10k-images.idx3-ubyte,  t10k-labels.idx1-ubyte
        labels_path = os.path.join( path, "%s-labels.idx1-ubyte" % kind )
        images_path = os.path.join( path, "%s-images.idx3-ubyte" % kind )

        #------------------------------------------
        # open() 関数と with 構文でラベルデータ（教師データ）の読み込み
        # "rb" : バイナリーモードで読み込み
        #------------------------------------------
        with open( labels_path, 'rb' ) as lbpath:
            # struct.unpack(...) : バイナリーデータを読み込み文字列に変換
            # magic : マジックナンバー （先頭 から 4byte）
            # num : サンプル数（magicの後の 4byte）
            magic, n = \
            struct.unpack(
                '>II',           # > : ビッグエンディアン, I : 符号なし整数, >II : 4byte + 4byte
                lbpath.read(8)   # 8byte
            )
            
            # numpy.fromfile(...) : numpy 配列にバイトデータを読み込んむ
            # dtype : numpy 配列のデータ形式
            labels = numpy.fromfile( file = lbpath, dtype = numpy.uint8 )

        #------------------------------------------
        # open() 関数と with 構文で画像データの読み込む
        # "rb" : バイナリーモードで読み込み
        #------------------------------------------
        with open( images_path, "rb" ) as imgpath:
            # struct.unpack(...) : バイナリーデータを読み込み文字列に変換
            # magic : マジックナンバー （先頭 から 4byte）
            # num : サンプル数（magicの後の 4byte）
            # rows : ?
            # cols : ?
            magic, num, rows, cols = \
            struct.unpack(
                ">IIII",           # > : ビッグエンディアン, I : 符号なし整数, >IIII : 4byte + 4byte + 4byte + 4byte
                imgpath.read(16)   # 16byte
            )

            # numpy.fromfile(...) : numpy 配列にバイトデータを読み込んでいき,
            # 読み込んだデータを shape = [labels, 784] に reshape
            images = numpy.fromfile( file = imgpath, dtype = numpy.uint8 )
            images = images.reshape( len(labels), 784 )
       
            
        return images, labels
    

    @staticmethod
    def load_cifar10( path, kind = "tain" ):
        """
        検証データ用の CIFAR-10 データを読み込む。
        バイナリ形式 : CIFAR-10 binary version (suitable for C programs)

        [Input]
            path : str
                CIFAR-10 データセットが格納されているフォルダへのパス
            kind : str
                読み込みたいデータの種類（トレーニング用データ or テスト用データ）
                "train" : トレーニング用データ
                "test" : テスト用データ
        """
        # 読み込みファイル名の設定
        # トレーニング用のデータ
        if( kind == "train" ):
            # data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5
            files = [ os.path.join( path, "data_batch_{}.bin".format(i) ) for i in range(1,6) ]
        # テスト用のデータ
        elif( kind == "test" ):
            # test_batch
            files = [ os.path.join( path, "test_batch.bin") ]
        else:
            files = [ os.path.join( path, "data_batch_{}.bin".format(i) ) for i in range(1,6) ]

        print( "files :", files )

        # 内部データサイズの設定 
        image_height = 32   # CIFAR-10 画像の高さ (pixel)
        image_width = 32    #
        n_channels = 3      # RGB の 3 チャンネル

        image_bytes = image_height * image_width * n_channels
        labels_byte = 1
        record_bytes = image_bytes + labels_byte

        images = []
        labels = []

        # トレーニング用のデータ
        if ( kind == "train" ):
            # data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5 に関しての loop
            for i in range( 5 ):
                #print( "i=" , i )
                # バイナリーモードでファイルオープン
                byte_stream = open( files[i], mode="rb" )

                # 全レコード長に関しての loop
                for record in range(10000):
                    #
                    byte_stream.seek( record_bytes * record , 0 )

                    # バッファに割り当て
                    label_buffer = numpy.frombuffer( byte_stream.read(labels_byte), dtype=numpy.uint8 )
                    image_buffer = numpy.frombuffer( byte_stream.read(image_bytes), dtype=numpy.uint8 )
                    image_buffer = image_buffer.astype( numpy.float32 )

                    images.append( image_buffer )
                    labels.append( label_buffer )
                    #images[record] = image_buffer

                # 
                byte_stream.close()

        # テスト用のデータ
        elif ( kind == "test" ):
            byte_stream = open( files[0], mode="rb" )
            # 全レコード長に関しての loop
            for record in range(10000):
                #
                byte_stream.seek( record_bytes * record , 0 )

                # バッファに割り当て
                label_buffer = numpy.frombuffer( byte_stream.read(labels_byte), dtype=numpy.uint8 )
                image_buffer = numpy.frombuffer( byte_stream.read(image_bytes), dtype=numpy.uint8 )
                image_buffer = image_buffer.astype( numpy.float32 )

                images.append( image_buffer )
                labels.append( label_buffer )
                #images[record] = image_buffer

            # 
            byte_stream.close()

        return images, labels


    @staticmethod
    def load_cifar10_tensorflow( path, kind = "tain", bReshape = False, bTensor = False ):
        """
        TensorFlow を用いて, 検証データ用の CIFAR-10 データを読み込む。
        バイナリ形式 : CIFAR-10 binary version (suitable for C programs)

        [Input]
            path : str
                CIFAR-10 データセットが格納されているフォルダへのパス
            kind : str
                読み込みたいデータの種類（トレーニング用データ or テスト用データ）
                "train" : トレーニング用データ
                "test" : テスト用データ
            bReshape : Bool
                image データの reshape を行うか否か
            bTensor : Bool
                戻り値を Tensor のままにするか否か
                
        [Output]
            images : Tensor / shape = [n_samples = 50,000, n_features = 3*32*32 = 3*1024]
                トレーニングデータ用の画像データ
                n_features = n_channels * image_height * image_width
            labels : Tensor / [n_samples = 10,000,]
                トレーニングデータ用のラベルデータ（教師データ）
                0~9 の数字ラベル
                0 : 飛行機 [airplane] 
                1 : 
                2 :
                ...
                7 : 馬 [horse]
                8 :
                9 : 
        """
        # 読み込みファイル名の設定
        # トレーニング用のデータ
        if( kind == "train" ):
            # data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5
            files = [ os.path.join( path, "data_batch_{}.bin".format(i) ) for i in range(1,6) ]
        # テスト用のデータ
        elif( kind == "test" ):
            # test_batch
            files = [ os.path.join( path, "test_batch") ]
        else:
            files = [ os.path.join( path, "data_batch_{}.bin".format(i) ) for i in range(1,6) ]

        print( "files :", files )

        # 内部データサイズの設定 
        image_height = 32   # CIFAR-10 画像の高さ (pixel)
        image_width = 32    #
        n_channels = 3      # RGB の 3 チャンネル

        image_size = image_height * image_width * n_channels
        record_size = 1 + image_size    # 1 は、ラベル (0~9) に対応する値
        
        # 固定の長さのバイトを読み取るレコードリーダーオブジェクトを作成
        reader = tf.FixedLengthRecordReader( record_bytes = record_size )
        #print( "reader :", reader)

        # tf.FixedLengthRecordReader.read(...) で key と string の Tensor を生成
        # Returns the next record (key, value) pair produced by a reader.
        # queue: A Queue or a mutable string Tensor representing a handle to a Queue, with string work items.
        # ファイル名（のキュー）を渡すことで、ファイルの内容（の一部（を表す tensor））が得られる
        filename_queue = tf.train.string_input_producer( files )     # 1ファイルでもキュー生成が必要
        key_tsr, record_str_tsr = reader.read( queue = filename_queue )
        # tf.decode_raw(...) : 文字列から uint8 Tensor に変換
        record_bytes_tsr = tf.decode_raw( record_str_tsr, tf.uint8 )
        #print( "filename_queue :", filename_queue )
        #print( "key_tsr :", key_tsr )
        #print( "record_str_tsr :", record_str_tsr )

        # ラベルを抽出
        # tf.slice(...) : Tensorの一部分を取り出す。beginで開始場所、sizeで切り出す大きさを指定する。
        labels = tf.cast( 
                     tf.slice( input_ = record_bytes_tsr, begin = [0], size = [1] ),         # ?
                     tf.int32 
                 )
        # 画像データを抽出
        images = tf.slice( input_ = record_bytes_tsr, begin = [1], size = [image_size] )     # ?
        if ( bReshape == True ):
            image = tf.reshape( image, [n_channels, image_height, image_width] )

        if( bTensor == False ):
            # Session を生成＆run() して、Tensor の実際の値を取得
            session = tf.Session()
            images = session.run( images )
            labels = session.run( labels )

        return images, labels


    @staticmethod
    def load_cifar10_with_transform_tensorflow( 
        path, 
        kind = "tain", 
        bTensor = False,
        crop_height = 24,
        crop_width = 24
    ):
        """
        TensorFlow を用いて, 検証データ用の CIFAR-10 データを読み込み、ランダムに加工する。
        バイナリ形式 : CIFAR-10 binary version (suitable for C programs)

        [Input]
            path : str
                CIFAR-10 データセットが格納されているフォルダへのパス
            kind : str
                読み込みたいデータの種類（トレーニング用データ or テスト用データ）
                "train" : トレーニング用データ
                "test" : テスト用データ
            bTensor : Bool
                戻り値を Tensor のままにするか否か

            crop_height : int
                画像加工時の画像のの切り取り高さ
            crop_width : int
                画像加工時の画像のの切り取り幅

        [Output]
            images : Tensor / shape = [n_samples = 50,000, n_features = 3*32*32 = 3*1024]
                トレーニングデータ用の画像データ
                n_features = n_channels * image_height * image_width
            labels : Tensor / [n_samples = 10,000,]
                トレーニングデータ用のラベルデータ（教師データ）
        """
        #tf.set_random_seed(12)

        # Tensor の状態のままの images, labels データを読み込み
        image, labels = load_cifar10_tensorflow( path, kind, bReshape = True, bTensor = True )

        # 画像を変形（転置）
        image = tf.transpose( image, [1,2,0] )  # [1,2,0] : ?
        image = tf.cast( image, tf.float32 )    # ?

        # 画像をランダムに切り取る
        image = tf.image.resize_image_with_crop_or_pad( 
                    image, 
                    target_height = crop_height,    # 切り取り高さ
                    target_width = crop_width       # 切り取り幅 
                )

        # 画像の左右をランダムに反転
        image = tf.image.random_flip_left_right( image )

        # 明るさをランダムに変更
        image = tf.image.random_brightness( image, max_delta = 63, seed = 12 )

        # コントラストをランダムに変更
        image = tf.image.random_contrast( image, lower = 0.2, upper = 1.8, seed = 12 )

        # 画像を正規化
        image = tf.image.per_image_standardization( image )

        if( bTensor == False ):
            # Session を生成＆run() して、Tensor の実際の値を取得
            session = tf.Session()
            images = session.run( images )
            labels = session.run( labels )

        return image, labels


    #---------------------------------------------------------
    # 欠損値の処理を行う関数群
    #---------------------------------------------------------
    def meanImputationNaN( self, axis = 0 ):
        """
        欠損値 [NaN] を平均値で補完する
        [Input]
            axis : int
                0 : NaN を列の平均値で補完
                1 : NaN を行の平均値で補完
        """
        imputer = Imputer( 
                      missing_values = 'NaN', 
                      strategy = 'mean', 
                      axis = axis       # 0 : 列の平均値, 1 : 行の平均値
                  )
        
        imputer.fit( self.df_ )         # self.df_ は１次配列に変換されることに注意

        self.df_ = imputer.transform( self.df_ )

        return self
    
    #---------------------------------------------------------
    # カテゴリデータの処理を行う関数群
    #---------------------------------------------------------
    def setColumns( self, columns ):
        """
        データフレームにコラム（列）を設定する。
        """
        self.df_.columns = columns
        
        return self
    
    def mappingOrdinalFeatures( self, key, input_dict ):
        """
        順序特徴量のマッピング（整数への変換）
        
        [Input]
            key : string
                順序特徴量を表すキー（文字列）

            dict : dictionary { "" : 1, "" : 2, ... }
        
        """
        self.df_[key] = self.df_[key].map( dict(input_dict) )   # 整数に変換

        return self

    def encodeClassLabel( self, key ):
        """
        クラスラベルを表す文字列を 0,1,2,.. の順に整数化する.（ディクショナリマッピング方式）

        [Input]
            key : string
                整数化したいクラスラベルの文字列
        """
        mapping = { label: idx for idx, label in enumerate( numpy.unique( self.df_[key]) ) }
        self.df_[key] = self.df_[key].map( mapping )

        return self

    def encodeClassLabelByLabelEncoder( self, colum, bPrint = True ):
        """
        クラスラベルを表す文字列を sklearn.preprocessing.LabelEncoder クラスを用いてエンコードする.

        [input]
            colum : int
                エンコードしたいクラスラベルが存在する列番号
            bPrint : bool
            エンコード対象を print するか否か
        """
        encoder = LabelEncoder()
        encoder.fit_transform( self.df_.loc[:, colum].values )  # ? fit_transform() の結果の再代入が必要？
        encoder.transform( encoder.classes_ )

        if ( bPrint == True):
            print( "encodeClassLabelByLabelEncoder() encoder.classes_ : ", encoder.classes_ )
            print( "encoder.transform", encoder.transform( encoder.classes_ ) )
            #print( "encodeClassLabelByLabelEncoder() encoder.classes_[0] : ", encoder.classes_[0] )
            #print( "encodeClassLabelByLabelEncoder() encoder.classes_[1] : ", encoder.classes_[1] )
        return self

    def oneHotEncode( self, categories, col ):
        """
        カテゴリデータ（名義特徴量, 順序特徴量）の One-hot Encoding を行う.

        [Input]
            categories : list
                カテゴリデータの list

            col : int
                特徴行列の変換する変数の列位置 : 0 ~

        """
        X_values = self.df_[categories].values    # カテゴリデータ（特徴行列）を抽出
        #print( X_values )
        #print( self.df_[categories] )

        # one-hot Encoder の生成
        ohEncode = OneHotEncoder( 
                      categorical_features = [col],    # 変換する変数の列位置：[0] = 特徴行列 X_values の最初の列
                      sparse = False                   # ?  False : 通常の行列を返すようにする。
                   )

        # one-hot Encoding を実行
        #self.df_ = ohEncode.fit_transform( X_values ).toarray()   # ? sparse = True の場合の処理
        self.df_ = pandas.get_dummies( self.df_[categories] )     # 文字列値を持つ行だけ数値に変換する
        
        return self

    #---------------------------------------------------------
    # データセットの分割を行う関数群
    #---------------------------------------------------------
    @staticmethod
    def dataTrainTestSplit( X_input, y_input, ratio_test = 0.3, input_random_state = 0 ):
        """
        データをトレーニングデータとテストデータに分割する。
        分割は, ランダムサンプリングで行う.

        [Input]
            X_input : Matrix (行と列からなる配列)
                特徴行列

            y_input : 配列
                教師データ

            ratio_test : float
                テストデータの割合 (0.0 ~ 1.0)

        [Output]
            X_train : トレーニングデータ用の Matrix (行と列からなる配列)
            X_test  : テストデータの Matrix (行と列からなる配列)
            y_train : トレーニングデータ用教師データ配列
            y_test  : テストデータ用教師データ配列
        """        
        X_train, X_test, y_train, y_test \
        = train_test_split(
            X_input,  y_input, 
            test_size = ratio_test, 
            random_state = input_random_state             # 
          )
        
        return X_train, X_test, y_train, y_test

    #---------------------------------------------------------
    # データのスケーリングを行う関数群
    #---------------------------------------------------------
    @staticmethod
    def normalizedTrainTest( X_train, X_test ):
        """
        指定したトレーニングデータ, テストにデータ（データフレーム）を正規化 [nomalize] する.
        ここでの正規化は, min-maxスケーリング [0,1] 範囲を指す.
        トレーニングデータは正規化だけでなく欠損値処理も行う.
        テストデータは,トレーニングデータに対する fit() の結果で, transform() を行う.

        [Input]
            X_train : トレーニングデータ用の Matrix (行と列からなる配列)
            X_test  : テストデータの Matrix (行と列からなる配列)

        [Output]
            X_train_norm : 正規化されたトレーニングデータ用の Matrix (行と列からなる配列)
            X_test_norm  : 正規化されたテストデータの Matrix (行と列からなる配列)
        """
        mms = MinMaxScaler()

        # fit_transform() : fit() を実施した後に, 同じデータに対して transform() を実施する。
        # トレーニングデータの場合は、それ自体の統計を基に正規化や欠損値処理を行っても問題ないので、fit_transform() を使って構わない。
        X_train_norm = mms.fit_transform( X_train )

        # transform() :  fit() で取得した統計情報を使って, 渡されたデータを実際に書き換える.
        # テストデータの場合は, 比較的データ数が少なく, トレーニングデータの統計を使って正規化や欠損値処理を行うべきなので,
        # トレーニングデータに対する fit() の結果で、transform() を行う必要がある。
        X_test_norm = mms.transform( X_test )

        return X_train_norm, X_test_norm
        
    @staticmethod
    def standardizeTrainTest( X_train, X_test ):
        """
        指定したトレーニングデータ, テストにデータ（データフレーム）を標準化 [standardize] する.
        ここでの標準化は, 平均値 : 0 , 分散値 : 1 への変換指す.
        トレーニングデータは標準化だけでなく欠損値処理も行う.
        テストデータは,トレーニングデータに対する fit() の結果で, transform() を行う.

        [Input]
            X_train : トレーニングデータ用の Matrix (行と列からなる配列)
            X_test  : テストデータの Matrix (行と列からなる配列)

        [Output]
            X_train_std : 標準化された [standardized] トレーニングデータ用の Matrix (行と列からなる配列)
            X_test_std  : 標準化された [standardized] テストデータの Matrix (行と列からなる配列)
        """
        stdsc = StandardScaler()

        # fit_transform() : fit() を実施した後に, 同じデータに対して transform() を実施する。
        # トレーニングデータの場合は, それ自体の統計を基に標準化や欠損値処理を行っても問題ないので, fit_transform() を使って構わない。
        X_train_std = stdsc.fit_transform( X_train )

        # transform() :  fit() で取得した統計情報を使って, 渡されたデータを実際に書き換える.
        # テストデータの場合は, 比較的データ数が少なく, トレーニングデータの統計を使って標準化や欠損値処理を行うべきなので,
        # トレーニングデータに対する fit() の結果で、transform() を行う必要がある。
        X_test_std = stdsc.transform( X_test )

        return X_train_std, X_test_std
    