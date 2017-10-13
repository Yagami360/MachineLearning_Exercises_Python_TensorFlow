# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境

"""
    更新情報
    [17/08/16] : 検証用のサンプルデータセット生成関数を追加
    [17/08/31] : クラス名を DataPreProcess → MLDreProcess に改名
"""

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
    def generateMoonsDataSet( input_n_samples = 100, input_random_state = 123 ):
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

        dat_X, dat_y = make_moons( n_samples = input_n_samples, random_state = input_random_state )

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
    