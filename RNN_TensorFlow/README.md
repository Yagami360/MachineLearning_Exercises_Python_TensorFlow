## TensorFlow を用いたリカレントニューラルネットワーク（RNN）の実装と簡単な応用

TensorFlow を用いた、リカレントニューラルネットワーク（RNN）による時系列モデルの予想、画像識別、自然言語処理の練習用実装コード集。

この README.md ファイルには、各コードの実行結果、概要、RNN の背景理論の説明を記載しています。
分かりやすいように main.py ファイル毎に１つの完結した実行コードにしています。

### 項目 [Contents]

1. [使用するライブラリ](#ID_1)
1. [使用するデータセット](#ID_2)
1. [コードの説明＆実行結果](#ID_3)
    1. [RNN によるノイズ付き sin 波形（時系列データ）からの波形の予想（生成）処理 : `main1.py`](#ID_3-1)
    1. LSTM による sin 波形（時系列データ）の生成処理 : `main2.py`
    1. LSTM による Adding Problem : `main3.py`
    1. GNU による sin 波形（時系列データ）の生成処理 : `main4.py`
    1. 双方向 RNN による MNIST データセットの識別処理 : `main5.py`
    1. RNN Encoder-Decoder による自然言語処理（足し算の応答） : `main6.py` 
1. [背景理論](#ID_4)
    1. [リカレントニューラルネットワーク [RNN : Recursive Neural Network]<br>＜階層型ニューラルネットワーク＞](#ID_5)
        1. [リカレントニューラルネットワークのアーキテクチャの種類](#ID_5-1)
            1. [隠れ層間で回帰構造をもつネットワーク](#ID_5-1-1)
                1. [通時的誤差逆伝搬法 [BPTT : back-propagation through time]](#ID_5-1-1-1)
        1. [長・短期記憶（LSTM [long short-term memory]）モデル](#ID_5-2)
            1. [CEC [constant error carousel]](#ID_5-2-1)
            1. [重み衝突 [weight conflict] と入力ゲート [input gate]、出力ゲート [output gate]](#ID_5-2-2)
            1. [忘却ゲート [forget gate]](#ID_5-2-3)
            1. [覗き穴結合 [peephole connections]](#ID_5-2-4)
            1. [LSTM モデルの定式化](#ID_5-2-5)
        1. [GRU [gated recurrent unit]](#ID_5-3)
        1. [双方向 RNN [BiRNN : Bidirectional RNN]](#ID_5-4)
        1. [RNN Encoder-Decoder (Seqenence-to-sequence models)](#ID_5-5)


<a id="#ID_1"></a>

### 使用するライブラリ

> TensorFlow ライブラリ <br>
>> 

> その他ライブラリ
>>

<br>

<a id="#ID_2"></a>

### 使用するデータセット

> ノイズ付き sin 波形（時系列データ）

<br>

<a id="#ID_3"></a>

## コードの説明＆実行結果

<a id="#ID_3-1"></a>

## RNN によるノイズ付き sin 波形（時系列データ）からの波形の予想（生成）処理 : `main1.py`
> 実装中...

RNN による時系列モデルの取り扱いの簡単な例として、まず、ノイズ付き sin 波形の近似を考える。


- 以下のような、ノイズ付き sin 波形を生成する。<br>
    - 周期 `T = 100`, ノイズ幅 `noize_size = 0.05` 
![rnn_1-1](https://user-images.githubusercontent.com/25688193/33363817-48d2f754-d525-11e7-84f4-218d2601667e.png)
    ```python
    [MLPreProcess.py]
    @staticmethod
    def generate_sin_with_noize( t, T = 100, noize_size = 0.05, seed = 12 ):
        """
        ノイズ付き sin 波形（時系列データ）を生成する
        [Input]
            t : array
                時間のリスト
            T : float
                波形の周期
            noize_size : float
                ノイズ幅の乗数
        """
        numpy.random.seed( seed = seed )

        # numpy.random.uniform(...) : 一様分布に従う乱数
        noize = noize_size * numpy.random.uniform( low = -1.0, high = 1.0, size = len(t),  )
        sin = numpy.sin( 2.0 * numpy.pi * (t / T) )

        X_features = t
        y_labels = sin + noize

        return X_features, y_labels
    ```
    ```python
    [main1.py]
    T1 = 100                # ノイズ付き sin 波形の周期
    noize_size1 = 0.05      # ノイズ付き sin 波形のノイズ幅
    times = numpy.arange( 2.5 * T1 + 1 )    # 時間 t の配列
    X_features, y_labels = MLPreProcess.generate_sin_with_noize( t = times, T = T1, noize_size = noize_size1, seed = 12 )
    ```
- 通時的誤差逆伝搬法 [BPTT : back-propagation through time] での計算負荷の関係上、
時系列データを一定間隔に区切る。
    - 全時系列データの長さ : `len_sequences = len( X_features )`
    - １つの時系列データの長さ τ : `len_one_sequence = 25`

<!--
- 特徴行列 `X_features` は、特徴数 x 個 × サンプル数 x 個 :<br> `X_features = `
- 教師データ `y_labels` は、サンプル数 x 個 : <br >`y_labels = `
- トレーニングデータ xx% 、テストデータ xx% の割合で分割 : <br>`sklearn.cross_validation.train_test_split( test_size = , random_state =  )`
- 
-->

<br>

<a id="#ID_3-2"></a>

## 実行結果２ : `main2.py`
> 実装中...

<br>

---

<a id="#ID_4"></a>

## 背景理論

<a id="ID_5"></a>

## リカレントニューラルネットワーク [RNN : Recursive Neural Network]<br>＜階層型ニューラルネットワーク＞の概要
![image](https://user-images.githubusercontent.com/25688193/30980712-f06a0906-a4bc-11e7-9b15-4c46834dd6d2.png)
![image](https://user-images.githubusercontent.com/25688193/30981066-22f53124-a4be-11e7-9111-9514f04aed7c.png)

<a id="ID_5-1"></a>

### リカレントニューラルネットワーク（RNN）のアーキテクチャの種類

<a id="ID_5-1-1"></a>

#### 隠れ層間で回帰構造をもつネットワーク
![image](https://user-images.githubusercontent.com/25688193/31013864-0baf82cc-a553-11e7-9296-870f600f0381.png)
![image](https://user-images.githubusercontent.com/25688193/31013877-185db822-a553-11e7-9c5f-625acace78f8.png)
![image](https://user-images.githubusercontent.com/25688193/31016204-bcbc0cb0-a55e-11e7-86df-3ba574fa8bc2.png)
![image](https://user-images.githubusercontent.com/25688193/31017867-f379d312-a564-11e7-9d67-fc79a7dda26d.png)

<a id="ID_5-1-1-1"></a>

##### 通時的誤差逆伝搬法 [BPTT : back-propagation through time]
![image](https://user-images.githubusercontent.com/25688193/31018664-dacf44d4-a567-11e7-8014-34523646bfca.png)
![image](https://user-images.githubusercontent.com/25688193/31019688-7725288c-a56b-11e7-919d-d0be44d4be33.png)

<a id="ID_5-1-1-1-1"></a>

###### 通時的誤差逆伝搬法によるパラメータの更新式（誤差関数が最小２乗誤差）
![image](https://user-images.githubusercontent.com/25688193/31189494-64bc3a80-a973-11e7-90a8-348e97f93f47.png)
![image](https://user-images.githubusercontent.com/25688193/31189294-c48db61a-a972-11e7-9673-a7805c53eaf5.png)
![image](https://user-images.githubusercontent.com/25688193/31190398-f0e5337a-a975-11e7-8eff-ff74adf3a6ff.png)
![image](https://user-images.githubusercontent.com/25688193/31190919-835e326e-a977-11e7-966e-d3675cb83452.png)
![image](https://user-images.githubusercontent.com/25688193/31211718-661ae058-a9d6-11e7-96ae-075f35981fd1.png)


<a id="ID_5-2"></a>

### 長・短期記憶（LSTM [long short-term memory]）モデル

<a id="ID_5-2-1"></a>

#### CEC [constant error carousel]
![image](https://user-images.githubusercontent.com/25688193/31226189-2d62a892-aa10-11e7-93e5-b32902d83702.png)
![image](https://user-images.githubusercontent.com/25688193/31226163-0eb9927a-aa10-11e7-9d06-306e4443c5a8.png)
![image](https://user-images.githubusercontent.com/25688193/31235831-6fa44284-aa2d-11e7-9377-845ea30837c5.png)
![image](https://user-images.githubusercontent.com/25688193/31226906-eb4288bc-aa12-11e7-9f16-621ed4d50063.png)

<a id="ID_5-2-2"></a>

#### 重み衝突 [weight conflict] と入力ゲート [input gate]、出力ゲート [output gate]
![image](https://user-images.githubusercontent.com/25688193/31236796-16687124-aa30-11e7-89b5-2da158274de7.png)
![image](https://user-images.githubusercontent.com/25688193/31246908-ed52d18e-aa49-11e7-946f-44f3fa177eb3.png)
![image](https://user-images.githubusercontent.com/25688193/31246932-fa855dc2-aa49-11e7-882d-462dd22be03d.png)

<a id="ID_5-2-3"></a>

#### 忘却ゲート [forget gate]
![image](https://user-images.githubusercontent.com/25688193/31247911-036bc036-aa4d-11e7-9f5f-117eaab0b738.png)
![image](https://user-images.githubusercontent.com/25688193/31247928-130b98b8-aa4d-11e7-89aa-ac27b1667666.png)
![image](https://user-images.githubusercontent.com/25688193/31248855-2cf3eb7e-aa50-11e7-99b7-4c81a093f679.png)
![image](https://user-images.githubusercontent.com/25688193/31249125-2453757e-aa51-11e7-9ce2-715edddf8232.png)

<a id="ID_5-2-4"></a>

#### 覗き穴結合 [peephole connections]
![image](https://user-images.githubusercontent.com/25688193/31272328-83122b86-aac5-11e7-84db-6a52bd8d2c44.png)
![image](https://user-images.githubusercontent.com/25688193/31272347-8f9d67bc-aac5-11e7-9fda-640bdb6a9d7f.png)
![image](https://user-images.githubusercontent.com/25688193/31279596-941088d2-aae4-11e7-9e30-dc28771800c4.png)

<a id="ID_5-2-5"></a>

#### LSTM モデルの定式化
![image](https://user-images.githubusercontent.com/25688193/31278352-91da316c-aadf-11e7-8ad6-963e7e235852.png)
![image](https://user-images.githubusercontent.com/25688193/31283264-169b4f16-aaf0-11e7-9f19-976dc2e09bc9.png)
![image](https://user-images.githubusercontent.com/25688193/31284097-8a2e6e84-aaf2-11e7-8e7d-df00110c5bf6.png)
![image](https://user-images.githubusercontent.com/25688193/31293857-b20586f6-ab13-11e7-85b2-460f9bab5e62.png)
![image](https://user-images.githubusercontent.com/25688193/31294053-706d763a-ab14-11e7-8aed-1fed8327d58c.png)


<a id="ID_5-3"></a>

### GRU [gated recurrent unit]
![image](https://user-images.githubusercontent.com/25688193/31338072-e514030c-ad38-11e7-908c-2446c32b60c6.png)
![image](https://user-images.githubusercontent.com/25688193/31306417-cfa02a3c-ab8a-11e7-8fb1-0579fe5aa0be.png)
![image](https://user-images.githubusercontent.com/25688193/31307146-b1ce34fa-ab98-11e7-862a-b139d330222e.png)
![image](https://user-images.githubusercontent.com/25688193/31308026-2bd77ff2-abaa-11e7-967e-04cff1579a36.png)


<a id="ID_5-4"></a>

### 双方向 RNN [BiRNN : Bidirectional RNN]
![image](https://user-images.githubusercontent.com/25688193/31332068-edadd682-ad1f-11e7-9f11-e7374b83465e.png)
![image](https://user-images.githubusercontent.com/25688193/31334064-78437f7a-ad27-11e7-84f2-decd65d1599d.png)
![image](https://user-images.githubusercontent.com/25688193/31335870-68a806d2-ad2f-11e7-9cd2-36648536cc64.png)
![image](https://user-images.githubusercontent.com/25688193/31335226-9d1c925a-ad2c-11e7-8f79-dccd9d931c41.png)
![image](https://user-images.githubusercontent.com/25688193/31335735-d0a5b780-ad2e-11e7-82ae-17cd33f2546c.png)


<a id="ID_5-5"></a>

### RNN Encoder-Decoder (Seqenence-to-sequence models)
![image](https://user-images.githubusercontent.com/25688193/31340555-7cd2efac-ad41-11e7-85f0-d70f0f9c7bee.png)
![image](https://user-images.githubusercontent.com/25688193/31370123-203bf512-adc4-11e7-8bc1-d65df760a43f.png)
![image](https://user-images.githubusercontent.com/25688193/31370130-2c510356-adc4-11e7-9a59-d2b93cfa4698.png)
![image](https://user-images.githubusercontent.com/25688193/31370139-372bbfd2-adc4-11e7-965c-96bc88661505.png)
![image](https://user-images.githubusercontent.com/25688193/31371878-45210ec6-adce-11e7-9096-3bbd77dee065.png)
![image](https://user-images.githubusercontent.com/25688193/31376678-b29f4ff0-ade0-11e7-9988-88602f28b32c.png)



