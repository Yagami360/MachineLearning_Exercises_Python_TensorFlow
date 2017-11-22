## TensorFlow を用いた CNN-StyleNet / NeuralStyle による画像生成の実装

TensorFlow を用いた CNN-StyleNet / NeuralStyle による画像生成の練習用実装コード。<br>

この README.md ファイルには、各コードの実行結果、概要、CNN の背景理論の説明を記載しています。<br>
分かりやすいように `main.py` ファイル毎に１つの完結した実行コードにしています。

StyleNet / NeuralStyle（ニューラルスタイル）は、１つ目の画像から「画像スタイル」を学習し、２つ目の画像の「構造（内容）を維持」した上で、１つ目の画像スタイルを２つ目の画像に適用可能な手法である。

これは、一部の CNN に「２種類の中間層」が存在するという特性に基いている。<br>
この２種類の中間層とは、「画像スタイルをエンコード（符号化）するような中間層」と「画像内容をエンコード（符号化）するような中間層」である。<br>
この２つの中間層（それぞれ、スタイル層、内容層と名付ける）に対して、スタイル画像と内容画像でトレーニングし、従来のニューラルネットワークと同様にして、損失関数値をバックプロパゲーションすれば、２つの画像を合成した新たな画像を生成出来る。<br>
そして、この CNN モデルの構築には、事前に学習された CNN モデルを使用する。

- 元論文「A Neural Algorithm of Artistic Style」
    - https://arxiv.org/abs/1508.06576

- 参考サイト
    - https://research.preferred.jp/2015/09/chainer-gogh/
    
### 項目 [Contents]

1. [使用するライブラリ](#ID_1)
1. [使用するデータセット](#ID_2)
1. [コードの説明＆実行結果](#ID_3)
    1. [CNN-StyleNet / NeuralStyle（ニューラルスタイル）による画像生成処理 : `main1.py`](#ID_3-1)
1. [背景理論](#ID_4)


<a id="ID_1"></a>

## 使用するライブラリ

> TensorFlow ライブラリ </br>
>> 

> Scipy ライブラリ
>> `scipy.misc.imread(...)` : Read an image from a file as an array.
>>> https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.imread.html<br>
>> `scipy.io.loadmat` : MATLAB 用のオブジェクトファイルを読み込む
>>> https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.io.loadmat.html

> その他ライブラリ
>>


<a id="ID_2"></a>

### 使用するデータセット

- 学習済み CNN モデルのデータ : MATLAB オブジェクトファイル
    - [imagenet-vgg-verydee-19.mat]( http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat)

> 

<a id="ID_3"></a>

## コードの説明＆実行結果

<a id="ID_3-1"></a>

## CNN-StyleNet / NeuralStyle（ニューラルスタイル）による画像生成処理 : `main1.py`

- まずは、学習済みの CNN モデルのデータ `imagenet-vgg-verydee-19.mat` を詠み込む。
- 




<br>

---

<a id="#背景理論"></a>

## 背景理論
