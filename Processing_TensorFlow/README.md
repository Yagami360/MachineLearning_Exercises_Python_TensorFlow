## TensorFlow ライブラリの処理フローの練習コード集


### 項目 [Contents]

1. [TensorFlow での全体ワークフロー](#全体ワークフロー)
1. [使用するライブラリ](#使用するライブラリ)
1. [使用するデータセット](#使用するデータセット)
1. [コードの実行結果](#コードの実行結果)

    1. [テンソルの設定、及び計算グラフとセッション](#テンソルの設定、及び計算グラフとセッション)
    1. [変数とプレースホルダ、及び計算グラフとセッション](#変数とプレースホルダ、及び計算グラフとセッション)
    1. [計算グラフ](#計算グラフ)
    1. [](#)

## 全体ワークフロー : `main1.py`

基本的に、計算グラフという有向グラフを構築し、それをセッションで動かすという処理フローになる。</br>
これは、MATLAB Simulink とよく似たモデルベースな設計。但し、MATLAB Simulink とは異なりモデル作成のための GUI はないので、頭の中（コード）で計算グラフを考えながら設計していくスタイルになる。</br>
（TensorBoard でコードの結果の計算グラフを可視化することは可能。）

- データセットを読み込み or 生成する。
- データを変換、正規化する。 </br>
  （例）正規化処理 : `data = tf.nn.batch_norm_with_global_normalization(...)`
- データセットをトレーニングデータ、テストデータ、検証データセットに分割する。
- アルゴリズム（モデル）のパラメータ（ハイパーパラメータ）を設定する。
- 変数とプレースホルダを設定
    - TensorFlow は、損失関数を最小化するための最適化において、</br>
      変数と重みベクトルを変更 or 調整する。
    - この変更や調整を実現するためには、</br>
      "プレースホルダ [placeholder]" を通じてデータを供給（フィード）する必要がある。
    - そして、これらの変数とプレースホルダと型について初期化する必要がある。
- モデルの構造を定義する。</br>
  データを取得 or 設定し、変数とプレースホルダを初期化した後は、</br>
  モデルを定義する必要がある。これは、計算グラフを作成するという方法で行う。 </br>
    （例）線形モデル : `y_predict = tf.add( tf.mul( x_input, weight_matrix), b_matrix )`
- 損失関数を設定する。
- モデルの初期化と学習（トレーニング）
    - ここまでの準備で, 実際に, 計算グラフ（有向グラフ）のオブジェクトを作成し,</br>プレースホルダを通じて, データを計算グラフ（有向グラフ）に供給する。</br>
    - そして、構築した計算グラフは　セッション [Session] により、その処理を実行する。</br>
    （例） 
    ```
    with tf.Session( graph = graph ) as session:
        ...
        session.run(...)
    ```
- モデルの評価 (Optional)
- ハイパーパラメータのチューニング (Optional)
- デプロイと新しい成果指標の予想 (Optional)

</br>

<a name="#使用するライブラリ"></a>

### 使用するライブラリ

> TensorFlow ライブラリ </br>
>> API 集 </br>
https://www.tensorflow.org/api_docs/python/ </br>

>> Session 関連 </br>
>>> `tf.Session` : A class for running TensorFlow operations. </br>
https://www.tensorflow.org/api_docs/python/tf/Session </br>
>>> `tf.Session.close()` : Session を close する。</br>
https://www.tensorflow.org/api_docs/python/tf/Session#close </br>

>> Variable関連 </br>
>>> `tf.Variable` : </br>
https://www.tensorflow.org/api_docs/python/tf/Variable </br>

>> Tensor の生成関連 </br>
>>> 固定のテンソル </br>
>>>> `tf.zeros(...)` : 全ての要素が 0 からなる Tensor を作成する。</br>
https://www.tensorflow.org/api_docs/python/tf/zeros </br>
>>>> `tf.ones(...)` : 全ての要素が 1 からなる Tensor を作成する。</br>
https://www.tensorflow.org/api_docs/python/tf/ones </br>
>>>> `tf.fill(...)` : </br>
https://www.tensorflow.org/api_docs/python/tf/fill </br>
>>>> `tf.(...)` : </br>
https://www.tensorflow.org/api_docs/python/tf/ </br>

>>> シーケンステンソル </br>
>>>> `` : </br>
>>>> `` : </br>


> その他ライブラリ
>> scikit-learn ライブラリ


<a name="#使用するデータセット"></a>

### 使用するデータセット


<a name="#コードの実行結果"></a>

## コードの実行結果

<a name="#全体ワークフロー"></a>


<a name="#テンソルの設定、及び計算グラフとセッション"></a>

## テンソルの設定、及び計算グラフとセッション : `main2.py`
> コード実装中...

TensorFlow におけるテンソルとは、単に多次元配列であるが、計算グラフに対して何かしらのグラフ構造を与えると言う点で、他の多次元配列とは異なる。</br>
但し、ここで注意すべきことは、Tensor を設定したからといって、TensorFlow が計算グラフに何かしらを追加するわけではないということである。</br>
TensorFlow が計算グラフに何かを追加するのは Tensor が作成されて、利用可能になった後に限られる。

- 固定のテンソル
    - `tf.zeros(...)` : 全ての要素が 0 からなる Tensor を作成する。</br>
    
    （例）Session を生成せず、そのまま Tensor を print()　</br> 
    ```
    zero_tsr = tf.zeros( [3, 2] )
    print( zero_tsr )
    
    <出力>
    Tensor("zeros:0", shape=(3, 2), dtype=float32)
    → Tensor オブジェクト（Tensor 型）がそのまま出力されている。
    これは session を実行していないため。
    ```
    （例）Session を生成＆run() して、Tensor を print()　</br>
    ```
    zero_tsr = tf.zeros( [3, 2] )
    session = tf.Session()
    print( session.run( zero_tsr ) )
    
    <出力>
    [[ 0.  0.]
    [ 0.  0.]
    [ 0.  0.]]
    → 全ての要素が 0 からなるテンソル（この場合、２次配列）が出力されている。
    ```
    - `tf.ones(...)` : 全て 1 の要素からなる Tensor を作成する。</br>
    ```
    ones_tsr = tf.ones( [3, 2] )
    print(  " tf.ones(...) の Tensor 型 : ", ones_tsr )

    session = tf.Session()
    print( "tf.ones(...) の value : \n", session.run( ones_tsr ) )
    session.close()

    <出力>
    tf.ones(...) の Tensor 型 : Tensor("ones:0", shape=(3, 2), dtype=float32)
    tf.ones(...) の value :
    [[ 1.  1.]
    [ 1.  1.]
    [ 1.  1.]]
    ```
    - `tf.fill(...)` : 指定した定数で埋められた Tensor を作成する。
    ```
    filled_tsr = tf.fill( [3, 2], "const" )
    print( "tf.fill(...) の Tensor 型 : ", filled_tsr )

    session = tf.Session()
    print( "tf.fill(...) の value : ", session.run( filled_tsr ) )
    session.close()

    <出力>
    tf.fill(...) の Tensor 型 :  Tensor("Fill:0", shape=(3, 2), dtype=string)
    tf.fill(...) の value :  
    [[b'const' b'const']
    [b'const' b'const']
    [b'const' b'const']]
    ```

    - `tf.constant(...)` : 指定した既存の定数から Tensor を作成する。
    ```
    const_tsr = tf.constant( [1,2,3] )
    print( "tf.constant(...) の Tensor 型 : ", const_tsr )

    session = tf.Session()
    print( "tf.constant(...) の value \n: ", session.run( const_tsr ) )
    session.close()

    <出力>
    tf.constant(...) の Tensor 型 :  Tensor("Const:0", shape=(3,), dtype=int32)
    tf.constant(...) の value :  [1 2 3]
    ```

- シーケンステンソル
    - `tf.linespace(...)` : 
    ```
    liner_tsr = tf.linspace( start = 0.0, stop = 1.0, num = 3 )
    print( "tf.linspace(...) の Tensor 型 : ", liner_tsr )

    session = tf.Session()
    print( "tf.linspace(...) の value : \n", session.run( liner_tsr ) )
    session.close()

    <出力>
    tf.linspace(...) の Tensor 型 :  Tensor("LinSpace:0", shape=(3,), dtype=float32)
    tf.linspace(...) の value : 
    [ 0.   0.5  1. ]
    ```

- ランダムステンソル



<a name="#変数とプレースホルダの設定、及び計算グラフとセッション"></a>

## 変数とプレースホルダの設定、及び計算グラフとセッション : `main2.py`
> コード実装中...
