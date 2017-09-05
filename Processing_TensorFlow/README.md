## TensorFlow ライブラリの処理フローの練習コード集


### 項目 [Contents]
1. [TensorFlow での全体ワークフロー](#全体ワークフロー)
1. [使用するライブラリ](#使用するライブラリ)
1. [使用するデータセット](#使用するデータセット)
1. [コードの実行結果](#コードの実行結果)
    1. [テンソルの設定、及び計算グラフとセッション](##テンソルの設定、及び計算グラフとセッション)
    1. [変数とプレースホルダ、及び Op ノードと計算グラフ](##コードの実行結果２)
    1. [行列の操作](#行列の操作)
    1. [](#)


<a name="#全体ワークフロー"></a>

## TensorFlow での全体ワークフロー : `main1.py`

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
    ```python
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

>> Op ノード </br>
>>> `tf.global_variables_initializer` : </br>
https://www.tensorflow.org/api_docs/python/tf/global_variables_initializer </br>

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

<a name="##テンソルの設定、及び計算グラフとセッション"></a>

## テンソルの設定、及び計算グラフとセッション : `main2.py`

TensorFlow におけるテンソルとは、単に多次元配列であるが、計算グラフに対して何かしらのグラフ構造を与えると言う点で、他の多次元配列とは異なる。</br>
但し、ここで注意すべきことは、Tensor を設定したからといって、TensorFlow が計算グラフに何かしらを追加するわけではないということである。</br>
TensorFlow が計算グラフに何かを追加するのは Tensor が作成されて、利用可能になった後に限られる。

- 固定のテンソル
    - `tf.zeros(...)` : 全ての要素が 0 からなる Tensor を作成する。</br>
    
    （例）Session を生成せず、そのまま Tensor を print()　</br> 
    ```python
    zero_tsr = tf.zeros( [3, 2] )
    print( zero_tsr )
    
    <出力>
    Tensor("zeros:0", shape=(3, 2), dtype=float32)
    → Tensor オブジェクト（Tensor 型）がそのまま出力されている。
    これは session を実行していないため。
    ```
    （例）Session を生成＆run() して、Tensor を print()　</br>
    ```python
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
    ```python
    ones_tsr = tf.ones( [3, 2] )
    print(  " tf.ones(...) の Tensor 型 : ", ones_tsr )

    session = tf.Session()
    print( "tf.ones(...) の value : \n", session.run( ones_tsr ) )
    session.close()
    ```
    ---
    ```python
    <出力>
    tf.ones(...) の Tensor 型 : Tensor("ones:0", shape=(3, 2), dtype=float32)
    tf.ones(...) の value :
    [[ 1.  1.]
    [ 1.  1.]
    [ 1.  1.]]
    ```

    - `tf.fill(...)` : 指定した定数で埋められた Tensor を作成する。
    ```python
    filled_tsr = tf.fill( [3, 2], "const" )
    print( "tf.fill(...) の Tensor 型 : ", filled_tsr )

    session = tf.Session()
    print( "tf.fill(...) の value : ", session.run( filled_tsr ) )
    session.close()
    ```
    ```python
    <出力>
    tf.fill(...) の Tensor 型 :  Tensor("Fill:0", shape=(3, 2), dtype=string)
    tf.fill(...) の value :  
    [[b'const' b'const']
    [b'const' b'const']
    [b'const' b'const']]
    ```

    - `tf.constant(...)` : 指定した既存の定数から Tensor を作成する。
    ```python
    const_tsr = tf.constant( [1,2,3] )
    print( "tf.constant(...) の Tensor 型 : ", const_tsr )

    session = tf.Session()
    print( "tf.constant(...) の value \n: ", session.run( const_tsr ) )
    session.close()
    ```
    ```python
    <出力>
    tf.constant(...) の Tensor 型 :  Tensor("Const:0", shape=(3,), dtype=int32)
    tf.constant(...) の value :  [1 2 3]
    ```

- シーケンステンソル </br>
    予め、指定した区間を含んだ Tensor のこと。</br>
    Numpy ライブラリでいうところの `range()` と似たような動作を行う。

    - `tf.linespace(...)` : stop 値のあるシーケンステンソルを作成する。
    ```python
    liner_tsr = tf.linspace( start = 0.0, stop = 1.0, num = 3 )
    print( "tf.linspace(...) の Tensor 型 : ", liner_tsr )

    session = tf.Session()
    print( "tf.linspace(...) の value : \n", session.run( liner_tsr ) )
    session.close()
    ```
    ```python
    <出力>
    tf.linspace(...) の Tensor 型 :  Tensor("LinSpace:0", shape=(3,), dtype=float32)
    tf.linspace(...) の value : 
    [ 0.   0.5  1. ]
    ```

    - `tf.range(...)` : stop 値のないシーケンステンソルを作成する。
    ```python
    int_seq_tsr = tf.range( start = 1, limit = 15, delta = 3 )
    print( "tf.range(...) の Tensor 型 : ", int_seq_tsr )
    
    session = tf.Session()
    print( "tf.range(...) の value : \n", session.run( int_seq_tsr ) )
    session.close()
    ```
    ```python
    [出力]
    tf.range(...) の Tensor 型 :  Tensor("range:0", shape=(5,), dtype=int32)
    tf.range(...) の value : [ 1  4  7 10 13]
    ```

- ランダムテンソル
    - `tf.random_uniform(...)` : 一様分布に基づく乱数の Tensor を作成する。
    - `tf.random_normal(...)` : 正規分布に基づく乱数の Tensor を作成する。

</br>


<a name="#コードの実行結果２"></a>

## 変数とプレースホルダ、及び Op ノードと計算グラフ : `main3.py`

### ① 変数の設定と、Op ノード、計算グラフ 

- 変数の作成は、` tf.Variable(...)` を使用する。
- 変数の初期化は、`tf.global_variables_initializer()` で作成した Op ノードを、</br>
  Session の `run(...)` に設定し、計算グラフをオペレーションさせることで行う。</br>
  `session.run( tf.global_variables_initializer() )`
    - 尚、変数の初期化は、各変数の `initializer` メソッドでも可能。 </br>
      この場合も同様に、Session の `run(...)` に設定し、計算グラフをオペレーションさせる必要あり。</br>
    （例）`session.run( zeros_tsr.initializer )`
- 変数の値の代入、変更は、`tf.assign(...)` を使用する。
- 変数に値が代入されるタイミングは、</br>
  Session の `run(...)` に指定したオペレーションがすべて完了した後になる。

（抜粋コード）
```python
    # Reset graph
    ops.reset_default_graph()

    # Session の設定
    session = tf.Session()

    # Tensor の設定
    zeros_tsr = tf.zeros( [3, 2] )
    
    # Variable の設定
    zeros_var = tf.Variable( zeros_tsr )
    print( "tf.Variable() :", zeros_var )
    
    # Opノード [op : operation] の作成 変数の初期化
    init_op = tf.global_variables_initializer()
    print( "tf.global_variables_initializer() :\n", init_op )

    # Session を run
    output = session.run( init_op )
    print( "session.run( init_op ) :\n", output )
    
    session.close()
```
```python
    <出力>
    tf.Variable() : 
    <tf.Variable 'Variable:0' shape=(3, 2) dtype=float32_ref>

    tf.global_variables_initializer() :
    name: "init"
    op: "NoOp"
    input: "^Variable/Assign"

    session.run( init_op ) : 
    None
    → Variable の初期化演算（オペレーション）の結果、計算グラフから None 値が Output されている。
```

> 構築した計算グラフを TensorBoard を用いた描写
![tensorboard_graph_variable-zero](https://user-images.githubusercontent.com/25688193/30039823-663e1be0-9211-11e7-989c-3766957b56fa.png)
>> Variable ( `zeros_var = tf.Variable( zeros_tsr )` )に対し、</br>
zero テンソル( `zeros_var = tf.Variable( zeros_tsr )` )が、Assign （割り当て）られて、</br>
初期化の オペレーション（Opノード）`( init_op = tf.global_variables_initializer() )` にフローされている。

- TensorBoard での計算グラフの記号の意味
    - 単方向矢印（➞）: オペレーション間のデータフロー </br>
    - 双方向矢印（↔）: 入力テンソルを変更することができるノードへの参照</br>

- TensorBoard を用いて、構築した計算グラフの表示する。コード側の処理は以下の通り。
    - `tf.summary.merge_all()` で Session の summary を TensorBoard に加える。
    - その後、`tf.summary.FileWriter(...)` で指定したフォルダに </br>
    Session の計算グラフ `session.graph` を書き込む。</br>
    `tf.summary.FileWriter( "./TensorBoard", graph = session.graph )`

</br>

### ② プレースホルダの設定と、Op ノード、計算グラフ 

TensorFlow におけるプレースホルダー [placeholder] は、計算グラフに供給するためのデータの位置情報のみを保有しており、一時保管庫の役割を果たす。</br>
利用法としては、データは未定のまま計算グラフを構築し、具体的な値を実行するときに与えたいケースに利用する。

- プレースホルダの作成は、`tf.placeholder` で行う。</br>
  （例）tf.float32 型のプレースホルダ : `holder = tf.placeholder( tf.float32, shape = [3, 2] )`
- プレースホルダを、実際に計算グラフに配置するには、</br>
  プレースホルダで少なくとも１つの演算（Op ノード）を実行（設定）する必要がある。</br>
  - ここでは、この演算の１つの例として、`tf.identity(...)` を使用する。</br>
    `identity_op = tf.identity( holder )`
- プレースホルダは、Session の オペレーションの実行時に、`run(...)` で指定した引数 feed_dict を通じて具体的なデータを与える。</br>
この引数 `feed_dict` で指定するオブジェクトは、ディクショナリ構造 `{Key: value}` のオブジェクトとなる。


（抜粋コード）
```python
    # Reset graph
    ops.reset_default_graph()

    # Session の設定
    session = tf.Session()

    # プレースホルダの作成
    holder = tf.placeholder( tf.float32, shape = [2, 2] )
    print( "tf.placeholder( tf.float32, shape = [2, 2] ) :", holder)

    # 演算を設定（Opノード）
    identity_op = tf.identity( holder )

    #  
    random = numpy.random.rand( 2, 2 )
    print( "random :\n", random)

    # 構築した計算グラフを実行
    output = session.run( identity_op, feed_dict = { holder : random } )
    print( "session.run( identity_op, feed_dict = { holder : random } ) : \n", output )
    
    session.close()
```
```python
    <出力>
    Tensor("Placeholder:0", shape=(2, 2), dtype=float32)
    
    random :
    [[ 0.84719631  0.53525849]
    [ 0.52465215  0.6306719 ]]
    
    session.run( identity_op, feed_dict = { holder : random } ) : 
     [[ 0.84719634  0.53525847]
    [ 0.52465212  0.63067192]]
    → Iditity 演算（オペレーション）の結果、計算グラフから等しい値が Output させている。
```

> 構築した計算グラフを TensorBoard を用いた描写</br>
![tensorboard_graph_placeholder-identity](https://user-images.githubusercontent.com/25688193/30051670-dd5da28a-925d-11e7-9de5-954376cde1ad.png)
>> Placeholder : `tf.placeholder( tf.float32, shape = [2, 2] )` を、オペレーション（Opノード）`identity_op = tf.identity( holder )` に矢印（オペレーション間のデータフロー）で設定している。 


<a name="#行列の操作"></a>

## 行列の操作 : `main4.py`
TensorFlow において、行列は、２次元配列構造を持つ Tensor の一種である。</br>
従って、行列の型は、全て Tensor 型になる。</br>
TensorFlow の用途的に行列は多用されるため、TensorFlow ではそれらを簡単に演算（加算、減算等）するための構文（演算子）が用意されている。

- 行列の作成は、２次元配列構造の Tensor の作成方法と同じである。
    - `tf.diag(...)` : list から対角行列（Tensor）を作成する。</br>
    `Identity_matrix = tf.diag( [1.0, 1.0, 1.0] )`
    - `tf.truncated_normal(...)` : </br>
    `A_matrix = tf.truncated_normal( [2, 3] )`
    - `tf.fill(...)` : </br>
    ` B_matrix = tf.fill( [2,3], 5.0)`
    - `tf.random_uniform(...)` : </br>
    `C_matrix = tf.random_uniform( [3,2] )`
    - ` tf.convert_to_tensor(...)` : list や numpy.array 等から Tensor （この場合、行列）を作成</br>
        ```python
        D_matrix = tf.convert_to_tensor(
                       numpy.array( [[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]] )
                   )
        ```
- 行列の加算、減算は、以下のように Session の `run(...)` 内の簡単なオペレーション `+`, `-` で実現できる。
    - 行列の加算 ( `+` ) : `session.run( A_matrix + B_matrix )`
    - 行列の減算 ( `-` ) : `session.run( A_matrix - B_matrix )`
- 行列の乗算には、Session の `run(...)` 内のオペレーション `tf.matmul(...)` を使用する。</br>
  `session.run( tf.matmul( B_matrix, Identity_matrix ) )`
- 行列の転置には、

＜抜粋コード＞
```python
    # Reset graph
    ops.reset_default_graph()

    # Session の設定
    session = tf.Session()

    # 各種行列 Tensor の作成
    Identity_matrix = tf.diag( [1.0, 1.0, 1.0] )    # tf.diag(...) : list から対角行列を作成
    A_matrix = tf.truncated_normal( [2, 3] )        # tf.truncated_normal(...) : 
    B_matrix = tf.fill( [2,3], 5.0)                 # 
    C_matrix = tf.random_uniform( [3,2] )           # tf.random_uniform(...) :
    D_matrix = tf.convert_to_tensor(                # tf.convert_to_tensor(...) : 
                   numpy.array( [[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]] )
               )

    print( "Identity_matrix <Tensor型> : ", Identity_matrix )
    print( "A_matrix <Tensor型> : ", A_matrix )
    print( "B_matrix <Tensor型> : ", B_matrix )
    print( "C_matrix <Tensor型> : ", C_matrix )
    print( "D_matrix <Tensor型> : ", D_matrix )

    # Session を run して値を設定後 print 出力
    print( "session.run( Identity_matrix ) :\n", session.run( Identity_matrix ) )
    print( "session.run( A_matrix ) :\n", session.run( A_matrix ) )
    print( "session.run( B_matrix ) :\n", session.run( B_matrix ) )
    print( "session.run( C_matrix ) :\n", session.run( C_matrix ) )
    print( "session.run( D_matrix ) :\n", session.run( D_matrix ) )

    # 行列の加算をして print
    print( "A_matrix + B_marix : \n", session.run( A_matrix + B_matrix) )

    # 行列の減算をして print
    print( "A_matrix - B_marix : \n", session.run( A_matrix - B_matrix) )

    # 行列の乗算をして print
    print( 
        "tf.matmul( B_matrix, Identity_matrix ) : \n", 
        session.run( 
            tf.matmul( B_matrix, Identity_matrix ) 
        ) 
    )
```
```python
    <出力>
    Identity_matrix <Tensor型> :  Tensor("Diag:0", shape=(3, 3), dtype=float32)
    A_matrix <Tensor型> :  Tensor("truncated_normal:0", shape=(2, 3), dtype=float32)
    B_matrix <Tensor型> :  Tensor("Fill:0", shape=(2, 3), dtype=float32)
    C_matrix <Tensor型> :  Tensor("random_uniform:0", shape=(3, 2), dtype=float32)
    D_matrix <Tensor型> :  Tensor("Const:0", shape=(3, 3), dtype=float64)
    
    session.run( Identity_matrix ) :
    [[ 1.  0.  0.]
    [ 0.  1.  0.]
    [ 0.  0.  1.]]
    
    session.run( A_matrix ) :
    [[-1.09035075 -0.32866415  1.57157266]
    [ 1.44946873 -0.15195866  1.64530897]]
    
    session.run( B_matrix ) :
    [[ 5.  5.  5.]
    [ 5.  5.  5.]]

    session.run( C_matrix ) :
    [[ 0.20161021  0.62964642]
     [ 0.10564721  0.49840438]
    [ 0.44993746  0.30875087]]

    session.run( D_matrix ) :
    [[ 1.  2.  3.]
    [-3. -7. -1.]
    [ 0.  5. -2.]]
    
    A_matrix + B_marix : 
    [[ 5.64082575  5.76026011  5.32105875]
    [ 5.50101185  5.78847122  3.64809322]]

    A_matrix - B_marix : 
    [[-5.78341341 -5.21996593 -5.4638133 ]
    [-4.55047417 -4.764575   -5.60554838]]

    tf.matmul( B_matrix, Identity_matrix ) : 
    [[ 5.  5.  5.]
    [ 5.  5.  5.]]
```