# TensorFlow ライブラリの処理フローの練習コード集

TensorFlow における基本的な処理の練習用コード集。（※機械学習のコードではありません。）</br>
この README.md ファイルには、各コードの実行結果、概要の説明を記載しています。</br>
分かりやすいように `main.py` ファイル毎に１つの完結したコードにしています。

## 項目 [Contents]
1. [使用するライブラリ](#ID_1)
1. [使用するデータセット](#ID_2)
1. [TensorFlow の概要](#ID_3)
1. [コードの実行結果](#ID_4)
    1. [テンソルの設定、及び計算グラフとセッション : `main2.py`](#ID_4-1)
    1. [変数とプレースホルダ、及び Op ノードと計算グラフ : `main3.py`](#ID_4-2)
    1. [行列の操作 : `main4.py`](#ID_4-3)
    1. [演算（オペレーション、Opノード）を設定 : `main5.py`](#ID_4-4)
    1. [計算グラフでの演算の設定、実行 : `main7.py`](#ID_4-6)
    1. [計算グラフでの入れ子の演算の階層化 : `main8.py`](#ID_4-7)
    1. [計算グラフでの複数の層の追加、操作 : `main9.py`](#ID_4-8)



<a id="ID_1"></a>

## 使用するライブラリ

> TensorFlow ライブラリ </br>
>> API 集 </br>
https://www.tensorflow.org/api_docs/python/ </br>

>> Session 関連 </br>
>>> `tf.Session` : A class for running TensorFlow operations. </br>
https://www.tensorflow.org/api_docs/python/tf/Session </br>
>>> `tf.Session.run(...)` : Session を run して、オペレーション、計算グラフを実行する。</br>
https://www.tensorflow.org/api_docs/python/tf/Session#run </br>
>>> `tf.Session.close()` : Session を close する。</br>
https://www.tensorflow.org/api_docs/python/tf/Session#close </br>

>> オペレーション（Op ノード）関連 </br>
>>> `tf.global_variables_initializer` : Variable の初期化を行うオペレーション</br>
https://www.tensorflow.org/api_docs/python/tf/global_variables_initializer </br>

>> Variable関連 </br>
>>> `tf.Variable` : Variable を作成する。</br>
https://www.tensorflow.org/api_docs/python/tf/Variable </br>

>> Placeholder 関連</br>
>>> `tf.placeholder` : Placeholder を作成する。</br>
https://www.tensorflow.org/api_docs/python/tf/placeholder </br>

>> Tensor の生成関連 </br>
https://www.tensorflow.org/api_guides/python/constant_op </br>
>>> 固定のテンソル </br>
>>>> `tf.zeros(...)` : 全ての要素が 0 からなる Tensor を作成する。</br>
https://www.tensorflow.org/api_docs/python/tf/zeros </br>
>>>> `tf.ones(...)` : 全ての要素が 1 からなる Tensor を作成する。</br>
https://www.tensorflow.org/api_docs/python/tf/ones </br>
>>>> `tf.fill(...)` : 指定した定数で埋められた Tensor を作成する。</br>
https://www.tensorflow.org/api_docs/python/tf/fill </br>
>>> シーケンステンソル </br>
>>>> `tf.linespace(...)` : stop 値のあるシーケンステンソルを作成する。</br>
https://www.tensorflow.org/api_docs/python/tf/lin_space </br>
>>>> `tf.range(...)` : stop 値のないシーケンステンソルを作成する。</br>
https://www.tensorflow.org/api_docs/python/tf/range </br>
>>> ランダムテンソル

</br>

<a id="ID_2"></a>

## 使用するデータセット
- 

</br>
<a id="ID_3"></a>

## TensorFlow の概要

基本的に、計算グラフという有向グラフを構築し、それをセッションで動かすという処理フローになる。</br>
これは、MATLAB Simulink とよく似たモデルベースな設計手法。</br>
但し、MATLAB Simulink とは異なりモデル作成のための GUI ツールはないので、頭の中（コード）で計算グラフを考えながら設計していくスタイルになる。</br>
（TensorBoard でコードの結果の計算グラフを可視化することは可能。）</br>
以下、コードの実行結果と共に、TensorFlow 特有の概念である、Tensor, Variable, Placeholder, Session, オペレーション（Opノード）等を記述する。

</br>
<a name="#ID_4"></a>

## コードの実行結果

<a id="ID_4-1"></a>

## テンソルの設定、及び計算グラフとセッション : `main2.py`

TensorFlow におけるテンソルとは、単に多次元配列であるが、計算グラフに対して何かしらのグラフ構造を与えると言う点で、他の多次元配列とは異なる。</br>
但し、ここで注意すべきことは、Tensor を設定したからといって、TensorFlow が計算グラフに何かしらを追加するわけではないということである。</br>
TensorFlow が計算グラフに何かを追加するのは Tensor が作成されて、利用可能（ `session.run(...)` された後）になった後に限られる。

- 固定のテンソル
    - `tf.zeros(...)` : 全ての要素が 0 からなる Tensor を作成する。</br>
    
    （例）Session を生成せず、そのまま Tensor を print()　</br> 
    ```python
    zero_tsr = tf.zeros( [3, 2] )
    print( zero_tsr )
    ```
    ```python
    [出力]
    Tensor("zeros:0", shape=(3, 2), dtype=float32)
    → Tensor オブジェクト（Tensor 型）がそのまま出力されている。
    これは session を実行していないため。
    ```
    （例）Session を生成＆run() して、Tensor を print()　</br>
    ```python
    zero_tsr = tf.zeros( [3, 2] )
    session = tf.Session()
    print( session.run( zero_tsr ) )
    ```
    ```python    
    [出力]
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

    ```python
    [出力]
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
    [出力]
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
    [出力]
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
    [出力]
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


<a id="ID_4-2"></a>

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

<抜粋コード : `main3.py`>
```python
def main():
    ...

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
[出力]
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
![image](https://user-images.githubusercontent.com/25688193/30110476-2a6acf66-9345-11e7-9585-cd5296851f18.png)

>> Variable ( `zeros_var = tf.Variable( zeros_tsr )` )に対し、</br>
zero テンソル( `zeros_var = tf.Variable( zeros_tsr )` )が、Assign （割り当て）られて、</br>
初期化の オペレーション（Opノード）`( init_op = tf.global_variables_initializer() )` に制御フローされている。

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


<抜粋コード : `main3.py`>
```python
def main():
    ...
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
[出力]
Tensor("Placeholder:0", shape=(2, 2), dtype=float32)
    
random :
[[ 0.84719631  0.53525849]
[ 0.52465215  0.6306719 ]]
    
session.run( identity_op, feed_dict = { holder : random } ) : 
[[ 0.84719634  0.53525847]
[ 0.52465212  0.63067192]]
→ Iditity 演算（オペレーション）の結果、計算グラフから等しい値が Output させている。
```

> 構築した計算グラフを TensorBoard で描写</br>
![image](https://user-images.githubusercontent.com/25688193/30110498-40ae91ea-9345-11e7-8cf7-f187577d45d3.png)
>> Placeholder : `tf.placeholder( tf.float32, shape = [2, 2] )` が、オペレーション（Opノード）`identity_op = tf.identity( holder )` に矢印（オペレーション間のデータフロー）で設定されており、</br>
Placeholder から `session.run(...)` の引数 `feed_dict = { holder : random }` を通じて、Identity ノードにデータを供給していることがビジュアル的に分かる。

</br>


<a id="ID_4-3"></a>

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
- 行列の転置には、`...`


<抜粋コード : `main4.py`> 
```python
def main():
    ...
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
[出力]
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

</br>


<a id="ID_4-4"></a>

## 演算（オペレーション、Opノード）の設定、実行 : `main5.py`
ここでは、ここまでに取り上げた TensorFlow における演算（オペレーション、Opノード）の他に、</br>
よく使うオペレーションを取り上げる。

- TensorFlow では、`tf.add(...)`, `tf.sub(...)`, `tf.mul(...)`, `tf.div(...)` を用いて、
  Opノードを作成し、</br>
  それらを Session で `session.run(...)` させることにより、四則演算を行うことが出来る。
- TensorFlow には、割り算演算である div に何種類かの div が用意されている。</br>
  これらの `div(...)` は、基本的に引数と戻り値の型が同じとなる。
    - `tf.div(x,y)` : 整数に対しての div（オペレーション、Op ノード）</br>
        ```python
        div_op = tf.div( 3, 4 )
        print( "session.run( div_op ) :\n", session.run( div_op ) )
        ```
        ```python
        [出力]
        session.run( div_op ) : 0
        ➞ 整数での演算なので、小数点以下切り捨てにより3/4 ➞ 0 になっている。
        ```
    - `tf.truediv(x,y)` : 浮動小数点数に対しての div（オペレーション、Op ノード）
        ```python
        truediv_op = tf.truediv( 3, 4 )
        print( "session.run( truediv_op ) :\n", session.run( truediv_op ) )
        ```
        ```python
        [出力]
        session.run( truediv_op ) : 0.75
        ```
    - `tf.floordiv(x,y)` : 浮動小数点数であるが、整数での演算を行いたい場合の div（オペレーション、Op ノード）
- 割り算の余り `mod(...)`
- 外積 `cross(...)`
- 数学関数 `abs(x)`, `cos(x)`等
- 統計、機械学習関連の基本演算 `erf()`等
- そして、これらの関数を組み合わせて、独自の関数を生成＆処理することが出来る。
    - その為には、Session の `run(...)` に複数のオペレーション（Opノード）を設定すれば良い。
        ```python
        comb_tan_op = tf.div( 
                          tf.sin( 3.1416/4. ), 
                          tf.cos( 3.1416/4. ) 
                       )
        
        output1 = session.run( comb_tan_op )
        print( "session.run( comb_tan_op ) : ", output1 )
        ```
        ```python
        [出力]
        session.run( comb_tan_op ) : 1.0
        ```
    - 更に複雑な関数を実現したい場合は、オペレーションを返す関数を def すれば良い。
        ```python
        def cusmom_polynormal( x ):
            ''' f(x) = 3 * x^2 - x + 10 '''
            cusmom_polynormal_op = ( tf.subtract( 3*) +10 )
            return cusmom_polynormal_op

        def main():
            ...
            cusmom_polynormal_op = cusmom_polynormal( x = 100 )
            output2 = session.run( cusmom_polynormal_op )
            print( "session.run( "cusmom_polynormal( x = 100 )", output2 )
        ```
        ```python
        [出力]
        session.run( cusmom_polynormal_op ) : 300
        ```


<抜粋コード : `main5.py`>
```python
def main():
    ...
    #======================================================================
    # 演算（オペレーション、Opノード）の設定、実行
    #======================================================================
    #----------------------------------------------------------------------
    # 単一の演算（オペレーション、Opノード）
    #----------------------------------------------------------------------
    # Reset graph
    ops.reset_default_graph()

    # Session の設定
    session = tf.Session()

    # オペレーション div 
    # 何種類かの 割り算演算 div の動作を確認する。
    div_op      = tf.div( 3, 4 )        # tf.div(x,y) : 整数演算での div
    truediv_op  = tf.truediv( 3, 4 )    # tf.truediv(x,y) : 浮動小数点数に対しての div
    floordiv_op = tf.floordiv( 3, 4 )   # 浮動小数点数であるが、整数での演算を行いたい場合の div

    # Session を run してオペレーションを実行後 print 出力
    # 何種類かの 割り算演算 div の動作を確認する。
    print( "session.run( div_op ) :\n", session.run( div_op ) )
    print( "session.run( truediv_op ) :\n", session.run( truediv_op ) )
    print( "session.run( truediv_op ) :\n", session.run( floordiv_op ) )
    
    session.close()
    
    #----------------------------------------------------------------------
    # 複数の演算（オペレーション、Opノード）の組み合わせ
    #----------------------------------------------------------------------
    # Reset graph
    ops.reset_default_graph()

    # Session の設定
    session = tf.Session()

    # オペレーションの組み合わせ 
    comb_tan_op = tf.div( 
                      tf.sin( 3.1416/4. ), 
                      tf.cos( 3.1416/4. ) 
                  )

    cusmom_polynormal_op = cusmom_polynormal( x = 10 )
    
    # Session を run してオペレーションを実行後 print 出力
    output1 = session.run( comb_tan_op )
    output2 = session.run( cusmom_polynormal_op )

    print( "session.run( comb_tan_op ) : ", output1 )
    print( "session.run( cusmom_polynormal_op ) : ", output2 )

    session.close()

def cusmom_polynormal( x ):
    '''
    f(x) = 3 * x^2 - x + 10
    '''
    cusmom_polynormal_op = ( tf.subtract( 3*tf.square(x), x) + 10 )
    return cusmom_polynormal_op
```

```python
[出力]
session.run( div_op ) :
 0
session.run( truediv_op ) :
 0.75
session.run( truediv_op ) :
 0
session.run( comb_tan_op ) :  1.0
session.run( cusmom_polynormal_op ) :  300
```

</br>


<a id="ID_4-6"></a>

## 計算グラフでの演算（オペレーション、Opノード）の設定、実行 : `main7.py`

ここまでの主に各種 TensorFlow オブジェクトの計算グラフへの配置に続き、</br>
より実践的な、計算グラフでの演算を簡単な例で行なってみる。

- 計算グラフに各種オブジェクト（Placeholder : `tf.placeholder(...)`、const : `tf.constant(...)`）を紐付ける。
- 計算グラフに１つのオペレーション Mul : `tf.multiply(...)` を追加する。
- 構築した計算グラフに対して、for ループ内で `session.run(...)` させ、計算グラフの Output を計算し続け、一連の値を取得する。
    ```python
    # 入力値の list を for ループして、list の各値に対し、オペレーション実行
    for value in float_list:
        # Session を run して、
        # Placeholder から feed_dict を通じて、データを供給しながら、オペレーション実行
        output = session.run( multipy_op, feed_dict = { float_holder: value } )
        
        # Session を run した結果（Output）を print 出力
        print( output )
    ```
    ```python
    [出力]
    3.0     ← 構築した計算グラフから Output される 1 回目の値
    9.0     ← 構築した計算グラフから Output される 2 回目の値
    15.0    ...
    21.0    
    27.0
    ```

> TensorBoard で描写した計算グラフ</br>
![image](https://user-images.githubusercontent.com/25688193/30110536-5b8159d0-9345-11e7-91aa-c4014e6d33fb.png)

>> Const 値 `float_const = tf.constant( 3. )` を Opノード : Mul にデータフローしながら、</br>
又、Placeholer から、`session.run(...)` の引数 `feed_dict = { float_holer, value }` を通じて、</br>
オペレーション Mul : `tf.multiply( float_holder, float_const )` にデータを供給して、`session.run(...)` でオペレーション Mul を実行し続けている。

<抜粋コード : `main7.py`>
```python
def main():
    ...
    # Reset graph
    ops.reset_default_graph()

    # Session の設定
    session = tf.Session()

    # 各種 通常の変数、Tensor、placeholder の作成
    float_list = numpy.array( [1., 3., 5, 7, 9.] )
    float_holder = tf.placeholder( tf.float32 )
    float_const = tf.constant( 3. )

    # オペレーション（Opノード）の作成
    multipy_op = tf.multiply( float_holder, float_const )

    # 入力値の list を for ループして、list の各値に対し、オペレーション実行
    for value in float_list:
        # Session を run して、
        # Placeholder から feed_dict を通じて、データを供給しながら、オペレーション実行
        output = session.run( multipy_op, feed_dict = { float_holder: value } )
        
        # Session を run した結果（Output）を print 出力
        print( output )
    
    # TensorBoard 用のファイル（フォルダ）を作成
    # Add summaries to tensorboard
    merged = tf.summary.merge_all()
    # tensorboard --logdir=${PWD}
    summary_writer = tf.summary.FileWriter( "./TensorBoard", graph = session.graph )

    session.close()
```
```python
[出力]
3.0
9.0
15.0
21.0
27.0
```

</br>

<a id="ID_4-7"></a>

## 計算グラフでの入れ子の演算の階層化 : `main8.py`
ここまでの計算グラフは、単一の演算（オペレーション）のみであったが、</br>
ここでは、複数のオペレーション（演算）からなる計算グラフを構築し、その処理を行う。</br>
この際、各オペレーション（演算）ノードを連結しながら計算グラフを構築していくことになる。

- 計算グラフに各種オブジェクト（Placeholder : `tf.placeholder(...)`、３つの const : `tf.constant(...)`）を紐付ける。
- 計算グラフに複数の演算（オペレーション）を追加し、それらを連結していく。
    - １つ目のオペレーション MatMul を作成する。</br>
    `matmul_op1 = tf.matmul( dim3_holder, const1 )`
    - ２つ目のオペレーション MatMul を作成し、先の１つ目のオペレーション MatMul を連結する。</br>
    具体的には、`tf.matmul(...)` の第１引数に、</br>
    先の１つ目のオペレーション MatMul を指定することで、連結することが出来る。</br>
    `matmul_op2 = tf.matmul( matmul_op1, const2 )` : 第１引数に `matmul_op1` を指定
    - ３つ目オペレーション Add を作成＆連結する。</br>
    同様にして、先の２つ目のオペレーション MatMul を指定することで、連結することが出来る。</br>
    `add_op3 = tf.add( matmul_op2, const3 )` : 第１引数に `matmul_op2` を指定
    - これらの３つの処理で、３つのオペレーションが作成＆連結される。
- 構築した計算グラフに対して, for ループで `session.run(...)` させ、計算グラフの Output を計算し続け、一連の値を取得する。
    ```python
    # 入力値の list を for ループして、list の各値に対し、
    # 構築した計算グラフのオペレーション実行
    for value in values:
        # Session を run して、
        # Placeholder から feed_dict を通じて、データを供給しながら、オペレーション実行
        output = session.run( add_op3, feed_dict = { dim3_holder : value } )
        
        # Session を run した結果（Output）を print 出力
        print( output )
    ```
    ```python
    [出力]
    [[ 102.]    ← 構築した計算グラフから Output される 1 回目の値
    [  66.]
    [  58.]]    

    [[ 114.]    ← 構築した計算グラフから Output される 2 回目の値
     [  78.]
    [  70.]]
    ```

> TensorBorad で描写した計算グラフ
![image](https://user-images.githubusercontent.com/25688193/30110959-eafe3bfe-9346-11e7-9522-a2517297cc19.png)
>> 

<抜粋コード : `main8.py`>
```python
def main():
    ...
    # Reset graph
    ops.reset_default_graph()

    # Session の設定
    session = tf.Session()

    # 各種 通常の変数、Tensor、placeholder の作成
    dim3_list = numpy.array( 
                    [
                        [ 1.,  3., 5,  7,  9.], 
                        [-2.,  0., 2., 4., 6.],
                        [-6., -3., 0., 3., 6.]
                    ]
                )
    
    # 後の for ループでの処理のため、
    # 入力を２つ（２回ループ）にするため、配列を複製
    values = numpy.array( [dim3_list, dim3_list + 1] )

    # 3*5 のデータを供給するための placeholder
    dim3_holder = tf.placeholder( tf.float32, shape = (3,5) )
    
    # 行列の演算用のConst 値
    const1 = tf.constant( [ [1.],[0.],[-1.],[2.],[4.] ] )
    const2 = tf.constant( [ [2.] ] )
    const3 = tf.constant( [ [10.] ] )

    print( "const1 : ", const1 )
    print( "const2 : ", const2 )
    print( "const3 : ", const3 )

    # オペレーション（Opノード）の作成＆連結
    matmul_op1 = tf.matmul( dim3_holder, const1 )
    matmul_op2 = tf.matmul( matmul_op1, const2 )
    add_op3 = tf.add( matmul_op2, const3 )

    print( "tf.matmul( dim3_holder, const1 ) : ", matmul_op1 )
    print( "tf.matmul( matmul_op1, const2 ) : ", matmul_op2 )
    print( "tf.add( matmul_op2, const1 ) : ", add_op3 )

    # 入力値の list を for ループして、list の各値に対し、
    # 構築した計算グラフのオペレーション実行
    for value in values:
        # Session を run して、
        # Placeholder から feed_dict を通じて、データを供給しながら、オペレーション実行
        output = session.run( add_op3, feed_dict = { dim3_holder : value } )
        
        # Session を run した結果（Output）を print 出力
        print( output )    
```
```python
[出力]
const1 :  Tensor("Const:0", shape=(5, 1), dtype=float32)
const2 :  Tensor("Const_1:0", shape=(1, 1), dtype=float32)
const3 :  Tensor("Const_2:0", shape=(1, 1), dtype=float32)

tf.matmul( dim3_holder, const1 ) :  Tensor("MatMul:0", shape=(3, 1), dtype=float32)
tf.matmul( matmul_op1, const2 ) :  Tensor("MatMul_1:0", shape=(3, 1), dtype=float32)
tf.add( matmul_op2, const1 ) :  Tensor("Add:0", shape=(3, 1), dtype=float32)

[[ 102.]
 [  66.]
 [  58.]]

[[ 114.]
 [  78.]
 [  70.]]
```

</br>

<a id="ID_4-8"></a>

## 計算グラフでの複数の層の追加、操作 : `main9.py`

ニューラルネットワークでは、一般的に複数の層を扱う。</br>
その為にも、ここではその基本として、計算グラフへの複数の層の追加、操作、及びカスタマイズした層の操作を取り上げる。

- 4×4 pixel の画像データをフィルタリング＆移動平均して、2×2 pixel の画像データとして、Output する計算グラフを考える。
- まず、データを供給するための Placeholder として、画像の数、画像の height、画像の width、画像の Channel 数からなる Placeholder : `image_holder` を設定する</br>
```python
# 画像のデータを供給するための placeholder
# 画像の形状 
# shape[0] : 画像の数
# shape[1] : 画像の height
# shape[2] : 画像の width
# shape[3] : 画像の Channel 数
image_shape = [ 1, 4, 4, 1 ] 
image_holder = tf.placeholder( tf.float32, shape = image_shape )
```
- 4×4 pixel の各 pixel 値がランダムな値からなるデータ `random_value` を作成する。
```python
# ランダムな画像を生成するためのランダム変数
numpy.random.seed(12)
random_value = numpy.random.uniform( size = image_shape )
print( "random_value : ", random_value )
```
```python
[出力] 4×4 の画像データ（各ピクセル値はランダムに生成）
[
    [
             
        [ [ 0.15416284] [ 0.7400497 ] [ 0.26331502] [ 0.53373939] ]  
        [ [ 0.01457496] [ 0.91874701] [ 0.90071485] [ 0.03342143] ]
        [ [ 0.95694934] [ 0.13720932] [ 0.28382835] [ 0.60608318] ]
        [ [ 0.94422514] [ 0.85273554] [ 0.00225923] [ 0.52122603] ]
    ]
]
```
- このランダムに生成した 4×4 pixel の画像データをフィルタリングして、
  上下左右の移動平均を算出することを考える。
    - その為には、`tf.nn.conv2d(...)` 関数を使用する。</br>
      この関数は、入力として placeholder : `input = image_holder` を受けとり、</br>
      フィルタ値 : `filter = filer_const`、スライド値 : `strides = stride_value` で、</br>
      データをフィルタリングして、上下左右の移動平均を算出する。
    - ここで、この Layer : `mov_avg_layer_op` の名前を `name = "Moving_Ave_Window"` としておく。
    - そして、この関数で作成した層の Output は、1×2×2×1 のデータになる。</br>
    `session.run( mov_avg_layer_op, feed_dict = { image_holder : random_value } )`
```python
# 画像のウインドウのフィルタ値、スライド値
filer_const = tf.constant( 0.25, shape = [2, 2, 1, 1] )
stride_value = [1, 2, 2, 1]

# layer（Opノード）の作成
# 画像のウインドウに定数を "畳み込む"（） 関数
# 画像のウインドウの要素（ピクセル）毎に、指定したフィルタで積を求め、
# 又、画像のウインドウを上下左右方向にスライドさせる。
mov_avg_layer_op = tf.nn.conv2d(
                       input = image_holder,       # 入力として Placeholder を指定
                       filter = filer_const,       # フィルタ値
                       strides = stride_value,     # スライド値
                       padding = "SAME",           #
                       name = "Moving_Ave_Window"  # 層の名前
                    )

# 計算グラフの実行
output1 = session.run( mov_avg_layer_op, feed_dict = { image_holder : random_value } )
print( "session.run( mov_avg_layer_op, feed_dict = { image_holder : random_value } ) : \n", output1 )
```
```python
[出力]
session.run( mov_avg_layer_op, feed_dict = { image_holder : random_value } ) : 
[
    [
        [ [ 0.45688361] [ 0.43279767] ]
        [ [ 0.72277981] [ 0.35334921] ]
    ]
]
```
- 上記の layer : `Moving_Ave_Window` で Output されるデータは、1×2×2×1 のデータなので、</br>
次に、このデータを 2×2 のデータに変換して、更に幾つかの変換処理を施すカスタマイズした layer を考える。
    - まず、オペレーション Squeeze : `tf.squeeze(...)` を使用して、</br>
    入力された 1×2×2×1 のデータの１次元要素を削除し、2×2 のデータに変換するオペレーションを作成する。</br>
    `custom_layer_sqeezed_op = tf.squeeze( input_tsr )`
    - 次に、これにオペレーション Matmul : `tf.matmul(...)` を作成＆結合する。</br>
    `custom_layer_matmul_op = tf.matmul( A_matrix_const, custom_layer_sqeezed_op )`
    - 更に、これにオペレーション Matmul : `tf.matmul(...)` を作成＆結合する。</br>
    `custom_layer_matmul_op = tf.matmul( A_matrix_const, custom_layer_sqeezed_op )`
    - 更に、これにオペレーション Add : `tf.add(...)` を作成＆結合する。</br>
    `custom_layer_add_op = tf.add( custom_layer_matmul_op, B_matrix_const )`
    - 最後に、オペレーション Sigmoid : `tf.sigmoid(...)` を使用して、値を 0~1 の範囲に変換するオペレーションを作成＆結合する。</br>
    `custom_layer_sigmoid_op = tf.sigmoid( custom_layer_add_op )`
    - これらの処理を関数 `def custom_layer( input_tsr ):` で実現する。
        - この時の引数である Tensor値 : `input_tsr` で、</br>
          カスタム層の初めのオペレーション Squeeze との接続が出来る。
    -  そして、`tf.name_scope('Custom_Layer')` で、このカスタム層の名前を設定出来る
```python
# 画像のウインドウの移動平均の２×２出力を行うカスタム層を作成する。
# 練習用コードの可読性のため、main() 関数内にて関数定義する。
def custom_layer( input_tsr ):
    # tf.squeeze(...) : sizeが 1 の次元の要素を削除し次元数を減らす
    custom_layer_sqeezed_op = tf.squeeze( input_tsr )

    A_matrix_const = tf.constant( [ [1., 2.], [-1., 3.] ] )
    B_matrix_const = tf.constant( 1., shape = [2, 2] )
        
    # オペレーション
    custom_layer_matmul_op = tf.matmul( A_matrix_const, custom_layer_sqeezed_op )
    custom_layer_add_op = tf.add( custom_layer_matmul_op, B_matrix_const )
    # シグモイド関数で 0 ~ 1 の範囲値に変換
    custom_layer_sigmoid_op = tf.sigmoid( custom_layer_add_op )
        
    return custom_layer_sigmoid_op

# Add custom layer to graph
with tf.name_scope('Custom_Layer') as scope:
    custom_layer_op = custom_layer( mov_avg_layer_op )
```
- 最後に、構築した計算グラフに対し、Session を `session.run(...)`して、この計算グラフでの処理を行う。

> TensorBorad で描写した計算グラフ
![image](https://user-images.githubusercontent.com/25688193/30134860-95fd2546-9393-11e7-8790-ac501ed587dc.png)
>> 

<抜粋コード : `main9.py`>
```python
def main():
    ...
    # Reset graph
    ops.reset_default_graph()

    # Session の設定
    session = tf.Session()

    #-------------------------------------------------
    # 各種 通常の変数、Tensor、placeholder の作成
    #--------------------------------------------------

    # 画像のデータを供給するための placeholder
    # 画像の形状 
    # shape[0] : 画像の数
    # shape[1] : 画像の height
    # shape[2] : 画像の width
    # shape[3] : 画像の Channel 数
    image_shape = [ 1, 4, 4, 1 ] 
    image_holder = tf.placeholder( tf.float32, shape = image_shape )
    print( "image_holder :", image_holder )
    
    # ランダムな画像を生成するためのランダム変数
    numpy.random.seed(12)
    random_value = numpy.random.uniform( size = image_shape )
    print( "random_value : ", random_value )
    
    # 画像のウインドウのフィルタ値、スライド値
    filer_const = tf.constant( 0.25, shape = [2, 2, 1, 1] )
    stride_value = [1, 2, 2, 1]

    #----------------------------------------------------------------------
    # layer（Opノード）の作成  
    #----------------------------------------------------------------------
    # 画像のウインドウに定数を "畳み込む"（） 関数
    # 画像のウインドウの要素（ピクセル）毎に、指定したフィルタで積を求め、
    # 又、画像のウインドウを上下左右方向にスライドさせる。
    mov_avg_layer_op = tf.nn.conv2d(
                           input = image_holder,       #
                           filter = filer_const,       #
                           strides = stride_value,     #
                           padding = "SAME",           #
                           name = "Moving_Ave_Window"  # 層の名前
                       )
    
    print( "filer_const : ", filer_const )
    print( "mov_avg_layer_op : ", mov_avg_layer_op )

    # 画像のウインドウの移動平均の２×２出力を行うカスタム層を作成する。
    # 練習用コードの可読性のため、main() 関数内にて関数定義する。
    def custom_layer( input_tsr ):
        # tf.squeeze(...) : sizeが 1 の次元の要素を削除し次元数を減らす
        custom_layer_sqeezed_op = tf.squeeze( input_tsr )

        A_matrix_const = tf.constant( [ [1., 2.], [-1., 3.] ] )
        B_matrix_const = tf.constant( 1., shape = [2, 2] )
        
        # オペレーション
        custom_layer_matmul_op = tf.matmul( A_matrix_const, custom_layer_sqeezed_op )
        custom_layer_add_op = tf.add( custom_layer_matmul_op, B_matrix_const )
        custom_layer_sigmoid_op = tf.sigmoid( custom_layer_add_op )     # シグモイド関数で 0 ~ 1 の範囲値に変換

        # オペレーションの output 値を確認
        output2 = session.run( custom_layer_sqeezed_op, feed_dict = { image_holder : random_value } )
        output3 = session.run( custom_layer_matmul_op, feed_dict = { image_holder : random_value } )
        output4 = session.run( custom_layer_add_op, feed_dict = { image_holder : random_value } )
        output5 = session.run( custom_layer_sigmoid_op, feed_dict = { image_holder : random_value } )

        print( "session.run( custom_layer_sqeezed_op, feed_dict = { image_holder : random_value } ) : \n", output2 )
        print( "session.run( custom_layer_matmul_op, feed_dict = { image_holder : random_value } ) : \n", output2 )
        print( "session.run( custom_layer_matmul_op, feed_dict = { image_holder : random_value } ) : \n", output3 )
        print( "session.run( custom_layer_add_op, feed_dict = { image_holder : random_value } ) : \n", output4 )
        print( "session.run( custom_layer_sigmoid_op, feed_dict = { image_holder : random_value } ) : \n", output5 )
        
        return custom_layer_sigmoid_op

    # Add custom layer to graph
    with tf.name_scope('Custom_Layer') as scope:
        custom_layer_op = custom_layer( mov_avg_layer_op )
    
    #----------------------------------------------------------------------
    # 計算グラフの実行
    #----------------------------------------------------------------------
    output1 = session.run( mov_avg_layer_op, feed_dict = { image_holder : random_value } )
    output = session.run( custom_layer_op, feed_dict = { image_holder : random_value } )

    print( "session.run( mov_avg_layer_op, feed_dict = { image_holder : random_value } ) : \n", output1 )
    print( "session.run( custom_layer_op, feed_dict = { image_holder : random_value } ) : \n", output )
    ...
```
```python
[出力]
image_holder : Tensor("Placeholder:0", shape=(1, 4, 4, 1), dtype=float32)
random_value :  [[[[ 0.15416284]
   [ 0.7400497 ]
   [ 0.26331502]
   [ 0.53373939]]

  [[ 0.01457496]
   [ 0.91874701]
   [ 0.90071485]
   [ 0.03342143]]

  [[ 0.95694934]
   [ 0.13720932]
   [ 0.28382835]
   [ 0.60608318]]

  [[ 0.94422514]
   [ 0.85273554]
   [ 0.00225923]
   [ 0.52122603]]]]

filer_const :  Tensor("Const:0", shape=(2, 2, 1, 1), dtype=float32)
mov_avg_layer_op :  Tensor("Moving_Ave_Window:0", shape=(1, 2, 2, 1), dtype=float32)

session.run( mov_avg_layer_op, feed_dict = { image_holder : random_value } ) : 
 [[[[ 0.45688361]
   [ 0.43279767]]

  [[ 0.72277981]
   [ 0.35334921]]]]

session.run( custom_layer_sqeezed_op, feed_dict = { image_holder : random_value } ) : 
 [[ 0.45688361  0.43279767]
 [ 0.72277981  0.35334921]]

session.run( custom_layer_matmul_op, feed_dict = { image_holder : random_value } ) : 
 [[ 0.45688361  0.43279767]
 [ 0.72277981  0.35334921]]

session.run( custom_layer_matmul_op, feed_dict = { image_holder : random_value } ) : 
 [[ 1.90244317  1.13949609]
 [ 1.71145582  0.62724996]]

session.run( custom_layer_add_op, feed_dict = { image_holder : random_value } ) : 
 [[ 2.90244317  2.13949609]
 [ 2.71145582  1.62724996]]

session.run( custom_layer_sigmoid_op, feed_dict = { image_holder : random_value } ) : 
 [[ 0.94796705  0.89468312]
 [ 0.93769926  0.8357926 ]]

session.run( custom_layer_op, feed_dict = { image_holder : random_value } ) : 
 [[ 0.94796705  0.89468312]
 [ 0.93769926  0.8357926 ]]
 ➞ 目的である 4×4 の画像をフィルタリング＆移動平均等をした後の、2×2 の画像の pixcel 値が出力されている。
```