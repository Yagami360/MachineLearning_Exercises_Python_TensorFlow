# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境 (TensorFlow インストール済み)
#     <Anaconda Prompt>
#     conda create -n tensorflow python=3.5
#     activate tensorflow
#     pip install --ignore-installed --upgrade tensorflow
#     pip install --ignore-installed --upgrade tensorflow-gpu

import numpy as np
import pandas
import matplotlib.pyplot as plt

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

from MLPreProcess import MLPreProcess                   # 機械学習の前処理群を表すクラス


def main():
    """
    CNNを用いた、MNIST 画像データの識別
    """
    print("Enter main()")

    sess = tf.Session()
    #======================================================================
    # データセットを読み込み or 生成
    # Import or generate data.
    #======================================================================
    #======================================================================
    # データセットをトレーニングデータ、テストデータ、検証データセットに分割
    #======================================================================
    # MNIST データが格納されているフォルダへのパス
    mnist_path = "D:\Data\MachineLearning_DataSet\MNIST"

    """
    X_train, y_train = MLPreProcess.load_mnist( mnist_path, "train" )
    X_test, y_test = MLPreProcess.load_mnist( mnist_path, "t10k" )

    X_train = np.array( [np.reshape(x, (28,28)) for x in X_train] )
    X_test = np.array( [np.reshape(x, (28,28)) for x in X_test] )
    """
    # TensorFlow のサポート関数を使用して, MNIST データを読み込み
    mnist = read_data_sets( mnist_path )
    print( "mnist :\n", mnist )
    X_train = np.array( [np.reshape(x, (28,28)) for x in mnist.train.images] )
    X_test = np.array( [np.reshape(x, (28,28)) for x in mnist.test.images] )
    y_train = mnist.train.labels
    y_test = mnist.test.labels

    print( "X_train.shape : ", X_train.shape )
    print( "y_train.shape : ", y_train.shape )
    print( "X_test.shape : ", X_test.shape )
    print( "y_test.shape : ", y_test.shape )

    print( "X_train : \n", X_train )
    print( "y_train : \n", y_train )

    #======================================================================
    # データを変換、正規化
    # Transform and normalize data.
    # ex) data = tf.nn.batch_norm_with_global_normalization(...)
    #======================================================================
    

    #======================================================================
    # アルゴリズム（モデル）のパラメータを設定
    # Set algorithm parameters.
    # ex) learning_rate = 0.01  iterations = 1000
    #======================================================================
    batch_size = 100
    learning_rate = 0.005
    evaluation_size = 500
    image_width = X_train[0].shape[0]
    image_height = X_train[0].shape[1]
    target_size = max(y_train) + 1
    num_channels = 1 # greyscale = 1 channel
    generations = 500
    eval_every = 5
    conv1_features = 25
    conv2_features = 50
    max_pool_size1 = 2 # NxN window for 1st max pool layer
    max_pool_size2 = 2 # NxN window for 2nd max pool layer
    fully_connected_size1 = 100

    #======================================================================
    # 変数とプレースホルダを設定
    # Initialize variables and placeholders.
    # TensorFlow は, 損失関数を最小化するための最適化において,
    # 変数と重みベクトルを変更 or 調整する。
    # この変更や調整を実現するためには, 
    # "プレースホルダ [placeholder]" を通じてデータを供給（フィード）する必要がある。
    # そして, これらの変数とプレースホルダと型について初期化する必要がある。
    # ex) a_var = tf.constant(42)
    #     x_input_holder = tf.placeholder(tf.float32, [None, input_size])
    #     y_input_holder = tf.placeholder(tf.fload32, [None, num_classes])
    #======================================================================
    x_input_shape = (batch_size, image_width, image_height, num_channels)
    x_input = tf.placeholder(tf.float32, shape=x_input_shape)
    y_target = tf.placeholder(tf.int32, shape=(batch_size))

    eval_input_shape = (evaluation_size, image_width, image_height, num_channels)
    eval_input = tf.placeholder(tf.float32, shape=eval_input_shape)
    eval_target = tf.placeholder(tf.int32, shape=(evaluation_size))

    #======================================================================
    # モデルの構造を定義する。
    # Define the model structure.
    # ex) add_op = tf.add(tf.mul(x_input_holder, weight_matrix), b_matrix)
    #======================================================================
    # Convolutional layer variables
    conv1_weight = tf.Variable(tf.truncated_normal([4, 4, num_channels, conv1_features],
                                                   stddev=0.1, dtype=tf.float32))
    conv1_bias = tf.Variable(tf.zeros([conv1_features], dtype=tf.float32))

    conv2_weight = tf.Variable(tf.truncated_normal([4, 4, conv1_features, conv2_features],
                                                   stddev=0.1, dtype=tf.float32))
    conv2_bias = tf.Variable(tf.zeros([conv2_features], dtype=tf.float32))

    # fully connected variables
    resulting_width = image_width // (max_pool_size1 * max_pool_size2)
    resulting_height = image_height // (max_pool_size1 * max_pool_size2)
    full1_input_size = resulting_width * resulting_height * conv2_features
    full1_weight = tf.Variable(tf.truncated_normal([full1_input_size, fully_connected_size1],
                              stddev=0.1, dtype=tf.float32))
    full1_bias = tf.Variable(tf.truncated_normal([fully_connected_size1], stddev=0.1, dtype=tf.float32))
    full2_weight = tf.Variable(tf.truncated_normal([fully_connected_size1, target_size],
                                                   stddev=0.1, dtype=tf.float32))
    full2_bias = tf.Variable(tf.truncated_normal([target_size], stddev=0.1, dtype=tf.float32))

    print( "resulting_width :", resulting_width )
    print( "resulting_height :", resulting_height )
    print( "full1_input_size :", full1_input_size )


    # Initialize Model Operations
    def my_conv_net(input_data):
        # First Conv-ReLU-MaxPool Layer
        conv1 = tf.nn.conv2d(input_data, conv1_weight, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
        max_pool1 = tf.nn.max_pool(relu1, ksize=[1, max_pool_size1, max_pool_size1, 1],
                                   strides=[1, max_pool_size1, max_pool_size1, 1], padding='SAME')

        # Second Conv-ReLU-MaxPool Layer
        conv2 = tf.nn.conv2d(max_pool1, conv2_weight, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
        max_pool2 = tf.nn.max_pool(relu2, ksize=[1, max_pool_size2, max_pool_size2, 1],
                                   strides=[1, max_pool_size2, max_pool_size2, 1], padding='SAME')

        # Transform Output into a 1xN layer for next fully connected layer
        final_conv_shape = max_pool2.get_shape().as_list()
        final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
        flat_output = tf.reshape(max_pool2, [final_conv_shape[0], final_shape])

        # First Fully Connected Layer
        fully_connected1 = tf.nn.relu(tf.add(tf.matmul(flat_output, full1_weight), full1_bias))

        # Second Fully Connected Layer
        final_model_output = tf.add(tf.matmul(fully_connected1, full2_weight), full2_bias)
    
        return(final_model_output)

    model_output = my_conv_net(x_input)
    test_model_output = my_conv_net(eval_input)

    #======================================================================
    # 損失関数を設定する。
    # Declare the loss functions.
    #======================================================================
    # Declare Loss Function (softmax cross entropy)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=y_target))

    #======================================================================
    #
    #======================================================================
    # Create a prediction function
    prediction = tf.nn.softmax(model_output)
    test_prediction = tf.nn.softmax(test_model_output)

    # Create accuracy function
    def get_accuracy(logits, targets):
        batch_predictions = np.argmax(logits, axis=1)
        num_correct = np.sum(np.equal(batch_predictions, targets))
        return(100. * num_correct/batch_predictions.shape[0])

    #======================================================================
    # モデルの初期化と学習（トレーニング）
    # ここまでの準備で, 実際に, 計算グラフ（有向グラフ）のオブジェクトを作成し,
    # プレースホルダを通じて, データを計算グラフ（有向グラフ）に供給する。
    # Initialize and train the model.
    #
    # ex) 計算グラフを初期化する方法の１つの例
    #     with tf.Session( graph = graph ) as session:
    #         ...
    #         session.run(...)
    #         ...
    #     session = tf.Session( graph = graph )  
    #     session.run(…)
    #======================================================================
    # Create an optimizer
    my_optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_step = my_optimizer.minimize(loss)

    # Initialize Variables
    init = tf.global_variables_initializer()
    sess.run(init)


    # Start training loop
    train_loss = []
    train_acc = []
    test_acc = []
    for i in range(generations):
        rand_index = np.random.choice(len(X_train), size=batch_size)
        rand_x = X_train[rand_index]
        rand_x = np.expand_dims(rand_x, 3)
        rand_y = y_train[rand_index]
        train_dict = {x_input: rand_x, y_target: rand_y}
    
        sess.run(train_step, feed_dict=train_dict)
        temp_train_loss, temp_train_preds = sess.run([loss, prediction], feed_dict=train_dict)
        temp_train_acc = get_accuracy(temp_train_preds, rand_y)
    
        if (i+1) % eval_every == 0:
            eval_index = np.random.choice(len(X_test), size=evaluation_size)
            eval_x = X_test[eval_index]
            eval_x = np.expand_dims(eval_x, 3)
            eval_y = y_test[eval_index]
            test_dict = {eval_input: eval_x, eval_target: eval_y}
            test_preds = sess.run(test_prediction, feed_dict=test_dict)
            temp_test_acc = get_accuracy(test_preds, eval_y)
        
            # Record and print results
            train_loss.append(temp_train_loss)
            train_acc.append(temp_train_acc)
            test_acc.append(temp_test_acc)
            acc_and_loss = [(i+1), temp_train_loss, temp_train_acc, temp_test_acc]
            acc_and_loss = [np.round(x,2) for x in acc_and_loss]
            print('Generation # {}. Train Loss: {:.2f}. Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))
    
    #======================================================================
    # モデルの評価
    # (Optional) Evaluate the model.
    #======================================================================
    # Matlotlib code to plot the loss and accuracies
    eval_indices = range(0, generations, eval_every)
    # Plot loss over time
    plt.plot(eval_indices, train_loss, 'k-')
    plt.title('Softmax Loss per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Softmax Loss')
    plt.show()

    # Plot train and test accuracy
    plt.plot(eval_indices, train_acc, 'k-', label='Train Set Accuracy')
    plt.plot(eval_indices, test_acc, 'r--', label='Test Set Accuracy')
    plt.title('Train and Test Accuracy')
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    #======================================================================
    # ハイパーパラメータのチューニング (Optional)
    #======================================================================


    #======================================================================
    # デプロイと新しい成果指標の予想 (Optional)
    #======================================================================


    print("Finish main()")
    return
    

if __name__ == '__main__':
     main()
