#!/user/bin/env python
# -*- coding:utf-8 -*-
# author :Guoz time:2019/7/19
"""
本程序为导入mimo_ofdm_im模型，进行训练，Rayleigh信道
"""
import tensorflow as tf
from scipy.special import binom
from mimo_ofdm_im_funcs import mimo_ofdm_im_train,mimo_ofdm_im_test,feature_genator,frange
from data_saver import *

import numpy as np

#参数设置

batch_size = 1000 #每个batch的大小
N = 4  # 每组子载波个数
K = 2  # 激活子载波
M = 2  # M阶调制
nTx = 2
nRx = 2
SNR_db = 5

traing_epochs = 11 #训练的代数
l_rate = 0.01
total_batch = 20 # 每代的batch数

n_hidden_1 = 128   # 小用Tanh大用ReLu
#输入尺寸
n_input = N
n_output = N #两个发射天线，每个天线中子组的子载波数N=4
#打印设置
display_step = 5

X = tf.placeholder("float", [None, n_input]) #占位符并没有初始值，它只会分配必要的内存。通过feed_dict字典送入数据
Y = tf.placeholder("float", [None, n_output])
initializer = tf.contrib.layers.xavier_initializer() #参数矩阵w,b通过该函数初始化

#把深度神经网络的构造定义为dl_layers函数
def dl_layers(x,Q):
    weights = {
        'encoder_h1': tf.Variable(initializer([n_input, Q])),
        'encoder_h2': tf.Variable(initializer([Q, n_output])),
    }
    biases = {
        'encoder_b1': tf.Variable(initializer([Q])),
        'encoder_b2': tf.Variable(initializer([n_output])),

    }

    if Q == 16 or Q == 32 or Q == 64:
        print('隐藏层1的激励函数为tanh!')
        layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))

    else:
        print('隐藏层1的激励函数为relu!')
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))

    return layer_2

y_pred = dl_layers(X,n_hidden_1) #经过深度学习(DL）后的预测值
y_true = Y #真实值

cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2)) #损失函数
learning_rate = tf.placeholder(tf.float32, shape=[]) #学习速率
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) #优化器
init = tf.global_variables_initializer() #添加节点用于初始化所有的变量

# 训练和测试
with tf.Session() as sess:
    # Training
    sess.run(init)
    for epoch in range(traing_epochs):
        avg_cost = 0.
        for index_m in range(total_batch):
            input_samples = []
            input_labels = []
            for index_k in range(0, batch_size):
                bits, bits_Tx1, bits_Tx2, Hg, h_con, Yg, Yg_Rx1, Yg_Rx2 = mimo_ofdm_im_train(N, K, M, SNR_db)
                Y_H_features,Yg_Rx1_bar = feature_genator(N,Hg,Yg,Yg_Rx1)
                input_labels.append(bits_Tx1)
                input_samples.append(Yg_Rx1_bar)

            batch_x = np.asarray(input_samples)#预处理后的输出值,训练数据集
            batch_y = np.asarray(input_labels)#输入比特作为样本，训练数据集
            _, cs = sess.run([optimizer, cost], feed_dict={X: batch_x,
                                                           Y: batch_y,
                                                           learning_rate: l_rate})
            avg_cost += cs / total_batch
        if epoch % display_step == 0:
            print("Training-->Epoch:", '%04d' % (epoch + 1), "cost=", \
                  "{:.9f}".format(avg_cost))
    print("Training has been done!")
    # Testing
    EbNodB_range = list(frange(0,11, 1))
    ber = [None] * len(EbNodB_range)
    for n in range(0, len(EbNodB_range)):
        input_samples_test = []
        input_labels_test = []
        if n<=10:
            test_number = 100000
        elif n > 10 and n<15: #低信噪高的时候，多跑一些点
            test_number = 1000000
        elif n >= 15 and n<20:
            test_number = 1500000
        elif n >= 20 and n<25:
            test_number = 2000000
        elif n >= 25 and n < 35:
            test_number = 3000000
        else:
            test_number = 5000000

        for i_test_number in range(0, test_number):
            bits, bits_Tx1, bits_Tx2, Hg, h_con, Yg, Yg_Rx1, Yg_Rx2 = mimo_ofdm_im_test(N, K, M,EbNodB_range[n])
            Y_H_features,Yg_Rx1_bar = feature_genator(N,Hg,Yg,Yg_Rx1)
            input_labels_test.append(bits_Tx1)  # all_bits
            input_samples_test.append(Yg_Rx1_bar)  # y_

        batch_x = np.asarray(input_samples_test)  #把输出信号signal_output作为测试集的样本
        batch_y = np.asarray(input_labels_test)   #把输入比特信号bits测试集的标签
        mean_error = tf.reduce_mean(abs(y_pred - batch_y))  # mean_error.eval({X:batch_x}),
        print("Testing...")
        mean_error_rate = 1 - tf.reduce_mean(
            tf.reduce_mean(
                tf.to_float(
                    tf.equal(
                        tf.sign(y_pred - 0.5), tf.cast(tf.sign(batch_y - 0.5), tf.float32)
                    )
                ),
            1)
        )
        ber[n] = mean_error_rate.eval({X: batch_x})

        print("SNR=", EbNodB_range[n], "BER:", ber[n])

    import matplotlib.pyplot as plt
    import time
    time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())

    if n_hidden_1 == 16 or n_hidden_1 == 32 or n_hidden_1 == 64:
        activation_func = "tanh"
    else:
        activation_func = "relu"
    fig_name = 'MIMO-OFDM-IM_Rayleigh({},{},{}) SNR_db{} EbNodB_range{} Q={} {} {} {}'\
        .format(N,K,M,SNR_db,EbNodB_range,n_hidden_1,activation_func,traing_epochs-1,time_str)
    data_save_write(ber, fig_name)

    fig_label = 'DeepIM,Q={},{},epochs={}'.format(n_hidden_1,activation_func,traing_epochs-1)
    plt.plot(EbNodB_range, ber, 'bo-', label= fig_label)
    plt.yscale('log')
    plt.xlabel('SNR Range')
    plt.ylabel('BER')
    fig_title = 'MIMO-OFDM-IM_Rayleigh({},{},{})'.format(N,K,M)
    plt.title(fig_title)
    plt.grid()
    plt.legend(loc='upper right', ncol=1)
    plt.savefig('./figures/'+ fig_name + '.png')
    plt.show()
    plt.close()
    print("图像成功保存！")


