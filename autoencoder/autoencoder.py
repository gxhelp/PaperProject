#!/user/bin/env python
# -*- coding:utf-8 -*-
# author :Guoz time:2019/9/13

# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
#
# from tensorflow.examples.tutorials.mnist import input_data
#
# mnist = input_data.read_data_sets("/tmp/data/",one_hot=False)
#
# #Visualizing decoder setting
# learning_rate = 0.001
# training_epochs = 2
# batch_size = 256
# display_step = 1
# examples_to_show = 10
#
# n_input = 784
#
# X = tf.placeholder('float',[None,n_input])
#
# n_hidden_1 = 256
# n_hidden_2 = 128
# n_hidden_3 = 64
# n_hidden_4 = 16
#
# weight = {
#     'encoder_h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
#     'encoder_h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
#     'encoder_h3':tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3])),
#     'encoder_h4':tf.Variable(tf.random_normal([n_hidden_3,n_hidden_4])),
#
#     'decoder_h1':tf.Variable(tf.random_normal([n_hidden_4,n_hidden_3])),
#     'decoder_h2':tf.Variable(tf.random_normal([n_hidden_3,n_hidden_2])),
#     'decoder_h3':tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1])),
#     'decoder_h4':tf.Variable(tf.random_normal([n_hidden_1,n_input])),
# }
#
# biases = {
#     'encoder_b1':tf.Variable(tf.random_normal([n_hidden_1])),
#     'encoder_b2':tf.Variable(tf.random_normal([n_hidden_2])),
#     'encoder_b3':tf.Variable(tf.random_normal([n_hidden_3])),
#     'encoder_b4':tf.Variable(tf.random_normal([n_hidden_4])),
#
#     'decoder_b1':tf.Variable(tf.random_normal([n_hidden_3])),
#     'decoder_b2':tf.Variable(tf.random_normal([n_hidden_2])),
#     'decoder_b3':tf.Variable(tf.random_normal([n_hidden_1])),
#     'decoder_b4':tf.Variable(tf.random_normal([n_input]))
# }
#
# def encoder(x):
#     layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weight['encoder_h1']),biases['encoder_b1']))
#     layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weight['encoder_h2']),biases['encoder_b2']))
#     layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2,weight['encoder_h3']),biases['encoder_b3']))
#     layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3,weight['encoder_h4']),biases['encoder_b4']))
#     return layer_4
#
# def decoder(x):
#     layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weight['decoder_h1']),biases['decoder_b1']))
#     layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weight['decoder_h2']),biases['decoder_b2']))
#     layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2,weight['decoder_h3']),biases['decoder_b3']))
#     layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3,weight['decoder_h4']),biases['decoder_b4']))
#     return layer_4
#
# #construct model
# encoder_op = encoder(X)
# decoder_op = decoder(encoder_op)
#
# #Prediction
# y_pred = decoder_op
# y_true = X
#
# #Define loss and optimizer
# cost = tf.reduce_mean(tf.pow(y_true - y_pred,2))
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
#
# init = tf.initialize_all_variables()
#
# with tf.Session() as sess:
#     sess.run(init)
#     total_batch = int(mnist.train.num_examples/batch_size)
#     #training cycle
#     for epoch in range(training_epochs):
#         for i in range(total_batch):
#             batch_xs,batch_ys = mnist.train.next_batch(batch_size)
#             _,c = sess.run([optimizer,cost],feed_dict={X:batch_xs})
#
#             if epoch %display_step == 0:
#                 print("Epoch",'%04d'%(epoch+1),
#                       "cost=","{:.9f}".format(c))
#     print("Optimization Finished!")
#
#     encode_decode = sess.run(
#         y_pred,feed_dict= {X:mnist.test.images[:examples_to_show]}
#     )
#     f,a = plt.subplots(2,10,figsize = (10,2))
#     for i in range(examples_to_show):
#         a[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
#         a[1][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
#     plt.show()

"""
以上是Autoencoder的基本结构，现在最主要的是要产生数据，MIMO-OFDM-IM的数据
"""
#创建MIMO-OFDM-IM系统模型
# k = 3 # Number of information bits per channel usage
M = 2
n = 1 # Number of complex channel uses per message
seed = 1 # Seed RNG reproduce identical results
nTx = 2 #发射天线个数
nRx = 2 #接收天线个数
N = 4 #每个subblock中子载波总数
K = 2 #每个subblock中激活子载波个数
model_file_1 = 'MIMO_OFDM_IM/'
# model_file_2 = '_k_{}_n_{}_tx_{}_rx_{}_s_{}'.format(k, n, mt, mr, seed)


assert n is 1, "n does not equal 1"
assert nTx == nRx, "Number of Transmit and Receive antennas is not equal"


# ## Build Autoencoder with subfunctions

# In[ ]:
import numpy as np

class AutoEncoder(object):
    def __init__(self, M, n, mt, mr, N, K, seed=None, filename=None):
        '''This function is the initialization function of the Autoencoder class'''
        import time
        from scipy.special import binom

        self.M = M
        self.n = n
        self.nTx = nTx
        self.nRx = nRx
        self.N = N
        self.K = K
        self.index_bits = int( np.log2( binom(N, K) ))  # 索引比特数目
        self.info_bits = np.log2(self.M)*self.K  # 信息比特数目

        self.bits_per_symbol = self.info_bits / self.n
        # self.info_bits = np.log2(self.M)
        self.T = 2 ** (self.nTx * self.info_bits)  # Anzahl möglicher Tupel
        self.seed = seed if (seed is not None) else int(time.time())
        self.graph = None
        self.sess = None
        self.vars = None
        self.saver = None
        self.constellations = None
        # self.create_graph()
        # self.create_session()
        # self.sum = self.sum_writer()
        # if filename is not None:
        #     self.load(filename)
        return

    def source_bits(self):
        q = self.index_bits + self.info_bits
        # print(q)
        q = int(q)
        bits = np.random.binomial(n=1, p=0.5, size=(q,))
        bit_index = bits[0:self.index_bits:1]

        bit_info = bits[self.index_bits:q:1]
        return bits,bit_index,bit_info

    def qam_constellion_gen(self,M):
        a = 1 / np.sqrt(2)
        # qam_factor = 0
        if M == 4:
            qam_constellions = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=complex)  # gray mapping
            qam_factor = (2 / 3) * (M - 1)
        elif M == 8:
            qam_constellions = np.array([1, a + a * 1j, -a + a * 1j, 1j, a - a * 1j, -1j, -1, -a - a * 1j],
                           dtype=complex)  # 8PSK, not 8QAM indeed
            qam_factor = 1
        elif M == 16:
            qam_constellions = np.array([-3 + 3j, -3 + 1j, -3 - 3j, -3 - 1j,
                            -1 + 3j, -1 + 1j, -1 - 3j, -1 - 1j,
                            3 + 3j, 3 + 1j, 3 - 3j, 3 - 1j,
                            1 + 3j, 1 + 1j, 1 - 3j, 1 - 1j], dtype=complex)
            qam_factor = (2 / 3) * (M - 1)
        else:
            qam_constellions = np.array([1, -1], dtype=complex)  # BPSK
            qam_factor = 1

        return qam_constellions, qam_factor

    def index_lookup_table_gen(self,N,K):
        idx = []
        if N == 4:
            if K == 1:
                idx = np.array([[0], [1], [2], [3]])
            elif K == 2:
                idx = np.array([[0, 1], [2, 3], [0, 2], [1, 3]])
            else:
                idx = np.array([[0, 1, 2], [1, 2, 3], [0, 2, 3], [0, 1, 3]])

        assert N is 4, "N, the num of subcarrier, does not equal 4"
        return idx

    def ofdm_im(self,bit_index,bit_info):
        qam_constellions, qam_factor = self.qam_constellion_gen(self.M)
        index_lookup_table = self.index_lookup_table_gen(self.N,self.K)
        index_decimal = bit_index.dot(2 ** np.arange(bit_index.size)[::-1])
        num_info_bit_per_active_subcarrier = int(self.info_bits/self.K)
        bit_K = bit_info.reshape(-1, num_info_bit_per_active_subcarrier)
        info_decimal = np.zeros((K,), dtype=int)
        sym = np.zeros((K,), dtype=complex)
        for i in range(K):
            bit_sy_i = bit_K[i, :]
            info_decimal[i] = bit_sy_i.dot(2 ** np.arange(bit_sy_i.size)[::-1])
            sym[i] = qam_constellions[info_decimal[i]]

        power = np.sqrt(self.N / self.K / qam_factor)  # power allocation factor
        tx_symbols = np.zeros((N,), dtype=complex)
        tx_symbols[index_lookup_table[index_decimal, :]] = sym*power

        return tx_symbols

    def mimo(self):
        symbols_for_antenas = np.zeros([self.nTx,self.N],dtype = complex)
        source_bits_for_antenas = np.zeros([self.nTx, int(self.index_bits+self.info_bits)])
        for i_tx in range(self.nTx):
            bits, bit_index, bit_info = self.source_bits()
            source_bits_for_antenas[i_tx] = bits
            tx_symbols = self.ofdm_im(bit_index, bit_info)
            symbols_for_antenas[i_tx] = tx_symbols #每一行代表1个发射天线的符号，每一列代表每个子载波

        return source_bits_for_antenas,symbols_for_antenas

    def channel_gen(self,channel_type):
        pass

    def pass_channel(self):
        pass

# ae = AutoEncoder(M, n, nTx, nRx, N, K)
# # bits,bit_index,bit_info = ae.source_bits()
# # print(bits,bit_index,bit_info)
# #
# # # index_lookup_table = ae.index_lookup_table_gen(N,K)
# # # print(index_lookup_table)
# #
# # tx_symbols = ae.ofdm_im(bit_index,bit_info)
# # print(tx_symbols)
#
# source_bits,mimo_symbols = ae.mimo()
# print(source_bits,'\n',mimo_symbols)
