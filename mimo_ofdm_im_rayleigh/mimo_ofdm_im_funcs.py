#!/user/bin/env python
# -*- coding:utf-8 -*-
# author :Guoz time:2019/7/19

def mimo_ofdm_im_train(N,K,M,SNRdb):

    import numpy as np
    from scipy.special import binom
    from qam import mqam

    nRx = 2
    nTx = 2

    m = int(np.log2(M))
    cnk = binom(N, K)
    c = int(np.log2(cnk))
    q = K * m + c  # number of bits per OFDM-IM symbol

    nTx_q = nTx*q

    bits = np.random.binomial(n=1, p=0.5, size=(nTx_q,))
    bits2 = bits.reshape(nTx,q)
    QAM,qam_factor = mqam(M)
    if M==8 or M==2:
        qam_factor = 1
    else:
        qam_factor = (2 / 3) * (M - 1)
    power = np.sqrt(N / K / qam_factor)  # power allocation factor

    # index patterns for N=4 and K=1,2,3 only
    if K == 1:
        idx = np.array([[0], [1], [2], [3]])
    elif K == 2:
        idx = np.array([[0, 1], [2, 3], [0, 2], [1, 3]])
    else:
        idx = np.array([[0, 1, 2], [1, 2, 3], [0, 2, 3], [0, 1, 3]])

    tx_sym = np.zeros((nTx,N), dtype=complex)
    for i_nTx in range(nTx):
        bits_use = bits2[i_nTx,:]
        bit_id = bits_use[0:c:1]
        id_de = bit_id.dot(2 ** np.arange(bit_id.size)[::-1])

        bit_sy = bits_use[c:q:1]
        bit_K = bit_sy.reshape(-1, m)
        sy_de = np.zeros((K,), dtype=int)
        sym = np.zeros((K,), dtype=complex)
        for i in range(K):
            bit_sy_i = bit_K[i, :]
            sy_de[i] = bit_sy_i.dot(2 ** np.arange(bit_sy_i.size)[::-1])
            sym[i] = QAM[sy_de[i]]
        tx_sym[i_nTx,idx[id_de, :]] = sym*power

    tx_subblock = tx_sym.reshape(-1, 1)

    SNR = 10 ** (SNRdb / 10)
    sigma = np.sqrt(1 / SNR)
    # eps = 1./(1 + SNR) # imperfect CSI
    # eps = 0.0

    #MIMO Rayleigh Block-Fading Channel
    h11 = 1 / np.sqrt(2) * (np.random.randn(1, 1) + 1j * np.random.randn(1, 1))[0][0]
    h12 = 1 / np.sqrt(2) * (np.random.randn(1, 1) + 1j * np.random.randn(1, 1))[0][0]
    h21 = 1 / np.sqrt(2) * (np.random.randn(1, 1) + 1j * np.random.randn(1, 1))[0][0]
    h22 = 1 / np.sqrt(2) * (np.random.randn(1, 1) + 1j * np.random.randn(1, 1))[0][0]
    # h11 = 1
    # h12 = 0
    # h21 = 0
    # h22 = 1
    h_con = [h11,h12,h21,h22]
    Hg = np.array([[h11,h12,0,0,0,0,0,0], [h21,h22,0,0,0,0,0,0],[0,0,h11,h12,0,0,0,0],[0,0,h21,h22,0,0,0,0],
                   [0,0,0,0,h11,h12,0,0],[0,0,0,0,h21,h22,0,0],[0,0,0,0,0,0,h11,h12],[0,0,0,0,0,0,h21,h22]])

    Wg = 1 / np.sqrt(2) * (np.random.randn(nRx * N, 1) + 1j * np.random.randn(nRx * N, 1)) # 得到on multi-input multi-output文章中的公式（6）中的Wg, 尺寸是RN * 1
    Yg = Hg.dot(tx_subblock) + sigma*np.sqrt(1/nRx)*Wg
    Yg = Yg.reshape(1, -1)



    Yg_Rx1 = Yg[0][:N] #接收天线1的y信号
    Yg_Rx2 = Yg[0][N:]

    bits_Tx1 = bits2[0,:]#发射天线1的x比特
    bits_Tx2 = bits2[1, :]

    Yg = Yg.reshape(-1, 1)
    h_inv = np.linalg.inv(Hg)
    Yg_bar = h_inv.dot(Yg)

    return bits,bits_Tx1,bits_Tx2,Hg,h_con,Yg,Yg_Rx1,Yg_Rx2


def feature_genator(N,Hg,Yg,Yg_Rx1):
    import numpy as np
    Yg = Yg.reshape(-1, 1)
    h_inv = np.linalg.inv(Hg)
    Yg_bar = h_inv.dot(Yg)

    Yg_bar_temp = Yg_bar[:,0]
    Yg_Rx1_bar = Yg_bar_temp[:N]
    Yg_Rx2_bar = Yg_bar_temp[N:] #不用
    #提取特征值
    y_con = np.concatenate((np.real(Yg_Rx1_bar), np.imag(Yg_Rx1_bar)))  #Y的实部和虚部
    y_m = np.absolute((Yg_Rx1)) #y_m能量值
    Yg_Rx1_features = np.concatenate((y_con, y_m))

    Yg_Rx1_features = Yg_Rx1_features.reshape(-1, 1)

    return Yg_Rx1_features,Yg_Rx1_bar


def mimo_ofdm_im_test(N,K,M,SNRdb_range):

    import numpy as np
    from scipy.special import binom
    from qam import mqam

    SNRdb = SNRdb_range
    nRx = 2
    nTx = 2

    m = int(np.log2(M))
    cnk = binom(N, K)
    c = int(np.log2(cnk))
    q = K * m + c  # number of bits per OFDM-IM symbol

    nTx_q = nTx*q

    bits = np.random.binomial(n=1, p=0.5, size=(nTx_q,))
    bits2 = bits.reshape(nTx,q)
    QAM,qam_factor = mqam(M)

    # print(64,QAM,qam_factor)
    # power = np.sqrt(N / K / qam_factor)  # power allocation factor

    # index patterns for N=4 and K=1,2,3 only
    if K == 1:
        idx = np.array([[0], [1], [2], [3]])
    elif K == 2:
        idx = np.array([[0, 1], [2, 3], [0, 2], [1, 3]])
    else:
        idx = np.array([[0, 1, 2], [1, 2, 3], [0, 2, 3], [0, 1, 3]])

    tx_sym = np.zeros((nTx,N), dtype=complex)
    for i_nTx in range(nTx):
        bits_use = bits2[i_nTx,:]
        bit_id = bits_use[0:c:1]
        id_de = bit_id.dot(2 ** np.arange(bit_id.size)[::-1])

        bit_sy = bits_use[c:q:1]
        bit_K = bit_sy.reshape(-1, m)
        sy_de = np.zeros((K,), dtype=int)
        sym = np.zeros((K,), dtype=complex)
        for i in range(K):
            bit_sy_i = bit_K[i, :]
            sy_de[i] = bit_sy_i.dot(2 ** np.arange(bit_sy_i.size)[::-1])
            sym[i] = QAM[sy_de[i]]
        tx_sym[i_nTx,idx[id_de, :]] = sym

    tx_subblock = tx_sym.reshape(-1, 1)

    SNR = 10 ** (SNRdb / 10)
    sigma = np.sqrt(1 / SNR)
    # eps = 1./(1 + SNR) # imperfect CSI
    # eps = 0.0

    #MIMO Rayleigh Block-Fading Channel
    h11 = 1 / np.sqrt(2) * (np.random.randn(1, 1) + 1j * np.random.randn(1, 1))[0][0]
    h12 = 1 / np.sqrt(2) * (np.random.randn(1, 1) + 1j * np.random.randn(1, 1))[0][0]
    h21 = 1 / np.sqrt(2) * (np.random.randn(1, 1) + 1j * np.random.randn(1, 1))[0][0]
    h22 = 1 / np.sqrt(2) * (np.random.randn(1, 1) + 1j * np.random.randn(1, 1))[0][0]
    # h11 = 1
    # h12 = 0
    # h21 = 0
    # h22 = 1
    h_con = [h11,h12,h21,h22]
    Hg = np.array([[h11,h12,0,0,0,0,0,0], [h21,h22,0,0,0,0,0,0],[0,0,h11,h12,0,0,0,0],[0,0,h21,h22,0,0,0,0],
                   [0,0,0,0,h11,h12,0,0],[0,0,0,0,h21,h22,0,0],[0,0,0,0,0,0,h11,h12],[0,0,0,0,0,0,h21,h22]])

    Wg = 1 / np.sqrt(2) * (np.random.randn(nRx * N, 1) + 1j * np.random.randn(nRx * N, 1)) # 得到on multi-input multi-output文章中的公式（6）中的Wg, 尺寸是RN * 1
    Yg = Hg.dot(tx_subblock) + sigma*np.sqrt(1/nRx)*Wg
    Yg = Yg.reshape(1, -1)

    Yg_Rx1 = Yg[0][:N]
    Yg_Rx2 = Yg[0][N:]

    bits_Tx1 = bits2[0,:]#发射天线1的x比特
    bits_Tx2 = bits2[1, :]

    return bits,bits_Tx1,bits_Tx2,Hg,h_con,Yg,Yg_Rx1,Yg_Rx2

def frange(x, y, jump):
    while x < y:
        yield x
        x += jump