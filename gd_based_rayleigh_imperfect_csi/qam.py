#!/user/bin/env python
# -*- coding:utf-8 -*-
# author :Guoz time:2019/7/18

# M-ary modulations
def mqam(M):
    import numpy as np
    a = 1 / np.sqrt(2)
    # qam_factor = 0
    if M == 4:
        QAM = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=complex)  # gray mapping
        qam_factor = (2 / 3) * (M - 1)
    elif M == 8:
        QAM = np.array([1, a + a * 1j, -a + a * 1j, 1j, a - a * 1j, -1j, -1, -a - a * 1j],
                       dtype=complex)  # 8PSK, not 8QAM indeed
        qam_factor = 1
    elif M == 16:
        QAM = np.array([-3 + 3j, -3 + 1j, -3 - 3j, -3 - 1j,
                        -1 + 3j, -1 + 1j, -1 - 3j, -1 - 1j,
                        3 + 3j, 3 + 1j, 3 - 3j, 3 - 1j,
                        1 + 3j, 1 + 1j, 1 - 3j, 1 - 1j], dtype=complex)
        qam_factor = (2 / 3) * (M - 1)
    else:
        QAM = np.array([1, -1], dtype=complex)  # BPSK
        qam_factor = 1

    return QAM,qam_factor