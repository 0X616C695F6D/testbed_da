#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:06:31 2024

@author: ash
"""


import numpy as np
import math
import matplotlib.pyplot as plt

data = np.fromfile('/home/ash/output.iq')
real_part = data[0::2]
imag_part = data[1::2]

sig_len = math.floor(len(real_part))
frame_len = 1024

total_frames = sig_len / frame_len

for i in range(10):
    start = i*frame_len
    end = start + frame_len
    plt.figure(figsize=(12, 6))
    plt.plot(real_part[start:end], label='In-phase (I)')
    plt.plot(imag_part[start:end], label='Quadrature (Q)')
    plt.show()