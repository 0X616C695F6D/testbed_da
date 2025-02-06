"""
Script to convert raw.iq files into npy.

Size of output npy is (4096, 2, 1024) ; (samples, iq, frames).
Signal is also normalized.
"""


import numpy as np
import matplotlib.pyplot as plt

data = np.fromfile('/home/ash/raw.iq')
real_part = data[0::2]
imag_part = data[1::2]

frame_len = 1024
total_frames = 4096
num_samples = total_frames * frame_len

I_frames = real_part[:num_samples].reshape(total_frames, frame_len)
Q_frames = imag_part[:num_samples].reshape(total_frames, frame_len)

IQ_frames = np.stack((I_frames, Q_frames), axis=-1)
IQ_frames = np.transpose(IQ_frames, (0, 2, 1))

peak_value = np.max(np.abs(IQ_frames))
IQ_frames_normalized = IQ_frames / peak_value

assert np.isclose(np.max(np.abs(IQ_frames_normalized)), 1.0, atol=1e-7)

for i in range(10):
    plt.figure(figsize=(12, 6))
    plt.plot(IQ_frames_normalized[i, 0, :], label='In-phase (I)')
    plt.plot(IQ_frames_normalized[i, 1, :], label='Quadrature (Q)')
    plt.legend()
    plt.title(f'Normalized IQ Data - Frame {i*10}')
    plt.show()

np.save('sig', IQ_frames_normalized)
