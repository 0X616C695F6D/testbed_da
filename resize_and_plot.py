"""
Script to convert raw.iq files into npy.

Size of output npy is (4096, 2, 1024) ; (samples, iq, frames).
Signal is also normalized.
"""

#%% New normalization method - based on local (per-frame) max ^_^
import numpy as np
import matplotlib.pyplot as plt

# Read raw IQ data from file
data = np.fromfile('/home/ash/raw.iq')

# Separate real and imaginary parts
real_part = data[0::2]
imag_part = data[1::2]

# Change frame length from 1024 to 4096
frame_len = 4096  
total_frames = 2100  # number of frames (samples)
num_samples = total_frames * frame_len

# Reshape into frames
I_frames = real_part[:num_samples].reshape(total_frames, frame_len)
Q_frames = imag_part[:num_samples].reshape(total_frames, frame_len)

# Stack I and Q so that each sample is a 2 x frame_len array.
# Current stacking: shape will be (total_frames, frame_len, 2)
IQ_frames = np.stack((I_frames, Q_frames), axis=-1)

# Transpose to have shape (total_frames, 2, frame_len)
IQ_frames = np.transpose(IQ_frames, (0, 2, 1))

# Per-frame normalization: For each frame, divide by its own max absolute value
IQ_frames_normalized = np.empty_like(IQ_frames)
for i in range(total_frames):
    frame = IQ_frames[i]  # shape (2, frame_len)
    max_val = np.max(np.abs(frame))
    if max_val != 0:
        IQ_frames_normalized[i] = frame / max_val
    else:
        IQ_frames_normalized[i] = frame  # if the frame is all zeros, leave it as is

final_frames = IQ_frames_normalized[16:]

# Optionally, check that each frame's max value is 1 (or very close)
frame_max_values = np.max(np.abs(final_frames), axis=(1, 2))
if not np.all(np.isclose(frame_max_values, 1.0, atol=1e-7)):
    print("Warning: Not all frames are normalized to 1.")

# Plot a few normalized frames to visually inspect the result
for i in range(10):
    plt.figure(figsize=(12, 6))
    plt.plot(final_frames[i, 0, :], label='In-phase (I)')
    plt.plot(final_frames[i, 1, :], label='Quadrature (Q)')
    plt.legend()
    plt.title(f'Normalized IQ Data - Frame {i}')
    plt.show()

# Save the normalized IQ data to a NPY file
np.save('sig', final_frames)


#%%  Old normalization method - based on global max T_T
# import numpy as np
# import matplotlib.pyplot as plt

# data = np.fromfile('/home/ash/raw.iq')
# real_part = data[0::2]
# imag_part = data[1::2]

# frame_len = 1024
# total_frames = 4096
# num_samples = total_frames * frame_len

# I_frames = real_part[:num_samples].reshape(total_frames, frame_len)
# Q_frames = imag_part[:num_samples].reshape(total_frames, frame_len)

# IQ_frames = np.stack((I_frames, Q_frames), axis=-1)
# IQ_frames = np.transpose(IQ_frames, (0, 2, 1))

# peak_value = np.max(np.abs(IQ_frames))
# IQ_frames_normalized = IQ_frames / peak_value

# assert np.isclose(np.max(np.abs(IQ_frames_normalized)), 1.0, atol=1e-7)

# for i in range(10):
#     plt.figure(figsize=(12, 6))
#     plt.plot(IQ_frames_normalized[i, 0, :], label='In-phase (I)')
#     plt.plot(IQ_frames_normalized[i, 1, :], label='Quadrature (Q)')
#     plt.legend()
#     plt.title(f'Normalized IQ Data - Frame {i*10}')
#     plt.show()

# np.save('sig', IQ_frames_normalized)
