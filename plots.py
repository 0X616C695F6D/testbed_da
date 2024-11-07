import numpy as np
import matplotlib.pyplot as plt

def plot_signal(file_path, signal_index=0):
    """
    Plot a waveform sample.

    Parameters
    ----------
    file_path : Path to file containing feature set
    signal_index : Index of waveform to sample

    Returns
    -------
    Plot of waveform

    """
    X = np.load(file_path)
    
    signal = X[signal_index]
    
    plt.figure(figsize=(12, 6))
    plt.plot(signal[:, 0], label='In-phase (I)')
    plt.plot(signal[:, 1], label='Quadrature (Q)')
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    plt.show()