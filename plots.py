import numpy as np
import matplotlib.pyplot as plt

def plot_signal(file_path, signal_index=0):
    """
    Plot a waveform sample.

    Parameters
    ----------
    file_path : Path to file containing feature set.
        Has to be a numpy file containing IQ data.
    signal_index : Index of waveform to sample.
        Waveform selected to be plotted.

    Returns
    -------
    Plot of waveform
        For best visibilty and abstraction, spines are
        invisible and tickers are off. Could be changed later.

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