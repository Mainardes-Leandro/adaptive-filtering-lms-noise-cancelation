import numpy as np
import matplotlib.pyplot as plt

from lms import LMSFilter
from signal_generation import generate_dataset

def run_experiment():
    
    # Parameters
    fs = 500
    duration = 5
    num_taps = 16
    mu = 0.001

    # Create results folder
    if save_figures:
        os.makedirs("results/figures", exist_ok=True)

    # Generate signals
    x, d, s = generate_dataset(fs, duration)

    # LMS
    lms = LMSFilter(num_taps=num_taps, mu=mu)
    y, e, w_history = lms.adapt(x, d)

    # Plot signals
    plt.figure()
    plt.plot(d, label="Noisy signal (d)")
    plt.plot(s, label="Clean signal (s)", linestyle="--")
    plt.plot(e, label="Estimated clean signal (e)", alpha=0.8)
    plt.legend()
    plt.title("Signal Comparison")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid()

    if save_figures:
        plt.savefig("results/figures/signal_comparison.png", dpi=300)

    # Error convergence
    mse = e**2

    plt.figure()
    plt.plot(mse)
    plt.title("Error Convergence (Instantaneous MSE)")
    plt.xlabel("Samples")
    plt.ylabel("Error^2")
    plt.grid()

    if save_figures:
        plt.savefig("results/figures/error_convergence.png", dpi=300)

    # Weight evolution
    plt.figure()
    for i in range(w_history.shape[1]):
        plt.plot(w_history[:, i], alpha=0.6)
    
    plt.title("Weight Evolution")
    plt.xlabel("Samples")
    plt.ylabel("Coefficient Value")
    plt.grid()

    if save_figures:
        plt.savefig("results/figures/weight_evolution.png", dpi=300)
    
    plt.show()

if __name__ == "__main__":
    run_experiment()
