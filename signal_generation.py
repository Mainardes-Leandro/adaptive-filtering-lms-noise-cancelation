import numpy as np

def generate_time_vector(fs: int, duration: float):
    """
    Generates a time vector.

    Parameters:
    fs: Sampling frequency
    duration: Signal duration (seconds)

    Returns:
    t: Time vector
    """

    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs

    return t

def generate_clean_signal(t, freq=5):
    """
    Generates a clean sinusoidal signal.

    Parameters:
    t: Time vector
    freq: Signal frequency

    Returns:
    A sinusoidal signal
    """

    return np.sin(2 * np.pi * freq * t)


def generate_noise(n_samples, noise_std=0.5):
    """
    Generates white Gaussian noise.

    Parameters:
    n_samples: number of samples of the noise
    noise_std: Limit value of the noise

    Returns:
    Random noise
    """

    return np.random.normal(0, noise_std, n_samples)


def generate_correlated_noise(noise):
    """
    Creates correlated noise using a simple FIR filter.

    Parameters:
    noise: generated noise

    Returns:
    v: correlated noise
    """

    b = np.array([1.0, 0.5, -0.3])

    v = np.convolve(noise, b, mode="same")

    return v


def create_noisy_signal(signal, noise):
    """
    Adds noise to clean signal.
    """

    return signal + noise


def generate_dataset(fs=500, duration=5):
    """
    Generates all signals needed for the experiment.

    Returns:
    x: reference noise
    d: noisy signal
    s: clean signal
    """

    t = generate_time_vector(fs, duration)

    s = generate_clean_signal(t)

    noise = generate_noise(len(t))

    v = generate_correlated_noise(noise)

    d = create_noisy_signal(s, v)

    x = noise

    return x, d, s
