import numpy as np
"""
Filtro adaptativo LMS
num_taps: Number of cofficients
mu: Adaptation step size

Return: None

The run_filter function is responsible for running the filter
x: Input signal.
d: Desired signal.

Return:
y: Output signal.
e: Error.
w_history: weight matrix

"""
class LMSFilter:
    def __init__(self, num_taps, mu):
        self.num_taps = num_taps
        self.mu = mu
        self.w = np.zeros(num_taps)

    def run_filter(self, x, d):
        n_samples = len(x)

        y = np.zeros(n_samples)
        e = np.zeros(n_samples)

        w_history = np.zeros((n_samples, self.num_taps))

        x_buffer = np.zeros(self.num_taps)

        for n in range(n_samples):

            x_buffer[1:] = x_buffer[:-1]
            x_buffer[0] = x[n]

            y[n] = np.dot(self.w, x_buffer)

            e[n] = d[n] - y[n]

            self.w = self.w + self.mu * x_buffer * e[n]

            w_history[n] = self.w

        return y, e, w_history
