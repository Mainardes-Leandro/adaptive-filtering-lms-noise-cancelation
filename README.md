# Adaptive Filtering with LMS for Noise Cancellation

## Description

This project demonstrates the implementation of an adaptive filter using the Least Mean Squares (LMS) algorithm for noise cancellation in time-series signals.

The goal is to recover a clean signal from a noisy observation by adaptively estimating and removing correlated noise.

## The Problem

We consider a classical signal processing scenario:

d(n) = s(n) + v(n)

Where:
- s(n): clean signal
- v(n): noise
- d(n): observed noisy signal

The LMS filter uses a reference noise signal to estimate v(n) and subtract it from d(n), recovering an estimate of s(n).

## The Algorithm

The LMS algorithm updates the filter coefficients as:

w(n+1) = w(n) + μ x(n) e(n)

Where:
- μ: step size (learning rate)
- x(n): input vector
- e(n): error signal

The objective is to minimize the mean squared error (MSE).

## Project Structure

project/
│
├── src/
│   ├── lms.py
│   ├── signal_generation.py
│   └── experiment.py
│
├── results/
│   └── figures/
│
├── requirements.txt
└── README.md

## Execution

1. Clone the repository:
   git clone <your-repo-url>

2. Install dependencies:
   pip install -r requirements.txt

3. Run the experiment:
   python src/experiment.py
