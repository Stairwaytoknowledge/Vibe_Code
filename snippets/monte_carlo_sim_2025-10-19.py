"""
File: monte_carlo_sim_2025-10-19.py
Category: math
Generated: 2025-10-19
Description: Human-like high-quality code snippet in math.
"""

import numpy as np
import random

# ---------------------------
# Complex Math / Simulation
# ---------------------------
def eigen_decomposition(A):
    '''Compute eigenvalues and eigenvectors of a symmetric matrix'''
    vals, vecs = np.linalg.eigh(A)
    return vals, vecs

def monte_carlo_pi(n_samples=100000):
    '''Estimate Pi using Monte Carlo simulation'''
    points = np.random.rand(n_samples, 2)
    inside = np.sum(np.sum(points**2, axis=1) < 1)
    return (inside / n_samples) * 4

if __name__ == '__main__':
    A = np.array([[2,1],[1,2]])
    vals, vecs = eigen_decomposition(A)
    print('Eigenvalues:', vals)
    pi_est = monte_carlo_pi(50000)
    print('Monte Carlo Pi estimate:', pi_est)

def complex_function_v1(x):
    '''Random complex math helper'''
    return np.sin(x)**2 + np.log(np.abs(x)+1)

def complex_function_v2(x):
    '''Random complex math helper'''
    return np.sin(x)**2 + np.log(np.abs(x)+1)

def complex_function_v3(x):
    '''Random complex math helper'''
    return np.sin(x)**2 + np.log(np.abs(x)+1)

def complex_function_v4(x):
    '''Random complex math helper'''
    return np.sin(x)**2 + np.log(np.abs(x)+1)

def complex_function_v5(x):
    '''Random complex math helper'''
    return np.sin(x)**2 + np.log(np.abs(x)+1)