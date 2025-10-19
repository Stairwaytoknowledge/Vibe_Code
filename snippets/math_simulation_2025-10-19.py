import numpy as np

# Complex math simulations: Eigenvalues & Monte Carlo
def eigen_decomposition(A):
    '''Compute eigenvalues and eigenvectors of a symmetric matrix'''
    vals, vecs = np.linalg.eigh(A)
    return vals, vecs

def monte_carlo_pi(n_samples=100000):
    '''Estimate Pi using Monte Carlo simulation'''
    points = np.random.rand(n_samples, 2)
    inside = np.sum(np.sum(points**2, axis=1) < 1)
    return (inside / n_samples) * 4

def complex_function_v1(x):
    return np.sin(x)**2 + np.log(np.abs(x)+1)

if __name__ == '__main__':
    A = np.array([[2,1],[1,2]])
    vals, vecs = eigen_decomposition(A)
    print('Eigenvalues:', vals)
    print('Monte Carlo Pi estimate:', monte_carlo_pi(50000))