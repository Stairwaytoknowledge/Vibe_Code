import numpy as np

    """Compute eigenvalues and eigenvectors of a symmetric matrix"""

    def eigen_decomposition(A):
        vals, vecs = np.linalg.eigh(A)
        return vals, vecs
    