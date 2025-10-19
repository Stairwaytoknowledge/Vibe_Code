
# File: snippets/gen_ai_2025-10-19_2241.py
# Category: Generative AI
# Description: Generate procedural patterns and simulate noise images for AI tasks.

import numpy as np
import matplotlib.pyplot as plt
import itertools

# Generate procedural coordinates
def procedural_pattern(n=300):
    return [(i**2 % n, i*3 % n) for i in range(n)]

# Generate noise image
def noise_image(shape=(128,128)):
    return np.random.randn(*shape)

# Generate patterns and noise
pattern = procedural_pattern(500)
img = noise_image((256,256))

# Combine multiple noise images for variation
img_combined = sum([noise_image((256,256)) for _ in range(3)]) / 3.0

# Visualize procedural patterns
plt.figure(figsize=(8,8))
plt.scatter(*zip(*pattern), c='red', alpha=0.6, s=10)
plt.title("Procedural Pattern Points")
plt.show()

# Visualize noise image
plt.imshow(img_combined, cmap='gray')
plt.title("Combined Noise Image")
plt.colorbar()
plt.show()

# Generate higher dimensional coordinates for potential generative AI tasks
high_dim_coords = np.random.rand(500,5)
covariance = np.cov(high_dim_coords.T)
eigvals, eigvecs = np.linalg.eigh(covariance)
print("Top 3 eigenvalues of high-dimensional coordinates:", eigvals[-3:])
