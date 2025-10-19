
"""
File: snippets/gen_ai_2025-10-19_2223.py
Category: Generative AI
Generated: 2025-10-19
Description: Generates a simple procedural image and pattern for AI tasks.
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate random noise image
def noise_image(shape=(64,64)):
    return np.random.randn(*shape)

# Generate procedural coordinates
def procedural_pattern(n=200):
    return [(i**2 % n, i*3 % n) for i in range(n)]

# Visualize pattern
pattern = procedural_pattern(300)
img = noise_image((128,128))

# Plot procedural pattern
plt.scatter(*zip(*pattern), c='blue', alpha=0.5)
plt.title("Procedural Pattern")
plt.show()

# Show noise image
plt.imshow(img, cmap='gray')
plt.title("Noise Image")
plt.show()
