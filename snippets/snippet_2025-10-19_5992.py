import numpy as np

# Generate a random noise image of given shape
def generate_noise_image(shape=(64,64)):
    return np.random.randn(*shape)

# Generate procedural coordinates for visualization
def procedural_pattern(n=100):
    return [(i**2 % n, i*3 % n) for i in range(n)]