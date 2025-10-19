"""
File: snippets/gen_ai_2025-10-19_xyd1.py
Category: gen_ai
Generated: 2025-10-19
Description: Complex human-style snippet in gen_ai.
"""

import numpy as np
    import random

    # Procedural content / generative simulation
    def procedural_pattern(n=100):
        return [(i**2 % n, i*3 % n) for i in range(n)]

    def noise_image(shape=(64,64)):
        return np.random.randn(*shape)

    if __name__=='__main__':
        pattern = procedural_pattern(200)
        img = noise_image((128,128))
        print("Generated procedural pattern points:", pattern[:5])
        print("Generated noise image shape:", img.shape)
    