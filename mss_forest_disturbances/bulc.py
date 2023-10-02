"""Numpy implementation of BULC-P.
"""

import numpy as np


def bulcp(array, leveler=0.8):
    assert leveler >= 0 and leveler <= 1

    output = np.zeros_like(array)
    output[0] = array[0]

    num_classes = array.shape[-1]
    min_prob = (1 - leveler) / num_classes

    for i in range(1, array.shape[0]):
        output[i] = output[i - 1] * array[i]
        output[i] /= np.sum(output[i - 1] * array[i], axis=-1, keepdims=True)
        output[i] *= leveler
        output[i] += min_prob

    return output


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    probability_array = np.abs(rng.normal(0, 1, (200, 228, 228, 9)))
    probability_array /= probability_array.sum(axis=-1, keepdims=True)
    array = np.array(
        [
            [0.3, 0.7],
            [0.3, 0.7],
            [0.3, 0.7],
            [0.9, 0.1],
            [0.4, 0.6],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
        ]
    )
    print(bulcp(array))
    print(bulcp(probability_array).shape)
