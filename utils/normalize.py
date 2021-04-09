import numpy as np


def old_normalize(data):
    return keras.utils.normalize(data, order=2, axis=(-2, -1))


def normalize(data):
    normalized_data = []
    max_values = np.amax(data, axis=(-2, -1))
    for i in range(len(data)):
        normalized_data.append(data[i] / max_values[i])
    return np.array(normalized_data)