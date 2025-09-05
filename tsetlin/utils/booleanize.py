import numpy as np

def booleanize(x):
    # Get binary as string of 8 bits
    binary_str = format(x, '08b')

    # Convert each character ('0' or '1') to boolean
    x = [bit == '1' for bit in binary_str]

    return np.array(x, dtype=bool)

def booleanize_features(X, mean, std):
    # Normalization to [0, 255]
    X = (X - mean) / std
    X = np.array(X * 255, dtype=np.uint8)

    X_bool = []
    for x_features in X:
        bool_features = []
        for x in x_features:
            bool_features.append(booleanize(x))
        X_bool.append(np.concatenate(bool_features))

    return np.array(X_bool, dtype=bool)
