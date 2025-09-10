def booleanize(x):
    # Get binary as string of 8 bits
    binary_str = format(x, '08b')

    # Convert each character ('0' or '1') to boolean
    return [bit == '1' for bit in binary_str]

def booleanize_features(X, mean, std):
    # Normalization to [0, 255]
    X = (X - mean) / std
    X = [[int(x* 255) for x in row] for row in X]

    X_bool = []
    for x_features in X:
        bool_features = [booleanize(x) for x in x_features]
        X_bool.append([b for row in bool_features for b in row])

    return X_bool
