import numpy as np

def boolieanize(x):
    # Get binary as string of 8 bits
    binary_str = format(x, '08b')

    # Convert each character ('0' or '1') to boolean
    x = [bit == '1' for bit in binary_str]

    return np.array(x, dtype=bool)
