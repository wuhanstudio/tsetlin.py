
def mean(matrix):
    return [sum(col) / len(col) for col in zip(*matrix)]

def std(matrix):
    n_rows = len(matrix)
    
    means = [sum(col) / n_rows for col in zip(*matrix)]
    variances = []
    
    for col_idx, mean_val in enumerate(means):
        variance_sum = 0
        for row in range(n_rows):
            diff = matrix[row][col_idx] - mean_val
            variance_sum += diff * diff
        variances.append(variance_sum / n_rows)
    
    return [v ** 0.5 for v in variances]

def clip(x, a_min, a_max):
    return max(a_min, min(x, a_max))

def argmax(x):
    return max(range(len(x)), key=x.__getitem__)

def to_int32(val):
    val = int(val)
    return max(-2**31, min(val, 2**31 - 1))
