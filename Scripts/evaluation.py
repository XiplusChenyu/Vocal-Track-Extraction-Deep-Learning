import numpy as np
from config import PARAS

test_a = np.array([[1, 1], [0, 1]])
test_b = np.array([[1, 0], [0, 1]])


def calculate_score(binary_mask, mask):
    same = binary_mask == mask
    same = np.array(same, dtype=int)
    return np.sum(same) / (same.shape[0] * same.shape[1])


def calculate_double_score(binary_mask, mask1, mask2):
    same1 = binary_mask == mask1
    same1 = np.array(same1, dtype=int)

    same2 = binary_mask == mask2
    same2 = np.array(same2, dtype=int)

    return max((np.sum(same1) / (same1.shape[0] * same1.shape[1])),
               (np.sum(same2) / (same2.shape[0] * same2.shape[1]))
)