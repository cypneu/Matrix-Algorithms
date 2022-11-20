from collections import Counter

import numpy as np

counter = Counter()


def binet_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if (n := A.shape[0]) == 1:
        counter["*"] += 1
        return np.array([[A[0, 0] * B[0, 0]]])

    counter["+"] += n ** 2
    n //= 2

    C_11 = binet_matmul(A[:n, :n], B[:n, :n]) + binet_matmul(A[:n, n:], B[n:, :n])
    C_12 = binet_matmul(A[:n, :n], B[:n, n:]) + binet_matmul(A[:n, n:], B[n:, n:])
    C_21 = binet_matmul(A[n:, :n], B[:n, :n]) + binet_matmul(A[n:, n:], B[n:, :n])
    C_22 = binet_matmul(A[n:, :n], B[:n, n:]) + binet_matmul(A[n:, n:], B[n:, n:])

    return np.block([[C_11, C_12], [C_21, C_22]])


print(binet_matmul(np.arange(0, 16).reshape(4, 4), np.arange(0, 16).reshape(4, 4)))
print(counter)
