from collections import Counter

import numpy as np

counter = Counter()


def strassen_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if (n := A.shape[0]) == 1:
        counter["*"] += 1
        return np.array([[A[0, 0] * B[0, 0]]])

    n //= 2

    A_11, A_12, A_21, A_22 = A[:n, :n], A[:n, n:], A[n:, :n], A[n:, n:]
    B_11, B_12, B_21, B_22 = B[:n, :n], B[:n, n:], B[n:, :n], B[n:, n:]

    P_1 = strassen_matmul(A_11 + A_22, B_11 + B_22)
    P_2 = strassen_matmul(A_21 + A_22, B_11)
    P_3 = strassen_matmul(A_11, B_12 - B_22)
    P_4 = strassen_matmul(A_22, B_21 - B_11)
    P_5 = strassen_matmul(A_11 + A_12, B_22)
    P_6 = strassen_matmul(A_21 - A_11, B_11 + B_12)
    P_7 = strassen_matmul(A_12 - A_22, B_21 + B_22)

    C_11 = P_1 + P_4 - P_5 + P_7
    C_12 = P_3 + P_5
    C_21 = P_2 + P_4
    C_22 = P_1 - P_2 + P_3 + P_6

    counter["+"] += 12 * n ** 2
    counter["-"] += 6 * n ** 2

    return np.block([[C_11, C_12], [C_21, C_22]])
