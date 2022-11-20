from collections import Counter

import numpy as np

counter = Counter()


def det(A: np.ndarray) -> float:
    _, U = lu(A)
    counter["*"] += A.shape[0]
    return np.prod(np.diag(U))


def lu(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if (n := A.shape[0]) == 1:
        return np.array([[1]]), np.array([[A[0, 0]]])

    n //= 2
    A_11, A_12, A_21, A_22 = A[:n, :n], A[:n, n:], A[n:, :n], A[n:, n:]

    L_11, U_11 = lu(A_11)
    U_11_inv = inverse(U_11)
    L_11_inv = inverse(L_11)

    U_12 = mul(L_11_inv, A_12)
    L_21 = mul(A_21, U_11_inv)
    L_22 = A_22 - mul(mul(mul(A_21, U_11_inv), L_11_inv), A_12)
    L_s, U_s = lu(L_22)

    counter["-"] += n ** 2

    return (
        np.block([[L_11, np.zeros((n, n))], [L_21, L_s]]),
        np.block([[U_11, U_12], [np.zeros((n, n)), U_s]]),
    )


def inverse(A: np.ndarray) -> np.ndarray:
    if (n := A.shape[0]) == 1:
        counter["/"] += 1
        return np.array([[1 / A[0, 0]]])

    n //= 2
    A_11, A_12, A_21, A_22 = A[:n, :n], A[:n, n:], A[n:, :n], A[n:, n:]

    A_11_inv = inverse(A_11)

    S_22 = A_22 - mul(mul(A_21, A_11_inv), A_12)
    S_22_inv = inverse(S_22)

    B_11 = A_11_inv + mul(mul(mul(mul(A_11_inv, A_12), S_22_inv), A_21), A_11_inv)
    B_12 = -mul(mul(A_11_inv, A_12), S_22_inv)
    B_21 = -mul(mul(S_22_inv, A_21), A_11_inv)
    B_22 = S_22_inv

    counter["+"] += n ** 2
    counter["-"] += 3 * n ** 2

    return np.block([[B_11, B_12], [B_21, B_22]])


def mul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if (n := A.shape[0]) == 1:
        counter["*"] += 1
        return np.array([[A[0, 0] * B[0, 0]]])

    n //= 2

    A_11, A_12, A_21, A_22 = A[:n, :n], A[:n, n:], A[n:, :n], A[n:, n:]
    B_11, B_12, B_21, B_22 = B[:n, :n], B[:n, n:], B[n:, :n], B[n:, n:]

    P_1 = mul(A_11 + A_22, B_11 + B_22)
    P_2 = mul(A_21 + A_22, B_11)
    P_3 = mul(A_11, B_12 - B_22)
    P_4 = mul(A_22, B_21 - B_11)
    P_5 = mul(A_11 + A_12, B_22)
    P_6 = mul(A_21 - A_11, B_11 + B_12)
    P_7 = mul(A_12 - A_22, B_21 + B_22)

    C_11 = P_1 + P_4 - P_5 + P_7
    C_12 = P_3 + P_5
    C_21 = P_2 + P_4
    C_22 = P_1 - P_2 + P_3 + P_6

    counter["+"] += 12 * n ** 2
    counter["-"] += 6 * n ** 2

    return np.block([[C_11, C_12], [C_21, C_22]])
