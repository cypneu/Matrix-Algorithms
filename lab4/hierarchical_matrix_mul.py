from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.utils.extmath import randomized_svd as rsvd

EPSILON = 0.0004
R = 5


@dataclass
class MatrixNode:
    rank: Optional[int] = None
    shape: Optional[tuple[int, int]] = None
    singular_values: Optional[np.ndarray] = None
    U: Optional[np.ndarray] = None
    V: Optional[np.ndarray] = None
    children: list["MatrixNode"] = field(default_factory=list)


def create_tree(M: np.ndarray, r: int, epsilon: float) -> MatrixNode:
    U, D, V = rsvd(M, r, random_state=None)
    if (n := M.shape[0] // 2) < r or D[r - 1] < epsilon:
        return compress_matrix(M, U, D, V, r)

    return MatrixNode(
        children=[
            create_tree(M[:n, :n], r, epsilon),
            create_tree(M[:n, n:], r, epsilon),
            create_tree(M[n:, :n], r, epsilon),
            create_tree(M[n:, n:], r, epsilon),
        ],
    )


def compress_matrix(
    M: np.ndarray, U: np.ndarray, D: np.ndarray, V: np.ndarray, r: int
) -> MatrixNode:
    if np.all(np.abs(M) < EPSILON):
        return MatrixNode(rank=0, shape=M.shape)

    return MatrixNode(
        rank=r,
        singular_values=np.diag(D),
        U=U,
        V=np.diag(D) @ V,
        shape=M.shape,
    )


def decompress_matrix(compressed_M: MatrixNode) -> np.ndarray:
    if compressed_M.rank is not None:
        return (
            compressed_M.U @ compressed_M.V
            if compressed_M.rank != 0
            else np.zeros(compressed_M.shape)
        )

    return np.block(
        [
            [
                decompress_matrix(compressed_M.children[0]),
                decompress_matrix(compressed_M.children[1]),
            ],
            [
                decompress_matrix(compressed_M.children[2]),
                decompress_matrix(compressed_M.children[3]),
            ],
        ]
    )


def vector_mul(node: MatrixNode, X: np.ndarray) -> np.ndarray:
    if not node.children:
        if node.rank == 0:
            return np.zeros(X.shape[0])
        return node.U @ (node.V @ X)

    rows = X.shape[0]
    X1 = X[: rows // 2, :]
    X2 = X[rows // 2 :, :]

    Y11 = vector_mul(node.children[0], X1)
    Y12 = vector_mul(node.children[1], X2)
    Y21 = vector_mul(node.children[2], X1)
    Y22 = vector_mul(node.children[3], X2)

    return np.block([[Y11 + Y12], [Y21 + Y22]])


def matrix_add(M: MatrixNode, N: MatrixNode) -> MatrixNode:
    if M.rank == 0 or N.rank == 0:
        return M if M.rank != 0 else N

    if not M.children and not N.children:
        return _recompress(M, N)

    if M.children and N.children:
        return MatrixNode(
            children=[
                matrix_add(M.children[0], N.children[0]),
                matrix_add(M.children[1], N.children[1]),
                matrix_add(M.children[2], N.children[2]),
                matrix_add(M.children[3], N.children[3]),
            ]
        )

    if M.children and not N.children:
        return _divide_add(M, N)

    if not M.children and N.children:
        return _divide_add(N, M)


def _recompress(M: MatrixNode, N: MatrixNode) -> MatrixNode:
    A, B = np.hstack((M.U, N.U)), np.vstack((M.V, N.V))
    Q_A, R_A = np.linalg.qr(A, mode="reduced")
    Q_B, R_B = np.linalg.qr(B.T, mode="reduced")

    U_prim, D, V_prim = rsvd(R_A @ R_B.T, M.rank, random_state=None)

    return MatrixNode(rank=M.rank, U=Q_A @ U_prim, V=np.diag(D) @ (Q_B @ V_prim.T).T)


def _divide_add(A: MatrixNode, B: MatrixNode) -> MatrixNode:
    U1, U2 = _split_in_half_rows(B.U)
    V1, V2 = _split_in_half_cols(B.V)

    return MatrixNode(
        children=[
            matrix_add(MatrixNode(B.rank, U=U1, V=V1), A.children[0]),
            matrix_add(MatrixNode(B.rank, U=U1, V=V2), A.children[1]),
            matrix_add(MatrixNode(B.rank, U=U2, V=V1), A.children[2]),
            matrix_add(MatrixNode(B.rank, U=U2, V=V2), A.children[3]),
        ]
    )


def matrix_mul(M: MatrixNode, N: MatrixNode) -> MatrixNode:
    if M.rank == 0 or N.rank == 0:
        return M if M.rank == 0 else N

    if not M.children and not N.children:
        return MatrixNode(rank=M.rank, U=M.U, V=(M.V @ N.U) @ N.V)

    if M.children and N.children:
        M1, M2, M3, M4 = M.children
        N1, N2, N3, N4 = N.children
        return MatrixNode(
            children=[
                matrix_add(matrix_mul(M1, N1), matrix_mul(M2, N3)),
                matrix_add(matrix_mul(M1, N2), matrix_mul(M2, N4)),
                matrix_add(matrix_mul(M3, N1), matrix_mul(M4, N3)),
                matrix_add(matrix_mul(M3, N2), matrix_mul(M4, N4)),
            ]
        )

    if M.children and not N.children:
        M1, M2, M3, M4 = M.children

        U1, U2 = _split_in_half_rows(N.U)
        V1, V2 = _split_in_half_cols(N.V)

        return MatrixNode(
            children=[
                matrix_add(
                    matrix_mul(M1, MatrixNode(N.rank, U=U1, V=V1)),
                    matrix_mul(M2, MatrixNode(N.rank, U=U2, V=V1)),
                ),
                matrix_add(
                    matrix_mul(M1, MatrixNode(N.rank, U=U1, V=V2)),
                    matrix_mul(M2, MatrixNode(N.rank, U=U2, V=V2)),
                ),
                matrix_add(
                    matrix_mul(M3, MatrixNode(N.rank, U=U1, V=V1)),
                    matrix_mul(M4, MatrixNode(N.rank, U=U2, V=V1)),
                ),
                matrix_add(
                    matrix_mul(M3, MatrixNode(N.rank, U=U1, V=V2)),
                    matrix_mul(M4, MatrixNode(N.rank, U=U2, V=V2)),
                ),
            ]
        )

    if not M.children and N.children:
        N1, N2, N3, N4 = N.children

        U1, U2 = _split_in_half_rows(M.U)
        V1, V2 = _split_in_half_cols(M.V)

        return MatrixNode(
            children=[
                matrix_add(
                    matrix_mul(MatrixNode(M.rank, U=U1, V=V1), N1),
                    matrix_mul(MatrixNode(M.rank, U=U1, V=V2), N3),
                ),
                matrix_add(
                    matrix_mul(MatrixNode(M.rank, U=U1, V=V1), N2),
                    matrix_mul(MatrixNode(M.rank, U=U1, V=V2), N4),
                ),
                matrix_add(
                    matrix_mul(MatrixNode(M.rank, U=U2, V=V1), N1),
                    matrix_mul(MatrixNode(M.rank, U=U2, V=V2), N3),
                ),
                matrix_add(
                    matrix_mul(MatrixNode(M.rank, U=U2, V=V1), N2),
                    matrix_mul(MatrixNode(M.rank, U=U2, V=V2), N4),
                ),
            ]
        )


def _split_in_half_cols(A: np.ndarray) -> np.ndarray:
    mid = A.shape[1] // 2
    return A[:, :mid], A[:, mid:]


def _split_in_half_rows(A: np.ndarray) -> np.ndarray:
    mid = A.shape[0] // 2
    return A[:mid, :], A[mid:, :]
