from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.utils.extmath import randomized_svd

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
    U, D, V = randomized_svd(M, r, random_state=None)
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
