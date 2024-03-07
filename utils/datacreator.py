# Importing Libraries
import torch
import numpy as np


# dataCreator function
def dataCreator(N=512, error_percent=0.1, shuffle=False):
    if N < 3 or type(N) != int:
        raise ValueError("Expected Integer Value about 2 for N.")
    if 0 <= error_percent <= 0.25:
        pass
    elif error_percent > 1 and error_percent < 25:
        error_percent = error_percent / 100
    else:
        raise ValueError("error_percent needs to be a float value in [0,0.25].")
    P = np.zeros(N)
    P[N // 2 :] = 1
    D = np.zeros(N)
    D[N // 4 : N // 2] = 1
    D[3 * N // 4 :] = 1
    M1 = D.copy()
    M2 = D.copy()
    num_errors = int(N * error_percent)
    num_errors = num_errors
    A_pos = np.array([i for i in range(0, N // 4)])
    C_pos = np.array([i for i in range(N // 2, 3 * N // 4)])
    A_swaps_1 = np.random.choice(A_pos, num_errors // 2, replace=False)
    A_swaps_2 = np.random.choice(A_pos, num_errors, replace=False)
    C_swaps_1 = np.random.choice(C_pos, num_errors - num_errors // 2, replace=False)
    M1[A_swaps_1] = 1
    M1[A_swaps_1 + (N // 4)] = 0
    M1[C_swaps_1] = 1
    M1[C_swaps_1 + (N // 4)] = 0
    M2[A_swaps_2] = 1
    M2[A_swaps_2 + (3 * N // 4)] = 0
    if shuffle:
        permut = np.random.permutation(N)
        P = P[permut]
        D = D[permut]
        M1 = M1[permut]
        M2 = M2[permut]
    return P, D, M1, M2


if __name__ == "__main__":
    P, D, M1, M2 = dataCreator(16, 0.1)
