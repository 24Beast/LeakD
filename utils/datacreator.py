# Importing Libraries
import numpy as np
import pandas as pd


# validator
def validate_error_percent(error_percent, name):
    if 0 <= error_percent <= 0.25:
        pass
    elif error_percent > 1 and error_percent < 25:
        error_percent = error_percent / 100
    else:
        raise ValueError(f"{name}_percent needs to be a float value in [0,0.25].")


# dataCreator function
def dataCreator(N=512, error_percent=0.1, shuffle=False, data_error_percent=None):
    error_percent = error_percent / 2
    if N < 3 or type(N) != int:
        raise ValueError("Expected Integer Value about 2 for N.")
    validate_error_percent(error_percent, "error")
    if data_error_percent == None:
        data_error_percent = error_percent
    else:
        validate_error_percent(data_error_percent, "data_error")
    P = np.zeros(N)
    P[N // 2 :] = 1
    D = np.zeros(N)
    D[N // 4 : N // 2] = 1
    D[3 * N // 4 :] = 1
    M_unbias = D.copy()
    M2 = D.copy()
    D_bias = D.copy()
    num_errors = int(N * error_percent)
    num_data_errors = int(N * data_error_percent)
    A_pos = np.array([i for i in range(0, N // 4)])
    C_pos = np.array([i for i in range(N // 2, 3 * N // 4)])
    A_swaps_1 = np.random.choice(A_pos, num_errors // 2, replace=False)
    A_swaps_2 = np.random.choice(A_pos, num_errors, replace=False)
    C_swaps_1 = np.random.choice(C_pos, num_errors - num_errors // 2, replace=False)
    A_swaps_bias = np.random.choice(A_pos, num_data_errors, replace=False)
    M_unbias[A_swaps_1] = 1
    M_unbias[A_swaps_1 + (N // 4)] = 0
    M_unbias[C_swaps_1] = 1
    M_unbias[C_swaps_1 + (N // 4)] = 0
    M2[A_swaps_2] = 1
    M2[A_swaps_2 + (3 * N // 4)] = 0
    D_bias[A_swaps_bias] = 1
    D_bias[A_swaps_bias + (3 * N // 4)] = 0
    if shuffle:
        permut = np.random.permutation(N)
        P = P[permut]
        D = D[permut]
        M_unbias = M_unbias[permut]
        M2 = M2[permut]
    return P, D, D_bias, M_unbias, M2


# COMPAS Dataset
COMPAS_SENSITIVE_ATTRS = [""]


def COMPASData(attribute="race"):
    df = pd.read_csv(
        "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    )
    """
    Removing records where charged for alternative reasons. Reference below:
    Jeff Larson, Surya Mattu, Lauren Kirchner, and Julia Angwin. How we analyzed the compas recidivism algorithm. 2016. URL: https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm.
    """
    df = df[
        (df["days_b_screening_arrest"] <= 30)
        & (df["days_b_screening_arrest"] >= -30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ].reset_index(drop=True)

    return df


if __name__ == "__main__":
    P, D, D_bias, M_unbias, M2 = dataCreator(32, 0.1, False, 0.2)
    df = COMPASData()
