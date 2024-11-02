# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

# Constants
array_type = np.typing.NDArray
MIN_ALPHA = -0.25
MAX_ALPHA = 0.25

# Helper Functions
def calc_y_at(a_val: int, t_val: int, P_at: array_type) -> bool:
    P_a = np.sum(P_at[a_val])
    P_t = np.sum(P_at[:, t_val])
    return P_at[a_val][t_val] > (P_a * P_t)


def calc_delta(
    P_at: array_type, P_at_pred: array_type, a_val: int, t_val: int, AtoT: bool = True
) -> float:
    if AtoT:
        P_t_given_a = P_at[a_val][t_val] / np.sum(P_at[a_val])
        P_tpred_given_a = P_at_pred[a_val][t_val] / np.sum(P_at_pred[a_val])
        return P_tpred_given_a - P_t_given_a
    else:
        P_a_given_t = P_at[a_val][t_val] / np.sum(P_at[:, t_val])
        P_apred_given_t = P_at_pred[a_val][t_val] / np.sum(P_at_pred[:, t_val])
        return P_apred_given_t - P_a_given_t


def calc_DBA(
    P_at: array_type, P_atpred: array_type, P_apredt: array_type
) -> dict[str, float]:
    DBA_at = 0
    for a_val in [0, 1]:
        for t_val in [0, 1]:
            y_at = calc_y_at(a_val, t_val, P_at)
            delta = calc_delta(P_at, P_atpred, a_val, t_val, AtoT=True)
            DBA_at += (y_at * delta) - ((1 - y_at) * delta)
    DBA_ta = 0
    for a_val in [0, 1]:
        for t_val in [0, 1]:
            y_at = calc_y_at(a_val, t_val, P_at)
            delta = calc_delta(P_at, P_apredt, a_val, t_val, AtoT=False)
            DBA_ta += (y_at * delta) - ((1 - y_at) * delta)
    return {"AtoT": DBA_at, "TtoA": DBA_ta}


def calc_MDBA(
    P_at: array_type, P_atpred: array_type, P_apredt: array_type
) -> dict[str, float]:
    DBA_at = 0
    for a_val in [0, 1]:
        for t_val in [0, 1]:
            delta = calc_delta(P_at, P_atpred, a_val, t_val, AtoT=True)
            DBA_at += abs(delta)
    DBA_ta = 0
    for a_val in [0, 1]:
        for t_val in [0, 1]:
            delta = calc_delta(P_at, P_apredt, a_val, t_val, AtoT=False)
            DBA_ta += abs(delta)
    return {"AtoT": DBA_at, "TtoA": DBA_ta}


def generateP_mat(alpha : float) -> array_type:
    return np.array([[0.25 + alpha, 0.25],[0.25, 0.25 - alpha]])

def createHeatMap(calc_func: Callable, increments : float = 0.01) -> array_type:
    num = int((MAX_ALPHA - MIN_ALPHA)//increments)
    vals = np.zeros((num+1,num+1))
    for num_d in range(0, num + 1):
        alpha_d = MIN_ALPHA + (increments * num_d)
        P_at = generateP_mat(alpha_d)
        for num_m in range(0, num + 1):
            alpha_m = MIN_ALPHA + (increments * num_m)
            P_at_pred = generateP_mat(alpha_m)
            curr_val = calc_func(P_at, P_at_pred, P_at_pred)
            vals[num_d,num_m] = curr_val["AtoT"]
    return vals
                

if __name__ == "__main__":
    DBA_map = createHeatMap(calc_DBA)
    plt.title("DBA Map")
    plt.imshow(DBA_map,cmap="jet")
    plt.show()

    MDBA_map = createHeatMap(calc_MDBA)
    plt.title("MDBA Map")
    plt.imshow(MDBA_map,cmap="jet")
    plt.show()
