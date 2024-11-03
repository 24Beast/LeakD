# Importing Libraries
import numpy as np
import seaborn as sns
from typing import Callable
import matplotlib.pyplot as plt

# Constants
array_type = np.typing.NDArray
MIN_ALPHA = -0.25
MAX_ALPHA = 0.25
LABELS = [MIN_ALPHA, 0, MAX_ALPHA]
plt.rc("axes", labelsize=20)
plt.rc("xtick", labelsize=15)
plt.rc("ytick", labelsize=15)


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


def generateP_mat(alpha: float) -> array_type:
    return np.array([[0.25 + alpha, 0.25], [0.25, 0.25 - alpha]])


def createHeatMap(calc_func: Callable, increments: float = 0.01) -> array_type:
    num = int((MAX_ALPHA - MIN_ALPHA) // increments)
    vals = np.zeros((num + 1, num + 1))
    for num_d in range(0, num + 1):
        alpha_d = MIN_ALPHA + (increments * num_d)
        P_at = generateP_mat(alpha_d)
        for num_m in range(0, num + 1):
            alpha_m = MIN_ALPHA + (increments * num_m)
            P_at_pred = generateP_mat(alpha_m)
            curr_val = calc_func(P_at, P_at_pred, P_at_pred)
            vals[num_m, num_d] = curr_val["AtoT"]
    return vals


def plotHeatMap(
    heatmap: array_type,
    title: str,
    xlabel: str,
    ylabel: str,
    cmap: str,
    vmin: float,
    vmax: float,
    save_loc: str,
) -> None:
    l = len(heatmap)
    sns.heatmap(heatmap, cmap=cmap, vmax=vmax, vmin=vmin)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(ticks=[0, int(l // 2), l], labels=LABELS, rotation=90)
    plt.yticks(ticks=[0, int(l // 2), l], labels=LABELS[::-1])
    plt.savefig(save_loc + title)


if __name__ == "__main__":
    import os

    INCR = 0.01
    OUTPUT_DIR = "./results/"

    if not (os.path.exists(OUTPUT_DIR)):
        os.makedirs(OUTPUT_DIR)

    DBA_map = createHeatMap(calc_DBA, increments=INCR)
    plotHeatMap(
        DBA_map, "DBA", r"$\alpha_d$", r"$\alpha_m$", "jet", -0.7, 0.7, OUTPUT_DIR
    )

    MDBA_map = createHeatMap(calc_MDBA, increments=INCR)
    plotHeatMap(
        MDBA_map, "MDBA", r"$\alpha_d$", r"$\alpha_m$", "jet", -2.5, 2.5, OUTPUT_DIR
    )
