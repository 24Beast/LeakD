# Importing Libraries
import torch
import pandas
from DLA import DPA
from Leakage import Leakage
from attackerModels.ANN import simpleDenseModel
from tests.DBA import DirectionalBiasAmplification


# Defining Constants
BASE_DIR = "./models/mobile_v3_ratio_1_genderbal_1_bal_1/"


# Helper Function
def loadData(base_dir: str = BASE_DIR, split: str = "train"):
    gts = torch.load(base_dir + f"{split}_gts.pth")
    T = gts[:, :-2]
    A = gts[:, -2:]
    preds = torch.load(base_dir + f"{split}_preds.pth")
    T_pred = preds[:, :-2]
    A_pred = preds[:, -2:]
    return A, T, A_pred, T_pred


def calcCorr(A: torch.tensor, T: torch.tensor) -> torch.tensor:
    """
    Returns correlation coefficient for A[:,0] with all values of T.

    Typically has some rounding errors of the order < 1e-8. Check https://pytorch.org/docs/stable/generated/torch.corrcoef.html

    Parameters
    ----------
    A : torch.tensor
        of the shape (Num observations, num_A).
    T : torch.tensor
        of the shape (Num observations, num_T).

    Returns
    -------
    corr_vals : torch.tensor
        Correlation coefficients of T with A[:,0]. torch.tensor of the shape (num_T)

    """
    num_A = A.shape[1]
    corr_matrix = torch.corrcoef(torch.hstack([A, T]).T)
    corr_vals = corr_matrix[num_A:, 0]
    return corr_vals


# Load Data
A_train, T_train, A_pred_train, T_pred_train = loadData(BASE_DIR, "train")
A_test, T_test, A_pred_test, T_pred_test = loadData(BASE_DIR, "test")

# Accuracy Calc
AtoT_acc = ((1.0 * (T_pred_train > 0.5) == T_train) * 1.0).mean()
TtoA_acc = ((1.0 * (A_pred_train > 0.5) == A_train) * 1.0).mean()
model_acc = {"AtoT": AtoT_acc, "TtoA": TtoA_acc}

# Attacker Model Initialization
attackerModel_AtoT = simpleDenseModel(
    2, 205, 2, numFirst=4, activations=["relu", "sigmoid", "sigmoid"]
)
attackerModel_TtoA = simpleDenseModel(
    205, 2, 2, numFirst=4, activations=["relu", "sigmoid", "sigmoid"]
)

# Parameter Initialization
dla_obj = DPA(
    {"attacker_AtoT": attackerModel_AtoT, "attacker_TtoA": attackerModel_TtoA},
    {
        "learning_rate": 0.01,
        "loss_function": "bce",
        "epochs": 100,
        "batch_size": 128,
    },
    model_acc,
    "bce",
    threshold=False,
)
leak_AtoT = dla_obj.getAmortizedLeakage(
    A_train, T_train, T_pred_train, "AtoT"
)  # , feat_test = A_test, data_test = T_test, pred_test = T_pred_test)
print(f"leakage for AtoT: {leak_AtoT}")
print("______________________________________")
print("______________________________________")

leak_TtoA = dla_obj.getAmortizedLeakage(
    T_train, A_train, A_pred_train, "TtoA"
)  # , feat_test = T_test, data_test = A_test, pred_test = A_pred_test)
print(f"leakage for TtoA: {leak_TtoA}")
print("______________________________________")
print("______________________________________")

dba_metric = DirectionalBiasAmplification()

dba_vals_AtoT = dba_metric._compute(T_pred_train, T_train, A_train)
print(f"DBA AtoT : {dba_vals_AtoT['bias_amplification']}")
dba_vals_TtoA = dba_metric._compute(A_pred_train, A_train, T_train)
print(f"DBA TtoA : {dba_vals_TtoA['bias_amplification']}")

leakage_metric = Leakage(
    {"attacker_D": attackerModel_TtoA, "sameModel": True},
    {
        "learning_rate": 0.01,
        "loss_function": "bce",
        "epochs": 50,
        "batch_size": 128,
    },
    AtoT_acc,
    "accuracy",
    threshold=True,
)

leak_val = leakage_metric.getAmortizedLeakage(A_train, T_train, T_pred_train)
print(f"{leak_val=}")


corr_val_train = calcCorr(A_train, T_train).mean().item()
corr_val_test = calcCorr(A_test, T_test).mean().item()

print("Correlation\n---------------------------------------------------------")
print(f"Train Correlation = {corr_val_train}, Test Correlation = {corr_val_test}")
print(f"Correlation Amp. = {corr_val_test - corr_val_train}")

corr_val_train = calcCorr(A_train, T_train).abs().mean().item()
corr_val_test = calcCorr(A_test, T_test).abs().mean().item()

print("Absolute Correlation\n-------------------------------------------------")
print(
    f"Train Absolute Correlation = {corr_val_train}, Test Absolute Correlation = {corr_val_test}"
)
print(f"Absolute Correlation Amp. = {corr_val_test - corr_val_train}")
