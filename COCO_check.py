# Importing Libraries
import torch
import pickle
import pandas
from DLA import DPA
from Leakage import Leakage
from attackerModels.ANN import simpleDenseModel
from tests.DBA import DirectionalBiasAmplification


# Defining Constants
DATA_DIR = "models/mask9_leakage_unbalanced/"
print(f"Working on {DATA_DIR=}")


# Helper Function
def loadData(data_dir: str = DATA_DIR):
    with open(DATA_DIR + "loss_info.pkl", "rb") as f:
        data = pickle.load(f)
    gt = data["val_labels"][-1]
    preds = data["val_probs"][-1]
    T = torch.tensor(gt[:, :-1])
    A = torch.tensor(gt[:, -1:])
    A = torch.hstack([A, 1 - A])
    T_pred = torch.tensor(preds[:, :-1])
    A_pred = torch.tensor(preds[:, -1:])
    A_pred = torch.hstack([A_pred, 1 - A_pred])
    return A, T, A_pred, T_pred


# Load Data
A, T, A_pred, T_pred = loadData(DATA_DIR)
num_classes = T.shape[1]

# Accuracy Calc
AtoT_acc = ((1.0 * (T_pred > 0.5) == T) * 1.0).mean()
TtoA_acc = ((1.0 * (A_pred > 0.5) == A) * 1.0).mean()
model_acc = {"AtoT": AtoT_acc, "TtoA": TtoA_acc}

# Attacker Model Initialization
attackerModel_AtoT = simpleDenseModel(
    2, num_classes, 2, numFirst=4, activations=["relu", "sigmoid", "sigmoid"]
)
attackerModel_TtoA = simpleDenseModel(
    num_classes, 2, 2, numFirst=4, activations=["relu", "sigmoid", "sigmoid"]
)
"""
# Parameter Initialization
leakage = DPA(
    {"attacker_AtoT": attackerModel_AtoT, "attacker_TtoA": attackerModel_TtoA},
    {
        "learning_rate": 0.01,
        "loss_function": "bce",
        "epochs": 100,
        "batch_size": 128,
    },
    model_acc,
    "bce",
    threshold=False
)

leak_AtoT = leakage.getAmortizedLeakage(
    A, T, T_pred, "AtoT"
)  # , feat_test = A_test, data_test = T_test, pred_test = T_pred_test)
print(f"leakage for AtoT: {leak_AtoT}")
print("______________________________________")
print("______________________________________")

leak_TtoA = leakage.getAmortizedLeakage(
    T, A, A_pred, "TtoA"
)  # , feat_test = T_test, data_test = A_test, pred_test = A_pred_test)
print(f"leakage for TtoA: {leak_TtoA}")
print("______________________________________")
print("______________________________________")
"""


leakage = Leakage(
    {"attacker_D": attackerModel_TtoA, "sameModel": True},
    {
        "learning_rate": 0.1,
        "loss_function": "bce",
        "epochs": 100,
        "batch_size": 64,
    },
    AtoT_acc,
    "accuracy",
    threshold=True,
)

leak = leakage.getAmortizedLeakage(A, T, T_pred)
print(f"leakage: {leak}")
print("______________________________________")
print("______________________________________")

dba_metric = DirectionalBiasAmplification()

dba_vals_AtoT = dba_metric._compute(T_pred, T, A)
print(f"DBA AtoT : {dba_vals_AtoT['bias_amplification']}")
dba_vals_TtoA = dba_metric._compute(A_pred, A, T)
print(f"DBA TtoA : {dba_vals_TtoA['bias_amplification']}")
