# Importing Libraries
import torch
from DLA import DPA
import matplotlib.pyplot as plt
from attackerModels.ANN import simpleDenseModel

# Defining Constants
BASE_DIR = "./models/vit_ratio_3_genderbal_0_bal_0/"
WIDTHS = [2 + i * 2 for i in range(20)]
NORMALIZED = 0


# Helper Function
def loadData(base_dir: str = BASE_DIR, split: str = "train"):
    gts = torch.load(base_dir + f"{split}_gts.pth")
    T = gts[:, :-2]
    A = gts[:, -2:]
    preds = torch.load(base_dir + f"{split}_preds.pth")
    T_pred = preds[:, :-2]
    A_pred = preds[:, -2:]
    return A, T, A_pred, T_pred


# Load Data
A_train, T_train, A_pred_train, T_pred_train = loadData(BASE_DIR, "train")
A_test, T_test, A_pred_test, T_pred_test = loadData(BASE_DIR, "test")

# Accuracy Calc
AtoT_acc = ((1.0 * (T_pred_train > 0.5) == T_train) * 1.0).mean()
TtoA_acc = ((1.0 * (A_pred_train > 0.5) == A_train) * 1.0).mean()
model_acc = {"AtoT": AtoT_acc, "TtoA": TtoA_acc}

norm_vals = []
for width in WIDTHS:
    print(f"Working on {width=}")
    # Attacker Model Initialization
    attackerModel_AtoT = simpleDenseModel(
        2, 205, 2, numFirst=width, activations=["relu", "sigmoid", "sigmoid"]
    )
    attackerModel_TtoA = simpleDenseModel(
        205, 2, 2, numFirst=width, activations=["relu", "sigmoid", "sigmoid"]
    )

    # Parameter Initialization
    leakage = DPA(
        {"attacker_AtoT": attackerModel_AtoT, "attacker_TtoA": attackerModel_TtoA},
        {
            "learning_rate": 0.0005,
            "loss_function": "bce",
            "epochs": 10,
            "batch_size": 128,
        },
        model_acc,
        "bce",
        threshold=False,
        normalized=True,
    )

    leak_AtoT = leakage.getAmortizedLeakage(
        A_train, T_train, T_pred_train, "AtoT", num_trials=10
    )  # , feat_test = A_test, data_test = T_test, pred_test = T_pred_test)
    print(f"leakage for AtoT: {leak_AtoT}")
    print("______________________________________")
    print("______________________________________")

    norm_vals.append(leak_AtoT)

    """
    leak_TtoA = leakage.getAmortizedLeakage(
        T_train, A_train, A_pred_train, "TtoA"
    )  # , feat_test = T_test, data_test = A_test, pred_test = A_pred_test)
    print(f"leakage for TtoA: {leak_TtoA}")
    print("______________________________________")
    print("______________________________________")
    """

norm_vals = torch.tensor(norm_vals)
torch.save(norm_vals, "normalized.pt")

non_norm_vals = []
for width in WIDTHS:
    print(f"Working on {width=}")
    # Attacker Model Initialization
    attackerModel_AtoT = simpleDenseModel(
        2, 205, 2, numFirst=width, activations=["relu", "sigmoid", "sigmoid"]
    )
    attackerModel_TtoA = simpleDenseModel(
        205, 2, 2, numFirst=width, activations=["relu", "sigmoid", "sigmoid"]
    )

    # Parameter Initialization
    leakage = DPA(
        {"attacker_AtoT": attackerModel_AtoT, "attacker_TtoA": attackerModel_TtoA},
        {
            "learning_rate": 0.0005,
            "loss_function": "bce",
            "epochs": 10,
            "batch_size": 128,
        },
        model_acc,
        "bce",
        threshold=False,
        normalized=False,
    )

    leak_AtoT = leakage.getAmortizedLeakage(
        A_train, T_train, T_pred_train, "AtoT", num_trials=10
    )  # , feat_test = A_test, data_test = T_test, pred_test = T_pred_test)
    print(f"leakage for AtoT: {leak_AtoT}")
    print("______________________________________")
    print("______________________________________")

    non_norm_vals.append(leak_AtoT)

    """
    leak_TtoA = leakage.getAmortizedLeakage(
        T_train, A_train, A_pred_train, "TtoA"
    )  # , feat_test = T_test, data_test = A_test, pred_test = A_pred_test)
    print(f"leakage for TtoA: {leak_TtoA}")
    print("______________________________________")
    print("______________________________________")
    """

non_norm_vals = torch.tensor(non_norm_vals)
torch.save(non_norm_vals, "non_normalized.pt")

# Plotting
y1 = non_norm_vals[:, 0]
y2 = norm_vals[:, 0]
con1 = non_norm_vals[:, 1] * 0.619  # 1.96/sqrt(10) = 95% confidence interval
con2 = norm_vals[:, 1] * 0.619

plt.plot(WIDTHS, -1 * y1, label="Non-Normalized", color="b")
plt.fill_between(WIDTHS, -1 * (y1 - con1), -1 * (y1 + con1), color="b", alpha=0.1)
plt.plot(WIDTHS, -1 * y2, label="Normalized", color="r")
plt.fill_between(WIDTHS, -1 * (y2 - con2), -1 * (y2 + con2), color="r", alpha=0.1)
# plt.plot(x, y2-y1,label = "Difference")
plt.xlabel("Width", fontsize=25)
plt.ylabel("Bias Amplification", fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=15)
plt.show()
