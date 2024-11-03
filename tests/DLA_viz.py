# Importing Libraries
import copy
import math
import torch
import numpy as np
import torch.optim as optim
from typing import Callable, Union, Literal


# Defining Constants
array_type = np.typing.NDArray
MODEL_ACC = 1.0
NUM_SAMPLES = 1024


# Helper Function
def P_mat_to_data(
    P_at: array_type, num_points: int = 1024
) -> tuple[torch.Tensor, torch.Tensor]:
    A = torch.zeros((num_points, 1), dtype=torch.float)
    D = torch.zeros((num_points, 1), dtype=torch.float)
    num_zeros = int(num_points * np.sum(P_at[0]))
    A[num_zeros:] = 1
    num_00 = int(num_points * P_at[0, 0])
    num_10 = int(num_points * P_at[1, 0])
    D[num_00:num_zeros] = 1
    D[num_zeros + num_10 :] = 1
    return A, D


# Main class
class DLA_viz:
    def __init__(
        self,
        model_params: dict,
        train_params: dict,
        model_acc: float,
        eval_metric: Union[Callable, str] = "mse",
        threshold=True,
    ) -> None:
        """
        Parameters
        ----------
        model_params : dict
            Dictionary of the following forms-
            {"attacker_AtoT" : model_AT, "attacker_TtoA" : model_TA}
        train_params : dict
            {
                "AtoT":
                    {
                        "learning_rate": The learning rate hyperparameter,
                        "loss_function": The loss function to be used.
                                Existing options: ["mse", "cross-entropy"],
                        "epochs": Number of training epochs to be set,
                        "batch_size: Number of batches per epoch
                    },
                "TtoA": {same format as AtoT}
            }
        model_acc : float
            The accuracy of the model being tested for quality equalization.
        eval_metric : Union[Callable,str], optional
            Either a Callable of the form eval_metric(y_pred, y)
            or a string to utilize exiting methods.
            Existing options include ["accuracy"]
            The default is "mse".

        Returns
        -------
        None
            Initializes the class.

        """
        self.model_params = model_params
        self.train_params = train_params
        self.model_attacker_trained = False
        self.threshold = threshold
        self.model_acc = model_acc

        self.loss_functions = {
            "mse": torch.nn.MSELoss(),
            "cross-entropy": torch.nn.CrossEntropyLoss(),
            "bce": torch.nn.BCELoss(),
        }
        self.eval_functions = {
            "accuracy": lambda y_pred, y: (y_pred == y).float().mean(),
            "mse": lambda y_pred, y: ((y_pred - y) ** 2).float().mean(),
            "bce": torch.nn.BCELoss(),
        }
        self.initEvalMetric(eval_metric)
        self.defineModel()

    def calcLeak(
        self,
        feat_d: torch.tensor,
        feat_m: torch.tensor,
        data: torch.tensor,
        pred: torch.tensor,
        mode: Literal["AtoT", "TtoA"],
    ) -> torch.tensor:
        """
        Parameters
        ----------
        feat : torch.tensor
            Protected Attribute.
        data : torch.tensor
            Ground truth data.
        pred : torch.tensor
            Predicted Values.
        mode : Literal["AtoT","TtoA"]
            Sets Direction of calculation.

        Returns
        -------
        leakage : torch.tensor
            Evaluated Leakage.

        """
        pert_data = self.permuteData(data)
        self.train(pert_data, feat_d, "D_" + mode)
        lambda_d = self.calcLambda(
            getattr(self, "attacker_D_" + mode), pert_data, feat_d
        )
        self.train(pred, feat_m, "M_" + mode)
        lambda_m = self.calcLambda(getattr(self, "attacker_M_" + mode), pred, feat_m)
        print(f"{lambda_d=},\n{lambda_m=}")
        leakage = (lambda_m - lambda_d) / (lambda_m + lambda_d)
        return leakage

    def train(
        self,
        x: torch.tensor,
        y: torch.tensor,
        attacker_mode: str,
    ) -> torch.tensor:
        self.defineModel()
        model = getattr(self, "attacker_" + attacker_mode)
        criterion = self.loss_functions[self.train_params["loss_function"]]
        optimizer = optim.Adam(
            model.parameters(), lr=self.train_params["learning_rate"]
        )
        batches = math.ceil(len(x) / self.train_params["batch_size"])

        print(f"Training Activated for Mode: {attacker_mode}")

        # Training loop
        for epoch in range(1, self.train_params["epochs"] + 1):
            perm = torch.randperm(x.shape[0])
            x = x[perm]
            y = y[perm]
            start = 0
            running_loss = 0.0
            # print(batches)
            for batch_num in range(batches):
                x_batch = x[start : (start + self.train_params["batch_size"])]
                y_batch = y[start : (start + self.train_params["batch_size"])]

                optimizer.zero_grad()
                # Forward pass
                outputs = model(x_batch)
                # print(f"{outputs=}\n{y_batch=}")
                loss = criterion(outputs, y_batch)
                # print(f"{loss.item()=}")

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                start += self.train_params["batch_size"]
                running_loss += loss.item()

            avg_loss = running_loss / batches
            if epoch % 10 == 0:
                print(f"\rCurrent Epoch {epoch}: Loss = {avg_loss}", end="")

        print("\nModel training completed")

    def calcLambda(
        self, model: torch.nn.Module, x: torch.tensor, y: torch.tensor
    ) -> torch.tensor:
        y_pred = model(x)
        if self.threshold:
            y_pred = y_pred > 0.5
        return self.eval_metric(y_pred, y)

    def defineModel(self) -> None:
        if type(self.model_params.get("attacker_AtoT", None)) == None:
            raise Exception("attacker_AtoT Model Missing!")
        if type(self.model_params.get("attacker_TtoA", None)) == None:
            raise Exception("attacker_TtoA Model Missing!")
        self.attacker_D_AtoT = self.model_params["attacker_AtoT"]
        self.attacker_M_AtoT = copy.deepcopy(self.attacker_D_AtoT)
        self.attacker_D_TtoA = self.model_params["attacker_TtoA"]
        self.attacker_M_TtoA = copy.deepcopy(self.attacker_D_TtoA)

    def permuteData(self, data: torch.tensor) -> torch.tensor:
        """
        Currently assumes ground truth data to be binary values in a pytorch tensor.
        Should work for any NxM type array.

        Parameters
        ----------
        data : torch.tensor
            Original ground truth data.

        Returns
        -------
        new_data : torch.tensor
            Randomly pertubed data for quality equalization.
        """
        if self.model_acc > 1:
            self.model_acc = self.model_acc / 100
        num_observations = data.shape[0]
        rand_vect = torch.zeros((num_observations, 1))
        rand_vect[: int(self.model_acc * num_observations)] = 1
        rand_vect = rand_vect[torch.randperm(num_observations)]
        new_data = rand_vect * (data) + (1 - rand_vect) * (1 - data)
        return new_data

    def initEvalMetric(self, metric: Union[Callable, str]) -> None:
        if callable(metric):
            self.eval_metric = metric
        elif type(metric) == str:
            if metric in self.eval_functions.keys():
                self.eval_metric = self.eval_functions[metric]
            else:
                raise ValueError("Metric Option given is unavailable.")
        else:
            raise ValueError("Invalid Metric Given.")

    def getAmortizedLeakage(
        self,
        feat_d: torch.tensor,
        feat_m: torch.tensor,
        data: torch.tensor,
        pred: torch.tensor,
        mode: Literal["AtoT", "TtoA"],
        num_trials: int = 10,
        method: str = "mean",
    ) -> tuple[torch.tensor, torch.tensor]:
        vals = torch.zeros(num_trials)
        for i in range(num_trials):
            print(f"Working on Trial: {i}")
            vals[i] = self.calcLeak(feat_d, feat_m, data, pred, mode)
            print(f"Trial {i} val: {vals[i]}")
        if method == "mean":
            return torch.mean(vals), torch.std(vals)
        elif method == "median":
            return torch.median(vals), torch.std(vals)
        else:
            raise ValueError("Invalid Method given for Amortization.")


def calc_DPA(
    P_at: array_type, P_atpred: array_type, P_apredt: array_type
) -> dict[str, float]:
    A_d, D = P_mat_to_data(P_at, NUM_SAMPLES)
    A_m, M = P_mat_to_data(P_atpred, NUM_SAMPLES)

    # Attacker Model Initialization
    attackerModel = simpleDenseModel(
        1, 1, 1, numFirst=1, activations=["sigmoid", "sigmoid", "sigmoid"]
    )

    # Parameter Initialization
    leakage = DLA_viz(
        {"attacker_AtoT": attackerModel, "attacker_TtoA": attackerModel},
        {
            "learning_rate": 0.05,
            "loss_function": "bce",
            "epochs": 100,
            "batch_size": 64,
        },
        MODEL_ACC,
        "bce",
        threshold=True,
    )

    val = leakage.getAmortizedLeakage(A_d, A_m, D, M, "AtoT")

    return {"AtoT": val[0]}


if __name__ == "__main__":
    # Test case
    import os
    import sys

    sys.path.append("../")

    from attackerModels.ANN import simpleDenseModel
    from viz import createHeatMap, plotHeatMap

    # Parameter Initialization
    INCR = 0.01
    OUTPUT_DIR = "./results/"

    if not (os.path.exists(OUTPUT_DIR)):
        os.makedirs(OUTPUT_DIR)

    DPA_map = createHeatMap(calc_DPA, increments=INCR)
    plotHeatMap(
        DPA_map, "DPA", r"$\alpha_d$", r"$\alpha_m$", "jet", None, None, OUTPUT_DIR
    )
