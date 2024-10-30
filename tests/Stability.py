# Importing Libraries
import copy
import math
import torch
import numpy as np
import torch.optim as optim
from typing import Callable, Union, Literal


# Helper Function


def RMSE(x, y):
    if len(x) != len(y):
        raise ValueError(
            f"Length of x and y must be same! Current shapes: {x.shape=} & {y.shape=}"
        )
    return (torch.sum((x - y) ** 2) ** 0.5) / len(x)


# Main class
class Leakage:
    def __init__(
        self,
        model_params: dict,
        train_params: dict,
        model_acc: float,
        data_eq_method: Literal["permute", "noise"] = "permute",
        eval_metric: Union[Callable, str] = "mse",
        threshold=True,
    ) -> None:
        """
        Parameters
        ----------
        model_params : dict
            Dictionary of one of the following forms-
            {"attacker_D" : model, "sameModel" : True} or
            {"attacker_D" : model_d, "attacker_M" : model_m} or
            {"attacker_D" : model_d, "attacker_M" : model_m, "sameModel" : False}
        train_params : dict
            {"learning_rate": The learning rate hyperparameter,
            "loss_function": The loss function to be used.
                            Existing options: ["mse", "cross-entropy"],
            "epochs": Number of training epochs to be set,
            "batch_size: Number of batches per epoch}'
        model_acc : float
            The accuracy of the model being tested for quality equalization.
            In case of continuous variables use the rmse value.
        model_eq : Literal["permute", "noise"]
            Use permute for categorical data and noise for continuous.
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
        self.data_eq_method = data_eq_method

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
        self, feat: torch.tensor, data: torch.tensor, pred: torch.tensor
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

        Returns
        -------
        leakage : torch.tensor
            Evaluated Leakage.

        """
        if self.data_eq_method == "permute":
            pert_data = self.permuteData(data)
        else:
            pert_data = self.noisyData(data)
        self.train(pert_data, feat, "Data")
        lambda_d = self.calcLambda(self.attacker_D, pert_data, feat)
        self.train(pred, feat, "Model")
        lambda_m = self.calcLambda(self.attacker_M, pred, feat)
        print(f"{lambda_d=},\n{lambda_m=}")
        leakage = lambda_m - lambda_d
        return leakage

    def train(
        self,
        x: torch.tensor,
        y: torch.tensor,
        attacker_mode: str,
    ) -> torch.tensor:
        self.defineModel()
        if attacker_mode == "Model":
            model = self.attacker_M
        else:
            model = self.attacker_D

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
        if type(self.model_params.get("attacker_D", None)) == None:
            raise Exception("Attacker_D Missing!")
        self.attacker_D = self.model_params["attacker_D"]
        if type(self.model_params.get("sameModel", None)) == None:
            try:
                self.attacker_M = self.model_params["attacker_M"]
            except KeyError:
                raise Exception("Attacker_M is Missing!")
        else:
            self.attacker_M = copy.deepcopy(self.attacker_D)

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

    def noisyData(self, data: torch.tensor) -> torch.tensor:
        error_std = self.model_acc
        means = torch.zeros_like(data)
        error = torch.normal(means, error_std)
        new_data = data + error
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
        feat: torch.tensor,
        data: torch.tensor,
        pred: torch.tensor,
        num_trials: int = 10,
        method: str = "mean",
    ) -> tuple[torch.tensor, torch.tensor]:
        vals = torch.zeros(num_trials)
        for i in range(num_trials):
            print(f"Working on Trial: {i}")
            vals[i] = self.calcLeak(feat, data, pred)
            print(f"Trial {i} val: {vals[i]}")
        if method == "mean":
            return torch.mean(vals), torch.std(vals)
        elif method == "median":
            return torch.median(vals), torch.std(vals)
        else:
            raise ValueError("Invalid Method given for Amortization.")


if __name__ == "__main__":
    # Test case
    import json
    import pandas as pd
    from ..attackerModels.ANN import simpleDenseModel

    # Data Initialization
    from ..utils.datacreator import StabilityExp

    NUM_SAMPLES = 16384
    DATA_ERROR_W = 0.2
    MODEL_ERROR_W = 0.5
    POLY_POW = 4
    DATA_RANGE = (1, 2)
    ATTACKER_WIDTHS = [i for i in range(1, 20)]
    OUTFILE = "../results/Stability.json"

    P, D, M = StabilityExp(
        NUM_SAMPLES, DATA_ERROR_W, MODEL_ERROR_W, POLY_POW, DATA_RANGE
    )
    P = torch.tensor(P, dtype=torch.float).reshape(-1, 1)
    D = torch.tensor(D, dtype=torch.float).reshape(-1, 1)
    M = torch.tensor(M, dtype=torch.float).reshape(-1, 1)

    # Calculating Params
    model_mse = RMSE(M, D)
    leakages = {}

    # Parameter Initialization
    for num, width in enumerate(ATTACKER_WIDTHS, 1):
        print(f"Working on Iteration {num}", flush=True)
        # Attacker Model Initialization
        attackerModel = simpleDenseModel(
            1, width, 1, numFirst=1, activations=["relu", "relu", "relu"]
        )

        # Parameter Initialization
        leakage = Leakage(
            {"attacker_D": attackerModel, "sameModel": True},
            {
                "learning_rate": 0.05,
                "loss_function": "mse",
                "epochs": 100,
                "batch_size": 64,
            },
            model_mse,
            "noise",
            "mse",
            threshold=False,
        )

        leak = leakage.getAmortizedLeakage(P, D, M)
        print(f"leakage for case {num+1} ({width=}): {leak}")
        print("______________________________________")
        print("______________________________________")
        leakages[num] = {"width": width, "data": leak}

    print("Saving results!")
    with open(OUTFILE, "w") as f:
        json.dump(leakages, f)
