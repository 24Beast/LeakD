# Importing Libraries
import torch
import numpy as np
import math
import torch.optim as optim
from typing import Callable, Union
from attackerModels.ANN import simpleDenseModel


# Main class
class Leakage:
    def __init__(
        self,
        model_params: dict,
        train_params: dict,
        model_acc: float,
        eval_metric: Union[Callable, str] = "mse",
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
            "loss_function": The loss function to be used -- options: "mse" and "cross-entropy", 
            "epochs": Number of training epochs to be set,
            "batch_size: Number of batches per epoch}
        model_acc : float
            The accuracy of the model being tested for quality equalization.
        eval_metric : Union[Callable,str], optional
            Either a Callable of the form eval_metric(y_pred, y)
            or a string to utilize exiting methods.
            Existing options include ["mse"]
            The default is "mse".

        Returns
        -------
        None
            Initializes the class.

        """
        self.model_params = model_params
        self.train_params = train_params
        self.model_attacker_trained = False
        self.model_acc = model_acc

        self.loss_functions = {"mse": torch.nn.MSELoss(), 
                               "cross-entropy": torch.nn.CrossEntropyLoss()
                               } 

        self.initEvalMetric(eval_metric)

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
        pert_data = self.permuteData(data)
        self.train(self.attacker_D, pert_data, feat, "Data")
        self.train(self.attacker_M, pred, feat, "Model")
        lamba_d = self.calcLambda(self.attacker_D, data, feat)
        lamba_m = self.calcLambda(self.attacker_M, pred, feat)
        leakage = lamba_m - lamba_d
        return leakage

    def train(
        self,
        model: torch.nn.Module,
        x: torch.tensor,
        y: torch.tensor,
        attacker_mode: str,
    ) -> torch.tensor:
        if attacker_mode == "Model":
            if not (self.model_attacker_trained):
                self.model_attacker_trained = True
            else:
                return
        self.defineModel()
        pass

        criterion = self.loss_functions[self.train_params["loss_function"]]
        optimizer = optim.Adam(model.parameters(), lr=self.train_params["learning_rate"])
        batches = math.ceil(len(x) / self.train_params["batch_size"])

        # Training loop
        for epoch in range(self.train_params["epochs"]):
            start = 0
            for batch_num in batches:
                x_batch = x[start : (start + self.train_params["batch_size"])]
                y_batch = y[start : (start + self.train_params["batch_size"])]

                # Forward pass
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                start+=self.train_params["batch_size"]

        print("Model training completed")

    def calcLambda(
        self, model: torch.nn.Module, x: torch.tensor, y: torch.tensor
    ) -> torch.tensor:
        y_pred = model(x)
        return self.eval_metric(y_pred, y)

    def defineModel(self) -> None:
        print("Initializaing Model")
        if type(self.model_params.get("attacker_D", None)) == None:
            raise Exception("Attacker_D Missing!")
        self.attacker_D = self.model_params["attacker_D"]
        if type(self.model_params.get("sameModel", None)) == None:
            try:
                self.attacker_M = self.model_params["attacker_M"]
            except KeyError:
                raise Exception("Attacker_M is Missing!")
        else:
            self.attacker_M = self.attacker_D.copy()
        print("Model Initialized")

    def permuteData(self, data: torch.tensor) -> torch.tensor:
        """
        Currently assumes ground truth data to be binary values in a pytorch tensor.

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
        rand_vect = torch.zeros(num_observations)
        rand_vect[: int(self.model_acc * num_observations)] = 1
        rand_vect = rand_vect[torch.randperm(num_observations)]
        new_data = rand_vect * (data) + (1 - rand_vect) * (1 - data)
        return new_data

    def initEvalMetric(self, metric: Union[Callable, str]) -> None:
        if callable(metric):
            self.eval_metric = metric
        elif type(metric) == str:
            if metric == "mse":
                self.metric = torch.nn.MSELoss()
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
        if method == "mean":
            return torch.mean(vals), torch.std(vals)
        elif method == "median":
            return torch.median(vals), torch.std(vals)
        else:
            raise ValueError("Invalid Method given for Amortization.")


if "__name__" == "__main__":
    # Test case
    
    from attackerModels.ANN import simpleDenseModel

    # Data Initialization
    feat = torch.zeros(8, 1)
    feat[4:] = 1
    data = torch.zeros([i % 2 for i in range(8)]).reshape(8, 1)
    pred = data.copy()
    pred[2], pred[5] = pred[5], pred[2]

    # Calculating Params
    model_acc = torch.sum(pred == data)

    # Parameter Initialization
    attacker_model = simpleDenseModel(1, 1)

    leakage_obj = Leakage({"attacker_D" : attacker_model, "sameModel" : True}, 
                          {"learning_rate": 0.01, "loss_function": "cross-entropy", "epochs": 100, "batch_size": 10},
                           model_acc,
                  )
    
    leakage_obj.train(attacker_model, feat, pred, "Model")