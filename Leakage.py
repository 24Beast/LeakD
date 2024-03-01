# Importing Libraries
import torch
import numpy as np
from typing import Callable, Union


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
            DESCRIPTION. *Rahul can you please fill this*
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
                raise Exception("Metric Option given is unavailable.")
        else:
            raise Exception("Invalid Metric Given.")
