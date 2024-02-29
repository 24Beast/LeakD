# Importing Libraries
import torch
import numpy as np


# Main class
class Leakage:
    
    def __init__(self, model_params, train_params, eval_metric):
        self.model_params = model_params
        self.train_params = train_params
        self.eval_metric = eval_metric
        self.model_attacker_trained = False
    
    def calcLeak(self, feat, data, pred):
        pert_data = self.permuteData(data)
        self.train(self.attacker_D, pert_data, feat, "Data")
        self.train(self.attacker_M, pred, feat, "Model")
        lamba_d = self.calcLambda(self.attacker_D, data, feat)
        lamba_m = self.calcLambda(self.attacker_M, pred, feat)
        return lamba_m - lamba_d
        
    def train(self, model, x, y, attacker_mode):
        if(attacker_mode == "Model"):
            if not(self.model_attacker_trained):
                self.model_attacker_trained = True
            else:
                return
        self.defineModel()
        pass
    
    def calcLambda(self, model, x, y):
        y_pred = model(x)
        return self.eval_metric(y_pred, y)
    
    def defineModel(self):
        print("Initializaing Model")
        
    