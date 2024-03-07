# Importing Libraries
import torch
import torch.nn as nn


# Defining Models
class simpleDenseModel(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dims,
        num_layers=5,
        numFirst=16,
        activations=["relu", "relu", "sigmoid", "relu", "relu"],
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.activations = activations
        self.num_layers = num_layers
        self.numFirst = numFirst
        self.initLayers()
        self.filterActivations()

    def initLayers(self):
        self.layers = []
        num_in = self.input_dims
        num_out = self.numFirst
        for i in range(1, self.num_layers + 1):
            layer = nn.Linear(num_in, num_out)
            num_in = num_out
            if i == self.num_layers - 1:
                num_out = self.output_dims
            elif i >= (self.num_layers // 2):
                num_out = num_out // 2
            else:
                num_out = num_out * 2
            self.layers.append(layer)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.identity = lambda x: x

    def filterActivations(self):
        if len(self.activations) < self.num_layers:
            identity_list = [""] * (self.num_layers - len(self.activations))
            identity_list.extend(self.activations)
            self.activations = identity_list
        self.activations = ["identity" if i == "" else i for i in self.activations]

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layer[i](x)
            x = getattr(self, self.activations[i])
        return x


if __name__ == "__main__":
    attackerModel = simpleDenseModel(1, 1, 7)
    print("Model Layers:")
    print(*attackerModel.layers, sep="\n", end="\n\n")
    print("Model Activations:")
    print(*attackerModel.activations, sep="\n", end="\n\n")
