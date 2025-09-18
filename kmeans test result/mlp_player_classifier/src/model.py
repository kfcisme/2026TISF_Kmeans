import torch
import torch.nn as nn

ACTS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
}

class MLP(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_sizes=(256,128,64), dropout=0.1, activation="relu"):
        super().__init__()
        act = ACTS.get(activation, nn.ReLU)
        layers = []
        d = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(d, h), act(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # logits
