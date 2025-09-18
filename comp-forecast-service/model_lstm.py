# model_lstm.py
import torch
import torch.nn as nn
import numpy as np

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim=9, hidden=64, classes=8, num_layers=1, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        h_out = hidden * (2 if bidirectional else 1)
        self.fc = nn.Linear(h_out, classes)

    def forward(self, x):         # x: (B, L, D)
        out, _ = self.lstm(x)     # (B, L, H)
        last = out[:, -1, :]      # (B, H)
        logits = self.fc(last)    # (B, C)
        return logits

def prepare_input(comp_seq: np.ndarray, n_seq=None) -> torch.Tensor:
    # comp_seq: (L, 8)
    if n_seq is None:
        n = np.ones((comp_seq.shape[0], 1), dtype=np.float32)
    else:
        n = np.array(n_seq, dtype=np.float32).reshape(-1, 1)
    x = np.concatenate([comp_seq, n], axis=1)[None, ...]  # (1, L, 9)
    return torch.tensor(x, dtype=torch.float32)
