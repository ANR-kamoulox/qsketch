import torch
from torch import nn
import numpy as np


# very simple dense auto encoder structure
class DenseEncoder(nn.Module):
    def __init__(self, input_shape, bottleneck_size=64):
        super(DenseEncoder, self).__init__()
        self.input_shape = input_shape
        intermediate_size = max(64, bottleneck_size)
        self.fc1 = nn.Linear(np.prod(input_shape), intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, bottleneck_size)

    def forward(self, x):
        out = self.fc1(x.view(-1, np.prod(self.input_shape)))
        out = torch.relu(out)
        out = self.fc2(out)
        return torch.relu(out)


class DenseDecoder(nn.Module):
    def __init__(self, input_shape, bottleneck_size=64):
        super(DenseDecoder, self).__init__()
        self.input_shape = input_shape
        intermediate_size = max(64, bottleneck_size)
        self.fc1 = nn.Linear(bottleneck_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, np.prod(input_shape))

    def forward(self, x):
        out = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(out)).view(
            -1, self.input_shape[0], self.input_shape[1], self.input_shape[2]
        )
