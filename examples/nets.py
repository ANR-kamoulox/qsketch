import torch
from torch import nn
import numpy as np


def recycle_module(module, index):
    """ We need a particular randn that makes sure we use cuda if available,
    even if the net is to be on cpu. This is because the devices do not
    share the random sequence, so that we may not have the same output, which
    is required here"""
    torch.manual_seed(index)
    if torch.cuda.is_available():
        gen_device = torch.device('cuda')
    else:
        gen_device = torch.device('cpu')
    params = list(module.parameters())
    params = module.state_dict()
    for key in params:
        params[key] = torch.randn(params[key].shape,
                                  device=gen_device).to(params[key].device)
    module.load_state_dict(params)


# very simple dense auto encoder structure
class DenseEncoder(nn.Module):
    def __init__(self, input_shape, bottleneck_size=64, index=0):
        super(DenseEncoder, self).__init__()
        self.input_shape = input_shape
        intermediate_size = max(64, bottleneck_size)
        self.fc1 = nn.Linear(np.prod(input_shape), intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, bottleneck_size)
        self.recycle(index)

    def forward(self, x):
        out = self.fc1(x.view(-1, np.prod(self.input_shape)))
        out = torch.relu(out)
        out = self.fc2(out)
        return torch.relu(out)

    def recycle(self, index):
        # this is to accelerate modules dataset of qsketch
        recycle_module(self, index)


class DenseDecoder(nn.Module):
    def __init__(self, input_shape, bottleneck_size=64,  index=0):
        super(DenseDecoder, self).__init__()
        self.input_shape = input_shape
        intermediate_size = max(64, bottleneck_size)
        self.fc1 = nn.Linear(bottleneck_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, np.prod(input_shape))
        self.recycle(index)

    def forward(self, x):
        out = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(out)).view(
            -1, self.input_shape[0], self.input_shape[1], self.input_shape[2]
        )

    def recycle(self, index):
        recycle_module(self, index)
