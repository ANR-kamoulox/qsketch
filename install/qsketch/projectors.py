import torch
from torch import nn
from torchpercentile import Percentile


class Projectors:
    """A dataset of projectors, each of which is an instance of the class
    provided as a parameter. This class is a Module that should accept the
    following parameters in its constructor:
    `shape_in`: the shape of each input sample to the module
    `num_out`: the number of output features produced by the module
    `seed`: a number identifying the instance. Each seed should
            *deterministically* give the same network.
    """

    def __init__(self, num_thetas, data_shape, projector_class, device='cpu'):
        self.num_thetas = num_thetas
        self.data_shape = data_shape
        self.projector_class = projector_class
        self.device = device

    def __getitem__(self, indexes):
        device = torch.device(self.device)

        if isinstance(indexes, int):
            idx = [indexes]
        else:
            idx = indexes

        result = []
        for pos, id in enumerate(idx):
            result += [self.projector_class(
                                      shape_in=self.data_shape,
                                      num_out=self.num_thetas,
                                      seed=id).to(device)]
        return result[0] if isinstance(indexes, int) else result


class LinearProjector(nn.Linear):
    def __init__(self, shape_in, num_out, seed=0):
        super(LinearProjector, self).__init__(
            in_features=torch.prod(torch.tensor(shape_in)),
            out_features=num_out,
            bias=False)
        self.reset(seed)

    def forward(self, input):
        return super(LinearProjector, self).forward(
            input.view(input.shape[0], -1))

    def reset(self, seed):
        """ reseting the weights of the module in a reproductible way.
        The difficulty lies in the fact that the random sequences on
        CPU and GPU are not the same, even with the same seed."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            gen_device = torch.device('cuda')
        else:
            gen_device = torch.device('cpu')
        new_weights = torch.randn(
            self.weight.shape, device=gen_device).to(self.weight.device)
        self.weight = torch.nn.Parameter(
            new_weights/torch.norm(new_weights, dim=1, keepdim=True))

    def backward(self, grad):
        """Manually compute the gradient of the layer for any input"""
        return torch.mm(grad.view(grad.shape[0], -1), self.weight)


class SparseLinearProjector(LinearProjector):
    """ A linear projector such that only a few proportion of its elements
    are active"""
    def __init__(self, shape_in, num_out, seed=0, active_proportion=5):
        self.active_proportion = active_proportion
        super(SparseLinearProjector, self).__init__(
            shape_in=shape_in,
            num_out=num_out,
            seed=seed)
        self.reset(seed)

    def reset(self, seed):
        super(SparseLinearProjector, self).reset(seed)
        threshold = Percentile()(self.weight.abs().flatten()[:, None],
                                 [100-self.active_proportion, ])
        new_weights = self.weight.clone()
        new_weights[torch.abs(self.weight) < threshold] = 0
        self.weight = torch.nn.Parameter(
            new_weights/torch.norm(new_weights, dim=1, keepdim=True))
