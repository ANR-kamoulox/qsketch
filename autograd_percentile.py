# %%
import torch
import numpy as np


# Percentile autograd function
class NumpyPercentile(torch.autograd.Function):
    """
    numpy Percentile autograd Functions subclassing torch.autograd.Function
    """

    def __init__(self, percentiles, device):
        self.n_percentiles = percentiles
        self.percentiles = np.linspace(0, 100, percentiles)
        self.device = device

    def forward(self, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output.
        """
        percentiles = np.percentile(input.detach().cpu().numpy(),
                                    self.percentiles, axis=1).T
        percentiles = torch.Tensor(percentiles.astype(float)).to(self.device)
        self.save_for_backward(input, percentiles)
        return percentiles

    def backward(self, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient with
        respect to the output. We compute the gradient of the histogram
        with respect to the input.
        """
        input, percentiles = self.saved_tensors
        grad_input = torch.zeros(input.shape, device=self.device)
        grad_output = grad_output.detach().cpu().numpy()
        for i in range(input.shape[0]):
            bins_i = (np.digitize(input[i].detach().cpu().numpy(),
                                  percentiles[i].detach().cpu())-1)
            grad_input[i] = torch.Tensor(grad_output[i, bins_i]
                                        ).to(self.device)
        return grad_input



# Percentile autograd function
class Percentile(torch.autograd.Function):
    """
    torch Percentile autograd Functions subclassing torch.autograd.Function
    """

    def __init__(self, percentiles, device):
        self.n_percentiles = percentiles
        self.percentiles = torch.linspace(0, 100, percentiles).to(device)
        self.device = device

    def forward(self, input):
        """
        Find the percentile of a list of values.
        """
        in_sorted, in_argsort = torch.sort(input, dim=1)
        positions = self.percentiles * (input.shape[1]-1) / 100
        floored = torch.floor(positions)
        ceiled = torch.ceil(positions)
        weight_floored = ceiled-positions
        weight_ceiled = positions-floored
        identical = (ceiled == floored)
        weight_floored[identical] = 0.5
        weight_ceiled[identical] = 0.5
        d0 = in_sorted[:, floored.long()] * weight_floored
        d1 = in_sorted[:, ceiled.long()] * weight_ceiled
        self.save_for_backward(in_argsort, floored.long(), ceiled.long(),
                               weight_floored, weight_ceiled)

        return d0+d1

    def backward(self, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient with
        respect to the output. We compute the gradient of the percentile
        with respect to the input.
        """
        (in_argsort, floored, ceiled,
         weight_floored, weight_ceiled) = self.saved_tensors
        input_shape = in_argsort.shape

        # the argsort in the flattened in vector
        rows_offsets = (input_shape[1]
                        * torch.range(0,input_shape[0]-1,
                                      device=in_argsort.device))[:, None].long()
        in_argsort = (in_argsort + rows_offsets).view(-1)
        floored = (floored + rows_offsets).view(-1).long()
        ceiled = (ceiled + rows_offsets).view(-1).long()

        grad_input = torch.zeros((in_argsort.size()), device=self.device)
        grad_input[in_argsort[floored]] += (grad_output
                                            * weight_floored[None,:]).view(-1)
        grad_input[in_argsort[ceiled]] += (grad_output
                                           * weight_ceiled[None,:]).view(-1)

        grad_input = grad_input.view(*input_shape)
        return grad_input
