import torch
import sketch
import os


# A class for random normalized linear projections, which is the module under
class LinearProjector(torch.nn.Linear):
    def __init__(self, input_shape, num_projections):
        self.dim_in = torch.prod(torch.tensor(input_shape))
        self.dim_out = num_projections
        super(LinearProjector, self).__init__(
            in_features=self.dim_in,
            out_features=self.dim_out,
            bias=False)
        self.reset_parameters()

    def forward(self, input):
        return super(LinearProjector, self).forward(
            input.view(input.shape[0], -1))

    def reset_parameters(self):
        super(LinearProjector, self).reset_parameters()
        new_weight = self.weight

        # make sure each projector is normalized
        self.weight = torch.nn.Parameter(
           new_weight/torch.norm(new_weight, dim=1, keepdim=True))


def sw(batch1, batch2, num_projections=1000):
    """directly compute the sliced Wasserstein distance between two
    batches of samples. This is done by randomly picking random projections,
    sketching the batches with them, and compute the squared error between
    the batches. The number of percentiles taken is the smallest of the two
    number of samples.

    batch1: Tensor, (num_samples,) + shape
    batch2: Tensor, (num_samples,) + shape
    """

    # check that dimensions match
    if batch1.shape[1:] != batch2.shape[1:]:
        raise Exception('sw: except the first one, dimension of the batches '
                        'must match.')
    projectors = LinearProjector(input_shape=batch1.shape[1:],
                                 num_projections=num_projections)

    # pick the smallest of the two number of samples as the number of quantiles
    num_percentiles = min(batch1.shape[0], batch2.shape[0])
    percentiles = torch.linspace(0, 100, num_percentiles)

    # compute the percentiles on the two batches
    sketch1 = sketch.sketch(projectors, batch1, percentiles)
    sketch2 = sketch.sketch(projectors, batch2, percentiles)

    # return SW as the average of the squared error between them
    return torch.nn.MSELoss()(sketch1, sketch2)


class GSW():
    def __init__(self, dataset, num_percentiles=500, num_examples=5000,)
