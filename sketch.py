# imports
import numpy as np
import torch
from torch.utils.data import Dataset
from .autograd_percentile import Percentile


class Projectors:
    """Each projector is a set of unit-length random vector"""

    def __init__(self, num_thetas, data_shape):
        self.num_thetas = num_thetas
        self.data_shape = data_shape
        self.data_dim = np.prod(np.array(data_shape))

        # prepare the torch device (cuda or cpu ?)
        use_cuda = False  # torch.cuda.is_available()
        self.device = "cuda" if use_cuda else "cpu"

    def __getitem__(self, idx):
        device = torch.device(self.device)

        if isinstance(idx, int):
            idx = [idx]
        result = torch.empty((len(idx), self.num_thetas, self.data_dim),
                             device=device)
        for pos, id in enumerate(idx):
            torch.manual_seed(id)

            """nb_values = np.random.randint(low=3, high=500)
            values = torch.randn(nb_values, device=device)
            indices = torch.randint(low=0, high=nb_values,
                                    size=(self.num_thetas, self.data_dim)
                                    ).long()
            result[pos] = values[indices]"""
            result[pos] = torch.randn(self.num_thetas, self.data_dim,
                                      device=device)
            result[pos] /= (torch.norm(result[pos], dim=1, keepdim=True))
        return torch.squeeze(result)


class Sketcher(Dataset):
    """Sketcher class: takes a source of data, a dataset of projectors, and
    construct sketches.
    When accessing one of its elements, computes the corresponding sketch.
    When iterated upon, computes random batches of sketches.
    """
    def __init__(self,
                 dataloader,
                 projectors,
                 num_quantiles,
                 device,
                 requires_grad=False):
        self.data_source = dataloader
        self.projectors = projectors
        self.num_quantiles = num_quantiles
        self.requires_grad = requires_grad
        self.device = "cpu"

    def __iter__(self):
        return self

    def __next__(self):
        next_id = np.random.randint(np.iinfo(np.int32).max)
        return self.__getitem__(next_id)

    def __getitem__(self, index):
        # get the device
        device = torch.device(self.device)

        # get the projector
        projector = self.projectors[index].view([-1, self.projectors.data_dim])
        projector = projector.to(device)
        if self.requires_grad:
            projector.requires_grad_()

        # allocate the projectons variable
        projections = torch.empty((projector.shape[0],
                                   len(self.data_source.sampler)),
                                  device=device,
                                  requires_grad=self.requires_grad)

        # compute the projections by a loop over the data
        pos = 0
        for imgs, labels in self.data_source:
            # get a batch of images and send it to device
            imgs = imgs.to(device)

            # if required, flatten each sample
            if imgs.shape[-1] != self.projectors.data_dim:
                imgs = imgs.view([-1, self.projectors.data_dim])

            # aggregate the projections
            projections[:, pos:pos+len(imgs)] = \
                torch.mm(projector, imgs.transpose(0, 1))
            pos += len(imgs)

        # compute the quantiles for these projections
        return (Percentile(self.num_quantiles, device)(projections).float(),
                projector)


def add_sketch_arguments(parser):
    parser.add_argument("--num_thetas",
                        help="Number of thetas per sketch.",
                        type=int,
                        default=2000)
    parser.add_argument("--num_quantiles",
                        help="Number of quantiles to compute",
                        type=int,
                        default=100)
    parser.add_argument("--clip_to",
                        help="Number of datapoints used per sketch. If "
                             "negative, take all of them.",
                        type=int,
                        default=3000)

    return parser
