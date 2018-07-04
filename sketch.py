# imports
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.sampler import Sampler
from .autograd_percentile import Percentile
import multiprocessing
from multiprocessing import Queue
from .celeba import CelebA
from functools import reduce
import tqdm
import argparse
import os
import sys
from math import floor


class Projectors:
    """Each projector is a set of unit-length random vector"""
    def __init__(self, num_thetas, data_shape):
        self.num_thetas = num_thetas
        self.data_shape = data_shape
        self.data_dim = np.prod(np.array(data_shape))

        # prepare the torch device (cuda or cpu ?)
        use_cuda = False#torch.cuda.is_available()
        self.device = "cuda" if use_cuda else "cpu"

    def __getitem__(self, idx):
        device = torch.device(self.device)

        if isinstance(idx, int):
            idx = [idx]
        result = torch.empty((len(idx), self.num_thetas, self.data_dim),
                             device=device)
        for pos, id in enumerate(idx):
            torch.manual_seed(id)
            result[pos] = torch.randn(self.num_thetas, self.data_dim,
                                      device=device)
            result[pos] /= (torch.norm(result[pos], dim=1, keepdim=True))
        return torch.squeeze(result)


class DynamicSubsetRandomSampler(Sampler):
    r"""Samples a given number of elements randomly, from a given amount
    of indices, with replacement.

    Arguments:
        data_source (Dataset): elements to sample from
        nb_items (int): number of samples to draw
    """

    def __init__(self, data_source, nb_items):
        self.data_source = data_source
        self.nb_items = nb_items

    def __iter__(self):
        return iter(list(np.random.randint(low=0, high=len(self.data_source),
                                           size=self.nb_items)))

    def __len__(self):
        return self.nb_items


def load_data(dataset, data_dir="data", img_size=None,
              clipto=None, batch_size=640, use_cuda=False):
    if use_cuda:
        kwargs = {'num_workers': 1, 'pin_memory': True}
    else:
        num_workers = max(1, floor((multiprocessing.cpu_count()-2)/2))
        kwargs = {'num_workers': num_workers}

    # First load the DataSet
    if os.path.isfile(dataset):
        # this is a file, and hence should be a ndarray saved by numpy.save
        imgs = torch.tensor(np.load(dataset))
        data = TensorDataset(imgs, torch.zeros(imgs.shape[0]), **kwargs)
    else:
        # this is not a file. In this case, there will be a DataSet class to
        # handle it.

        # first define the transforms
        transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor()])

        # If it's a dir and is celebA, then we have a special loader
        if os.path.isdir(dataset):
            if os.path.basename(dataset).upper() == "CELEBA":
                data = CelebA(dataset, transform, mode="train")
        else:
            # Just assume it's a torchvision dataset
            DATASET = getattr(datasets, dataset)
            data = DATASET(data_dir, train=True, download=True,
                           transform=transform)

    # Now get a dataloader
    nb_items = len(data) if clipto < 0 else clipto
    sampler = DynamicSubsetRandomSampler(data, nb_items)
    data_loader = torch.utils.data.DataLoader(data,
                                              sampler=sampler,
                                              batch_size=batch_size,
                                              **kwargs)
    return data_loader


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
        self.current = np.random.randint(0, 10000, 1)

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        return self.__getitem__(self.current - 1)

    def __getitem__(self, index):
        """import os
        print(os.getpid(),index)"""
        device = torch.device(self.device)
        projector = self.projectors[index].view([-1, self.projectors.data_dim])
        projector = projector.to(device)
        if self.requires_grad:
            projector.requires_grad_()

        projections = torch.empty((projector.shape[0],
                                   len(self.data_source.sampler)),
                                  device=device,
                                  requires_grad=self.requires_grad)

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


def add_data_arguments(parser):
    parser.add_argument("dataset",
                        help="either: i/ a file saved by numpy.save "
                             "containing a ndarray of shape "
                             "(num_samples,)+data_shape, or ii/ the path "
                             "to the celebA directory (that must end with"
                             "`CelebA` or iii/ the name of one of the "
                             "datasets in torchvision.datasets")
    parser.add_argument("--img_size",
                        help="Images are resized as s x s",
                        type=int,
                        default=64)
    parser.add_argument("--root_data_dir",
                        help="Root directory of the dataset. Defaults to"
                             "`data/\{dataset\}`")
    return parser


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
