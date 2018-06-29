# imports
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, TensorDataset, DataLoader,
                             RandomSampler, Sampler
from .autograd_percentile import Percentile
import multiprocessing
from .celeba import CelebA
from functools import reduce
import tqdm
import argparse
import os
import sys


class Projectors(Projectors):
    """Each projector is a set of unit-length random vector"""
    def __init__(self, size, num_thetas, data_shape):
        super(Projectors, self).__init__()
        self.size = size
        self.num_thetas = num_thetas
        self.data_shape = data_shape
        self.data_dim = np.prod(np.array(data_shape))
        self.device = "cpu"

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        result = torch.empty((len(idx), self.num_thetas, self.data_dim),
                             device=self.device)
        for pos, id in enumerate(idx):
            torch.manual_seed(id)
            result[pos] = torch.randn(self.num_thetas, self.data_dim,
                                      device=self.device)
            result[pos] /= (torch.norm(result[pos], dim=1, keepdim=True))
        return torch.squeeze(result)

class DynamicSubsetRandomSampler(Sampler):
    r"""Samples a given number of elements randomly, from a given amount
    of indices, with replacement.

    Arguments:
        data_source (Dataset): elements to sample from
        nb_items (int): number of samples to draw
    """

    def __init__(self, indices):
        self.data_source = data_source
        self.nb_items = nb_items
        self.indices = None
        self.update()

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return self.nb_items

    def udate(self):
        self.indices = list(np.random.randint(low=0, high=len(self.data_source),
                                              size=self.nb_items))



def load_data(dataset, data_dir="data", img_size=None,
              clipto=None, batchsize=64, use_cuda=False):
    kwargs = {
        'num_workers': 1, 'pin_memory': True
    } if use_cuda else {'num_workers': multiprocessing.cpu_count()-1}

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
    if clipto < 0:
        sampler = DynamicSubsetRandomSampler(data, len(data))
    else:
        sampler = DynamicSubsetRandomSampler(data, clipto)

    data_loader = torch.utils.data.DataLoader(data,
                                              sampler=sampler
                                              batch_size=batch_size,
                                              **kwargs)
    return data_loader


class SketchIterator:
    def __init__(self,
                 dataloader, projectors, batch_size, num_quantiles,
                 start, stop, clipto, device):
        self.device = device
        self.dataloader = dataloader
        self.projectors = projectors
        self.num_quantiles = num_quantiles
        self.quantiles = torch.linspace(0, 100, num_quantiles)
        self.start = start
        self.stop = stop if stop >= 0 else None
        self.current = start
        self.batch_size = batch_size
        self.num_samples_per_batch = (clipto if clipto > 0
                                      else len(self.dataloader.dataset))
        self.clipto = clipto
        self.percentile_fn = Percentile(num_quantiles, device)

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop is not None and (self.current >= self.stop):
            raise StopIteration
        else:
            batch_proj = self.projectors[range(self.current,
                                         self.current+self.batch_size)]
            self.current += self.batch_size
            batch_proj = batch_proj.view([-1, self.projectors.data_dim])

            # loop over the data
            num_samples = len(self.dataloader.dataset)

            pos = 0
            projections = None
            for index, (img, labels) in enumerate(self.dataloader):
                img = img.to(self.device)
                if img.shape[-1] != self.projectors.data_dim:
                    img = img.view([-1, self.projectors.data_dim])
                num_batch = self.clipto if self.clipto > 0 else num_samples
                if img.shape != (num_batch, self.projectors.data_dim):
                    if projections is None:
                        projections = torch.empty((self.projectors.data_dim,
                                                   num_batch),
                                                  device=self.device)
                    projections[:, pos:pos+len(img)] = \
                        torch.mm(batch_proj, img.transpose(0, 1))
                    pos += len(img)
                else:
                    projections = torch.mm(batch_proj, img.transpose(0, 1))

            # compute the quantiles for each of these projections
            return self.percentile_fn(projections), batch_proj


def add_data_arguments(parser):
    parser.add_argument("dataset",
                        help="either: i/ a file saved by numpy.save "
                             "containing a ndarray of shape "
                             "(num_samples,)+data_shape, or ii/ the path "
                             "to the celebA directory (that must end with
                             "`CelebA` or iii/ the name of one of the "
                             "datasets in torchvision.datasets")
    parser.add_argument("--img_size",
                        help="Images are resized as s x s",
                        type=int,
                        default=64)
    parser.add_argument("--memory_usage",
                        help="RAM usage for batches in Gb",
                        type=int,
                        default=2)
    parser.add_argument("--root_data_dir",
                        help="Root directory of the dataset. Defaults to"
                             "`data/\{dataset\}`")
    return parser


def add_sketch_arguments(parser):
    parser.add_argument("--projectors",
                        help="Type of projectors, must be of the classes "
                             "overriding Projectors.",
                        default="RandomProjectors")
    parser.add_argument("--num_thetas",
                        help="Number of thetas per sketch.",
                        type=int,
                        default=50)
    parser.add_argument("--num_sketches",
                        help="Number of sketches.",
                        type=int,
                        default=400)
    parser.add_argument("--num_quantiles",
                        help="Number of quantiles to compute",
                        type=int,
                        default=100)
    parser.add_argument("--clipto",
                        help="Number of datapoints used per sketch. If "
                             "negative, take all of them.",
                        type=int,
                        default=-1)

    return parser
