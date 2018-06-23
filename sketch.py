# imports
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, TensorDataset, DataLoader
from joblib import Parallel, delayed
from .autograd_percentile import Percentile
import multiprocessing
from functools import reduce
import tqdm
import argparse
import os
import sys



class Projectors(Dataset):
    def __init__(self, size, num_thetas, data_shape):
        super(Dataset, self).__init__()
        self.size = size
        self.num_thetas = num_thetas
        self.data_shape = data_shape
        self.data_dim = np.prod(np.array(data_shape))
        self.device = "cpu"

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        pass

    def set_device(self, device):
        self.device = device


class RandomProjectors(Projectors):
    """Each projector is a set of unit-length random vector"""
    def __init__(self, size, num_thetas, data_shape):
        super(RandomProjectors, self).__init__(size, num_thetas,
                                               data_shape)

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


class RandomLocalizedProjectors(Projectors):
    """Each projector is a set of unit-length random vectors, that are
    active only in neighbourhoods of the data"""
    @staticmethod
    def get_factors(n):
        return np.unique(reduce(list.__add__,
                         ([i, int(n//i)] for i in range(1, int(n**0.5) + 1)
                          if not n % i)))[1:]

    def __init__(self, size, num_thetas, data_shape):
        print('Initializing localized projectors')
        super(RandomLocalizedProjectors, self).__init__(size, num_thetas,
                                                        data_shape)
        self.factors = [v for v in
                        RandomLocalizedProjectors.get_factors(self.data_dim)
                        if v > num_thetas]

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        result = np.zeros((len(idx), self.num_thetas, self.data_dim))
        for pos, id in enumerate(idx):
            # generate each set of projectors
            np.random.seed(id)
            if not id % 2:
                result[pos] = np.random.randn(self.num_thetas, self.data_dim)
                result[pos] /= (np.linalg.norm(result[pos], axis=1))[:, None]

            rtemp = np.zeros((self.data_dim, self.data_dim))
            size_patches = np.random.choice(self.factors)
            short_matrix = np.random.randn(self.data_dim, size_patches)
            short_matrix /= (np.linalg.norm(short_matrix, axis=1))[:, None]
            for k in range(int(self.data_dim/size_patches)):
                indices = slice(k*size_patches, (k+1)*size_patches)
                rtemp[indices, indices] = short_matrix[indices, :]
            subset = np.random.choice(self.data_dim, self.num_thetas,
                                      replace=False)
            result[pos] = rtemp[subset]
        return np.squeeze(result)


class UnitaryProjectors(Projectors):
    """Each projector is a unitary basis of the data-space"""
    def __getitem__(self, idx):
        if self.num_thetas != self.data_dim:
            raise ValueError('For unitary projectors, num_theta=data_dim')
        if isinstance(idx, int):
            idx = [idx]

        result = np.empty((len(idx), self.data_dim, self.data_dim))
        for pos, id in enumerate(idx):
            np.random.seed(id)
            '''A Random matrix distributed with Haar measure,
            From Francesco Mezzadri:
            @article{mezzadri2006generate,
                title={How to generate random matrices from the
                classical compact groups},
                author={Mezzadri, Francesco},
                journal={arXiv preprint math-ph/0609050},
                year={2006}}
            '''
            z = np.random.randn(self.data_dim, self.data_dim)
            q, r = np.linalg.qr(z)
            d = np.diagonal(r)
            ph = d/np.absolute(d)
            result[pos] = np.multiply(q, ph, q)
        return np.squeeze(result)


def load_data(dataset, clipto,
              data_dir="data", img_size=None, memory_usage=2, use_cuda=False):
    kwargs = {
        'num_workers': 1, 'pin_memory': True
    } if use_cuda else {'num_workers': 4}

    if os.path.exists(dataset):
        # this is a ndarray saved by numpy.save
        imgs = torch.tensor(np.load(dataset))
        data = TensorDataset(imgs, torch.zeros(imgs.shape[0]), **kwargs)
    else:
        # this is a torchvision dataset
        DATASET = getattr(datasets, dataset)
        transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor()])
        data = DATASET(data_dir, train=True, download=True,
                       transform=transform)

    data_shape = data[0][0].size()
    data_dim = int(np.prod(data_shape))

    # computing batch size, that fits in memory
    data_bytes = data_dim * data[0][0].element_size()
    nimg_batch = int(memory_usage*2**30 / data_bytes)
    num_samples = len(data)
    nimg_batch = max(1, min(nimg_batch, num_samples))

    batch_size = clipto if clipto > 0 else nimg_batch
    data_loader = torch.utils.data.DataLoader(data,
                                              batch_size=batch_size, **kwargs)
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


def write_sketch(data_loader, output, projectors_class, num_sketches,
                 num_thetas, num_quantiles, clipto):

    # load data
    data_shape = data_loader.dataset[0][0].shape

    # prepare the projectors
    ProjectorsClass = getattr(sys.modules[__name__], projectors_class)
    projectors = ProjectorsClass(args.num_sketches, num_thetas, data_shape)

    print('Sketching the data')
    # allocate the sketch variable (quantile function)
    qf = np.array([s for s, p in
                   tqdm.tqdm(SketchIterator(data_loader, projectors, 1,
                                            num_quantiles, 0, num_sketches,
                                            clipto))])

    # save sketch
    np.save(output, {'qf': qf, 'data_shape': data_shape,
                     'projectors_class': projectors.__class__.__name__})


def add_data_arguments(parser):
    parser.add_argument("dataset",
                        help="either a file saved by numpy.save "
                             "containing a ndarray of shape "
                             "(num_samples,)+data_shape, or the name "
                             "of the dataset to sketch, which "
                             "must be one of those supported in "
                             "torchvision.datasets")
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
    parser.add_argument("--clip",
                        help="Number of datapoints used per sketch. If "
                             "negative, take all of them.",
                        type=int,
                        default=-1)

    return parser


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description=
                                     'Sketch a torchvision dataset or a '
                                     'ndarray saved on disk.')
    parser = add_data_arguments(parser)
    parser = add_sketch_arguments(parser)
    parser.add_argument("--output",
                        help="Output file, defaults to the `dataset` argument")
    args = parser.parse_args()

    # parsing the dataset parameters and getting the data loader
    if args.root_data_dir is None:
        args.root_data_dir = 'data/'+args.dataset
    data_loader = load_data(args.dataset, args.clip, args.root_data_dir,
                            args.img_size, args.memory_usage)

    # setting the default value for the output as the dataset name
    if args.output is None:
        args.output = 'sketch_' + args.dataset

    # launch sketching
    write_sketch(data_loader,
                 args.output,
                 args.projectors, args.num_sketches, args.num_thetas,
                 args.num_quantiles, args.clip)
