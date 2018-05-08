# imports
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch import Tensor
from joblib import Parallel, delayed
import multiprocessing
from functools import reduce
import tqdm
import argparse
import os
import sys


class Projectors(Dataset):
    def __init__(self, size, data_dim):
        super(Dataset, self).__init__()
        self.size = size
        self.data_dim = data_dim

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        pass


class RandomProjectors(Projectors):
    """Each projector is a set of unit-length random vectors"""
    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]

        result = np.empty((len(idx), self.data_dim, self.data_dim))
        for pos, id in enumerate(idx):
            np.random.seed(id)
            result[pos] = np.random.randn(self.data_dim, self.data_dim)
            result[pos] /= (np.linalg.norm(result[pos], axis=1))[:, None]
        return np.squeeze(result)


class RandomLocalizedProjectors(Projectors):
    """Each projector is a set of unit-length random vectors, that are
    active only in neighbourhoods of the data"""
    @staticmethod
    def get_factors(n):
        return np.unique(reduce(list.__add__,
                         ([i, int(n//i)] for i in range(1, int(n**0.5) + 1)
                          if not n % i)))[1:]

    def __init__(self, size, data_dim):
        print('Initializing localized projectors')
        super(RandomLocalizedProjectors, self).__init__(size, data_dim)
        self.factors = RandomLocalizedProjectors.get_factors(self.data_dim)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        result = np.zeros((len(idx), self.data_dim, self.data_dim))
        for pos, id in enumerate(idx):
            # generate each set of projectors
            np.random.seed(id)
            size_patches = np.random.choice(self.factors)
            short_matrix = np.random.randn(self.data_dim, size_patches)
            short_matrix /= (np.linalg.norm(short_matrix, axis=1))[:, None]
            for k in range(int(self.data_dim/size_patches)):
                indices = slice(k*size_patches, (k+1)*size_patches)
                result[pos, indices, indices] = short_matrix[indices, :]
        return np.squeeze(result)


class UnitaryProjectors(Projectors):
    """Each projector is a unitary basis of the data-space"""
    def __getitem__(self, idx):
        if self.shape is None:
            raise ValueError('Projectors must be prepared before use.')

        if isinstance(idx, int):
            idx = [idx]

        result = np.empty((len(idx), *self.shape))
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
            z = np.random.randn(*self.shape)
            q, r = np.linalg.qr(z)
            d = np.diagonal(r)
            ph = d/np.absolute(d)
            result[pos] = np.multiply(q, ph, q)
        return np.squeeze(result)


class NumpyDataset(Dataset):
    def __init__(self, array):
        self.data = array

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return (self.data[idx], None)


class OneShotDataLoader(DataLoader):
    def __init__(self, dataset):
        self.dataset = dataset
        self.done = False

    def __iter__(self):
        self.done = False
        return self

    def __next__(self):
        if self.done:
            raise StopIteration
        else:
            self.done = True
            return (self.dataset.data, None)


def load_data(dataset, clipto,
              data_dir="data", img_size=None, memory_usage=2):
    # Data loading
    print('Loading data')
    if os.path.exists(dataset):
        # this is a ndarray saved by numpy.save. Just dataset and clipto are
        # useful
        imgs_npy = np.load(dataset)
        if clipto is not None:
            imgs_npy = imgs_npy[:min(imgs_npy.shape[0], clipto)]
        num_samples = imgs_npy.shape[0]
        npy_dataset = NumpyDataset(imgs_npy)
        data_loader = OneShotDataLoader(npy_dataset)
    else:
        # this is a torchvision dataset
        DATASET = getattr(datasets, dataset)
        transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor()])
        data = DATASET(data_dir, train=True, download=True,
                       transform=transform)
        data_dim = int(np.prod(data[0][0].size()))

        # computing batch size, that fits in memory
        data_bytes = data_dim * data[0][0].element_size()
        nimg_batch = int(memory_usage*2**30 / data_bytes)
        num_samples = len(data)
        if clipto is not None:
            num_samples = min(num_samples, clipto)
            nimg_batch = min(nimg_batch, clipto)
        nimg_batch = max(1, min(nimg_batch, num_samples))

        data_loader = torch.utils.data.DataLoader(data, batch_size=nimg_batch)

        if nimg_batch == num_samples:
            # Everything fits into memory: load only once
            for img, labels in data_loader:
                imgs_npy = torch.Tensor(img).view(-1, data_dim).numpy()
            npy_dataset = NumpyDataset(imgs_npy)
            data_loader = OneShotDataLoader(npy_dataset)
    return data_loader


def fast_percentile(V, quantiles):
    return np.array(Parallel(n_jobs=multiprocessing.cpu_count()-1)
                    (delayed(np.percentile)(v, quantiles)
                     for v in V))


class SketchIterator:
    def __init__(self,
                 dataloader, projectors, batch_size, num_quantiles,
                 start, stop=-1):
        self.dataloader = dataloader
        self.projectors = projectors
        self.quantiles = np.linspace(0, 100, num_quantiles)
        self.start = start
        self.stop = stop if stop >= 0 else None
        self.current = start
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop is not None and (self.current >= self.stop):
            raise StopIteration
        else:
            batch_proj = self.projectors[range(self.current,
                                         self.current+self.batch_size)]
            self.current += self.batch_size
            batch_proj = np.reshape(batch_proj, [-1, self.projectors.data_dim])

            # loop over the data
            num_samples = len(self.dataloader.dataset)
            data_dim = int(np.prod(self.dataloader.dataset[0][0].shape))

            pos = 0
            projections = None
            for img, labels in self.dataloader:
                # load img numpy data
                if isinstance(img, torch.Tensor):
                    img = torch.Tensor(img).numpy()
                if img.shape[-1] != data_dim:
                    img = np.reshape(img, [-1, data_dim])
                if img.shape != (num_samples, data_dim):
                    if projections is None:
                        projections = np.empty((data_dim, num_samples))
                    projections[:, pos:pos+len(img)] = batch_proj.dot(img.T)
                    pos += len(img)
                else:
                    projections = batch_proj.dot(img.T)

            # compute the quantiles for each of these projections
            return fast_percentile(projections, self.quantiles), batch_proj


def write_sketch(data_loader, output, projectors_class, num_sketches,
                 num_quantiles):

    # load data
    data_dim = int(np.prod(data_loader.dataset[0][0].shape))

    # prepare the projectors
    ProjectorsClass = getattr(sys.modules[__name__], projectors_class)
    projectors = ProjectorsClass(args.num_sketches, data_dim)

    print('Sketching the data')
    # allocate the sketch variable (quantile function)
    qf = np.array([s for s, p in
                   tqdm.tqdm(SketchIterator(data_loader, projectors, 1,
                                            num_quantiles, 0, num_sketches))])

    # save sketch
    np.save(output, {'qf': qf,
                     'projectors_class': projectors.__class__.__name__})


def add_data_arguments(parser):
    parser.add_argument("dataset",
                        help="either a file saved by numpy.save "
                             "containing a ndarray of shape "
                             "num_samples x data_dim, or the name "
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
    parser.add_argument("--num_sketches",
                        help="Number of sketches. Each sketch gets a number "
                             "thetas equal to the data dimension.",
                        type=int,
                        default=400)
    parser.add_argument("--num_quantiles",
                        help="Number of quantiles to compute",
                        type=int,
                        default=100)
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
    if args.dataset is None:
        raise ValueError('Need a dataset argument. Aborting.')
    if args.root_data_dir is None:
        args.root_data_dir = 'data/'+args.dataset
    data_loader = load_data(args.dataset, None, args.root_data_dir,
                            args.img_size, args.memory_usage)

    # setting the default value for the output as the dataset name
    if args.output is None:
        args.output = 'sketch_' + args.dataset

    # launch sketching
    import cProfile
    pr = cProfile.Profile()
    pr.enable()

    write_sketch(data_loader,
                 args.output,
                 args.projectors, args.num_sketches, args.num_quantiles)

    pr.disable()
    pr.dump_stats('sketch_profile.prof')
