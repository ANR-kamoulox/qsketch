# imports
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
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


def load_data(dataset, clipto,
              data_dir="data", img_size=None, memory_usage=2):
    # Data loading
    if os.path.exists(dataset):
        # this is a ndarray saved by numpy.save. Just dataset and clipto are
        # useful
        imgs_npy = np.load(dataset)
        (num_samples, data_dim) = imgs_npy.shape
        if clipto is not None:
            imgs_npy = imgs_npy[:min(num_samples, clipto)]
            (num_samples, data_dim) = imgs_npy.shape
        data_loader = None
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
        nimg_batch = min(nimg_batch, num_samples)

        data_loader = torch.utils.data.DataLoader(data, batch_size=nimg_batch)

        if nimg_batch == num_samples:
            # Everything fits into memory: load only once
            for img, labels in data_loader:
                imgs_npy = torch.Tensor(img).view(-1, data_dim).numpy()
            data_loader = None
            data = None
        else:
            # Will have to load several times
            imgs_npy = None
    return imgs_npy, data_loader, num_samples, data_dim

def percentile(v, q):
    return np.percentile(v,q)
def fast_percentile(V,quantiles):
    return np.array(Parallel(n_jobs=multiprocessing.cpu_count()-1)
                    (delayed(percentile)(v,quantiles)
                     for v in V))

def main_sketch(dataset, output, projectors_class, num_sketches,
                num_quantiles, img_size,
                memory_usage, data_dir, clipto=None):

    # load data
    (imgs_npy, data_loader,
     num_samples, data_dim) = load_data(dataset, clipto, data_dir,
                                        img_size, memory_usage)

    # prepare the projectors
    ProjectorsClass = getattr(sys.modules[__name__], projectors_class)
    projectors = ProjectorsClass(args.num_sketches, data_dim)
    sketch_loader = DataLoader(range(len(projectors)))

    print('Sketching the data')
    # allocate the sketch variable (quantile function)
    quantiles = np.linspace(0, 100, num_quantiles)
    qf = np.zeros((len(projectors), data_dim, num_quantiles))

    # proceed to projection
    for batch in tqdm.tqdm(sketch_loader):
        # initialize projections
        batch_proj = projectors[batch]
        if imgs_npy is None:
            # loop over the data if not loaded once (torchvision dataset)
            pos = 0
            projections = np.zeros((data_dim, num_samples))
            for img, labels in tqdm.tqdm(data_loader):
                # load img numpy data
                imgs_npy = torch.Tensor(img).view(-1, data_dim).numpy()
                projections[:, pos:pos+len(img)] = batch_proj.dot(imgs_npy.T)
                pos += len(img)
        else:
            # data is in memory as a ndarray
            projections = batch_proj.dot(imgs_npy.T)
        # compute the quantiles for each of these projections
        qf[batch] = fast_percentile(projections, quantiles)
        #qf[batch] = np.percentile(projections, quantiles, axis=1).T

    # save sketch
    np.save(output, {'qf': qf,
                     'projectors_class': projectors.__class__.__name__})   
    '''with open(output, 'wb') as f:
        np.save(f, {'qf': qf,
                     'projectors_class': projectors.__class__.__name__},
                allow_pickle=True)'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     'Sketch a torchvision dataset or a '
                                     'ndarray saved on disk.')
    parser.add_argument("dataset", help="either a file saved by numpy.save "
                                        "containing a ndarray of shape "
                                        "num_samples x data_dim, or the name "
                                        "of the dataset to sketch, which "
                                        "must be one of those supported in "
                                        "torchvision.datasets")
    parser.add_argument("-s", "--img_size",
                        help="Images are resized as s x s",
                        type=int,
                        default=64)
    parser.add_argument("-m", "--memory_usage",
                        help="RAM usage for batches in Gb",
                        type=int,
                        default=2)
    parser.add_argument("-r", "--root_data_dir",
                        help="Root directory of the dataset. Defaults to"
                             "`data/\{dataset\}`")
    parser.add_argument("-p", "--projectors",
                        help="Type of projectors, must be of the classes "
                             "overriding Projectors.",
                        default="RandomProjectors")
    parser.add_argument("-n", "--num_sketches",
                        help="Number of sketches. Each sketch gets a number "
                             "thetas equal to the data dimension.",
                        type=int,
                        default=400)
    parser.add_argument("-q", "--num_quantiles",
                        help="Number of quantiles to compute",
                        type=int,
                        default=100)
    parser.add_argument("-o", "--output",
                        help="Output file, defaults to the `dataset` argument")

    args = parser.parse_args()
    if args.root_data_dir is None:
        args.root_data_dir = 'data/'+args.dataset
    if args.output is None:
        args.output = args.dataset

    import cProfile
    pr = cProfile.Profile()
    pr.enable()
    main_sketch(args.dataset,
                args.output,
                args.projectors, args.num_sketches,
                args.num_quantiles, args.img_size,
                args.memory_usage, args.root_data_dir)
    pr.disable()
    pr.dump_stats('profile.prof')
