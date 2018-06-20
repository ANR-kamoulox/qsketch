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
    def __init__(self, size, num_thetas, data_shape):
        super(Dataset, self).__init__()
        self.size = size
        self.num_thetas = num_thetas
        self.data_dim = np.prod(np.array(data_shape))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        pass


class RandomProjectors(Projectors):
    """Each projector is a set of unit-length random vectors"""
    def __init__(self, size, num_thetas, data_shape):
        super(RandomProjectors, self).__init__(size, num_thetas,
                                               data_shape)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        result = np.empty((len(idx), self.num_thetas, self.data_dim))
        for pos, id in enumerate(idx):
            np.random.seed(id)
            result[pos] = np.random.randn(self.num_thetas, self.data_dim)
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


class NumpyDataset(Dataset):
    def __init__(self, array):
        self.data = array

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return (self.data[idx], None)


class OneShotDataLoader(DataLoader):
    def __init__(self, dataset, clipto):
        self.dataset = dataset
        self.done = False
        self.clipto = clipto

    def __iter__(self):
        self.done = False
        return self

    def __next__(self):
        if self.done:
            raise StopIteration
        else:
            self.done = True
            if self.clipto > 0:
                order = np.random.permutation(self.dataset.data.shape[0])
                return (self.dataset.data[order[:self.clipto], ...], None)
            else:
                return (self.dataset.data, None)


def load_data(dataset, clipto,
              data_dir="data", img_size=None, memory_usage=2):
    # Data loading
    if os.path.exists(dataset):
        # this is a ndarray saved by numpy.save. Just dataset and clipto are
        # useful
        imgs_npy = np.load(dataset)
        num_samples = imgs_npy.shape[0]
        npy_dataset = NumpyDataset(imgs_npy)
        data_loader = OneShotDataLoader(npy_dataset, clipto)
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

        if nimg_batch == num_samples:
            # Everything fits into memory: load only once
            data_loader = torch.utils.data.DataLoader(data,
                                                      batch_size=nimg_batch)
            for img, labels in data_loader:
                #imgs_npy = torch.Tensor(img).view(-1, data_dim).numpy()
                imgs_npy = torch.Tensor(img).numpy()

            npy_dataset = NumpyDataset(imgs_npy)
            data_loader = OneShotDataLoader(npy_dataset, clipto)
        else:
            data_loader = torch.utils.data.DataLoader(data, batch_size=clipto)

    return data_loader


def fast_percentile(V, quantiles):
    if V.shape[1] < 10000:
        return np.percentile(V, quantiles, axis=1).T
    return np.array(Parallel(n_jobs=multiprocessing.cpu_count()-1)
                    (delayed(np.percentile)(v, quantiles)
                     for v in V))


class SketchIterator:
    def __init__(self,
                 dataloader, projectors, batch_size, num_quantiles,
                 start, stop=-1, clipto=-1):
        self.dataloader = dataloader
        self.projectors = projectors
        self.quantiles = np.linspace(0, 100, num_quantiles)
        self.start = start
        self.stop = stop if stop >= 0 else None
        self.current = start
        self.batch_size = batch_size
        self.num_samples_per_batch = (clipto if clipto > 0
                                      else len(self.dataloader.dataset))
        self.clipto = clipto

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop is not None and (self.current >= self.stop):
            raise StopIteration
        else:
            #   import ipdb; ipdb.set_trace()
            batch_proj = self.projectors[range(self.current,
                                         self.current+self.batch_size)]
            self.current += self.batch_size
            batch_proj = np.reshape(batch_proj, [-1, self.projectors.data_dim])

            # loop over the data
            num_samples = len(self.dataloader.dataset)
            data_shape = self.dataloader.dataset[0][0].shape
            data_dim = int(np.prod(data_shape))

            pos = 0
            projections = None
            for index, (img, labels) in enumerate(self.dataloader):
                # load img numpy data
                if isinstance(img, torch.Tensor):
                    img = torch.Tensor(img).numpy()
                if img.shape[-1] != data_dim:
                    img = np.reshape(img, [-1, data_dim])
                num_batch = self.clipto if self.clipto > 0 else num_samples
                if img.shape != (num_batch, data_dim):
                    if projections is None:
                        projections = np.empty((data_dim, num_batch))
                    projections[:, pos:pos+len(img)] = batch_proj.dot(img.T)
                    pos += len(img)
                else:
                    projections = batch_proj.dot(img.T)

                """
                code for plotting the data, when it's a mono image
                from torchvision.utils import save_image, make_grid
                samples = img
                [num_samples, data_dim] = samples.shape
                import ipdb; ipdb.set_trace()
                samples = samples[:min(208, num_samples)]
                num_samples = samples.shape[0]

                imsize = int(np.sqrt(data_dim))
                samples = np.reshape(samples,
                                     [num_samples, 1, imsize,imsize])
                pic = make_grid(torch.Tensor(samples),
                                nrow=8, padding=2, normalize=True, scale_each=True)
                save_image(pic, 'dataset_example.png')
                import ipdb; ipdb.set_trace()"""

            # compute the quantiles for each of these projections
            return fast_percentile(projections, self.quantiles), batch_proj


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
