# imports
import numpy as np
import torch
from torch.utils.data import Dataset
from .autograd_percentile import Percentile
import atexit
import queue
import torch.multiprocessing as mp
import functools
import time


class Projectors:
    """Each projector is a set of unit-length random vector"""

    def __init__(self, num_thetas, data_shape):
        self.num_thetas = num_thetas
        self.data_shape = data_shape
        self.data_dim = np.prod(np.array(data_shape))
        # for now, always use the CPU for generating projectors
        self.device = "cpu"

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
                 requires_grad=False):
        self.data_source = dataloader
        self.projectors = projectors
        self.num_quantiles = num_quantiles
        self.requires_grad = requires_grad

        # only use the cpu for now for the sketcher
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


def exit_handler(processes, shared_data):
    print('Terminating sketchers...')
    for p in processes:
        p.join()
    print('done')


def stream_sketches(sketcher, data_queue, shared_data):
    print(shared_data)
    for (target_qf, projector) in sketcher:
        # while we didn't put the current sketch into the queue, we loop
        done = False
        while not done:
            try:
                # put the data into the queue
                data_queue.put(((target_qf, projector)), timeout=1)

                # normally, the data is now put and we may continue
                done = True
                if 'counter' in shared_data:
                    shared_data['counter'] -= 1
                    print('value of counter', shared_data['counter'])
                    if shared_data['counter'] < 0:
                        print('Limit reached, ending sketching.')
                        # they can die.
                        # we reached the limit, we let the other workers know
                        shared_data['sleep'] = True

            except queue.Full:
                # the queue was full and we couldn't put the data in time.
                # we have to go on with the same data in that case
                pass

            print(shared_data)
            if ('sleep' in shared_data) or ('die' in shared_data):
                print('breaking the sketch loop')
                break

        while 'sleep' in shared_data:
            print('sleeping', data_queue.qsize())
            time.sleep(2)
            if 'die' in shared_data:
                break


def start_sketching(num_workers, queue_size, counter,
                    dataloader, projectors, num_quantiles):
    """ starts the sketchers, that will put data into the sketch_queue.
    Each entry of the sketch queue will consist of a tuple
    (target_qf, projector), stored on cpu.
    If counter is None: the sketching will loop forever"""
    print(num_workers, 'num_workers')
    # go into a start method that works with pytorch queues
    ctx = mp.get_context('fork')

    # Allocate the sketch queue and start the sketchers jobs
    sketch_queue = ctx.Queue(maxsize=queue_size)
    manager = mp.Manager()
    shared_data = manager.dict()
    if counter is not None:
        shared_data['counter'] = counter
    processes = [ctx.Process(target=stream_sketches,
                             kwargs={'sketcher': Sketcher(dataloader,
                                                          projectors,
                                                          num_quantiles),
                                     'data_queue': sketch_queue,
                                     'shared_data': shared_data})
                 for n in range(num_workers)]

    atexit.register(functools.partial(exit_handler,
                                      processes=processes,
                                      shared_data=shared_data))
    for p in processes:
        p.start()

    return sketch_queue


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
