# imports
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from math import floor
import atexit
from functools import partial
from contextlib import contextmanager


class DataStream:
    "A DataStream object puts items from a Dataset into a queue"

    def __init__(self,
                 dataset,
                 device='cpu',
                 num_workers=2,
                 num_epochs=-1, queue=None):
        """creates a new datastream object. If num_epoch is negative, will
        loop endlessly. If the queue object is None, will create a new one"""

        # Allocate the data queue if not provided
        if queue is None:
            self.queue = mp.Queue(maxsize=30)

        # create a new multiprocessing manager
        self.manager = mp.Manager()

        # if the dataset has a `_pack` function, we call it now
        packfn = getattr(dataset, '_pack', None)
        if packfn is not None and callable(packfn):
            print('we call pack')
            packfn()

        # prepare some data for the synchronization of the workers
        self.params = self.manager.dict()

        self.params['dataset'] = dataset
        self.params['die'] = False
        self.params['num_epochs'] = num_epochs

        # create a lock
        self.lock = mp.Lock()

        self.device = device
        self.num_workers = num_workers

    def stream(self):
        # let's go
        self.process = mp.Process(
                            target=data_worker,
                            kwargs={'device': self.device,
                                    'num_workers': self.num_workers,
                                    'lock': self.lock,
                                    'params': self.params,
                                    'data_queue': self.queue})
        #atexit.register(partial(exit_handler, stream=self))
        self.process.start()


def exit_handler(stream):
    print('Terminating data worker...')
    if stream.params is not None:
        stream.params['die'] = True
    stream.process.join()
    print('done')


def data_worker(device, num_workers, lock, params, data_queue):
    @contextmanager
    def getlock():
        # get the lock of the stream to manipulate the stream.data
        result = lock.acquire(block=True)
        yield result
        if result:
            lock.release()

    epoch = 0
    with getlock():
        dataset = params['dataset']
        num_epochs = params['num_epochs']

    device_obj = torch.device(device)
    if device == 'cuda' and not dataset[0][0].is_cuda:
        print('[DataStream] the dataset is on CPU, and CUDA is asked. Pinning'
              ' memory.')
        # we will pin memory only if the dataset is on CPU
        kwargs = {'num_workers': 1, 'pin_memory': True}
        num_workers = 1
    elif dataset[0][0].is_cuda:
        print('[DataStream] The dataset is on CUDA, picking 0 workers.')
        kwargs = {'num_workers': 0}
        num_workers = 0
    else:
        print('[DataStream] the dataset is on CPU, and CPU is asked. '
              'Multiprocessing.')
        kwargs = {'num_workers': num_workers}
    data_source = DataLoader(dataset, batch_size=600, **kwargs)

    print('[DataStream] Starting the sampling with %d workers'
          % num_workers)
    while num_epochs < 0 or epoch < num_epochs:
        check = 100
        for (X, Y) in data_source:
            data_queue.put((X.to(device_obj), Y.to(device_obj)))
            check -= 1
            if check == 0:
                check = 100
                with getlock():
                    if params['die']:
                        return
        epoch += 1
