# imports
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
                 num_epochs=-1, queue=None):
        """creates a new datastream object. If num_epoch is negative, will
        loop endlessly. If the queue object is None, will create a new one"""

        # Allocate the data queue if not provided
        if queue is None:
            self.queue = mp.Queue(maxsize=30)

        # create a new multiprocessing manager
        self.manager = mp.Manager()

        # prepare some data for the synchronization of the workers
        self.params = self.manager.dict()
        self.params['dataset'] = dataset
        self.params['die'] = False
        self.params['num_epochs'] = num_epochs

        # create a lock
        self.lock = mp.Lock()

    def start(self):
        self.process = mp.Process(
                            target=data_worker,
                            kwargs={'lock': self.lock,
                                    'params': self.params,
                                    'data_queue': self.queue})
        atexit.register(partial(exit_handler, stream=self))
        self.process.start()


def exit_handler(stream):
    print('Terminating data worker...')
    if stream.params is not None:
        stream.params['die'] = True
    stream.process.join()
    print('done')


def data_worker(lock, params, data_queue):
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

    use_cuda = False
    if use_cuda:
        kwargs = {'num_workers': 1, 'pin_memory': True}
    else:
        num_workers = max(1, floor((mp.cpu_count()-1)/2))
        kwargs = {'num_workers': num_workers}
    data_source = DataLoader(dataset, batch_size=600, **kwargs)

    print('Starting the DataStream worker')
    while num_epochs < 0 or epoch < num_epochs:
        check = 100
        for (X, Y) in data_source:
            data_queue.put((X, Y))
            check -= 1
            if check == 0:
                check = 100
                with getlock():
                    if params['die']:
                        break
        epoch += 1
