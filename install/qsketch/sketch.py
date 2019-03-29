# imports
import numpy as np
import torch
from torch.utils.data import Dataset
from torchpercentile import Percentile
import atexit
import queue
import torch.multiprocessing as mp
from contextlib import contextmanager
from functools import partial
import time


class Sketcher(Dataset):
    """Sketcher class: takes a source of data, a dataset of projectors, and
    construct sketches, which are the quantiles of the output of the projectors
    when applied on the specified dataset.

    When accessing one of its elements, computes the corresponding sketch,
    which means using the projector with that seed.

    When iterated upon, compute sketches with projectors with seeds in
    a deterministic sequence.
    """

    def __init__(self,
                 data_source,
                 projectors,
                 num_quantiles,
                 num_examples,
                 seed=0):
        """
            Create a new sketcher.
            data_source: Queue object
                this is where we get the data from. Each item from this queue
                should be a tuple of (X, y) values. The sentinel is None.
            projectors: Projectors object
                this is the dataset of Projectors.
            num_quantiles: int
                the number of quantiles to compute the sketch with. They will
                be picked as regularly spaced between 0 and 100
            num_examples: int
                the number of samples to use for computing each sketch.
                The data_source is assumed to produce data endlessly.
        """
        self.data_source = data_source
        self.projectors = projectors
        self.num_quantiles = num_quantiles
        self.num_examples = num_examples

        # only use the cpu for now for the sketcher
        self.device = "cpu"

        # create a random sequence with provided seed
        self.pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        # get the next id
        self.pos += 1

        # finally get the sketch
        return self.__getitem__(self.pos)

    def __getitem__(self, indexes):
        if isinstance(indexes, int):
            index = [indexes]
        else:
            index = indexes

        # get the device
        device = torch.device(self.device)

        # get the projector
        sketches = []
        percentiles = torch.linspace(0, 100, self.num_quantiles)
        for id in index:
            projector = self.projectors[id].to(device)

            # allocate the projectons variable
            projections = torch.empty((self.num_examples,
                                       self.projectors.num_thetas),
                                      device=device)

            # compute the projections by a loop over the data
            pos = 0

            while pos < self.num_examples:
                (imgs, labels) = self.data_source.get()
                # get a batch of images and send it to device
                imgs = imgs.to(device)

                # aggregate the projections
                n_imgs = min(len(imgs), self.num_examples - pos)
                projections[pos:pos+n_imgs] = projector(imgs[:n_imgs])
                pos += n_imgs
            # compute the quantiles for these projections
            sketches += [
                (Percentile()(projections, percentiles).float(),
                 id)]
        return sketches[0] if isinstance(indexes, int) else sketches


class SketchStream:
    """A SketchStream object constructs a queue attribute that will be
    filled with sketches, which are quantiles of random projections"""

    def __init__(self):
        self.processes = []
        self.data = None

        # Allocate the sketch queue
        self.queue = mp.Queue()

    def start(self, num_workers, num_epochs, num_sketches,
              data_stream, projectors, num_quantiles, num_examples):
        # first stop if it was started before
        self.stop()

        # get the number of workers
        if num_workers < 0 or num_workers is None:
            # if not defined, take at least 1 and at most half of the cores
            num_workers = np.inf
            num_workers = max(1, min(num_workers,
                              int((mp.cpu_count()-1)/2)))

        print('SketchStream using ', num_workers, 'workers')
        # now create a queue with a maxsize corresponding to a few times
        # the number of workers
        self.queue = mp.Queue(maxsize=2*num_workers)
        self.manager = mp.Manager()
        self.projectors = projectors

        # prepare some data for the synchronization of the workers
        self.data = self.manager.dict()
        self.data['num_epochs'] = num_epochs
        self.data['pause'] = False
        self.data['current_pick_epoch'] = 0
        self.data['current_put_epoch'] = 0
        self.data['current_sketch'] = 0
        self.data['done_in_current_epoch'] = 0
        self.data['num_sketches'] = (num_sketches if num_sketches > 0
                                     else np.inf)
        self.data['sketch_list'] = np.random.randint(
                np.iinfo(np.int16).max,
                size=self.data['num_sketches']).astype(int)
        self.lock = mp.Lock()

        # prepare the workers
        self.processes = [mp.Process(target=sketch_worker,
                                     kwargs={'sketcher':
                                             Sketcher(data_stream.queue,
                                                      projectors,
                                                      num_quantiles,
                                                      num_examples),
                                             'data': self.data,
                                             'lock': self.lock,
                                             'stream_queue': self.queue})
                          for n in range(num_workers)]

        atexit.register(partial(exit_handler, stream=self))

        # go
        for p in self.processes:
            p.start()

    def pause(self):
        if self.data is None:
            return
        self.data['pause'] = True

    def restart(self):
        self.data['counter'] = 0
        self.resume()

    def resume(self):
        if self.data is not None:
            self.data['pause'] = False

    def stop(self):
        if self.data is not None:
            self.data['die'] = True


def exit_handler(stream):
    print('Terminating sketchers...')
    if stream.data is not None:
        stream.data['die'] = True
    for p in stream.processes:
        p.join()
    print('done')


def sketch_worker(sketcher, data, lock, stream_queue):
    """ Actual worker for the sketch stream.
    Will get sketch ids, get data from the data queue and put sketches in the
    stream queue"""

    @contextmanager
    def getlock():
        # get the lock of the stream to manipulate the stream.data
        result = lock.acquire(block=True)
        yield result
        if result:
            lock.release()

    pause_displayed = False
    while True:
        # not dying, unless we see that later
        worker_dying = False

        if not data['pause']:
            # not in pause

            if pause_displayed:
                # we were in pause previously. Output that we're no more
                print('Sketch worker back from sleep')
                pause_displayed = False

            # print('sketch: trying to get lock')
            # With the lock, so that only one worker can manipulate the
            # counting in the stream data at the same time.
            with getlock():
                # get both the id to compute, and the number of the pick_epoch
                # (the epoch we are currently asked to compute, as opposed
                # to the put epoch, which is the epoch whose sketches we are
                # currently putting in the queue.)
                id = data['current_sketch']
                sketch_id = data['sketch_list'][id].item()
                epoch = data['current_pick_epoch']
                # print('sketch: got lock, epoch %d and id %d' % (epoch, id))
                if epoch >= data['num_epochs']:
                    # the picked epoch is larger than the number of epochs.
                    # sketching is finished.
                    #print('epoch', epoch, 'greater than the number of epochs:',
                    #      stream.num_epochs, 'dying now.')
                    worker_dying = True
                else:
                    if id == data['num_sketches'] - 1:
                        # we reached the number of sketches per epoch.
                        # we let the other workers know and increment the
                        # pick epoch.
                        #print("Obtained id %d is last for this epoch. "
                        #      "Reseting the counter and incrementing current "
                        #      "epoch " % id)
                        data['current_sketch'] = 0
                        data['current_pick_epoch'] += 1
                        data['sketch_list'] = np.random.randint(
                            np.iinfo(np.int16).max,
                            size=data['num_sketches']).astype(int)
                    else:
                        # we just increment the current sketch to pick for
                        # the next worker.
                        data['current_sketch'] += 1
            if worker_dying:
                # dying has been asked for. we'll just loop infinitely.
                # this is because there is apparently some issues raised when
                # we just kill the worker, in case some data in the queue
                # originated from him has not been taken out ?
                #print(
                #    id, epoch, 'Reached the desired amount of epochs. Dying.')
                while True:
                    time.sleep(10)
                return

            # now to the thing. We compute the sketch that has been asked for.
            #print('sketch: now trying to compute id', id)
            (target_qf, sketch_id) = sketcher[sketch_id]

            # print('sketch: we computed the sketch with id', id)
            # we need to wait until the current put epoch is the epoch we
            # picked. It may indeed happen that we are several epochs ahead.
            can_put = False
            while not can_put:
                with getlock():
                    current_put_epoch = data['current_put_epoch']
                if current_put_epoch == epoch:
                    can_put = True
                else:
                    time.sleep(1)

            #print('sketch: trying to put id',id,'epoch',epoch)
            # now we actually put the sketch in the queue.
            stream_queue.put((target_qf.detach(), sketch_id))
            #print('sketch: we put id', id, 'epoch', epoch)

            with getlock():
                # we put the data, now update the counting
                data['done_in_current_epoch'] += 1
                #print('sketch: after put, got lock. id', id, 'epoch', epoch, 'done in current epoch',data['done_in_current_epoch'])
                if (
                        data['done_in_current_epoch']
                        == data['num_sketches']):
                    # This item was the last of its epoch, we put the sentinel
                    #print('Sketch: sending the sentinel')
                    stream_queue.put(None)
                    data['done_in_current_epoch'] = 0
                    data['current_put_epoch'] += 1

        if 'die' in data:
            print('Sketch worker dying')
            break

        if data['pause']:
            if not pause_displayed:
                print('Sketch worker going to sleep')
                pause_displayed = True
            time.sleep(2)


def add_sketch_arguments(parser):
    parser.add_argument("--num_thetas",
                        help="Number of thetas per sketch.",
                        type=int,
                        default=2000)
    parser.add_argument("--num_quantiles",
                        help="Number of quantiles to compute",
                        type=int,
                        default=100)
    parser.add_argument("--num_examples",
                        help="Number of datapoints used per sketch. If "
                             "negative, take all of them.",
                        type=int,
                        default=3000)
    parser.add_argument("--num_sketches",
                        help="Number of sketches per epoch. If negative, "
                             "take an infinite number of them.",
                        type=int,
                        default=-1)
    parser.add_argument("--num_workers",
                        help="Number of workers for computing the sketches. "
                        "If not provided, use all CPUs",
                        type=int,
                        default=-1)
    return parser
