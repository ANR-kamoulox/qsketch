# imports
import numpy as np
import torch
from torchpercentile import Percentile
import atexit
import queue
import torch.multiprocessing as mp
from contextlib import contextmanager
from functools import partial
import time


class FunctionsDataset:
    """A dataset of functions, each of which is constructed by a factory
    provided as a parameter.
    This factory should accept an `index` parameter, corresponding to the
    index of the function to construct.
    If a `refactory` is provided, it is called to update the current element,
    instead of creating a new one. This may be useful to save allocation time.

    In all cases, the factories should *deterministically* generate the same
    function given the same index.
    """

    def __init__(self, factory, refactory=None):
        self.factory = factory
        self.refactory = refactory
        self.pos = 0
        self.current = None

    def __iter__(self):
        return self

    def __next__(self):
        self.pos += 1
        if self.refactory is not None and self.current is not None:
            self.current = self.refactory(self.current, self.pos)
        else:
            self.current = self.factory(self.pos)
        return self.current

    def __getitem__(self, indexes):
        if isinstance(indexes, int):
            # only keeping a `current` for singleton queries, not for list
            if self.refactory is not None and self.current is not None:
                self.current = self.refactory(self.current, indexes)
            else:
                self.current = self.factory(indexes)
            return self.current

        return [self.factory(index=id) for id in indexes]


class Sketcher:
    """Sketcher class: from a given DataStream object, constructs sketches
    for any provided function, which are the quantiles of the output of
    this function when applied on the dataset.

    A Sketcher is accessed through with a function as an index.

    Optionally, a stream can be started, when a Dataset of functions
    """

    def __init__(self,
                 data_source,
                 num_quantiles,
                 num_examples):
        """
            Create a new sketcher.
            data_source: Queue object
                this is where we get the data from. Each item from this queue
                should be a tuple of (X, y) values. The sentinel is None.
            num_quantiles: int
                the number of quantiles to compute the sketch with. They will
                be picked as regularly spaced between 0 and 100
            num_examples: int
                the number of samples to use for computing each sketch.
                The data_source is assumed to produce data endlessly.
        """
        self.data_source = data_source
        self.num_quantiles = num_quantiles
        self.num_examples = num_examples
        self.queue = None
        self.shared_data = None

    def __getitem__(self, functions):
        try:
            _ = iter(functions)
            iterable = True
        except TypeError as te:
            functions = [functions]
            iterable = False

        # get the projector
        sketches = []
        percentiles = torch.linspace(0, 100, self.num_quantiles)
        for function in functions:
            # allocate the processed variable, to None
            processed = None

            # compute the projections by a loop over the data
            pos = 0
            while pos < self.num_examples:
                (imgs, labels) = self.data_source.get()

                # aggregate the projections
                n_imgs = min(len(imgs), self.num_examples - pos)
                computed = function(imgs[:n_imgs])
                computed = computed.view(n_imgs, -1)
                if processed is None:
                    # now that we know the dimension, compute the object
                    processed = torch.empty((self.num_examples,
                                             computed.shape[1]),
                                            device=computed.device)
                processed[pos:pos+n_imgs] = computed
                pos += n_imgs

            # flatten the samples to a matrix for the quantiles
            processed = processed.view(self.num_examples, -1)
            # compute the quantiles for these projections
            sketches += [
                (Percentile()(processed, percentiles).float(),
                 id)]
        return sketches[0] if not iterable else sketches

    def stream(self, functions, num_sketches, num_epochs, num_workers=-1):
        """starts a stream of sketches

        functions: FunctionsDataset object
            the dataset of function to iterate upon
        num_sketches: int
            the number of sketches to compute per epoch: each sketch
            corresponds to one particular functions.
        num_epochs: int
            the number of epochs
        num_workers: int
            the number of workers to have. a negative value will lead to
            picking half of the local cores
        """
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

        # prepare some data for the synchronization of the workers
        self.shared_data = self.manager.dict()
        self.shared_data['num_epochs'] = num_epochs
        self.shared_data['pause'] = False
        self.shared_data['current_pick_epoch'] = 0
        self.shared_data['current_put_epoch'] = 0
        self.shared_data['current_sketch'] = 0
        self.shared_data['done_in_current_epoch'] = 0
        self.shared_data['num_sketches'] = (num_sketches if num_sketches > 0
                                            else np.inf)
        self.shared_data['sketch_list'] = np.random.randint(
                np.iinfo(np.int16).max,
                size=self.shared_data['num_sketches']).astype(int)
        self.lock = mp.Lock()

        # prepare the workers
        self.processes = [mp.Process(target=sketch_worker,
                                     kwargs={'sketcher': self,
                                             'functions': functions})
                          for n in range(num_workers)]

        atexit.register(partial(exit_handler, stream=self))

        # go
        for p in self.processes:
            p.start()

    def pause(self):
        if self.shared_data is None:
            return
        self.shared_data['pause'] = True

    def restart(self):
        self.shared_data['counter'] = 0
        self.resume()

    def resume(self):
        if self.shared_data is not None:
            self.shared_data['pause'] = False

    def stop(self):
        if self.shared_data is not None:
            self.shared_data['die'] = True


def exit_handler(stream):
    print('Terminating sketchers...')
    if stream.shared_data is not None:
        stream.shared_data['die'] = True
    for p in stream.processes:
        p.join()
    print('done')


def sketch_worker(sketcher, functions):
    """ Actual worker for the sketch stream.
    Will get sketch ids, get data from the data queue and put sketches in the
    stream queue"""

    @contextmanager
    def getlock():
        # get the lock of the sketcher to manipulate the sketch.shared_data
        result = sketcher.lock.acquire(block=True)
        yield result
        if result:
            sketcher.lock.release()

    pause_displayed = False
    while True:
        # not dying, unless we see that later
        worker_dying = False

        if not sketcher.shared_data['pause']:
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
                id = sketcher.shared_data['current_sketch']
                sketch_id = sketcher.shared_data['sketch_list'][id].item()
                epoch = sketcher.shared_data['current_pick_epoch']
                # print('sketch: got lock, epoch %d and id %d' % (epoch, id))
                if epoch >= sketcher.shared_data['num_epochs']:
                    # the picked epoch is larger than the number of epochs.
                    # sketching is finished.
                    #print('epoch', epoch, 'greater than the number of epochs:',
                    #      stream.num_epochs, 'dying now.')
                    worker_dying = True
                else:
                    if id == sketcher.shared_data['num_sketches'] - 1:
                        # we reached the number of sketches per epoch.
                        # we let the other workers know and increment the
                        # pick epoch.
                        #print("Obtained id %d is last for this epoch. "
                        #      "Reseting the counter and incrementing current "
                        #      "epoch " % id)
                        sketcher.shared_data['current_sketch'] = 0
                        sketcher.shared_data['current_pick_epoch'] += 1
                        sketcher.shared_data['sketch_list'] = (
                            np.random.randint(
                                np.iinfo(np.int16).max,
                                size=sketcher.shared_data['num_sketches']
                                ).astype(int))
                    else:
                        # we just increment the current sketch to pick for
                        # the next worker.
                        sketcher.shared_data['current_sketch'] += 1
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
                    current_put_epoch = (
                        sketcher.shared_data['current_put_epoch'])
                if current_put_epoch == epoch:
                    can_put = True
                else:
                    time.sleep(1)

            #print('sketch: trying to put id',id,'epoch',epoch)
            # now we actually put the sketch in the queue.
            sketcher.queue.put((target_qf.detach(), sketch_id))
            #print('sketch: we put id', id, 'epoch', epoch)

            with getlock():
                # we put the data, now update the counting
                sketcher.shared_data['done_in_current_epoch'] += 1
                #print('sketch: after put, got lock. id', id, 'epoch', epoch, 'done in current epoch',sketcher.shared_data['done_in_current_epoch'])
                if (sketcher.shared_data['done_in_current_epoch']
                        == sketcher.shared_data['num_sketches']):
                    # This item was the last of its epoch, we put the sentinel
                    #print('Sketch: sending the sentinel')
                    sketcher.queue.put(None)
                    sketcher.shared_data['done_in_current_epoch'] = 0
                    sketcher.shared_data['current_put_epoch'] += 1

        if 'die' in sketcher.shared_data:
            print('Sketch worker dying')
            break

        if sketcher.shared_data['pause']:
            if not pause_displayed:
                print('Sketch worker going to sleep')
                pause_displayed = True
            time.sleep(2)


def add_sketch_arguments(parser):
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
