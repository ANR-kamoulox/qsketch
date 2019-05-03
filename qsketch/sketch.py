# imports
import torch
from torch.utils.data import Dataset, DataLoader
from torchpercentile import Percentile
import atexit
import queue
from .datastream import DataStream
import multiprocessing.queues as queues
import torch.multiprocessing as mp
from contextlib import contextmanager
import collections
import time
import warnings


class ModulesDataset:
    """A dataset of torch modules.
    module_class: a torch.nn.Module
        this is the class name to turn into a dataset.
    device: string
        the device on which to generate the modules. Be careful that two
        ModulesDataset objects are equal if and only if they are using the same
        device.
    recycle: boolean
        whether or not to use recycling, which significantly accelerates
        accessing elements. If recycling is activated, a current element
        is kept in memory, which is reassigned new values on demand, instead
        of a new allocation.
    kwargs: dict
        parameters to provide to the class constructor when creating new
        elements
    """

    def __init__(self, module_class, device='cpu', recycle=True, **kwargs):
        self.module_class = module_class
        self.parameters = kwargs
        self.pos = 0
        self.current = None
        self.device = device
        self.recycle = recycle

    def __iter__(self):
        return self

    def __next__(self):
        self.pos += 1
        return self[self.pos]

    def recycle_module(self, module, index):
        """ default recycling method for modules.
        We need to make sure all recycling are performed with the same
        random sequence, which means it must be done on the same device.
        self.device is used here for this reason."""
        params = module.state_dict()
        torch.manual_seed(index)
        if (
                hasattr(module, 'reset_parameters')
                and callable(module.reset_parameters)):
            # we have a reset_parameters function. we need to call it after
            # being sure the module is on the right device
            module = module.to(self.device)
            module.reset_parameters()
        else:
            params = module.state_dict()
            for key in params:
                params[key] = torch.randn(
                    params[key].shape,
                    device=self.device)
            module.load_state_dict(params)

    def __getitem__(self, indexes):
        # get items, possibly using recycling
        if isinstance(indexes, int):
            if self.recycle:
                # only keeping track of a `current`
                # for singleton queries, not for list
                if self.current is None:
                    self.current = self.module_class(**self.parameters)
                result = self.current.to(self.device)
            else:
                result = self.module_class(**self.parameters).to(self.device)
            self.recycle_module(result, indexes)
            return result
        else:
            result = []
            for index in indexes:
                new_module = (self.module_class(**self.parameters)
                              .to(self.device))
                ModulesDataset.recycle_module(new_module, index)
                result += [new_module]
            return result

    def __call__(self, indexes):
        # get items, without recycling (new items necessarily)
        recycle_state = self.recycle
        self.recycle = False
        result = self[indexes]
        self.recycle = recycle_state
        return result


def to_iterator(data_source):
    if data_source is None:
        return None

    # try known stuff to make an iterator out of it
    if isinstance(data_source, DataStream):
        data_iterator = iter(data_source.queue.get, None)
    elif isinstance(data_source, queues.Queue):
        data_iterator = iter(data_source.get, None)
    elif isinstance(data_source, torch.Tensor):
        data_iterator = iter([[data_source, None]])
    elif isinstance(data_source, Dataset):
        data_iterator = DataLoader(data_source, batch_size=5000)
    elif isinstance(data_source, DataLoader):
        data_iterator = data_source
    else:
        if isinstance(data_source, collections.Iterable):
            # it's iterable, assuming it's ok
            data_iterator = data_source
        else:
            raise Exception('Sketcher: data_source type is not understood')
    return data_iterator


def sketch(modules, data, percentiles, num_examples=None):
    # check whether we want to sketch several modules or just one
    try:
        _ = iter(modules)
        iterable = True
    except TypeError as te:
        modules = [modules]
        iterable = False

    data_iterator = to_iterator(data)

    # for each module
    sketches = []
    for module in modules:
        # allocate the processed variable, to None
        processed = None

        pos = 0
        # compute the projections by a loop over the data. By default, use
        # all data except if num_examples is provided
        while (True if num_examples is None
               else pos < num_examples):

            # getting the next items
            try:
                (imgs, labels) = next(data_iterator)
            except StopIteration:
                if num_examples is not None:
                    warnings.warn(
                        'Number of datapoints not reaching %d, but'
                        'only %d. Using this and continuing.' % (
                               num_examples, pos))
                break

            # aggregate the projections. get only what's necessary if
            # num_examples is provided (batch is possibly too large)
            if num_examples is not None:
                n_imgs = min(len(imgs), num_examples - pos)
            else:
                n_imgs = len(imgs)

            # bring the module to the data device if it's not done already
            module = module.to(imgs.device)

            # apply the module after putting it on the data device
            computed = module(imgs[:n_imgs])
            # turn the output into a matrix
            computed = computed.view(n_imgs, -1)

            if processed is None:
                # we computed for the first time. Now we have several
                # options
                if num_examples is not None:
                    # We know the total number of elements. preallocate this
                    processed = torch.empty((num_examples,
                                             computed.shape[1]),
                                            device=computed.device)
                else:
                    # We don't know the total number of elements. Just
                    # allocate an empty tensor
                    processed = torch.Tensor().to(computed.device)
            if num_examples is not None:
                # if the computations are preallocated, store them at the right
                # place (faster)
                processed[pos:pos+n_imgs] = computed
            else:
                # concatenate the result:
                processed = torch.cat((processed, computed))
            # in any case, augment the position
            pos += n_imgs

        if processed is None:
            raise Exception('Did not get any data from data_source. '
                            'Cannot sketch.')

        # truncating in case we don't get enough. Possibly no-op
        processed = processed[:pos]

        # flatten the samples to a matrix for the quantiles
        processed = processed.view(processed.shape[0], -1)

        # compute the quantiles for these projections
        sketches += [Percentile()(processed, percentiles).float(), ]
    return sketches[0] if not iterable else sketches


class Sketcher:
    """Sketcher class: from a given DataStream object, constructs sketches
    for any provided function, which are the quantiles of the output of
    this function when applied on the dataset.

    A Sketcher is accessed through with a function as an index.

    Optionally, a stream can be started, when a Dataset of functions
    """

    def __init__(self,
                 data_source,
                 percentiles,
                 num_examples=None):
        """
            Create a new sketcher.
            data_source: either None, or a DataStream, a Queue, a Tensor,
                         a Dataset or a DataLoader
                If None, this sketcher cannot be streamed, because no
                default data is provided. Otherwise, this is where we get the
                data from. Each item should be a tuple of (X, y) values.
                The sentinel is None.
            percentiles: int
                the percentiles (between 0 and 100) to compute.
            num_examples: int or None
                the number of samples to use for computing each sketch.
                If None or if the data_source does not produce enough data,
                all data will be used for sketching
        """
        self.data_iterator = to_iterator(data_source)
        self.percentiles = percentiles
        self.num_examples = num_examples
        self.queue = None
        self.functions = None
        self.shared_data = None

    def __call__(self, modules, data=None, percentiles=None):
        # Use default if some parameters are not provided
        if data is None:
            data_iterator = self.data_iterator
            if data_iterator is None:
                raise Exception('Sketcher has no default data. Aborting.')
            num_examples = self.num_examples
        else:
            data_iterator = to_iterator(data)
            num_examples = None
        if percentiles is None:
            percentiles = self.percentiles

        return sketch(modules=modules,
                      data=data_iterator,
                      percentiles=percentiles,
                      num_examples=num_examples)

    def __getitem__(self, modules):
        # call the sketcher with default parameters
        return self(modules=modules,
                    data=None,
                    percentiles=None)

    def stream(self, modules, num_sketches, num_epochs,
               num_workers=-1, max_id=None):
        """starts a stream of sketches

        modules: ModulesDataset object
            the dataset of function to iterate upon
        num_sketches: int
            the number of sketches to compute per epoch: each sketch
            corresponds to one particular functions.
        num_epochs: int
            the number of epochs
        num_workers: int
            the number of workers to have. a negative value will lead to
            picking half of the local cores
        max_id: int or None
            the maximum index for modules.
        """
        # first stop if it was started before
        self.stop()

        # get the number of workers
        if num_workers < 0 or num_workers is None:
            # if not defined, take at least 1 and at most half of the cores
            num_workers = 1e7  # should be enough as a max number =)
            num_workers = max(1, min(num_workers,
                              int((mp.cpu_count()-1)/2)))

        print('SketchStream using ', num_workers, 'workers')
        # now create a queue with a maxsize corresponding to a few times
        # the number of workers
        self.queue = mp.Queue(maxsize=2*num_workers)
        manager = mp.Manager()

        # prepare some data for the synchronization of the workers
        self.shared_data = manager.dict()
        self.shared_data['num_epochs'] = num_epochs
        self.shared_data['max_id'] = (
            torch.iinfo(torch.int16).max if max_id is None
            else max_id)
        self.shared_data['pause'] = False
        self.shared_data['current_pick_epoch'] = 0
        self.shared_data['current_put_epoch'] = 0
        self.shared_data['current_sketch'] = 0
        self.shared_data['done_in_current_epoch'] = 0
        self.shared_data['num_sketches'] = (num_sketches if num_sketches > 0
                                            else -1)
        self.shared_data['sketch_list'] = torch.randint(
                low=0,
                high=self.shared_data['max_id'],
                size=(self.shared_data['num_sketches'],)).int()
        self.lock = mp.Lock()

        # prepare the workers
        processes = [mp.Process(target=sketch_worker,
                                kwargs={'sketcher': self,
                                        'modules': modules})
                     for n in range(num_workers)]
        #
        # atexit.register(partial(exit_handler, stream=self,
        #                         processes=processes))

        # go
        for p in processes:
            p.start()

        return self.queue

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


def exit_handler(stream, processes):
    print('Terminating sketchers...')
    if stream.shared_data is not None:
        stream.shared_data['die'] = True
    for p in processes:
        p.join()
    print('done')


def sketch_worker(sketcher, modules):
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
                # print('sketch: got lock, epoch %d and id %d' % (epoch, sketch_id))
                if epoch >= sketcher.shared_data['num_epochs']:
                    # the picked epoch is larger than the number of epochs.
                    # sketching is finished.
                    # print('epoch', epoch, 'greater than the number of epochs:',
                    #       sketcher.shared_data['num_epochs'], 'dying now.')
                    worker_dying = True
                else:
                    if id == sketcher.shared_data['num_sketches'] - 1:
                        # we reached the number of sketches per epoch.
                        # we let the other workers know and increment the
                        # pick epoch.
                        # print("Obtained id %d is last for this epoch. "
                        #       "Reseting the counter and incrementing current "
                        #       "epoch " % id)
                        sketcher.shared_data['current_sketch'] = 0
                        sketcher.shared_data['current_pick_epoch'] += 1
                        sketcher.shared_data['sketch_list'] = (
                            torch.randint(
                                low=0,
                                high=sketcher.shared_data['max_id'],
                                size=(sketcher.shared_data['num_sketches'],)
                                ).int())
                    else:
                        # we just increment the current sketch to pick for
                        # the next worker.
                        sketcher.shared_data['current_sketch'] += 1
            if worker_dying:
                # dying has been asked for. we'll just loop infinitely.
                # this is because there is apparently some issues raised when
                # we just kill the worker, in case some data in the queue
                # originated from him has not been taken out ?
                # print(
                #     id, epoch, 'Reached the desired amount of epochs. Dying.')
                while not sketcher.queue.empty():
                    pass
                # HERE WE SHOULD ACTUALLY DIE. However, there seems to be
                # an issue of the code not running (queue disappears) when
                # we do. Hence, I will infinitely loop here. BUG TO FIX
                while True:
                    time.sleep(10)
                return

            # now to the thing. We compute the sketch that has been asked for.
            # print('sketch: now trying to compute %d with id %d'
            #       % (id, sketch_id))
            module = modules[sketch_id]
            target_qf = sketcher[module]

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
                    if 'die' in sketcher.shared_data:
                        # print('Sketch worker dying wait for put')
                        while True:
                            # see line 388
                            time.sleep(10)
                        return
                    time.sleep(1)

            # print('sketch: trying to put id', id, 'epoch', epoch)
            # now we actually put the sketch in the queue.
            sketcher.queue.put((target_qf.detach(), sketch_id))
            # print('sketch: we put id', id, 'epoch', epoch)

            with getlock():
                # we put the data, now update the counting
                sketcher.shared_data['done_in_current_epoch'] += 1
                # print('sketch: after put, got lock. id', id, 'epoch', epoch, 'done in current epoch',sketcher.shared_data['done_in_current_epoch'])
                if (sketcher.shared_data['done_in_current_epoch']
                        == sketcher.shared_data['num_sketches']):
                    # This item was the last of its epoch, we put the sentinel
                    # print('Sketch: sending the sentinel')
                    sketcher.queue.put(None)
                    sketcher.shared_data['done_in_current_epoch'] = 0
                    sketcher.shared_data['current_put_epoch'] += 1

        if 'die' in sketcher.shared_data:
            # print('Sketch worker dying')
            while True:
                # see line 388
                time.sleep(10)
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
