import torch
from torchvision import datasets, transforms
import qsketch
import torch.multiprocessing as mp
import matplotlib.pylab as pl


class Multiply:
    def __init__(self, index):
        self.mul = index

    def __call__(self, x):
        return torch.sigmoid(torch.log(x ** (self.mul/1000)))


if __name__ == "__main__":
    # this is important to do this at the very beginning of the program
    mp.set_start_method('spawn', force=True)

    # prepare the torch device (cuda or cpu ?)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # load the data
    data = datasets.MNIST('~/data/MNIST',
                          transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])
                         )

    # Launch the data stream
    data_stream = qsketch.DataStream(data)
    data_stream.stream()

    # prepare the sketcher
    sketcher = qsketch.Sketcher(data_source=data_stream,
                                num_quantiles=10,
                                num_examples=1000)

    result = sketcher[lambda x: torch.mean((x > 10).float().view(x.shape[0], -1), dim=1)]

    import ipdb; ipdb.set_trace()
    # prepare the functions
    functions = qsketch.FunctionsDataset(
        qsketch.Factory(Multiply))

    sketch_queue = sketcher.stream(functions=functions,
                                   num_sketches=10,
                                   num_epochs=1,
                                   num_workers=2,
                                   max_id=10)

    for (quantiles, projector_id) in iter(sketch_queue.get, None):
        # Getting quantiles and plotting them
        #import ipdb; ipdb.set_trace()
        pl.plot(quantiles.numpy().T)
        pl.xlabel('quantile')
        pl.ylabel('value')
        pl.show()
