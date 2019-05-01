import torch
from torchvision import datasets, transforms
import qsketch
import torch.multiprocessing as mp
import matplotlib.pylab as pl
import nets


if __name__ == "__main__":
    # this is important to do this at the very beginning of the program
    mp.set_start_method('spawn', force=True)

    # prepare the torch device (cuda or cpu ?)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # load the data
    data = datasets.MNIST('~/data/MNIST',
                          transform=transforms.ToTensor())

    # Create a data stream and launch it
    data_stream = qsketch.DataStream(data)
    data_stream.stream()

    # prepare the random networks dataset
    randomcoders = qsketch.ModulesDataset(
                        nets.DenseEncoder,
                        device='cuda',
                        input_shape=data[0][0].shape,
                        bottleneck_size=100)
    import time
    start = time.time(); test = randomcoders[0]; print(time.time()-start)
    print('0', test.fc1.weight[:5])
    start = time.time(); test = randomcoders[10]; print(time.time()-start)
    print('10', test.fc1.weight[:5])
    start = time.time(); test = randomcoders[0]; print(time.time()-start)
    print('0', test.fc1.weight[:5])
    start = time.time(); test = randomcoders(10); print(time.time()-start)
    print('10 call', test.fc1.weight[:5])

    import ipdb; ipdb.set_trace()
    # prepare the sketcher
    sketcher = qsketch.Sketcher(data_source=data_stream,
                                percentiles=torch.linspace(0, 100, 300),
                                num_examples=1000)

    sketcher.stream(modules=randomcoders,
                    num_sketches=10,
                    num_epochs=1,
                    num_workers=2)

    for (quantiles, projector_id) in iter(sketcher.queue.get, None):
        # Getting quantiles and plotting them
        import ipdb; ipdb.set_trace()
        pl.plot(quantiles.numpy().T)
        pl.xlabel('quantile')
        pl.ylabel('value')
        pl.show()
