import torch
from torchvision import datasets, transforms
import qsketch
from torchvision.utils import make_grid
import torch.multiprocessing as mp
import numpy as np
import os
import nets
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.pad_inches'] = 0


if __name__ == "__main__":
    # this is important to do this at the very beginning of the program
    mp.set_start_method('spawn', force=True)
    num_samples = 5000
    plot_path = '~/swmin_samples_MNIST'

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
                        device='cpu',
                        input_shape=data[0][0].shape,
                        bottleneck_size=100)
    # import time
    # start = time.time(); test = randomcoders[0]; print(time.time()-start)
    # print('0', test.fc1.weight[:5])
    # start = time.time(); test = randomcoders[10]; print(time.time()-start)
    # print('10', test.fc1.weight[:5])
    # start = time.time(); test = randomcoders[0]; print(time.time()-start)
    # print('0', test.fc1.weight[:5])
    # start = time.time(); test = randomcoders(10); print(time.time()-start)
    # print('10 call', test.fc1.weight[:5])
    #
    # import ipdb; ipdb.set_trace()
    # prepare the sketcher
    sketcher = qsketch.Sketcher(data_source=data_stream,
                                percentiles=torch.linspace(0, 100, 300),
                                num_examples=5000)

    # for a in range(10):
    #     print(sketcher[randomcoders[0]])
    #     import ipdb; ipdb.set_trace()
    sketcher.stream(modules=randomcoders,
                    num_sketches=10,
                    num_epochs=1000,
                    num_workers=2)

    particles = torch.randn((num_samples,) + data[0][0].shape)
    particles = torch.nn.Parameter(particles)
    optimizer = torch.optim.Adam((particles,), lr=1e-3)
    criterion = torch.nn.MSELoss()

    plot_path = os.path.expanduser(plot_path)
    for epoch in range(1000):
        train_loss = 0
        for (target_quantiles, projector_id) in iter(sketcher.queue.get, None):
            projector = randomcoders[projector_id]
            optimizer.zero_grad()
            quantiles = sketcher(projector, particles)
            loss = criterion(target_quantiles, quantiles)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        if epoch % 10 == 1:
            # now do some plotting of current particles
            fig = plt.figure(1, figsize=(5, 8), dpi=200)
            fig.clf()
            axes = plt.gca()
            axes.xaxis.set_major_formatter(plt.NullFormatter())
            axes.yaxis.set_major_formatter(plt.NullFormatter())
            axes.tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False)
            # it's image data
            pic = make_grid(
                particles[:104].detach().clone(),
                nrow=8,
                padding=2, normalize=True, scale_each=True
                )
            pic_npy = pic.numpy()
            axes.imshow(np.transpose(pic_npy, (1, 2, 0)),
                        interpolation='nearest',
                        aspect='auto')
            #plt.axis('off')
            plt.title('Generated samples, iteration %04d' % epoch)
            fig.tight_layout()
            if not os.path.exists(plot_path):
                os.mkdir(plot_path)
            fig.savefig(os.path.join(plot_path, 'samples_%04d' % epoch),
                        bbox_inches='tight', pad_inches=0)
        print('epoch %d, loss: %0.4f' % (epoch, train_loss))
