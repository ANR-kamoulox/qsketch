import torch
from torchvision import datasets, transforms
import qsketch
from torchvision.utils import make_grid
import torch.multiprocessing as mp
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.pad_inches'] = 0


# A class for random linear projections
class LinearProjector(torch.nn.Linear):
    def __init__(self, in_features, out_features):
        self.dim_in = torch.prod(torch.tensor(in_features))
        self.dim_out = out_features
        super(LinearProjector, self).__init__(
            in_features=self.dim_in,
            out_features=self.dim_out,
            bias=False)
        self.reset_parameters()

    def forward(self, input):
        return super(LinearProjector, self).forward(
            input.view(input.shape[0], -1))

    def reset_parameters(self):
        super(LinearProjector, self).reset_parameters()
        new_weight = self.weight

        # make sure each projector is normalized
        self.weight = torch.nn.Parameter(
            new_weight/torch.norm(new_weight, dim=1, keepdim=True))


if __name__ == "__main__":
    # this is important to do this at the very beginning of the program
    mp.set_start_method('spawn', force=True)

    # some hard-wired parameters
    num_samples = 5000
    num_epoch = 10000

    plot_path = '~/swmin_samples_MNIST'
    plot_path = os.path.expanduser(plot_path)

    # prepare the torch device (cuda or cpu ?)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # load the data
    data = datasets.MNIST('~/data/MNIST',
                          transform=transforms.ToTensor())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create a data stream and launch it
    data_stream = qsketch.DataStream(data, device=device)
    data_stream.stream()

    # prepare the random networks dataset
    randomcoders = qsketch.ModulesDataset(
                        LinearProjector,
                        device=device,
                        in_features=data[0][0].shape,
                        out_features=3000)

    # prepare the sketcher
    sketcher = qsketch.Sketcher(data_source=data_stream,
                                percentiles=torch.linspace(0, 100, 100),
                                num_examples=5000)

    # start sketching the dataset
    sketcher.stream(modules=randomcoders,
                    num_sketches=10,
                    num_epochs=num_epoch,
                    num_workers=2)

    # initialize particles and the optimizer
    particles = torch.randn((num_samples,) + data[0][0].shape, device=device)
    particles = torch.nn.Parameter(particles)
    optimizer = torch.optim.Adam((particles,), lr=1e-1)
    criterion = torch.nn.MSELoss()

    # now doing the training, optimizing the particles
    for epoch in range(num_epoch):
        train_loss = 0
        for (target_quantiles, projector_id) in iter(sketcher.queue.get, None):
            projector = randomcoders[projector_id]
            optimizer.zero_grad()
            quantiles = sketcher(projector, particles)
            loss = criterion(target_quantiles, quantiles)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print('epoch %d, loss: %0.6f' % (epoch, train_loss))

        if epoch % 5 == 1:
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
            pic_npy = pic.cpu().numpy()
            axes.imshow(np.transpose(pic_npy, (1, 2, 0)),
                        interpolation='nearest',
                        aspect='auto')
            plt.title('Generated samples, iteration %04d' % epoch)
            fig.tight_layout()
            if not os.path.exists(plot_path):
                os.mkdir(plot_path)
            fig.savefig(os.path.join(plot_path, 'samples_%04d' % epoch),
                        bbox_inches='tight', pad_inches=0)
