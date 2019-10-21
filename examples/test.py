import torch
from torch import nn
from torchvision import datasets, transforms
import qsketch
from torchvision.utils import save_image, make_grid
import torch.multiprocessing as mp
from pathlib import Path
from tqdm import trange


class ConvGenerator(nn.Module):
    """
    A simple convolutive network for generating square images given
    some input features.
    """

    def __init__(self, data_shape, bottleneck_size=64):
        """
        Creates a convolutive generator that produces images of desired
        shape given flat inputs of specified shape.

        Parameters:
        -----------
        data_shape: tuple of int (nb_channels, dimension, dimension)
            should be a tuple of int of length 3, where the two last items
            are equal.
        bottleneck_size: int [scalar]
            specifies the dimensionality of the input
        """
        super(ConvGenerator, self).__init__()
        self.data_shape = data_shape
        d = data_shape[-1]

        self.fc4 = nn.Linear(bottleneck_size, int(d/2 * d/2 * d))
        self.deconv1 = nn.ConvTranspose2d(d, d,
                                          kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(d, d,
                                          kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(d, d,
                                          kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.Conv2d(d, self.data_shape[0],
                               kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        d = self.data_shape[-1]
        out = torch.relu(self.fc4(x))
        out = out.view(-1, d, int(d/2), int(d/2))
        out = torch.relu(self.deconv1(out))
        out = torch.relu(self.deconv2(out))
        out = torch.relu(self.deconv3(out))
        return torch.sigmoid(self.conv5(out))


if __name__ == "__main__":
    """Illustrates the use of the qsketch package to learn a vanilla generative
    network using the sliced Wasserstein distance.

    The approach is the one described in:
    @inproceedings{deshpande2018generative,
        title={Generative modeling using the sliced wasserstein distance},
        author={Deshpande, Ishan and Zhang, Ziyu and Schwing, Alexander G},
        booktitle={Proceedings of the IEEE Conference on Computer Vision and
                   Pattern Recognition},
        pages={3483--3491},
        year={2018}
    }

    except that only the real true Sliced Wasserstein distance is used, i.e.
    with random projections, excluding the trick presented in section 3.2
    of this aforementioned paper.
    """
    # this is important to do this at the very beginning of the program
    mp.set_start_method('spawn', force=True)

    # whether to output to disk (~/test_generative_SWcost folder) or just plot
    disk_save = False

    # refresh plot every... iterations
    plot_rate = 1000

    # input dimension to the generative model
    input_dim = 64

    # number of samples to use per update
    num_samples = 512

    # number of iterations
    num_iters = 100000

    # Sliced-Wasserstein parameters
    num_projections = 50
    num_sw_batches = 1
    num_percentiles = num_samples
    refresh_rate = 50  # we change the target projections every few iterations

    #
    if disk_save:
        plot_path = Path('~/testgenerative_SWcost').expanduser()
        plot_path.mkdir(exist_ok=True, parents=True)
    else:
        import matplotlib.pyplot as plt
        plt.ion()
        plt.show()

    # load the data
    data = datasets.MNIST('~/data/',
                          transform=transforms.ToTensor())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create the model
    model = ConvGenerator(data_shape=data[0][0].shape,
                          bottleneck_size=input_dim).to(device)

    # create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # create a sliced wasserstein loss object
    if torch.cuda.is_available():
        sketcher_device = 'cuda'
        num_sketchers = 2
    else:
        sketcher_device = 'cpu'
        num_sketchers = max(1, int(mp.cpu_count()/2))
    sw = qsketch.gsw.GSW(dataset=data,
                         num_percentiles=num_percentiles,
                         num_examples=num_samples,
                         projectors=num_projections,
                         batchsize=num_sw_batches,
                         manual_refresh=True,
                         device=sketcher_device,
                         num_workers_data=max(1, int(mp.cpu_count()/2)),
                         num_sketchers=num_sketchers)

    # monitoring the average training loss every few iterations
    train_loss = 0

    # now doing the training, optimizing the model
    bar = trange(num_iters, desc='Training with SW.', leave=True)
    for iteration in bar:

        # sample new input
        input = torch.randn((num_samples, input_dim),
                            device=device)
        samples = model(input)
        optimizer.zero_grad()
        loss = sw(samples)
        loss.backward()
        optimizer.step()

        # now display every few epochs
        if not iteration % refresh_rate:
            sw.refresh()
            bar.set_description('Training with SW. loss=%f' % loss.item())
            bar.refresh()

        if iteration and not iteration % plot_rate:
            if not disk_save:
                grid_img = make_grid(samples[:104].detach(), nrow=8, padding=2,
                                     normalize=True, scale_each=True)
                plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy(),
                           aspect='auto')
                plt.gca().get_xaxis().set_visible(False)
                plt.gca().get_yaxis().set_visible(False)
                plt.title('Particles generated after %d iterations '
                          % iteration)
                plt.draw()
                plt.pause(0.001)
            else:
                save_image(samples[:104],
                           filename=(plot_path /
                                     Path('samples_%04d.png' % iteration
                                          ).with_suffix('.png')),
                           nrow=8, padding=2, normalize=True, scale_each=True)
            train_loss = 0
