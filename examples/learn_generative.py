import torch
from torch import nn
from torchvision import datasets, transforms
import qsketch
from torchvision.utils import save_image
import torch.multiprocessing as mp
import os
from pathlib import Path


class ConvDecoder(nn.Module):
    def __init__(self, input_shape, bottleneck_size=64):
        super(ConvDecoder, self).__init__()
        self.input_shape = input_shape
        d = input_shape[-1]

        self.fc4 = nn.Linear(bottleneck_size, int(d/2 * d/2 * d))
        self.deconv1 = nn.ConvTranspose2d(d, d,
                                          kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(d, d,
                                          kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(d, d,
                                          kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.Conv2d(d, self.input_shape[0],
                               kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        d = self.input_shape[-1]
        out = torch.relu(self.fc4(x))
        out = out.view(-1, d, int(d/2), int(d/2))
        out = torch.relu(self.deconv1(out))
        out = torch.relu(self.deconv2(out))
        out = torch.relu(self.deconv3(out))
        return torch.sigmoid(self.conv5(out))


if __name__ == "__main__":
    # this is important to do this at the very beginning of the program
    mp.set_start_method('spawn', force=True)

    # some hard-wired parameters
    input_dim = 64
    num_samples = 5000
    num_iters = 100000

    #SW parameters
    num_projections = 5000
    num_percentiles = 300

    plot_path = Path(os.path.expanduser('~/generative_SWcost'))
    plot_path.mkdir(exist_ok=True, parents=True)

    # load the data
    data = datasets.MNIST('~/data/',
                          transform=transforms.ToTensor())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ConvDecoder(input_shape=data[0][0].shape,
                        bottleneck_size=input_dim)
    optimizer = torch.optim.Adam(model.parameters())

    # create a sliced wasserstein loss object
    sw = qsketch.gsw.GSW(data, device=device,
                         num_workers_data=10,
                         num_examples=num_samples,
                         num_percentiles=num_percentiles,
                         projectors=num_projections,
                         num_sketchers=2)

    # now doing the training, optimizing the particles
    train_loss = 0
    for epoch in range(num_iters):
        # initialize particles and the optimizer
        samples = torch.randn((num_samples, input_dim),
                              device=device)
        particles = model(samples)
        optimizer.zero_grad()
        loss = sw(particles)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 1:
            train_loss += loss.item()
            print('iteration %d, loss: %0.6f' % (epoch, loss.item()))
            save_image(particles[:104],
                       filename=(plot_path /
                                 Path('samples_%04d.png' % epoch
                                      ).with_suffix('.png')),
                       nrow=8, padding=2, normalize=True, scale_each=True)
            train_loss = 0
