import torch
from torchvision import datasets, transforms
import qsketch
from torchvision.utils import save_image
import torch.multiprocessing as mp
import os


if __name__ == "__main__":
    # this is important to do this at the very beginning of the program
    mp.set_start_method('spawn', force=True)

    # some hard-wired parameters
    num_samples = 5000
    num_iters = 100000

    plot_path = '~/swmin_samples_MNIST'
    plot_path = os.path.expanduser(plot_path)

    # load the data
    data = datasets.MNIST('~/data/',
                          transform=transforms.ToTensor())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # initialize particles and the optimizer
    particles = torch.randn((num_samples,) + data[0][0].shape, device=device)
    particles = torch.nn.Parameter(particles)

    optimizer = torch.optim.Adam([particles, ], lr=1e-1)

    # create a sliced wasserstein loss object
    sw = qsketch.gsw.GSW(data, device=device,
                         num_workers_data=20,
                         num_sketchers=2)

    # now doing the training, optimizing the particles
    for epoch in range(num_iters):
        train_loss = 0

        optimizer.zero_grad()
        loss = sw(particles)
        loss.backward()
        optimizer.step()
        print('iteration %d, loss: %0.6f' % (epoch, loss.item()))

        if epoch % 5 == 1:
            save_image(particles[:104],
                       filename=os.path.join(plot_path,
                                             'samples_%04d.png' % epoch),
                       nrow=8, padding=2, normalize=True, scale_each=True)
