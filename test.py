import torch
import qsketch
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pylab as pl


if __name__ == "__main__":
    # this is important to do this at the very beginning of the program
    mp.set_start_method('spawn', force=True)

    # prepare the torch device (cuda or cpu ?)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # load the data
    train_data = qsketch.load_image_dataset(
        dataset='MNIST',
        data_dir='~/data/',
        img_size=32
    )

    data_shape = train_data[0][0].shape

    # Launch the data stream
    data_stream = qsketch.DataStream(train_data)
    data_stream.start()

    # prepare the projectors
    projectors = qsketch.Projectors(
        num_thetas=10,
        data_shape=data_shape,
        projector_class=qsketch.SparseLinearProjector)

    # start sketching
    quantiles_stream = qsketch.SketchStream()
    quantiles_stream.start(num_workers=-1,
                           num_epochs=10,
                           num_sketches=500,
                           data_stream=data_stream,
                           projectors=projectors,
                           num_quantiles=100,
                           num_examples=1000)

    for (quantiles, projector_id) in iter(quantiles_stream.queue.get, None):
        # Getting quantiles and plotting them
        pl.plot(quantiles.numpy())
        pl.xlabel('quantile')
        pl.ylabel('value')
        pl.show()
