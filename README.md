# qsketch (_quantiles sketch_)
This package computes quantiles of the output of arbitrary pytorch Modules called _projectors_ over arbitrary datasets.

This package features multiprocessing capabilities which allow training deep models in an effective manner.

It is intended to be useful for research about training with sliced Wasserstein distances.


## installation
qsketch depends on torchpercentile, please check [https://github.com/aliutkus/torchpercentile](https://github.com/aliutkus/torchpercentile) for installation instructions.

Type `pip install -e .` in the root directory

## Usage

Checkout the example `test.py` for a simple example of using torch to learn a generative model
on MNIST that minimizes the Sliced Wasserstein distance.

The module comprises some key ingredients which are:
* `GSW` Generalized Sliced Wasserstein objects are created with a dataset as an argument, the specifications of the projections to use (either linear or arbitrary pytorch Modules), and will automatically compute the generalized sliced Wasserstein distance between a batch and this dataset. See the example.
* `Sketcher` objects encapsulate the application of some Module on the data, and the computation of quantiles of the corresponding outputs. A Sketcher object is initialized with a data source, which can notably be a DataStream (see below), and is directly accessed with the module of which output you want to compute the quantiles on the data:
  > quantiles = sketcher[test_module]

   It is also possible to start a stream of sketching through the `stream` method, in which case the sketcher will start sketching processes that will fill in a queue, that can be used for training.
* `DataStream` objects take a Dataset and continuously fill a queue from which one can get content. This is useful for multiprocessing and asynchronous training.
* `ModulesDataset` is a class that takes some torch Module classname as a parameter and creates a Dataset out of it. The idea is that each sample of a ModulesDataset is an instance of the provided class, initialized with a specific random seed. This is usefull for iterating over random projections, or more generally random pytorch Modules to be applied on the data.
