# Learning neural state-space models: do we need a state estimator?

This repository contains the Python code to reproduce the results of the paper [Learning neural state-space models: do we need a state estimator?](https://arxiv.org/abs/2206.12928) by Marco Forgione, Manas Mejari and Dario Piga.


# Folders:
* [torchid](torchis):  PyTorch implementation of neural state-space model. Adapted from the library https://github.com/forgi86/pytorch-ident developed by the first author.
* [examples](examples): experimentations and analyses of the paper: Wiener-Hammerstein circuit and pick-and-place machine.
* [doepy](doepy): library used for experiment planning. Adapted (with small bug fixes) from https://github.com/tirthajyoti/doepy
 <!--*  [doc](doc): paper latex files -->

Three [examples](examples) discussed in the paper are:

* [Wiener-Hammerstein Benchmark](examples/WH2009): A circuit with Wiener-Hammerstein behavior. Experimental dataset from http://www.nonlinearbenchmark.org
* [Pick & place machine](examples/EMPS): An
electronic component placement process in a pick-and-place
machine. Originally introduced in [A.Lj Juloski et al., Data-based
hybrid modelling of the component placement process in pick-and-place
machines, 2004.](https://www.sciencedirect.com/science/article/pii/S0967066104000784) Experimental dataset included in this repo.

# Software requirements:
Experiments were performed on a Python 3.9 conda environment with

 * numpy
 * scipy
 * matplotlib
 * pandas
 * pytorch (version 1.10)
 
These dependencies may be installed through the commands:

```
conda install numpy scipy pandas matplotlib
conda install pytorch torchvision -c pytorch
```

To run the software, please make sure that this repository's root folder is added to 
your PYTHONPATH.

# Citing

If you find this project useful, we encourage you to:

* Star this repository :star: 

* Cite the [paper](https://arxiv.org/abs/2206.12928) 
```
@article{forgione2022a,
  title={Learning neural state-space models: do we need a state estimator?},
  author={Forgione, M. and Mejari, M. and Piga, D.},
  journal={arXiv preprint arXiv:2206.12928},
  year={2022}
}
```
