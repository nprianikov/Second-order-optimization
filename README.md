# Second-order optimization for Image Classification

This repository contains the code used for my BSc thesis, "Second-order optimization for Image Classification". The purpose of this project was to explore the convergence behaviour and applicability of second-order optimization methods for image classification tasks, and to compare 
their performance with first-order stochastic gradient.

## Installation
Follow the steps:
1. Clone the repository to your local machine
2. Install the required dependancies:
```
pip install -r requirements.txt
```
3. (Optionally) Login to the [Weights and Biases](https://wandb.ai/home):
```
wandb login YOUR_API_KEY
```

## Usage
You can run the experiments with *train_models.py* by sepcifying relevant arguments. For examples:

```
python train_models.py --epochs 10 --lr 0.001 --batch_size 32 --dataset=mnist --optimizer=SGD --model=SmallCNN

```
If *--loop* is set to 1, experiments will run over all combinations of datasets, optimizers and models.

You can use 
```
train_models.py --help
```
to view detailed description of arguments.

<!-- 
Add later

## Results

## Known Issues

## Contact Information

-->

## Sources and acknowledgements

### External Optimizers used
* HesianFree optimizer. Adapted from https://github.com/fmeirinhos/pytorch-hessianfree
* Stochastic L-BFGS method. https://github.com/hjmshi/PyTorch-LBFGS
* Kronecker-factored Quasi-Newton. Extended the implementation from https://github.com/renyiryry/kbfgs_neurips2020_public to convolutional layers.

### Additional software
* https://github.com/amirgholami/PyHessian
