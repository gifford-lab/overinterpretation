# Overinterpretation

This repository contains the code for the paper:

[Overinterpretation reveals image classification model pathologies](https://arxiv.org/abs/2003.08907)
<br />
Authors: Brandon Carter, Siddhartha Jain, Jonas Mueller, David Gifford


## Introduction

Image classifiers are typically scored on their test set accuracy, but high accuracy can mask a subtle type of model failure. We find that high scoring convolutional neural networks (CNNs) on popular benchmarks exhibit troubling pathologies that allow them to display high accuracy even in the absence of semantically salient features. When a model provides a high-confidence decision without salient supporting input features, we say the classifier has overinterpreted its input, finding too much class-evidence in patterns that appear nonsensical to humans. Here, we demonstrate that neural networks trained on CIFAR-10 and ImageNet suffer from overinterpretation, and we find models on CIFAR-10 make confident predictions even when 95% of input images are masked and humans cannot discern salient features in the remaining pixel-subsets. Although these patterns portend potential model fragility in real-world deployment, they are in fact valid statistical patterns of the benchmark that alone suffice to attain high test accuracy. Unlike adversarial examples, overinterpretation relies upon unmodified image pixels.  We find ensembling and input dropout can each help mitigate overinterpretation.


## Usage

### Dependencies

Python 3.7<br>
PyTorch v1.5.0<br>
torchvision v0.5.0<br>

Full requirements in [requirements.txt](requirements.txt).


### Overview

The overinterpretation pipeline can be understood as:
1. Train models on full images ([train.py](train.py)).
2. Run backward selection for all training and test images ([run_sis_on_cifar.py](run_sis_on_cifar.py)).
3. Train new models on pixel-subsets of images and mask the remaining pixels ([train.py](train.py)).
4. Evaluate new models and compare accuracy to original models.

The relevant scripts for running this pipeline are [train.py](train.py) and [run_sis_on_cifar.py](run_sis_on_cifar.py).
Each script contains usage examples in the docstring.
[train.py](train.py) supports training models on full image data as well as pixel-subsets only (specified via command line arguments, usage examples in docstring).

Note that for CIFAR-10, when training models on pixel-subsets only, we keep 5% of pixels and mask the remaining 95% with zeros.


## Citation

If you use our methods or code, please cite:

```bib
@article{carter2020overinterpretation,
  title={Overinterpretation reveals image classification model pathologies},
  author={Carter, Brandon and Jain, Siddhartha and Mueller, Jonas and Gifford, David},
  journal={arXiv preprint arXiv:2003.08907},
  year={2020}
}
