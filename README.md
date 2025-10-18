# Environment Inference for Learning Generalizable Dynamical Systems (Neurips 2025) 

This repository contains a PyTorch implementation of DynaInfer, as described in the paper:

Shixuan Liu, Yue He, Haotian Wang, Wenjing Yang, Yunfei Wang, Peng Cui, Zhong Liu. [Environment Inference for Learning Generalizable Dynamical Systems]([https://ieeexplore.ieee.org/abstract/document/10613499/](https://openreview.net/forum?id=2M5dTDdGxl)). Neurips 2025. 

## Overview
![Overview](https://github.com/shixuanliu-andy/DynaInfer/blob/main/figure/fig_intro.pdf)
Data-driven methods offer efficient and robust solutions for analyzing complex dynamical systems but rely on the assumption of I.I.D. data, driving the development of generalization techniques for handling environmental differences. These techniques, however, are limited by their dependence on environment labels, which are often unavailable during training due to data acquisition challenges, privacy concerns, and environmental variability, particularly in large public datasets and privacy-sensitive domains. In response, we propose DynaInfer, a novel method that infers environment specifications by analyzing prediction errors from fixed neural networks within each training round, enabling environment assignments directly from data. We prove our algorithm effectively solves the alternating optimization problem in unlabeled scenarios and validate it through extensive experiments across diverse dynamical systems. Results show that DynaInfer outperforms existing environment assignment techniques, converges rapidly to true labels, and even achieves superior performance when environment labels are available.

## Results
