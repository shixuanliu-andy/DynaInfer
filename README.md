# Environment Inference for Learning Generalizable Dynamical System (NeurIPS 2025 Spotlight) 

This repository contains a PyTorch implementation of DynaInfer, as described in the paper:

Shixuan Liu, Yue He, Haotian Wang, Wenjing Yang, Yunfei Wang, Peng Cui, Zhong Liu. [Environment Inference for Learning Generalizable Dynamical System]([https://ieeexplore.ieee.org/abstract/document/10613499/](https://openreview.net/forum?id=2M5dTDdGxl)). NeurIPS 2025. 


## Overview
![Overview](https://github.com/shixuanliu-andy/DynaInfer/blob/main/figure/fig_intro.png)

Data-driven methods offer efficient and robust solutions for analyzing complex dynamical systems but rely on the assumption of I.I.D. data, driving the development of generalization techniques for handling environmental differences. These techniques, however, are limited by their dependence on environment labels, which are often unavailable during training due to data acquisition challenges, privacy concerns, and environmental variability, particularly in large public datasets and privacy-sensitive domains. In response, we propose DynaInfer, a novel method that infers environment specifications by analyzing prediction errors from fixed neural networks within each training round, enabling environment assignments directly from data. We prove our algorithm effectively solves the alternating optimization problem in unlabeled scenarios and validate it through extensive experiments across diverse dynamical systems. Results show that DynaInfer outperforms existing environment assignment techniques, converges rapidly to true labels, and even achieves superior performance when environment labels are available.
![Method](https://github.com/shixuanliu-andy/DynaInfer/blob/main/figure/fig_method.png)

## Usage
Here are some example commands to run the code:
```bash
python trainer.py --dataset lv --assumed_nenv 9 --device 0
python trainer.py --dataset lv --oracle --device 0
python trainer.py --dataset lv --coda --assumed_nenv 9 --device 0
python trainer.py --dataset lv --coda --coda_norm l2m-l2c --assumed_nenv 9 --device 0 
python trainer.py --dataset gs --assumed_nenv 3 --device 0
python trainer.py --dataset gs --oracle --device 0
python trainer.py --dataset ns --assumed_nenv 4 --device 0
python trainer.py --dataset ns --oracle --device 0
```

## Results
We illustrate the probability of environment assignments with DynaInfer over training time, respectively in LV, GS and NS experiments. Initially, our model may default to random assignments due to unoptimized neural networks. However, all the assignments in LV, GS and NS quickly converge to the true labels. Notably, systems with simpler dynamics, like LV compared to NS, enable quicker learning of base generalization methods, resulting in faster convergence of environment assignments.
![LV](https://github.com/shixuanliu-andy/DynaInfer/blob/main/figure/fig_ei_lv.png)
![GS](https://github.com/shixuanliu-andy/DynaInfer/blob/main/figure/fig_ei_gs.png)
![NS](https://github.com/shixuanliu-andy/DynaInfer/blob/main/figure/fig_ei_ns.png)

## Reference

If you find our code or paper useful, please cite:

```bibtex
@inproceedings{2025environment,
  title={Environment Inference for Learning Generalizable Dynamical System},
  author={Liu, Shixuan and He, Yue and Wang, Haotian and Yang, Wenjing and Wang, Yunfei and Cui, Peng and Liu, Zhong},
  booktitle={Advances in Neural Information Processing Systems},
  volume={39},
  year={2025},
  url={https://openreview.net/forum?id=2M5dTDdGxl}
}
```
