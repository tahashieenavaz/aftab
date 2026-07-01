<div align="center">
  <img src="https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/header.svg" />
</div>

<div align="center">
  Taha Shieenavaz | Shabnam Zareshahraki | Loris Nanni
</div>

<p align="center">
  <img src="https://img.shields.io/pypi/v/aftab" />
  <img src="https://img.shields.io/github/stars/tahashieenavaz/aftab?style=social" />
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" />
  <img src="https://img.shields.io/badge/license-CC--BY--NC--4.0-lightgrey" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/backend-PyTorch-ee4c2c" />
  <img src="https://img.shields.io/badge/citable-yes-success" />
  <img src="https://img.shields.io/badge/arXiv-coming%20soon-b31b1b" />
</p>


## Overview

**Aftab** (Persian: آفتاب, meaning "sun" or "sun rays") is a benchmarking framework for evaluating CNN-based encoders in PQN across Atari environments.  
It provides standardized training, evaluation, and reproducibility tools for deep reinforcement learning research.

We have compiled a few videos comparing PQN and Aftab agents. Watch them [here](./videos.md).


### Encoder Experiments

<div align="center">

| IQM HNS | IQM HNS (Last 50M Frames) |
| :---: | :---: |
| ![Global Performance](https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/encoder_experiments/global.png) | ![Last 50M Frames](https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/encoder_experiments/global_zoomed.png) |

</div>

### Hadamard Product Experiments

<div align="center">

| IQM HNS | IQM HNS (Last 50M Frames) |
| :---: | :---: |
| ![Global Performance](https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/hadamax_experiments/global.png) | ![Last 50M Frames](https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/hadamax_experiments/global_zoomed.png) |

</div>

References:
- [Hadamax Encoding: Elevating Performance in Model-Free Atari](https://arxiv.org/abs/2505.15345)

### Q-Values Experiments

<div align="center">

| IQM HNS | IQM HNS (Last 50M Frames) |
| :---: | :---: |
| ![Global Performance](https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/qvalue_experiments/global.png) | ![Last 50M Frames](https://raw.githubusercontent.com/tahashieenavaz/aftab/main/figures/qvalue_experiments/global_zoomed.png) |

</div>

References:
- [Stop Regressing](https://arxiv.org/abs/2403.03950)
- [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621)
- [Improving Regression Performance with Distributional Losses](https://arxiv.org/abs/1806.04613)

## Installation

Install via pip:

```bash
pip install aftab
```


## Usage

> Note that the JAX API is under development, but using current PyTorch version you need to expect training of your agents to take up to 13 hours for the best configuration. We hope we are going to get much faster results using JAX. 

```python
from aftab import Aftab
from aftab import aftab_environments

seeds = [1, 2, 3, 4]

for environment in aftab_environments:
    agent = Aftab(encoder="gamma", frames="pilot")
    for seed in seeds:
        agent.train(environment=environment, seed=seed)
        agent.log()
```


## Defining a Custom Encoder

You can define your own encoder as a PyTorch module and pass it to the agent:

```python
import torch
from aftab import Aftab

class CustomImageEncoder(torch.nn.Module):
    pass

agent = Aftab(encoder=CustomImageEncoder)
```


## Results

**Encoder Experiments**:

- Tables: 
  - [HNS](results/encoder_experiments/human_normalized_scores.md)
  - [Scores](results/encoder_experiments/scores.md)
- Charts:
  - [IQM HNS](https://github.com/tahashieenavaz/aftab/tree/main/figures/encoder_experiments/human_normalized_score)
  - [Loss Evolution](https://github.com/tahashieenavaz/aftab/tree/main/figures/encoder_experiments/loss)

**Hadamax Experiments**:

- Tables:
  - [HNS](results/hadamax_experiments/human_normalized_scores.md)
  - [Scores](results/hadamax_experiments/scores.md)
- Charts:
  - [IQM HNS](https://github.com/tahashieenavaz/aftab/tree/main/figures/hadamax_experiments/human_normalized_score)
  - [Loss Evolution](https://github.com/tahashieenavaz/aftab/tree/main/figures/hadamax_experiments/loss)

**Q-Value Experiments**:
- Tables:
  - [HNS](results/qvalue_experiments/human_normalized_scores.md)
  - [Scores](results/qvalue_experiments/scores.md)
- Charts:
  - [IQM HNS](https://github.com/tahashieenavaz/aftab/tree/main/figures/qvalue_experiments/human_normalized_score)
  - [Loss Evolution](https://github.com/tahashieenavaz/aftab/tree/main/figures/qvalue_experiments/loss)


## Model Complexity

### Base Variants

| Variant | Encoder Parameters | Regression Head Parameters | Total Parameters | Encoder FLOPs | Regression Head FLOPs | Total FLOPs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PQN** | 78,304 | 1,686,500 | 1,764,804 | 7.734 | 1.610 | 9.347 |
| **Alpha** | 174,752 | 1,782,948 | 1,957,700 | 27.541 | 1.610 | 29.151 |
| **Beta** | 89,008 | 1,782,948 | 1,871,956 | 61.515 | 1.610 | 63.126 |
| **Gamma** | 117,168 | 1,725,364 | 1,842,532 | 22.901 | 1.610 | 24.512 |
| **Delta** | 78,552 | 1,850,588 | 1,929,140 | 6.143 | 1.774 | 7.917 |
| **Epsilon** | 80,112 | 2,179,828 | 2,259,940 | 13.252 | 2.101 | 15.354 |
| **Zeta** | 77,232 | 2,537,396 | 2,614,628 | 25.362 | 2.462 | 27.824 |
| **Eta** | 78,400 | 23,739,460 | 23,817,860 | 28.422 | 23.663 | 52.085 |
| **Theta** | 76,288 | 1,127,428 | 1,203,716 | 9.065 | 1.053 | 10.118 |

> **Note:** The Eta variant has significantly more parameters than other variants, primarily due to the encoder producing a large number of features.

---

### Hadamax Variants

| Variant | Encoder Parameters | Regression Head Parameters | Total Parameters | Encoder FLOPs | Regression Head FLOPs | Total FLOPs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PQN Hadamax** | 156,608 | 3,968,516 | 4,125,124 | 159.014 | 3.969 | 162.984 |
| **Gamma Hadamax V1** | 234,336 | 1,609,220 | 1,843,556 | 122.001 | 1.610 | 123.611 |
| **Gamma Hadamax V2** | 234,336 | 3,280,388 | 3,514,724 | 129.300 | 3.281 | 132.581 |

## Hyperparameters

<div align="center">

| Hyperparameter | Value |
| :--- | :--- |
| Learning rate | $2.5 \times 10^{-4}$ |
| Training environments | 128 |
| Test environments | 8 |
| Optimizer | [Rectified Adam](https://arxiv.org/abs/1908.03265) |
| Weight decay | 0 |
| $\epsilon$ | $1 \times 10^{-5}$ |
| $\beta_{1}$ | 0.9 |
| $\beta_{2}$ | 0.999 |
| Total Frames | 200,000,000 |
| Loss function | Mean Squared Error |
| Scheduler | Linear Annealing |
| $\epsilon$-greedy exploration | 10% of total frames |
| Discount factor ($\gamma$) | 0.99 |
| GAE ($\lambda$) | 0.65 |
| Epochs | 2 |
| Batch size | 4096 |

</div>

<p align="center"><em>Used in encoder and Hadamax experiments.</em></p>

## Statistical Significance

### Encoder Experiments

|         |   PQN |   Alpha |   Beta |   Gamma |   Delta |   Epsilon |   Zeta |   Eta |   Theta |
|:--------|------:|--------:|-------:|--------:|--------:|----------:|-------:|------:|--------:|
| PQN     | -     |   -     |  -     |   -     |   -     |     -     |  -     | -     |   -     |
| Alpha   | 0     |   -     |  -     |   -     |   -     |     -     |  -     | -     |   -     |
| Beta    | 0     |   0.847 |  -     |   -     |   -     |     -     |  -     | -     |   -     |
| Gamma   | 0     |   0.295 |  0.802 |   -     |   -     |     -     |  -     | -     |   -     |
| Delta   | 0     |   0     |  0     |   0     |   -     |     -     |  -     | -     |   -     |
| Epsilon | 0     |   0.104 |  0.068 |   0.01  |   0     |     -     |  -     | -     |   -     |
| Zeta    | 0     |   0.145 |  0.293 |   0.024 |   0     |     0.552 |  -     | -     |   -     | 
| Eta     | 0.001 |   0.337 |  0.757 |   0.221 |   0     |     0.819 |  0.967 | -     |   -     | 
| Theta   | 0.431 |   0     |  0.004 |   0     |   0.046 |     0.001 |  0.001 | 0.002 |   -     |

### Hadamax Experiments

|                    |   Gamma |   Hadamax Gamma V1 |   Hadamax Gamma V2 |   Hadamax |
|:-------------------|--------:|-------------------:|-------------------:|----------:|
| Gamma              |       - |              -     |              -     |     -     |
| Hadamax Gamma V1   |       0 |              -     |              -     |     -     |
| Hadamax Gamma V2   |       0 |              0.72  |              -     |     -     |
| Hadamax Nature DQN |       0 |              0.078 |              0.151 |     -     |


### Q-Value Experiments

|                                      |   PQN |   Distributional Duelling |   Bootstrapped Duelling |   Distributional Bootstrapped Duelling |
|:-------------------------------------|------:|--------------------------:|------------------------:|---------------------------------------:|
| PQN                                  |     - |                     -     |                   -     |                                    -   |
| Distributional Duelling              |     0 |                     -     |                   - |                                    -   |
| Bootstrapped Duelling                |     0 |                     0.115 |                   -     |                                    - |
| Distributional Bootstrapped Duelling |     0 |                     0     |                   0.3   |                                    -   |

## Reproducibility

Due to the stochastic nature of deep reinforcement learning, exact reproducibility via fixed datasets is not feasible.  
Instead, we provide a set of random seeds used in our experiments.

```python
from aftab import aftab_seeds

print(aftab_seeds)
```

Full experiment replication:

```python
from aftab import Aftab
from aftab import aftab_environments
from aftab import aftab_seeds

for environment in aftab_environments:
    agent = Aftab()
    for seed in aftab_seeds:
        agent.train(environment=environment, seed=seed)
        agent.log()
```

A comprehensive set of Atari environments is available via EnvPool:  
https://envpool.readthedocs.io/en/latest/env/atari.html#available-tasks

## Hardware

[Nvidia A40](https://www.nvidia.com/en-us/data-center/a40) GPUs were used to run all the experiments in this experiment.

| Specification | Details |
|--------------|----------|
| GPU Memory | 48 GB GDDR6 with error-correcting code (ECC) |
| GPU Memory Bandwidth | 696 GB/s |
| Interconnect | NVIDIA NVLink 112.5 GB/s (bidirectional); PCIe Gen4: 64 GB/s |
| NVLink | 2-way low profile (2-slot) |
| Display Ports | 3x DisplayPort 1.4* |
| Max Power Consumption | 300 W |
| Form Factor | 4.4" (H) x 10.5" (L), Dual Slot |
| Thermal | Passive |
| vGPU Software Support | NVIDIA Virtual PC, NVIDIA Virtual Applications, NVIDIA RTX Virtual Workstation, NVIDIA Virtual Compute Server, NVIDIA AI Enterprise |
| vGPU Profiles Supported | See the Virtual GPU Licensing Guide |
| NVENC / NVDEC | 1x / 2x (includes AV1 decode) |
| Secure Boot | Secure and Measured Boot with Hardware Root of Trust (optional) |
| NEBS Ready | Level 3 |
| Power Connector | 8-pin CPU |

## Citation

```bibtex
@article{aftab2026benchmarking,
  title={Aftab: Benchmarking {CNN} Encoders in {PQN}},
  author={Shieenavaz, Taha and Zareshahraki, Shabnam and Nanni, Loris},
  journal={arXiv preprint arXiv:YYMM.NNNNN},
  year={2026}
}
```

### Related Works

```bibtex
@misc{2407.04811,
  Title = {Simplifying Deep Temporal Difference Learning},
  Author = {Matteo Gallici and Mattie Fellows and Benjamin Ellis and Bartomeu Pou and Ivan Masmitja and Jakob Nicolaus Foerster and Mario Martin},
  Year = {2024},
  Eprint = {arXiv:2407.04811},
}
```

```bibtex
@misc{2403.03950,
  Title = {Stop Regressing: Training Value Functions via Classification for Scalable Deep RL},
  Author = {Jesse Farebrother and Jordi Orbay and Quan Vuong and Adrien Ali Taïga and Yevgen Chebotar and Ted Xiao and Alex Irpan and Sergey Levine and Pablo Samuel Castro and Aleksandra Faust and Aviral Kumar and Rishabh Agarwal},
  Year = {2024},
  Eprint = {arXiv:2403.03950},
}
```

```bibtex
@misc{1511.06581,
  Title = {Dueling Network Architectures for Deep Reinforcement Learning},
  Author = {Ziyu Wang and Tom Schaul and Matteo Hessel and Hado van Hasselt and Marc Lanctot and Nando de Freitas},
  Year = {2015},
  Eprint = {arXiv:1511.06581},
}
```

```bibtex
@misc{1806.04613,
  Title = {Improving Regression Performance with Distributional Losses},
  Author = {Ehsan Imani and Martha White},
  Year = {2018},
  Eprint = {arXiv:1806.04613},
}
```

```bibtex
@misc{1602.04621,
  Title = {Deep Exploration via Bootstrapped DQN},
  Author = {Ian Osband and Charles Blundell and Alexander Pritzel and Benjamin Van Roy},
  Year = {2016},
  Eprint = {arXiv:1602.04621},
}
```

## License

© 2025 Taha Shieenavaz.  
Licensed under CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/