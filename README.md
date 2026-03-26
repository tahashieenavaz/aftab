<p align="center">

| Performance | Performance (Last 50M Frames) |
| :---: | :---: |
| ![Global Performance](figures/global.png) | ![Last 50M Frames](figures/global_zoomed.png) |

</p>

## Installation

We have composed the whole project inside an installable Python library. You can install the package using pip.

```terminal
pip install aftab
```

## Usage

You can import the agent and configure all the hyperparameters based on following guide.

```python
from aftab import Agent as AftabAgent

agent = AftabAgent(environment="pong")
agent.train()
agent.save("pong.model")
```

## Results

| **Game** | **PQN** | **Alpha** | **Beta** | **Gamma** | **Delta** | **Epsilon** | **Zeta** | **Eta** | **Theta** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Alien | 0.497 | 1.834 | 2.150 | 2.562 | 0.376 | 1.998 | 1.738 | 1.479 | 0.419 |
| Amidar | 0.603 | 1.101 | 0.872 | 0.825 | 0.539 | 1.173 | 0.866 | 1.042 | 0.665 |
| Assault | 31.821 | 31.993 | 30.422 | 31.027 | 21.389 | 28.884 | 32.934 | 32.382 | 30.384 |
| Asterix | 30.883 | 10.978 | 12.386 | 12.621 | 30.471 | 10.752 | 10.017 | 36.432 | 34.046 |
| Asteroids | 0.032 | 1.479 | 0.313 | 1.020 | 0.145 | 0.699 | 1.272 | 0.047 | 0.025 |
| Atlantis | 49.083 | 42.817 | 44.515 | 45.202 | 45.793 | 47.703 | 45.019 | 45.478 | 44.755 |
| Bankheist | 1.897 | 2.006 | 2.073 | 1.686 | 1.818 | 1.802 | 2.097 | 1.729 | 1.970 |
| Battlezone | 1.186 | 1.514 | 1.620 | 1.829 | 1.081 | 1.982 | 1.680 | 1.243 | 1.156 |
| Beamrider | 1.223 | 2.347 | 2.153 | 2.527 | 1.085 | 2.031 | 1.624 | 1.252 | 1.182 |
| Berzerk | 1.697 | 2.626 | 0.324 | 0.876 | 1.282 | 1.414 | 2.396 | 0.999 | 1.201 |
| Bowling | 0.044 | 0.111 | 0.035 | 0.128 | 0.068 | 0.063 | 0.077 | 0.112 | 0.056 |
| Boxing | 8.262 | 8.285 | 8.289 | 8.280 | 8.273 | 8.291 | 8.289 | 8.305 | 8.277 |
| Breakout | 13.343 | 15.398 | 15.057 | 16.470 | 12.962 | 14.560 | 13.055 | 17.899 | 15.067 |
| Centipede | 0.827 | 1.100 | 0.922 | 1.364 | 0.708 | 0.983 | 0.712 | 1.175 | 0.620 |
| Choppercommand | 2.473 | 15.875 | 24.620 | 27.135 | 1.178 | 21.830 | 15.754 | 3.251 | 1.405 |
| Crazyclimber | 6.303 | 7.185 | 7.318 | 6.473 | 6.349 | 7.263 | 7.292 | 7.034 | 6.277 |
| Defender | 3.189 | 4.174 | 3.798 | 5.893 | 2.937 | 4.517 | 3.760 | 3.711 | 5.128 |
| Demonattack | 72.278 | 72.840 | 73.037 | 72.948 | 70.825 | 72.463 | 71.976 | 71.360 | 69.148 |
| Doubledunk | 7.768 | 7.770 | 7.772 | 7.770 | 7.765 | 7.822 | 7.728 | 7.872 | 7.745 |
| Enduro | 2.694 | 2.696 | 2.723 | 2.718 | 2.696 | 2.718 | 2.686 | 2.693 | 2.669 |
| Fishingderby | 2.499 | 2.622 | 2.567 | 2.590 | 2.487 | 2.668 | 2.655 | 2.586 | 2.534 |
| Freeway | 1.134 | 1.142 | 1.141 | 1.128 | 1.134 | 1.138 | 1.141 | 1.139 | 1.133 |
| Frostbite | 1.555 | 2.791 | 2.069 | 2.011 | 1.279 | 1.606 | 2.257 | 1.720 | 1.025 |
| Gopher | 21.460 | 24.165 | 25.075 | 25.155 | 16.138 | 24.434 | 25.608 | 29.947 | 18.152 |
| Gravitar | 0.202 | 0.279 | 0.362 | 0.367 | 0.202 | 0.214 | 0.340 | 0.503 | 0.159 |
| Hero | 0.767 | 0.872 | 0.716 | 0.719 | 0.667 | 0.789 | 0.890 | 1.080 | 0.455 |
| Icehockey | 0.781 | 1.353 | 1.112 | 1.760 | 0.665 | 1.016 | 1.019 | 1.297 | 0.729 |
| Jamesbond | 6.612 | 10.531 | 8.682 | 17.401 | 5.962 | 21.986 | 11.612 | 11.526 | 6.595 |
| Kangaroo | 4.437 | 4.686 | 4.702 | 4.652 | 4.357 | 4.664 | 4.614 | 4.694 | 4.421 |
| Krull | 7.693 | 8.420 | 8.451 | 8.296 | 7.590 | 8.053 | 8.576 | 8.344 | 7.641 |
| Kungfumaster | 1.590 | 1.540 | 1.676 | 1.404 | 1.570 | 1.438 | 1.477 | 1.470 | 1.572 |
| Montezumarevenge | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | 0.008 | 0.000 |
| Mspacman | 0.537 | 0.628 | 0.904 | 0.822 | 0.590 | 0.715 | 0.867 | 1.057 | 0.606 |
| Namethisgame | 2.235 | 2.720 | 2.575 | 3.089 | 2.350 | 2.499 | 2.191 | 1.763 | 2.496 |
| Phoenix | 18.828 | 36.730 | 36.128 | 26.767 | 8.249 | 23.557 | 29.079 | 39.235 | 12.664 |
| Pitfall | 0.029 | 0.033 | 0.028 | 0.032 | 0.030 | 0.032 | 0.033 | 0.029 | 0.029 |
| Pong | 1.181 | 1.181 | 1.181 | 1.180 | 1.181 | 1.181 | 1.181 | 1.180 | 1.181 |
| Privateeye | 0.002 | 0.002 | 0.002 | 0.000 | -0.000 | 0.001 | 0.002 | 0.000 | 0.001 |
| Qbert | 1.640 | 1.855 | 1.846 | 1.865 | 1.370 | 1.805 | 1.818 | 1.874 | 1.530 |
| Riverraid | 1.388 | 1.627 | 1.666 | 1.707 | 1.354 | 1.671 | 1.741 | 1.578 | 1.357 |
| Roadrunner | 6.952 | 8.621 | 9.547 | 10.359 | 7.035 | 10.318 | 8.676 | 7.407 | 6.907 |
| Robotank | 6.895 | 7.223 | 7.348 | 7.373 | 6.930 | 7.274 | 7.203 | 6.830 | 6.849 |
| Seaquest | 0.193 | 0.245 | 0.203 | 0.208 | 0.189 | 0.207 | 0.376 | 0.374 | 0.197 |
| Skiing | -0.540 | 0.429 | -0.131 | -0.704 | -0.417 | 0.548 | 0.512 | 0.616 | -0.090 |
| Solaris | 0.063 | 0.124 | 0.117 | 0.128 | 0.109 | 0.164 | 0.141 | 0.119 | 0.128 |
| Spaceinvaders | 5.402 | 4.771 | 9.903 | 5.596 | 5.974 | 4.297 | 1.738 | 11.094 | 2.999 |
| Stargunner | 26.203 | 37.521 | 42.411 | 41.998 | 24.828 | 32.813 | 29.995 | 24.154 | 22.872 |
| Surround | 1.061 | 1.174 | 1.177 | 1.130 | 0.974 | 1.151 | 1.193 | 1.192 | 0.918 |
| Tennis | 1.403 | 2.274 | 1.421 | 2.671 | 1.452 | 2.267 | 1.449 | 1.870 | 1.396 |
| Timepilot | 5.783 | 14.973 | 19.375 | 14.375 | 4.683 | 12.621 | 14.647 | 14.401 | 4.516 |
| Tutankham | 1.531 | 1.533 | 1.547 | 1.516 | 1.509 | 1.538 | 1.549 | 1.562 | 1.528 |
| Upndown | 21.413 | 22.271 | 22.621 | 23.379 | 16.174 | 24.295 | 27.561 | 9.264 | 19.848 |
| Venture | 0.002 | 0.002 | 0.005 | 0.011 | 0.000 | 0.001 | 0.001 | 0.002 | 0.000 |
| Videopinball | 326.642 | 346.036 | 328.839 | 353.967 | 325.655 | 353.861 | 351.225 | 345.784 | 369.978 |
| Wizardofwor | 4.504 | 6.765 | 7.638 | 5.928 | 3.576 | 6.322 | 6.794 | 8.194 | 5.144 |
| Yarsrevenge | 2.184 | 2.559 | 2.629 | 2.574 | 1.950 | 2.266 | 2.628 | 2.535 | 2.046 |
| Zaxxon | 1.779 | 2.075 | 2.598 | 2.147 | 1.661 | 1.674 | 1.691 | 2.095 | 1.816 |
| **Median HNS** | **1.805** | **2.421** | **2.202** | **2.555** | **1.568** | **2.074** | **1.954** | **1.860** | **1.573** |

## Parameter Count

<div align="center">

| Variant  | Encoder Parameters | Regression Head | Total Parameters |
|----------|------------------|-----------------|------------------|
| PQN      | 78,304           | 1,686,500       | 1,764,804        |
| Alpha    | 174,752          | 1,782,948       | 1,957,700        |
| Beta     | 89,008           | 1,782,948       | 1,871,956        |
| Gamma    | 117,168          | 1,725,364       | 1,842,532        |
| Delta    | 78,552           | 1,850,588       | 1,929,140        |
| Epsilon  | 80,112           | 2,179,828       | 2,259,940        |
| Zeta     | 77,232           | 2,537,396       | 2,614,628        |
| Eta      | 78,400           | 23,739,460      | 23,817,860       |
| Theta    | 76,288           | 1,127,428       | 1,203,716        |

</div>

## Hyperparameters

<div align=center>

| Hyperparameter | Value |
| :--- | :--- |
| Learning rate | $2.5 \times 10^{-4}$ |
| Training environments | 128 |
| Test environments | 8 |
| Optimizer | Rectified Adam |
| Weight decay | 0 |
| Adam $\epsilon$ | $1 \times 10^{-5}$ |
| Total Frames | 200,000,000 |
| Loss function | Mean Squared Error |
| Scheduler | Linear Annealing |
| $\epsilon$-greedy exploration | 10% of total frames |
| Discount factor ($\gamma$) | 0.99 |
| GAE parameter ($\lambda$) | 0.65 |
| Epochs | 2 |
| Batch size | 4096 |

</div>

## Architectures

### Alpha

<img src="/figures/archs/alpha.png" />

```py
from aftab import AlphaEncoder

print(AlphaEncoder)
```

### Beta

<img src="/figures/archs/beta.png" />

```py
from aftab import BetaEncoder

print(BetaEncoder)
```
### Gamma

<img src="/figures/archs/gamma.png" />

```py
from aftab import GammaEncoder

print(GammaEncoder)
```

### Delta

<img src="/figures/archs/delta.png" />

```py
from aftab import DeltaEncoder

print(DeltaEncoder)
```

### Epsilon

<img src="/figures/archs/epsilon.png" />

```py
from aftab import EpsilonEncoder

print(EpsilonEncoder)
```


### Zeta

<img src="/figures/archs/zeta.png" />

```py
from aftab import ZetaEncoder

print(ZetaEncoder)
```

### Eta

<img src="/figures/archs/eta.png" />

```py
from aftab import EtaEncoder

print(EtaEncoder)
```

### Theta

<img src="/figures/archs/theta.png" />

```py
from aftab import ThetaEncoder

print(ThetaEncoder)
```


## Statistical Significance

<p align="center">
    <img src="./figures/statistical_significance.png" />
</p>

## Reproducibility

We run our experiments across four different seeds, and used <a href="https://github.com/tahashieenavaz/baloot">Baloot</a> library to enforce the seed across all the involved libraries.

```py
from baloot import seed_everything

seed_everything(10000)
```

The following numbers were used as our seeds for experiments. 

```py
from baloot import seed_everything

# first set of experiments
seed_everything(475284)

# second set of experiments
seed_everything(219842)

# third set of experiments
seed_everything(525975)

# fourth set of experiments
seed_everything(909314)
```

## Citation

Please cite this work should you find that useful.

```
@article{aftab2026benchmarking,
  title={Aftab: Benchmarking {CNN} Encoders in {PQN}},
  author={Shieenavaz, Taha and Zareshahraki, Shabnam and Nanni, Loris},
  journal={arXiv preprint arXiv:YYMM.NNNNN},
  year={2026}
}
```