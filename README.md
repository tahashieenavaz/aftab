<p align="center">
    <img src="./figures/global.png" />
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

## Specifications

<div align="center">

| Variant  | Encoder Parameters | Regression Head | Total Parameters |
|----------|------------------|-----------------|------------------|
| Base     | 78,304           | 1,686,500       | 1,764,804        |
| Alpha    | 174,752          | 1,782,948       | 1,957,700        |
| Beta     | 89,008           | 1,782,948       | 1,871,956        |
| Gamma    | 117,168          | 1,725,364       | 1,842,532        |
| Delta    | 78,552           | 1,850,588       | 1,929,140        |
| Epsilon  | 80,112           | 2,179,828       | 2,259,940        |
| Zeta     | 77,232           | 2,537,396       | 2,614,628        |
| Eta      | 78,400           | 23,739,460      | 23,817,860       |
| Theta    | 76,288           | 1,127,428       | 1,203,716        |

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