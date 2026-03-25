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

We used the following numbers as our seeds:
- 475284
- 219842
- 525975
- 909314

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