<p align="center">
    <img src="./figures/global.png" />
</p>

## Installation

We have composed the whole project inside an installable Python library. You can install the package using pip.

```terminal
pip install aftab
```

## Usage

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