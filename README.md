# Multivariate Extremes, Cascades and Simulation

**GAMEX - Edinburgh Summer School on Generative AI for Extremes**  
**The University of Edinburgh · 8-11 September 2026** <br>
[GAMEX Summer School website](https://gamex-network.github.io/school/)

This repository contains materials for a 2-hour practical session on **Multivariate extremes, cascades and simulation** with PyTorch by [**Clemente Ferrer**](https://github.com/clementeferrer), Pontificia Universidad Católica de Chile and [**Johnny Myung Won Lee**](https://johnnymdoubleu.github.io), The University of Edinburgh.

The practical continues the session's Transformer, now over continuous values, and uses it on a process in which every event is an extreme event.
The aim is to connect the probability models to the PyTorch and NumPy code used to simulate, estimate, generate from, and diagnose them.

---

## Practical materials

| File | Topic | Main representation |
| --- | --- | --- |
| [`02_gamex_continuous_transformer.ipynb`](02_gamex_continuous_transformer.ipynb) | Continuous autoregressive Transformer | Daily returns, Gaussian and Student-$t$ heads |
| [`03_gamex_hawkes_cascades.ipynb`](03_gamex_hawkes_cascades.ipynb) | Marked Hawkes process of extremes | Event times with marks on the limit set |
| [`04_gamex_cascade_estimation.ipynb`](04_gamex_cascade_estimation.ipynb) | Estimating and generating real cascades | Extreme days of three stock indices |

### Helper modules

The notebooks hold the models, the calls and the results; the machinery sits in three modules beside them, which you are welcome to open.

| Module | Contents |
| --- | --- |
| [`gamex_returns.py`](gamex_returns.py) | Data loading, the GamexCoin generator, training helpers, figures and the generation animation |
| [`gamex_cascades.py`](gamex_cascades.py) | The gauge and its surfaces, the von Mises-Fisher sampler, the `Process` class, EM, the gauge likelihood, the real-data pipeline and three animations |
| [`gamex_transformer.py`](gamex_transformer.py) | Sequences, `GaugeNet`, `CascadeTransformer`, the loss, training, sampling and the generation animation |

---

## Requirements

Required software and packages:

- Python 3.10+
- PyTorch 2.0+
- NumPy
- SciPy
- pandas
- Matplotlib
- JupyterLab

Recommended softwares:

- Git
- VScode
- Anaconda or Miniforge

A **CPU is sufficient** for the practical. The largest model has about 120,000 parameters and trains in roughly two minutes on a laptop.

---

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/gamex-network/gamex_summer_school_cascades.git
cd gamex_summer_school_cascades

python -m venv gamex
```

Activate it on macOS/Linux:

```bash
source gamex/bin/activate
```

or in Windows PowerShell:

```powershell
.\gamex\Scripts\Activate.ps1
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install torch numpy scipy pandas matplotlib jupyterlab
```

Then launch JupyterLab:

```bash
jupyter lab
```

Open the notebooks in order:

```text
02_gamex_continuous_transformer.ipynb
03_gamex_hawkes_cascades.ipynb
04_gamex_cascade_estimation.ipynb
```

Run the first code cell to confirm the installed PyTorch version and loaded compute device. Keep the notebooks in the same folder as the three `gamex_*.py` modules and the `data` folder; nothing is downloaded during the session.

---

## Data

`data/indices.csv` holds the daily closes of the S&P 500, the DAX and the Nikkei on their 9,009 common trading days, 1988 to 2026. The continuous Transformer uses the S&P 500 column and the cascade estimation uses all three. The Hawkes notebook simulates its own data from a known truth.

---
