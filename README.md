# Multivariate Extremes, Cascades and Simulation

**GAMEX - Edinburgh Summer School on Generative AI for Extremes**  
**The University of Edinburgh · 8-11 September 2026** <br>
[GAMEX Summer School website](https://gamex-network.github.io/school/)

This repository contains materials for a 2-hour practical session on **Multivariate extremes, cascades and simulation** with PyTorch by [**Clemente Ferrer**](https://github.com/clementeferrer), Pontificia Universidad Católica de Chile.

The practical continues the session's Transformer, now over continuous values, and uses it on a process in which every event is an extreme event:

1. a **continuous Transformer** whose head outputs the parameters of a density,
2. a **marked Hawkes process of extremes**, whose cascades are latent, and
3. the **Transformer as an estimator** of that process on three real stock indices.

The aim is to connect the probability models to the PyTorch and NumPy code used to simulate, estimate, generate from, and diagnose them.

---

## Practical materials

| File | Topic | Main representation |
| --- | --- | --- |
| [`01_gamex_continuous_transformer.ipynb`](01_gamex_continuous_transformer.ipynb) | Continuous autoregressive Transformer | Daily returns, Gaussian and Student-$t$ heads |
| [`02_gamex_hawkes_cascades.ipynb`](02_gamex_hawkes_cascades.ipynb) | Marked Hawkes process of extremes | Event times with marks on the limit set |
| [`03_gamex_cascade_estimation.ipynb`](03_gamex_cascade_estimation.ipynb) | Estimating and generating real cascades | Extreme days of three stock indices |

The three notebooks continue the session's numbering as Practicals 3, 4 and 5.

### Helper modules

The notebooks hold the models, the calls and the results; the machinery sits in three modules beside them, which you are welcome to open.

| Module | Used by | Contents |
| --- | --- | --- |
| [`gamex_returns.py`](gamex_returns.py) | Notebook 1 | Data loading, the GamexCoin generator, training helpers, figures and the generation animation |
| [`gamex_cascades.py`](gamex_cascades.py) | Notebooks 2 and 3 | The gauge and its surfaces, the von Mises-Fisher sampler, the `Process` class, EM, the gauge likelihood, the real-data pipeline and three animations |
| [`gamex_transformer.py`](gamex_transformer.py) | Notebook 3 | Sequences, `GaugeNet`, `CascadeTransformer`, the loss, training, sampling and the generation animation |

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
01_gamex_continuous_transformer.ipynb
02_gamex_hawkes_cascades.ipynb
03_gamex_cascade_estimation.ipynb
```

Run the first code cell to confirm the installed PyTorch version and loaded compute device. Keep the notebooks in the same folder as the three `gamex_*.py` modules and the `data` folder; nothing is downloaded during the session.

---

## Data

`data/indices.csv` holds the daily closes of the S&P 500, the DAX and the Nikkei on their 9,009 common trading days, 1988 to 2026. Notebook 1 uses the S&P 500 column; notebook 3 uses all three. Notebook 2 simulates its own data from a known truth.

---

## Runtimes

| Notebook | Runtime | Heaviest step |
| --- | --- | --- |
| 1 | about 30 seconds | Twelve epochs on 7,194 windows |
| 2 | about 15 seconds | Three animations |
| 3 | about 3 minutes | Thirty epochs on 3,185 simulated cascades |

The notebooks ship with their outputs, so every figure and animation can be read before running anything.
