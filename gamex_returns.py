"""Machinery for GAMEX Practical 3.

The notebook builds the model, the loss and the sampling loop; the data,
the figures and the animation live here.
"""

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, clear_output, display

import torch
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_returns(path):
    rows = [line.strip().split(",") for line in open(path)][1:]
    dates = [r[0] for r in rows][1:]              # aligned with diff(log price)
    price = np.array([float(r[1]) for r in rows])
    return dates, torch.tensor(100.0 * np.diff(np.log(price)), dtype=torch.float32)


def make_loader(inputs, targets, batch_size, seed):
    return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle=True,
                      generator=torch.Generator().manual_seed(seed))


def make_gamexcoin_returns(n_days=2000, nu=4.0, seed=20260908):
    g = torch.Generator().manual_seed(seed)
    omega, a, b = 1.2, 0.15, 0.80
    variance = torch.empty(n_days)
    gmx = torch.empty(n_days)
    variance[0] = omega / (1.0 - a - b)
    for t in range(n_days):
        if t > 0:
            variance[t] = omega + a * gmx[t - 1] ** 2 + b * variance[t - 1]
        chi2 = torch.randn(int(nu), generator=g).square().sum()          # chi-squared with nu dof
        t_draw = torch.randn(1, generator=g) * torch.sqrt(nu / chi2)     # Student-t with nu dof
        gmx[t] = variance[t].sqrt() * t_draw / math.sqrt(nu / (nu - 2.0))
    return gmx


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def live_loss_plot(figure, axis, history):
    axis.clear()
    for key, label in (("train_nll", "train"), ("val_nll", "validation")):
        axis.plot(range(1, len(history[key]) + 1), history[key], label=label)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Negative log-likelihood per day")
    axis.grid(alpha=0.3)
    axis.legend()
    clear_output(wait=True)
    display(figure)


def train_on(make_model, loader, loss_fn, n_epochs=10, seed=20260908, device="cpu"):
    torch.manual_seed(seed)
    net = make_model().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(n_epochs):
        net.train()
        for batch_inputs, batch_targets in loader:
            mu, log_sigma = net(batch_inputs.to(device))
            loss = loss_fn(mu, log_sigma, batch_targets.to(device)).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    return net


def own_sigma_exceedances(net, inputs, targets, k=4.0, device="cpu"):
    net.eval()
    with torch.inference_mode():
        mu, log_sigma = net(inputs.to(device))
    z = (targets[:, -1] - mu[:, -1].cpu()) / torch.exp(log_sigma[:, -1]).cpu()
    return int((z.abs() > k).sum()), len(z)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_series(values, dates, title, every=4, colour="C0"):
    ticks = [i for i, d in enumerate(dates) if d[:4] != dates[i - 1][:4] and int(d[:4]) % every == 0]
    plt.figure(figsize=(11, 3.2))
    plt.plot(values, linewidth=0.3, color=colour)
    plt.xticks(ticks, [dates[i][:4] for i in ticks])
    plt.xlabel("Year")
    plt.ylabel("Daily return (%)")
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_window(returns, dates, end, block_size):
    window = returns[end - block_size:end + 1].numpy()
    around = (list(range(end - block_size - 24, end - block_size))
              + list(range(end + 1, end + 9)))
    plt.figure(figsize=(9, 2.8))
    plt.bar(around, returns.numpy()[around], color="0.8")
    plt.bar(range(end - block_size, end), window[:-1], color="#4ea3ff", label=f"Input, {block_size} days")
    plt.bar([end], [window[-1]], color="#ff4d4d", label="Target")
    plt.xticks([end - block_size, end], [dates[end - block_size], dates[end]])
    plt.ylabel("Daily return (%)")
    plt.title("One training window, September 2008")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_band(day, observed, mu, sigma, dates, title):
    ticks = [i for i in day if dates[i][:4] != dates[i - 1][:4]]
    plt.figure(figsize=(11, 3.4))
    plt.plot(day, observed, linewidth=0.4, color="0.5", label="Observed return")
    plt.fill_between(day, mu - 2 * sigma, mu + 2 * sigma, alpha=0.35, color="C1",
                     label=r"Fitted $\mu \pm 2\sigma$")
    plt.xticks(ticks, [dates[i][:4] for i in ticks])
    plt.xlabel("Year")
    plt.ylabel("Daily return (%)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_paths(sampled, contexts, actuals, titles, block_size):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, group, context, actual, title in zip(axes, sampled, contexts, actuals, titles):
        for path in group:
            ax.plot(range(1, group.shape[1] + 1), path, linewidth=0.6, alpha=0.5, color="C0")
        ax.plot(range(1 - block_size, 1), context, linewidth=1.2, color="0.15", label="Observed context")
        ax.plot(range(1, group.shape[1] + 1), actual, linewidth=1.3, color="#ff4d4d", label="What happened")
        ax.set_title(title)
        ax.set_xlabel("Trading days from the start")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Daily return (%)")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def animate_continuation(context, mus, sigmas, draws, start_label, limit=14.0, interval=700):
    block_size = len(context)
    n_generated = len(draws)

    # ---- layout ----
    fig = plt.figure(figsize=(11, 5))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[2.6, 1.15], wspace=0.28)
    ax_path = fig.add_subplot(gs[0, 0])
    ax_density = fig.add_subplot(gs[0, 1])
    ax_path.plot(range(1 - block_size, 1), context, marker="o", markersize=4, linewidth=1.6,
                 color="#4ea3ff", label="Observed")

    # ---- animated artists ----
    generated_line, = ax_path.plot([], [], marker="o", markersize=4, linewidth=1.8,
                                   color="#ff4d4d", label="Generated")
    status_text = ax_path.text(0.02, 0.97, "", transform=ax_path.transAxes, ha="left", va="top",
                               fontsize=10,
                               bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.8"))
    ax_path.set_xlim(-block_size, n_generated + 1)
    ax_path.set_ylim(-limit, limit)
    ax_path.set_xlabel("Trading days relative to " + start_label)
    ax_path.set_ylabel("Daily return (%)")
    ax_path.set_title("Autoregressive generation, one density at a time")
    ax_path.grid(alpha=0.2)
    ax_path.legend(loc="lower left", fontsize=9)
    grid = np.linspace(-limit, limit, 240)

    def update(frame):
        generated_line.set_data(range(1, frame + 2), draws[:frame + 1])
        m, s = mus[frame], sigmas[frame]
        density = np.exp(-0.5 * ((grid - m) / s) ** 2) / (s * math.sqrt(2.0 * math.pi))
        ax_density.clear()
        ax_density.plot(grid, density, color="C1")
        ax_density.axvline(draws[frame], color="#ff4d4d", linewidth=1.4)
        ax_density.set_xlim(-limit, limit)
        ax_density.set_xlabel("Next-day return (%)")
        ax_density.set_title("Conditional density and the sampled value")
        ax_density.grid(alpha=0.2)
        status_text.set_text(f"Day {frame + 1}/{n_generated}\nmu {m:+.2f}%   sigma {s:.2f}%")
        return generated_line, status_text

    animation = FuncAnimation(fig, update, frames=n_generated, interval=interval, repeat=False, blit=False)
    plt.close(fig)
    display(HTML(animation.to_jshtml()))
