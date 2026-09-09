"""The cascade Transformer of GAMEX Practical 5.

One head per factor of the next-event likelihood: continuation and waiting
time from the Hawkes closed forms in `eta` and `beta`, direction from an
anchored von Mises-Fisher mixture in `kappa`, size from a truncated Gamma
whose rate is the learned gauge `g_psi`.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
from scipy import stats
from scipy.spatial import ConvexHull

import torch
from torch import nn
import torch.nn.functional as F

from gamex_cascades import sample_vmf, sphere_grid, star_surface, tree_layout

D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT = 64, 3, 4, 128, 0.1

LOSS_WEIGHTS = {"parent": 1.0, "angular": 1.0, "anchor": 1.0, "radial": 1.0,
                "cont": 0.5, "below": 10.0, "curvature": 5.0}


# ---------------------------------------------------------------------------
# Cascades as padded sequences
# ---------------------------------------------------------------------------


def build_sequences(events, max_len=256):
    seqs = []
    for _, block in events.groupby("cascade", sort=False):
        block = block.sort_values("t").iloc[:max_len]
        position = {e: i for i, e in enumerate(block["event"])}
        t = block["t"].to_numpy()
        seqs.append({
            "dt": np.diff(t, prepend=t[0]),
            "r": block["r"].to_numpy(),
            "W": block[["w1", "w2", "w3"]].to_numpy(),
            "parent_local": np.array([position.get(par, -1) for par in block["parent"]]),
        })
    return seqs


def collate(batch):
    B = len(batch)
    L = max(len(s["dt"]) for s in batch)
    out = {
        "dt": torch.zeros(B, L),
        "r": torch.ones(B, L),
        "W": torch.full((B, L, 3), 1.0 / np.sqrt(3.0)),
        "parent_local": torch.full((B, L), -1, dtype=torch.long),
        "valid": torch.zeros(B, L, dtype=torch.bool),
    }
    for b, s in enumerate(batch):
        n = len(s["dt"])
        out["dt"][b, :n] = torch.as_tensor(s["dt"], dtype=torch.float32)
        out["r"][b, :n] = torch.as_tensor(s["r"], dtype=torch.float32)
        out["W"][b, :n] = torch.as_tensor(s["W"], dtype=torch.float32)
        out["parent_local"][b, :n] = torch.as_tensor(s["parent_local"])
        out["valid"][b, :n] = True
    return out


def sinusoidal(x, n_freq=8):
    freqs = 2.0 ** torch.arange(n_freq, dtype=x.dtype)
    angle = x.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)


# ---------------------------------------------------------------------------
# The gauge as a 1-homogeneous network
# ---------------------------------------------------------------------------


class GaugeNet(nn.Module):
    def __init__(self, hidden=32, n_grid=300):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.c0 = nn.Parameter(torch.tensor(1.2))
        grid = sphere_grid(n_grid)
        adjacency = np.zeros((n_grid, n_grid), dtype=np.float32)
        for tri in ConvexHull(grid).simplices:
            for a, b in ((0, 1), (1, 2), (0, 2)):
                adjacency[tri[a], tri[b]] = adjacency[tri[b], tri[a]] = 1.0
        laplacian = np.eye(n_grid, dtype=np.float32) - adjacency / adjacency.sum(1, keepdims=True)
        self.register_buffer("grid", torch.as_tensor(grid, dtype=torch.float32))
        self.register_buffer("laplacian", torch.as_tensor(laplacian))

    def forward(self, x):
        r = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return r.squeeze(-1) * F.softplus(self.net(x / r).squeeze(-1) + self.c0)

    def curvature(self):
        return ((self.laplacian @ self.forward(self.grid)) ** 2).mean()   # (L g)^2, not g'L g: a sphere is not the prior


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class CascadeTransformer(nn.Module):
    def __init__(self, alpha, C_q, u_q_theta=None):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(float(alpha)))
        self.register_buffer("C_q", torch.tensor(float(C_q)))
        self.u_q_theta = None if u_q_theta is None else torch.as_tensor(np.asarray(u_q_theta), dtype=torch.float32)
        self.register_buffer("scales", torch.tensor([0.3, 1.2, 4.8]))
        self.embed = nn.Linear(2 * 8 + 1 + 1 + 3 + 3, D_MODEL)
        self.norm = nn.LayerNorm(D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=D_FF, dropout=DROPOUT,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS, enable_nested_tensor=False)
        # ---- continuation and time: the two scalars ARE (eta, beta) ----
        self.eta_raw = nn.Parameter(torch.tensor(0.3))
        self.beta_raw = nn.Parameter(torch.tensor(1.0))
        # ---- anchored angular head: next-parent attention plus one concentration ----
        self.next_q = nn.Linear(D_MODEL, D_MODEL)
        self.next_k = nn.Linear(D_MODEL, D_MODEL)
        self.next_edge = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 1))
        self.log_kappa = nn.Parameter(torch.log(torch.tensor(5.0)))
        # ---- radial head: the gauge, and nothing else ----
        self.gauge = GaugeNet()
        # ---- parent head: attention plus the pair statistics of the branching posterior ----
        self.parent_q = nn.Linear(D_MODEL, D_MODEL)
        self.parent_k = nn.Linear(D_MODEL, D_MODEL)
        self.parent_bg = nn.Linear(D_MODEL, 1)
        self.parent_edge = nn.Sequential(nn.Linear(6, 16), nn.GELU(), nn.Linear(16, 1))

    def eta(self):        # after training, this scalar is an estimate of eta
        return F.softplus(self.eta_raw)

    def beta(self):
        return F.softplus(self.beta_raw)

    def kappa(self):
        return self.log_kappa.exp().clamp(1e-2, 400.0)

    def lags(self, batch):
        t = torch.cumsum(batch["dt"], dim=1)
        lag = (t.unsqueeze(-1) - t.unsqueeze(-2)).clamp_min(0.0)
        L = lag.shape[-1]
        past = torch.tril(torch.ones(L, L, dtype=torch.bool)) & batch["valid"].unsqueeze(1)
        return lag, past

    def encode(self, batch):
        lag, past = self.lags(batch)
        excitation = torch.stack([(torch.exp(-s * lag) * past).sum(-1) for s in self.scales], dim=-1)
        log_dt = batch["dt"].clamp_min(1e-6).log()
        features = torch.cat(
            [sinusoidal(log_dt), log_dt.unsqueeze(-1), batch["r"].clamp_min(1e-8).log().unsqueeze(-1),
             batch["W"], torch.log1p(excitation)],
            dim=-1,
        )
        tokens = self.norm(self.embed(features))
        L = tokens.shape[1]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
        return self.encoder(tokens, mask=causal, src_key_padding_mask=~batch["valid"])

    def pending(self, batch):
        lag, past = self.lags(batch)
        return (torch.exp(-self.beta() * lag) * past).sum(-1)   # A_i = sum_j exp(-beta (t_i - t_j))

    def cont_prob(self, batch):   # P(at least one child still to come) = 1 - exp(-eta A_i)
        return (1.0 - torch.exp(-self.eta() * self.pending(batch))).clamp(1e-6, 1.0 - 1e-6)

    def time_logpdf(self, batch, dt_next):   # when the first of those pending children arrives
        eta, beta = self.eta(), self.beta()
        A = self.pending(batch).clamp_min(1e-8)
        u = dt_next.clamp_min(1e-6)
        one_minus = -torch.expm1(-beta * u)
        return (torch.log(eta * beta * A) - beta * u - eta * A * one_minus
                - torch.log(-torch.expm1(-eta * A)).clamp(min=-30.0))

    def next_parent_logq(self, h, batch, t_next):   # the anchor weights: which earlier event to imitate
        scores = torch.matmul(self.next_q(h) / np.sqrt(D_MODEL), self.next_k(h).transpose(1, 2))
        t = torch.cumsum(batch["dt"], dim=1)
        lag = (t_next.unsqueeze(-1) - t.unsqueeze(-2)).clamp_min(1e-6)
        scores = scores + self.next_edge(torch.stack([lag.clamp(max=20.0), lag.log()], dim=-1)).squeeze(-1)
        L = h.shape[1]
        scores = scores.masked_fill(~torch.tril(torch.ones(L, L, dtype=torch.bool)), float("-inf"))
        scores = scores.masked_fill(~batch["valid"].unsqueeze(1), float("-inf"))
        return F.log_softmax(scores, dim=-1)

    def angular_logpdf(self, batch, w_next, log_q):   # one vMF per candidate parent, mixed by q
        kappa = self.kappa()
        cos = torch.matmul(w_next, batch["W"].transpose(1, 2))
        log_vmf = torch.log(kappa) - np.log(2.0 * np.pi) - torch.log(-torch.expm1(-2.0 * kappa)) + kappa * (cos - 1.0)
        return torch.logsumexp(log_vmf + log_q, dim=-1)

    def u_q(self, W):
        w1, w2, w3 = W[..., 0], W[..., 1], W[..., 2]
        phi = torch.stack([torch.ones_like(w1), w1, w2, w3, w1 * w2, w1 * w3, w2 * w3,
                           w1 ** 2 - w3 ** 2, w2 ** 2 - w3 ** 2], dim=-1)
        return torch.exp(phi @ self.u_q_theta)

    def radial_logpdf(self, w_next, r_next):   # truncated Gamma; the cut is the fitted surface, not a free parameter
        g = self.gauge(w_next).clamp_min(1e-8)
        s = (g * r_next).clamp_min(1e-12)
        cut = self.C_q.expand_as(s) if self.u_q_theta is None else g * self.u_q(w_next)
        log_survival = torch.log(torch.special.gammaincc(self.alpha.expand_as(s), cut).clamp_min(1e-30))
        logp = torch.log(g) + (self.alpha - 1.0) * torch.log(s) - s - torch.lgamma(self.alpha) - log_survival
        return logp, (cut - s).clamp_min(0.0)

    def parent_logits(self, h, batch):   # scores earlier events as parents; trained, never read back here
        scores = torch.matmul(self.parent_q(h) / np.sqrt(D_MODEL), self.parent_k(h).transpose(1, 2))
        lag, _ = self.lags(batch)
        lag = lag.clamp_min(1e-6)
        W = batch["W"]
        L = h.shape[1]
        pair = torch.cat(
            [lag.clamp(max=20.0).unsqueeze(-1), lag.log().unsqueeze(-1),
             torch.matmul(W, W.transpose(1, 2)).unsqueeze(-1), W.unsqueeze(2).expand(-1, -1, L, -1)],
            dim=-1,
        )
        scores = scores + self.parent_edge(pair).squeeze(-1)
        scores = scores.masked_fill(~torch.tril(torch.ones(L, L, dtype=torch.bool), diagonal=-1), float("-inf"))
        return torch.cat([self.parent_bg(h), scores], dim=-1)


# ---------------------------------------------------------------------------
# The loss and the fit
# ---------------------------------------------------------------------------


def loss_fn(model, batch):
    h = model.encode(batch)
    valid = batch["valid"]
    has_next = valid & torch.roll(valid, -1, dims=1)
    has_next[:, -1] = False
    n_next = has_next.sum().clamp_min(1)

    def shift(x):
        return torch.roll(x, -1, dims=1)

    terms = {}
    # ---- continuation and time ----
    bce = F.binary_cross_entropy(model.cont_prob(batch), has_next.float(), reduction="none")
    terms["cont"] = (bce * valid).sum() / valid.sum()
    terms["time"] = -(model.time_logpdf(batch, shift(batch["dt"])) * has_next).sum() / n_next
    # ---- direction: anchored vMF, anchor supervised by the recorded parent ----
    w_next = shift(batch["W"])
    t_next = shift(torch.cumsum(batch["dt"], dim=1))
    log_q = model.next_parent_logq(h, batch, t_next)
    terms["angular"] = -(model.angular_logpdf(batch, w_next, log_q) * has_next).sum() / n_next
    target_next = shift(batch["parent_local"]).clamp_min(0)
    terms["anchor"] = -(log_q.gather(-1, target_next.unsqueeze(-1)).squeeze(-1) * has_next).sum() / n_next
    # ---- radius: the gauge ----
    logp, shortfall = model.radial_logpdf(w_next, shift(batch["r"]).clamp_min(1e-8))
    terms["radial"] = -(logp * has_next).sum() / n_next
    terms["below"] = ((shortfall ** 2) * has_next).sum() / n_next
    terms["curvature"] = model.gauge.curvature()
    # ---- parent ----
    target = batch["parent_local"]
    is_child = valid & (target >= 0)
    ce = F.cross_entropy(model.parent_logits(h, batch).transpose(1, 2), (1 + target).clamp_min(0), reduction="none")
    terms["parent"] = (ce * is_child).sum() / is_child.sum().clamp_min(1)
    terms["total"] = terms["time"] + sum(LOSS_WEIGHTS[k] * terms[k] for k in LOSS_WEIGHTS)
    return terms


@torch.no_grad()
def evaluate(model, seqs, batch_size=256):
    model.eval()
    total, n_batches = {}, 0
    for s in range(0, len(seqs), batch_size):
        for k, v in loss_fn(model, collate(seqs[s:s + batch_size])).items():
            total[k] = total.get(k, 0.0) + float(v)
        n_batches += 1
    return {k: v / n_batches for k, v in total.items()}


def train(model, train_seqs, val_seqs, epochs, batch_size=128, lr=2e-3, seed=123):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gauge_params = list(model.gauge.parameters())
    fast_params = (list(model.parent_edge.parameters()) + list(model.next_edge.parameters())
                   + [model.log_kappa, model.eta_raw, model.beta_raw])
    fast_ids = {id(par) for par in gauge_params + fast_params}
    others = [par for par in model.parameters() if id(par) not in fast_ids]
    optimiser = torch.optim.Adam(
        [{"params": others, "lr": lr},
         {"params": gauge_params, "lr": 3.0 * lr},     # a global object fed a few hundred radii per step
         {"params": fast_params, "lr": 10.0 * lr}]     # Adam moves ~lr per step; the scalars must travel
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    history, best, best_state = [], np.inf, None
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_seqs))
        running = 0.0
        for s in range(0, len(order), batch_size):
            terms = loss_fn(model, collate([train_seqs[i] for i in order[s:s + batch_size]]))
            optimiser.zero_grad()
            terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            running += float(terms["total"])
        scheduler.step()
        val = evaluate(model, val_seqs)["total"]
        if val < best:
            best, best_state = val, {k: v.detach().clone() for k, v in model.state_dict().items()}
        history.append((epoch, running / (len(order) // batch_size + 1), val))
        if (epoch + 1) % 5 == 0:
            print(f"epoch {epoch + 1:3d}/{epochs} | train {history[-1][1]:.4f} | validation {val:.4f}")
    model.load_state_dict(best_state)
    return np.array(history)


def finetune_gauge(model, W, R, steps=1500, lr=3e-3, curvature=None, u_q=None):
    if curvature is None:
        curvature = LOSS_WEIGHTS["curvature"]
    W = torch.as_tensor(np.atleast_2d(W), dtype=torch.float32)
    R = torch.as_tensor(np.asarray(R), dtype=torch.float32)
    u_q = None if u_q is None else torch.as_tensor(np.asarray(u_q), dtype=torch.float32)
    optimiser = torch.optim.Adam(model.gauge.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=steps)
    for _ in range(steps):
        g = model.gauge(W).clamp_min(1e-8)
        s = (g * R).clamp_min(1e-12)
        cut = model.C_q.expand_as(s) if u_q is None else g * u_q
        log_survival = torch.log(torch.special.gammaincc(model.alpha.expand_as(s), cut).clamp_min(1e-30))
        nll = -(torch.log(g) + (model.alpha - 1.0) * torch.log(s) - s - torch.lgamma(model.alpha) - log_survival).mean()
        loss = nll + curvature * model.gauge.curvature() + 10.0 * ((cut - s).clamp_min(0.0) ** 2).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        scheduler.step()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def sample_dt(model, batch, position, v):
    eta, beta = float(model.eta()), float(model.beta())
    eta_A = eta * float(model.pending(batch)[0, position])
    fraction = min(max(-np.log(1.0 - v * (1.0 - np.exp(-eta_A))) / eta_A, 1e-12), 1.0 - 1e-9)
    return -np.log(1.0 - fraction) / beta   # inverse cdf of the first-arrival law


def sample_radius(model, w, u):
    w = torch.as_tensor(w, dtype=torch.float32)[None]
    rate = float(model.gauge(w))
    cut = float(model.C_q) if model.u_q_theta is None else rate * float(model.u_q(w)[0])
    lo = stats.gamma.cdf(cut, float(model.alpha))
    u = lo + u * (1.0 - lo)
    return float(stats.gamma.ppf(min(u, 1.0 - 1e-12), float(model.alpha)) / rate)


def append_event(seq, dt, r, w, parent):
    return {
        "dt": np.append(seq["dt"], max(dt, 1e-6)),
        "r": np.append(seq["r"], r),
        "W": np.vstack([seq["W"], w]),
        "parent_local": np.append(seq["parent_local"], parent),
    }


@torch.no_grad()
def sample_continuation(model, prefix, rng, max_len=256, record=False):
    model.eval()
    seq = {k: np.array(v, copy=True) for k, v in prefix.items()}
    steps = []
    while len(seq["dt"]) < max_len:
        batch = collate([seq])
        h = model.encode(batch)
        n = len(seq["dt"])
        if rng.random() > float(model.cont_prob(batch)[0, n - 1]):
            break
        dt_new = sample_dt(model, batch, n - 1, rng.random())
        t_new = float(np.sum(seq["dt"])) + dt_new
        log_q = model.next_parent_logq(h, batch, torch.full((1, n), t_new))[0, n - 1, :n]
        q_weights = np.exp(log_q.numpy())
        parent_new = int(rng.choice(n, p=q_weights))   # the sampled anchor IS the parent: the genealogy comes free
        w_new = sample_vmf(seq["W"][parent_new][None], float(model.kappa()), rng)[0]
        r_new = sample_radius(model, w_new, rng.random())
        steps.append({"q": q_weights, "anchor": parent_new, "w": w_new, "r": r_new, "dt": dt_new})
        seq = append_event(seq, dt_new, r_new, w_new, parent_new)
    return (seq, steps) if record else seq


def generate_cascades(model, W0, R0, rng):
    out = []
    for w0, r0 in zip(W0, R0):
        prefix = {"dt": np.zeros(1), "r": np.array([r0]), "W": np.asarray(w0)[None],
                  "parent_local": np.array([-1])}
        out.append(sample_continuation(model, prefix, rng))
    return out


def summaries(seqs):
    rows = []
    for s in seqs:
        t = np.cumsum(s["dt"])
        par = s["parent_local"]
        depth = np.zeros(len(t), dtype=int)
        for i in range(1, len(t)):
            depth[i] = depth[par[i]] + 1 if par[i] >= 0 else 0
        rows.append({"size": len(t), "duration": t[-1] - t[0], "depth": depth.max(),
                     "max_r": np.max(s["r"])})
    return pd.DataFrame(rows)


def animate_generation(model, seq, steps, surface, labels=("$x_1$", "$x_2$", "$x_3$"), interval=800):
    t_gen = np.cumsum(seq["dt"])
    r_gen = seq["r"]
    X_gen = r_gen[:, None] * seq["W"]
    par_gen = seq["parent_local"]
    n_gen = len(t_gen)
    y_tree = tree_layout(par_gen)
    eta_gen, beta_gen = float(model.eta()), float(model.beta())
    grid_t = np.linspace(0.0, 1.06 * t_gen.max(), 500)

    def excitation(k, tg):    # eta~ beta~ sum_{i <= k} exp(-beta~ (t - T_i))
        lag = tg[:, None] - t_gen[None, :k + 1]
        return eta_gen * beta_gen * np.sum(np.exp(-beta_gen * np.clip(lag, 0.0, None)) * (lag >= 0), axis=1)

    exc_max = 1.15 * excitation(n_gen - 1, grid_t).max()
    lim = 1.1 * np.abs(X_gen).max()

    fig = plt.figure(figsize=(12.5, 5.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], hspace=0.42, wspace=0.16)
    ax_exc = fig.add_subplot(gs[0, 0])
    ax_tree = fig.add_subplot(gs[1, 0])
    ax_3d = fig.add_subplot(gs[:, 1], projection="3d")

    def update(frame):
        k = frame + 1                      # events visible: 0 .. k
        step = steps[frame]
        ax_exc.clear()
        ax_tree.clear()
        ax_3d.clear()

        sel = grid_t <= t_gen[k]
        ax_exc.plot(grid_t[sel], excitation(k, grid_t[sel]), color="C1", lw=1.6)
        ax_exc.scatter(t_gen[:k + 1], np.full(k + 1, 0.03 * exc_max), s=14 + 10 * r_gen[:k + 1],
                       color="C3", edgecolors="0.2", zorder=3)
        ax_exc.set_xlim(0, grid_t[-1])
        ax_exc.set_ylim(0, exc_max)
        ax_exc.set_ylabel("Excitation (events / day)")
        ax_exc.set_title(f"Event {k + 1}/{n_gen}: anchor = event {step['anchor'] + 1} "
                         f"(q = {step['q'][step['anchor']]:.2f}), size {step['r']:.1f} "
                         f"after {step['dt']:.1f} days")

        for i in range(1, k + 1):
            j = par_gen[i]
            ax_tree.plot([t_gen[j], t_gen[j], t_gen[i]], [y_tree[j], y_tree[i], y_tree[i]], color="0.5", lw=1.3)
        ax_tree.scatter(t_gen[:k + 1], y_tree[:k + 1], s=18 + 16 * r_gen[:k + 1],
                        color="C0", edgecolors="0.2", linewidths=0.6, zorder=3)
        ax_tree.set_yticks([])
        ax_tree.set_xlim(0, grid_t[-1])
        ax_tree.set_ylim(-0.8, y_tree.max() + 0.8)
        ax_tree.set_xlabel("Trading days since the first event")
        ax_tree.set_ylabel("The family tree")

        star_surface(ax_3d, surface, colour="0.55", alpha=0.28)
        ax_3d.scatter(X_gen[:k, 0], X_gen[:k, 1], X_gen[:k, 2], s=20 + 8 * r_gen[:k],
                      color="C3", alpha=0.8, linewidths=0, depthshade=False)
        ax_3d.scatter(*X_gen[k], s=130, color="C3", edgecolors="0.1", depthshade=False, zorder=5)
        ax_3d.set_xlim(-lim, lim); ax_3d.set_ylim(-lim, lim); ax_3d.set_zlim(-lim, lim)
        ax_3d.set_xlabel(labels[0]); ax_3d.set_ylabel(labels[1]); ax_3d.set_zlabel(labels[2])
        ax_3d.set_title("Marks, outside the fitted surface")
        return ()

    animation = FuncAnimation(fig, update, frames=n_gen - 1, interval=interval, repeat=False, blit=False)
    plt.close(fig)
    display(HTML(animation.to_jshtml()))
