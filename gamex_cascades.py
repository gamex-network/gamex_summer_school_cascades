"""Machinery for GAMEX Practicals 4 and 5.

The notebooks state the models and read the estimates; the plumbing lives
here.  Nothing below is specific to one practical: Practical 4 builds a
`Process` from a known truth, Practical 5 builds one from fitted estimates.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
from scipy import optimize, special, stats
from scipy.spatial import ConvexHull

# ---------------------------------------------------------------------------
# Geometry: the gauge and its surfaces
# ---------------------------------------------------------------------------


def lp_gauge(x, p, b_pos, b_neg):   # small in the directions where the law reaches far
    x = np.atleast_2d(x)
    term = np.clip(x, 0, None) / b_pos + np.clip(-x, 0, None) / b_neg
    return (term ** p).sum(axis=-1) ** (1.0 / p)


def sphere_grid(n):
    i = np.arange(n)
    z = 1.0 - 2.0 * (i + 0.5) / n
    s = np.sqrt(1.0 - z ** 2)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    return np.column_stack([s * np.cos(golden * i), s * np.sin(golden * i), z])


def star_surface(ax, gauge, colour="0.5", alpha=0.3, n=900):
    W = sphere_grid(n)
    P = W / np.asarray(gauge(W))[:, None]
    ax.plot_trisurf(
        P[:, 0], P[:, 1], P[:, 2],
        triangles=ConvexHull(W).simplices,
        color=colour, alpha=alpha, linewidth=0.05, edgecolor="none",
    )


def surface_of(radius):
    return lambda W: 1.0 / np.asarray(radius(W))   # the gauge whose unit ball has that radius


# ---------------------------------------------------------------------------
# Directions: the von Mises-Fisher law
# ---------------------------------------------------------------------------


def sample_vmf(mu, kappa, rng):
    mu = np.atleast_2d(mu)
    n = len(mu)
    u = rng.random(n)
    t = 1.0 + np.log(u + (1.0 - u) * np.exp(-2.0 * kappa)) / kappa
    t = np.clip(t, -1.0, 1.0)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    helper = np.where(np.abs(mu[:, :1]) < 0.9, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    v1 = np.cross(mu, helper)
    v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
    v2 = np.cross(mu, v1)
    s = np.sqrt(np.clip(1.0 - t ** 2, 0.0, None))
    w = t[:, None] * mu + s[:, None] * (np.cos(phi)[:, None] * v1 + np.sin(phi)[:, None] * v2)
    return w / np.linalg.norm(w, axis=1, keepdims=True)


def vmf_logpdf(cos, kappa):
    return (np.log(kappa) - np.log(2.0 * np.pi) - np.log1p(-np.exp(-2.0 * kappa))
            + kappa * (cos - 1.0))


def vertex_directions(pi=(0.5, 0.3, 0.2), kappa=12.0, p_pos=0.8):   # where a family starts
    def draw(n, rng):
        j = rng.choice(3, size=n, p=pi)
        sign = np.where(rng.random(n) < p_pos, 1.0, -1.0)
        return sample_vmf(sign[:, None] * np.eye(3)[j], kappa, rng)
    return draw


def pool_directions(W_pool):   # on real data, start families where the data started them
    def draw(n, rng):
        return W_pool[rng.integers(0, len(W_pool), size=n)]
    return draw


# ---------------------------------------------------------------------------
# The marked Hawkes process of extremes
# ---------------------------------------------------------------------------


class Process:
    def __init__(self, mu, eta, beta, kappa, alpha, gauge, immigrants, q=None, u_q=None):
        self.mu, self.eta, self.beta, self.kappa = mu, eta, beta, kappa
        self.alpha, self.gauge, self.immigrants = alpha, gauge, immigrants
        self.C_q = None if q is None else stats.gamma.ppf(q, alpha)
        self._u_q = u_q

    def u_q(self, W):   # the q-quantile in direction W; a fitted one wins over C_q / g(w)
        return self._u_q(W) if self._u_q is not None else self.C_q / self.gauge(W)

    def sample_radii(self, W, rng):   # every draw lands above its own quantile: extreme by construction
        scale = 1.0 / self.gauge(W)
        lo = stats.gamma.cdf(self.u_q(W), self.alpha, scale=scale)   # mass below the quantile
        u = rng.uniform(lo, 1.0)
        return stats.gamma.ppf(u, self.alpha, scale=scale)

    def sample_immigrants(self, n, rng):
        W = self.immigrants(n, rng)
        return W, self.sample_radii(W, rng)

    def intensity(self, t_grid, t_events):
        lag = np.asarray(t_grid)[:, None] - np.asarray(t_events)[None, :]
        return self.mu + self.eta * self.beta * np.sum(
            np.exp(-self.beta * np.clip(lag, 0.0, None)) * (lag > 0), axis=1)

    def simulate(self, T, rng):
        n_imm = rng.poisson(self.mu * T)
        t_imm = np.sort(rng.uniform(0.0, T, size=n_imm))
        W_imm, R_imm = self.sample_immigrants(n_imm, rng)
        rows = []
        for c in range(n_imm):   # one family at a time, and each event queues its own children
            queue = [(t_imm[c], W_imm[c], R_imm[c], -1, 0)]      # (t, w, r, parent key, generation)
            while queue:
                t0, w0, r0, parent_key, gen = queue.pop(0)
                key = len(rows)
                rows.append((t0, r0, w0[0], w0[1], w0[2], parent_key, key, c, gen))
                if len(rows) > 200_000:
                    raise RuntimeError("the cascades do not terminate: eta must be < 1")
                n_child = rng.poisson(self.eta)
                if n_child:
                    lags = rng.exponential(1.0 / self.beta, size=n_child)
                    W_child = sample_vmf(np.tile(w0, (n_child, 1)), self.kappa, rng)
                    R_child = self.sample_radii(W_child, rng)
                    for lag, w, r in zip(lags, W_child, R_child):
                        queue.append((t0 + lag, w, r, key, gen + 1))
        events = pd.DataFrame(rows, columns=["t", "r", "w1", "w2", "w3", "parent_key", "key", "cascade", "gen"])
        events = events.sort_values("t", ignore_index=True)
        rank = {k: i + 1 for i, k in enumerate(events["key"])}   # re-number the events by time
        events["event"] = np.arange(1, len(events) + 1)
        events["parent"] = [rank.get(k, 0) for k in events["parent_key"]]
        for j in range(3):
            events[f"x{j + 1}"] = events["r"] * events[f"w{j + 1}"]
        return events.drop(columns=["key", "parent_key"])


def directions(events):
    return events[["w1", "w2", "w3"]].to_numpy()


def marks(events):
    return events[["x1", "x2", "x3"]].to_numpy()


# ---------------------------------------------------------------------------
# Genealogies as trees
# ---------------------------------------------------------------------------


def tree_layout(par):
    n = len(par)
    children = [[] for _ in range(n)]
    for i in range(1, n):
        if par[i] >= 0:
            children[par[i]].append(i)
    y = np.zeros(n)
    next_row = [0.0]

    def place(i):
        if not children[i]:
            y[i] = next_row[0]
            next_row[0] += 1.0
        else:
            for c in children[i]:
                place(c)
            y[i] = np.mean([y[c] for c in children[i]])

    place(0)
    return y


def local_parents(block):
    position = {e: i for i, e in enumerate(block["event"])}
    return np.array([position.get(par_id, -1) for par_id in block["parent"]])


def draw_tree(ax, t, r, par, y=None):
    if y is None:
        y = tree_layout(par)
    for i in range(1, len(t)):
        j = par[i]
        if j >= 0:
            ax.plot([t[j], t[j], t[i]], [y[j], y[i], y[i]], color="0.5", lw=1.3)
    ax.scatter(t, y, s=18 + 16 * np.asarray(r), color="C0", edgecolors="0.2", linewidths=0.6, zorder=3)
    ax.set_yticks([])
    return y


# ---------------------------------------------------------------------------
# Estimation: the times, the concentration, the gauge
# ---------------------------------------------------------------------------


def fit_em(t, T, window=50.0, max_iter=200, tol=1e-7):
    t = np.sort(np.asarray(t)[np.asarray(t) <= T])   # the likelihood is for observation on (0, T]
    n = len(t)
    start = np.searchsorted(t, t - window)   # candidate parents of event i: the earlier events within the window
    child = np.concatenate([np.full(i - s, i) for i, s in enumerate(start)])
    parent = np.concatenate([np.arange(s, i) for i, s in enumerate(start)])
    lag = t[child] - t[parent]
    mu_hat, eta_hat, beta_hat = 0.5 * n / T, 0.5, 1.0
    loglik = []
    for _ in range(max_iter):
        # ---- E-step: branching probabilities ----
        w = eta_hat * beta_hat * np.exp(-beta_hat * lag)
        denom = np.full(n, mu_hat)
        np.add.at(denom, child, w)
        p_pair = w / denom[child]
        p_imm = mu_hat / denom
        compensator = mu_hat * T + eta_hat * np.sum(1.0 - np.exp(-beta_hat * (T - t)))
        loglik.append(np.sum(np.log(denom)) - compensator)
        if len(loglik) > 1 and abs(loglik[-1] - loglik[-2]) < tol * (1.0 + abs(loglik[-2])):
            break
        # ---- M-step, with the offspring censored by the end of the window ----
        mu_hat = p_imm.sum() / T
        A = p_pair.sum()
        B = np.sum(p_pair * lag)
        s = T - t

        def eta_of(b):
            return A / np.sum(1.0 - np.exp(-b * s))

        def dQ(b):
            return A / b - B - eta_of(b) * np.sum(s * np.exp(-b * s))

        beta_hat = optimize.brentq(dQ, 1e-4, 1e4) if dQ(1e-4) > 0 > dQ(1e4) else A / B
        eta_hat = eta_of(beta_hat)
    return {
        "mu": mu_hat, "eta": eta_hat, "beta": beta_hat, "loglik": np.array(loglik),
        "child": child, "parent": parent, "p_pair": p_pair, "p_imm": p_imm,
    }


def hard_parents(em):
    best = np.zeros(len(em["p_imm"]), dtype=int)
    best_p = em["p_imm"].copy()
    for i, j, prob in zip(em["child"], em["parent"], em["p_pair"]):
        if prob >= best_p[i]:
            best_p[i], best[i] = prob, j + 1
    return best


def fit_gauge(W, R, alpha, C_q):
    log_norm = special.gammaln(alpha) + np.log(special.gammaincc(alpha, C_q))

    def unpack(theta):
        return 1.0 + np.exp(theta[0]), np.exp(theta[1:4]), np.exp(theta[4:7])

    def nll(theta):
        gW = lp_gauge(W, *unpack(theta))
        s = R * gW
        below = np.clip(C_q - s, 0.0, None)   # an event inside the fitted surface is penalised, not impossible
        return -np.sum(np.log(gW) + (alpha - 1.0) * np.log(s) - s - log_norm) + 1e3 * np.sum(below ** 2)

    res = optimize.minimize(nll, np.zeros(7), method="L-BFGS-B")
    p_hat, b_pos_hat, b_neg_hat = unpack(res.x)

    def g_hat(x):
        return lp_gauge(x, p_hat, b_pos_hat, b_neg_hat)

    return g_hat, p_hat, b_pos_hat, b_neg_hat


def fit_marked_em(events, T, outer=12):
    t_all = events["t"].to_numpy()
    keep = t_all <= T
    order = np.argsort(t_all[keep], kind="stable")
    W = directions(events[keep])[order]
    em = fit_em(t_all, T)
    use = em["p_pair"] > 1e-3   # candidate pairs with a non-negligible temporal posterior
    child, parent = em["child"][use], em["parent"][use]
    cos = np.sum(W[child] * W[parent], axis=1)
    n = len(em["p_imm"])

    def e_step(kappa_hat):
        log_pair = np.log(em["p_pair"][use]) + vmf_logpdf(cos, kappa_hat)
        log_imm = np.log(em["p_imm"]) - np.log(4.0 * np.pi)   # uniform reference for the immigrant option
        m = np.full(n, -np.inf)
        np.maximum.at(m, child, log_pair)
        m = np.maximum(m, log_imm)
        w = np.exp(log_pair - m[child])
        denom = np.exp(log_imm - m)
        np.add.at(denom, child, w)
        return w / denom[child], log_pair, log_imm

    kappa_hat = 5.0
    for _ in range(outer):
        w, _, _ = e_step(kappa_hat)
        rbar = min(np.sum(w * cos) / w.sum(), 0.96)   # caps kappa_hat near 25, where the approximation is reliable
        kappa_hat = rbar * (3.0 - rbar ** 2) / (1.0 - rbar ** 2)
    return kappa_hat


# ---------------------------------------------------------------------------
# Real data: margins, the angle-dependent quantile, exceedances, declustering
# ---------------------------------------------------------------------------


def laplace_from_train(z, split):
    is_train = z.index < split
    out = {}
    for col in z.columns:
        x = z[col].to_numpy()
        sorted_train = np.sort(x[is_train])
        n = len(sorted_train)
        u = np.interp(x, sorted_train, np.arange(1, n + 1) / (n + 1.0),
                      left=1.0 / (n + 1.0), right=n / (n + 1.0))
        out[col] = np.where(u < 0.5, np.log(2.0 * u), -np.log(2.0 * (1.0 - u)))
    return pd.DataFrame(out, index=z.index)


def spherical_features(W):
    W = np.atleast_2d(W)
    w1, w2, w3 = W[:, 0], W[:, 1], W[:, 2]
    return np.column_stack([np.ones(len(W)), w1, w2, w3, w1 * w2, w1 * w3, w2 * w3,
                            w1 ** 2 - w3 ** 2, w2 ** 2 - w3 ** 2])


def fit_quantile_threshold(R, W, q):
    y = np.log(R)
    Phi = spherical_features(W)
    n, k = Phi.shape
    # minimise  q 1'u + (1 - q) 1'v   subject to  Phi theta + u - v = y,  u, v >= 0
    c = np.concatenate([np.zeros(k), np.full(n, q), np.full(n, 1.0 - q)])
    A_eq = np.hstack([Phi, np.eye(n), -np.eye(n)])
    res = optimize.linprog(c, A_eq=A_eq, b_eq=y, bounds=[(None, None)] * k + [(0.0, None)] * (2 * n),
                           method="highs")
    theta = res.x[:k]

    def u_q(Wq):
        return np.exp(spherical_features(Wq) @ theta)

    return u_q, theta


def fit_gauge_threshold(W, R, u_q, alpha, p_max=6.0):
    def unpack(theta):
        return 1.0 + (p_max - 1.0) / (1.0 + np.exp(-theta[0])), np.exp(theta[1:4]), np.exp(theta[4:7])

    def nll(theta):
        gW = lp_gauge(W, *unpack(theta))
        s = R * gW
        survival = np.maximum(special.gammaincc(alpha, gW * u_q), 1e-300)
        return -np.sum(np.log(gW) + (alpha - 1.0) * np.log(s) - s - special.gammaln(alpha) - np.log(survival))

    res = optimize.minimize(nll, np.zeros(7), method="L-BFGS-B")
    p_hat, b_pos_hat, b_neg_hat = unpack(res.x)

    def g_hat(x):
        return lp_gauge(x, p_hat, b_pos_hat, b_neg_hat)

    return g_hat, p_hat


def exceedance_pit(gauge, W, R, u_q):
    return -np.expm1(-gauge(W) * (R - u_q(W)))


def branching_posterior(t, T, mu, eta, beta, window=50.0):
    t = np.sort(np.asarray(t)[np.asarray(t) <= T])
    n = len(t)
    start = np.searchsorted(t, t - window)
    child = np.concatenate([np.full(i - s, i) for i, s in enumerate(start)])
    parent = np.concatenate([np.arange(s, i) for i, s in enumerate(start)])
    w = eta * beta * np.exp(-beta * (t[child] - t[parent]))
    denom = np.full(n, mu)
    np.add.at(denom, child, w)
    return {"mu": mu, "eta": eta, "beta": beta, "child": child, "parent": parent,
            "p_pair": w / denom[child], "p_imm": mu / denom}


def sample_parents(em, rng):
    n = len(em["p_imm"])
    out = np.zeros(n, dtype=int)
    for i in range(n):
        mask = em["child"] == i
        candidates = np.concatenate([[0], em["parent"][mask] + 1])
        probs = np.concatenate([[em["p_imm"][i]], em["p_pair"][mask]])
        out[i] = rng.choice(candidates, p=probs / probs.sum())
    return out


def attributed_events(t_days, X, parent):
    n = len(t_days)
    R = np.linalg.norm(X, axis=1)
    cascade, gen = np.arange(n), np.zeros(n, dtype=int)
    for i in range(n):
        if parent[i] > 0:
            cascade[i] = cascade[parent[i] - 1]
            gen[i] = gen[parent[i] - 1] + 1
    events = pd.DataFrame({"event": np.arange(1, n + 1), "t": t_days, "r": R, "parent": parent,
                           "cascade": pd.factorize(cascade)[0], "gen": gen})
    for j in range(3):
        events[f"w{j + 1}"] = X[:, j] / R
        events[f"x{j + 1}"] = X[:, j]
    return events


# ---------------------------------------------------------------------------
# Animations.  Each builds the frames, closes the figure and displays it.
# ---------------------------------------------------------------------------


def _show(animation, fig):
    plt.close(fig)
    display(HTML(animation.to_jshtml()))


def animate_limit_set(X, gauge, n_values, interval=600):
    fig = plt.figure(figsize=(6, 5.4))
    ax = fig.add_subplot(projection="3d")

    def update(frame):
        n = n_values[frame]
        S = X[:n] / np.log(n / 2.0)
        outside = gauge(S) > 1.0
        ax.clear()
        star_surface(ax, gauge, colour="0.55", alpha=0.30)
        ax.scatter(S[~outside, 0], S[~outside, 1], S[~outside, 2], s=3, alpha=0.25, color="C0", linewidths=0)
        ax.scatter(S[outside, 0], S[outside, 1], S[outside, 2], s=14, alpha=0.9, color="C3",
                   linewidths=0, depthshade=False)
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_zlim(-1.5, 1.5)
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_zticks([-1, 0, 1])
        ax.set_title(rf"$n = {n:,}$, shrunk by $\log(n/2)$: {int(outside.sum())} days outside $G$")
        return ()

    _show(FuncAnimation(fig, update, frames=len(n_values), interval=interval, repeat=False, blit=False), fig)


def animate_process(process, events, T, surface, n_frames=16, interval=600):
    show = events[events["t"] <= T]
    t_e, X_e, r_e = show["t"].to_numpy(), marks(show), show["r"].to_numpy()
    t_grid = np.linspace(0.0, T, 700)
    lam = process.intensity(t_grid, t_e)
    lim = 1.1 * np.abs(X_e).max()
    norm = plt.Normalize(0.0, T)

    fig = plt.figure(figsize=(11.5, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.16)
    ax_rate = fig.add_subplot(gs[0, 0])
    ax_3d = fig.add_subplot(gs[0, 1], projection="3d")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax_3d,
                 shrink=0.65, pad=0.13, label=r"Time $t$")

    def update(frame):
        t_now = T * (frame + 1) / n_frames
        sel = t_e <= t_now
        ax_rate.clear()
        ax_3d.clear()

        grid = t_grid <= t_now
        ax_rate.plot(t_grid[grid], lam[grid], color="0.25", lw=1.0)
        ax_rate.scatter(t_e[sel], np.full(int(sel.sum()), -0.05 * lam.max()), marker="|", s=60,
                        c=t_e[sel], cmap="viridis", norm=norm)
        ax_rate.set_xlim(0, T)
        ax_rate.set_ylim(-0.1 * lam.max(), 1.05 * lam.max())
        ax_rate.set_xlabel(r"Time $t$")
        ax_rate.set_ylabel(r"$\lambda(t)$")
        ax_rate.set_title(rf"$t = {t_now:.1f}$: {int(sel.sum())} extreme events so far")

        star_surface(ax_3d, surface, colour="0.55", alpha=0.22)
        ax_3d.plot(X_e[sel, 0], X_e[sel, 1], X_e[sel, 2], color="0.6", lw=0.5, alpha=0.7)
        ax_3d.scatter(X_e[sel, 0], X_e[sel, 1], X_e[sel, 2], c=t_e[sel], cmap="viridis", norm=norm,
                      s=14 + 6 * r_e[sel], depthshade=False)
        ax_3d.set_xlim(-lim, lim); ax_3d.set_ylim(-lim, lim); ax_3d.set_zlim(-lim, lim)
        ax_3d.set_xlabel(r"$x_1$"); ax_3d.set_ylabel(r"$x_2$"); ax_3d.set_zlabel(r"$x_3$")
        ax_3d.set_title("The marks so far")
        return ()

    _show(FuncAnimation(fig, update, frames=n_frames, interval=interval, repeat=False, blit=False), fig)


def animate_genealogy(events, T, n_frames=20, interval=600):
    window = events[events["t"] <= T].reset_index(drop=True)
    par_row = local_parents(window)

    # ---- one lane of tree rows per cascade ----
    cas = window["cascade"].to_numpy()
    y = np.zeros(len(window))
    colour = np.empty(len(window), dtype=object)
    base = 0.0
    for k, c in enumerate(window.groupby("cascade")["t"].min().sort_values().index):
        rows = np.flatnonzero(cas == c)
        local = {r: j for j, r in enumerate(rows)}
        y[rows] = base + tree_layout(np.array([local.get(par_row[r], -1) for r in rows]))
        colour[rows] = f"C{k % 10}"
        base = y[rows].max() + 2.0

    t_w, r_w = window["t"].to_numpy(), window["r"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 6))

    def update(frame):
        t_now = T * (frame + 1) / n_frames
        sel = t_w <= t_now                          # a parent always arrives before its child
        ax.clear()
        for i in np.flatnonzero(sel):
            j = par_row[i]
            if j >= 0:
                ax.plot([t_w[j], t_w[j], t_w[i]], [y[j], y[i], y[i]], color=colour[i], lw=1.0, alpha=0.7)
        ax.scatter(t_w[sel], y[sel], s=12 + 8 * r_w[sel], c=list(colour[sel]), edgecolors="0.2",
                   linewidths=0.4, zorder=3)
        ax.set_xlim(0, T)
        ax.set_ylim(-2, base)
        ax.set_yticks([])
        ax.set_xlabel(r"Time $t$")
        ax.set_ylabel("One colour per family")
        ax.set_title(f"t = {t_now:4.1f}: {int(sel.sum())} events in {len(np.unique(cas[sel]))} families")
        return ()

    _show(FuncAnimation(fig, update, frames=n_frames, interval=interval, repeat=False, blit=False), fig)
