#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core lottery validation on synthetic token trees.

What it does
------------
1) Builds synthetic token-level agent policies on a fixed B-ary tree of depth L.
   Each agent's next-token distribution is a softmax over (agent vector) · (state+token vector),
   giving multiplicative per-leaf utilities U_i^{prob}(X) = product of stepwise probs.

2) For a grid of polarization levels rho, solves:
     - Nash-welfare (NW) lottery p* := argmax_p sum_i log(U_i^T p)  (on simplex)
       via Frank–Wolfe with exact line search (1-D golden-section).
     - Baselines: uniform over leaves; utilitarian (argmax_j sum_i U_i(j)).

   For each lottery p and every coalition S ⊆ N (nonempty), solves a small LP to compute
     s_S^*(p) = max_{p' in simplex} min_{i in S} (|S|/n) * U_i^T p' / (U_i^T p),
   rewritten as a single LP with variable s:
     maximize s
     s.t.   U_i^T p' - (U_i^T p / alpha) * s >= 0   for all i in S,  alpha=|S|/n
            sum_j p'_j = 1,   p'_j >= 0
   Blocking exists if max_S s_S^*(p) > 1.

3) Plots the max coalition improvement factor across rho for NW vs. baselines, and
   prints a policy–lottery equivalence sanity check for one rho.

Produces
--------
- Figure: 'core_violation_plot.png'
- Console: total-variation distance between induced-policy rollouts and p* for one rho.

Dependencies
------------
numpy, matplotlib, scipy (optimize.linprog). Tested with SciPy HiGHS.
"""

import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.optimize import linprog


# ----------------------------
# Synthetic token-level setup
# ----------------------------


def generate_params(B, L, d, n_agents, seed=123):
    """Random token vectors v[t, a] and agent vectors w[i]. Unit-normalized."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(L, B, d))
    v /= np.linalg.norm(v, axis=2, keepdims=True) + 1e-12
    w = rng.normal(size=(n_agents, d))
    w /= np.linalg.norm(w, axis=1, keepdims=True) + 1e-12
    return v, w


def enumerate_leaves(B, L):
    """List of action tuples of length L."""
    return list(itertools.product(range(B), repeat=L))


def log_softmax_rows(M):
    """Row-wise log-softmax for a 2D array M (n x B)."""
    Mmax = M.max(axis=1, keepdims=True)
    S = M - Mmax
    return S - np.log(np.exp(S).sum(axis=1, keepdims=True))


def compute_utilities(v, w, rho):
    """
    Compute agent-by-leaf utilities U[i, j] = U_i^{prob}(X_j) > 0.
    v: (L,B,d), w: (n,d), rho: polarization.
    Returns U (n x m) and leaves list.
    """
    L, B, d = v.shape
    n = w.shape[0]
    leaves = enumerate_leaves(B, L)
    m = len(leaves)

    logu = np.zeros((n, m))
    for j, leaf in enumerate(leaves):
        z = np.zeros(d)  # running state embedding
        for t, a_t in enumerate(leaf):
            # logits for all actions at this state: (n x B)
            X = z[None, :] + v[t]  # (B, d)
            logits = rho * (w @ X.T)  # (n, B)
            ls = log_softmax_rows(logits)  # (n, B)
            logu[:, j] += ls[:, a_t]
            z = z + v[t, a_t]

    # Stabilize per-agent by subtracting max and exponentiate
    U = np.empty_like(logu)
    eps = 1e-300  # strictly positive
    for i in range(n):
        li = logu[i]
        li = li - li.max()
        U[i] = np.exp(li) + eps
    return U, leaves


# ----------------------------
# Nash welfare via Frank–Wolfe
# ----------------------------


def F_val(U, p):
    """Sum of logs: F(p) = sum_i log(U_i^T p)."""
    a = U @ p  # (n,)
    if np.any(a <= 0):
        return -np.inf
    return float(np.sum(np.log(a)))


def FW_nash_welfare(U, max_iters=1000, tol=1e-10):
    """
    Maximize F(p) = sum_i log(U_i^T p) over the simplex using Frank–Wolfe with
    1-D golden-section line search along segment to e_j with largest gradient.
    """
    n, m = U.shape
    p = np.ones(m) / m
    F_prev = F_val(U, p)

    for it in range(max_iters):
        a = U @ p  # (n,)
        g = (U / a[:, None]).sum(0)  # gradient in R^m
        j = int(np.argmax(g))
        b = U[:, j]  # column for leaf j

        # 1-D concave objective along gamma in [0,1]: F((1-gamma)p + gamma e_j)
        def F_gamma(gamma):
            return float(np.sum(np.log((1.0 - gamma) * a + gamma * b)))

        # Golden-section search for maximum on [0, 1]
        lo, hi = 0.0, 1.0
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
        Fc = F_gamma(c)
        Fd = F_gamma(d)
        for _ in range(80):
            if Fc < Fd:
                lo = c
                c = d
                Fc = Fd
                d = lo + gr * (hi - lo)
                Fd = F_gamma(d)
            else:
                hi = d
                d = c
                Fd = Fc
                c = hi - gr * (hi - lo)
                Fc = F_gamma(c)
            if hi - lo < 1e-9:
                break
        gamma_star = 0.5 * (lo + hi)
        p_new = (1.0 - gamma_star) * p
        p_new[j] += gamma_star

        F_new = F_val(U, p_new)
        if F_new - F_prev < tol:
            p = p_new
            break
        p = p_new
        F_prev = F_new

    return p


def egalitarian_lottery(U):
    """
    Compute egalitarian (maximin) lottery: argmax_p min_i (U_i^T p).
    Solved as LP:
      maximize t
      s.t.  U_i^T p >= t  for all i
            sum_j p_j = 1,  p_j >= 0
    """
    n, m = U.shape
    # Variables: [p_0, ..., p_{m-1}, t]
    # Maximize t => minimize -t
    c = np.zeros(m + 1)
    c[-1] = -1.0

    # Inequality: -U_i^T p + t <= 0  for all i  =>  U_i^T p - t >= 0
    A_ub = np.zeros((n, m + 1))
    A_ub[:, :m] = -U
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(n)

    # Equality: sum p_j = 1
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0])

    # Bounds: p_j in [0,1], t free
    bounds = [(0.0, 1.0)] * m + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')

    if res.success:
        return res.x[:m]
    else:
        # Fallback to uniform
        return np.ones(m) / m


# ----------------------------
# Coalition blocking metric
# ----------------------------


def max_coalition_improvement(U, p):
    """
    For lottery p, find the maximum α such that some coalition S can achieve:
      (|S|/N) · u_i(y) ≥ α · u_i(x) for all i in S
    with at least one strict inequality.

    This is equivalent to finding max α where coalition S can redistribute their
    |S|/N share to give everyone at least α times their original utility.

    Returns the maximum α across all coalitions. If α > 1, the lottery is blockable.
    """
    n, m = U.shape
    base = U @ p  # (n,)
    agents = range(n)
    max_alpha = 1.0

    for r in range(1, n + 1):
        budget = r / n  # |S|/N
        for S in itertools.combinations(agents, r):
            S_list = list(S)

            # Find max α such that all i in S can get α * base[i] utility
            # using budget share: maximize α
            # s.t. U_i^T p' >= α * base[i] for all i in S
            #      sum p'_j = budget, p'_j >= 0

            # Variables: [p'_0, ..., p'_{m-1}, α]
            # Maximize α => minimize -α
            c = np.zeros(m + 1)
            c[-1] = -1.0

            # Constraints: U_i^T p' - α * base[i] >= 0 => -U_i^T p' + α * base[i] <= 0
            A_ub = []
            b_ub = []
            for i in S_list:
                row = np.concatenate([-U[i], [base[i]]])
                A_ub.append(row)
                b_ub.append(0.0)

            A_ub = np.array(A_ub) if A_ub else None
            b_ub = np.array(b_ub) if b_ub else None

            # Equality: sum p'_j = budget
            A_eq = np.zeros((1, m + 1))
            A_eq[0, :m] = 1.0
            b_eq = np.array([budget])

            # Bounds: p'_j >= 0, α free
            bounds = [(0.0, None)] * m + [(None, None)]

            res = linprog(
                c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
            )

            if res.success:
                alpha_val = res.x[-1]
                if alpha_val > max_alpha:
                    max_alpha = float(alpha_val)

    return max_alpha


# ----------------------------
# Induced-policy sampling check
# ----------------------------


def build_prefix_index(leaves):
    """Map prefix tuple -> list of leaf indices that share that prefix."""
    prefix_to_idx = {}
    for j, leaf in enumerate(leaves):
        for t in range(len(leaf) + 1):
            pref = tuple(leaf[:t])
            prefix_to_idx.setdefault(pref, []).append(j)
    return prefix_to_idx


def induced_policy_rollout(p_star, leaves, B, L, num_samples=200_000, seed=7):
    """
    Sample leaves using the induced policy from p_star and compare frequencies to p_star.
    Returns empirical distribution and total variation distance TV = 0.5 ||p_hat - p_star||_1.
    """
    rng = np.random.default_rng(seed)
    prefix_to_idx = build_prefix_index(leaves)

    # Precompute masses for all prefixes
    mass = {}
    for pref, idxs in prefix_to_idx.items():
        mass[pref] = float(np.sum(p_star[idxs]))

    counts = np.zeros(len(leaves), dtype=np.int64)
    leaf_to_index = {leaf: j for j, leaf in enumerate(leaves)}

    for _ in range(num_samples):
        s = tuple()
        for t in range(L):
            denom = mass.get(s, 0.0)
            if denom <= 0.0:
                # Should not happen for prefixes with positive total mass, but guard anyway.
                # In that case, sample uniformly over enabled actions.
                probs = np.ones(B) / B
            else:
                probs = np.zeros(B)
                for a in range(B):
                    s_next = s + (a,)
                    probs[a] = mass.get(s_next, 0.0) / denom
            a = int(rng.choice(B, p=probs))
            s = s + (a,)
        counts[leaf_to_index[s]] += 1

    p_hat = counts / counts.sum()
    tv = 0.5 * np.sum(np.abs(p_hat - p_star))
    return p_hat, tv


# ----------------------------
# Main experiment
# ----------------------------


def main():
    # Tree and model sizes kept small for clarity/reproducibility
    B, L = 3, 4  # branching and depth -> m = B^L leaves
    d = 8  # embedding dimension
    n_agents = 6
    base_seed = 42
    n_runs = 3

    rho_grid = np.linspace(0.6, 5.0, 20)  # low to high polarization

    # Store results for averaging
    all_max_imp_NW = []
    all_max_imp_UTIL = []
    all_max_imp_UNIF = []

    print("Running polarization sweep and coalition-blocking checks...")
    for run in range(n_runs):
        print(f"\n=== Run {run + 1}/{n_runs} ===")
        # Generate different random params for each run
        v, w = generate_params(B, L, d, n_agents, seed=base_seed + run)

        max_imp_NW = []
        max_imp_UTIL = []
        max_imp_UNIF = []

        for rho in rho_grid:
            U, leaves = compute_utilities(v, w, rho)

            # Nash welfare lottery (Frank–Wolfe)
            p_nw = FW_nash_welfare(U, max_iters=1500, tol=1e-11)

            # Baselines
            m = U.shape[1]
            p_unif = np.ones(m) / m
            j_util = int(np.argmax(U.sum(axis=0)))
            p_util = np.zeros(m)
            p_util[j_util] = 1.0

            # Coalition blocking metric (max over all coalitions)
            max_imp_NW.append(max_coalition_improvement(U, p_nw))
            max_imp_UTIL.append(max_coalition_improvement(U, p_util))
            max_imp_UNIF.append(max_coalition_improvement(U, p_unif))

            print(
                f"  rho={rho:0.2f} | max α  NW={max_imp_NW[-1]:0.4f}  "
                f"UTIL={max_imp_UTIL[-1]:0.4f}  UNIF={max_imp_UNIF[-1]:0.4f}"
            )

        all_max_imp_NW.append(max_imp_NW)
        all_max_imp_UTIL.append(max_imp_UTIL)
        all_max_imp_UNIF.append(max_imp_UNIF)

    # Average and std across runs
    max_imp_NW = np.mean(all_max_imp_NW, axis=0)
    max_imp_UTIL = np.mean(all_max_imp_UTIL, axis=0)
    max_imp_UNIF = np.mean(all_max_imp_UNIF, axis=0)

    std_imp_NW = np.std(all_max_imp_NW, axis=0)
    std_imp_UTIL = np.std(all_max_imp_UTIL, axis=0)
    std_imp_UNIF = np.std(all_max_imp_UNIF, axis=0)

    print("\n=== Averaged Results ===")
    for i, rho in enumerate(rho_grid):
        print(
            f"  rho={rho:0.2f} | max α  NW={max_imp_NW[i]:0.4f}±{std_imp_NW[i]:0.4f}  "
            f"UTIL={max_imp_UTIL[i]:0.4f}±{std_imp_UTIL[i]:0.4f}  UNIF={max_imp_UNIF[i]:0.4f}±{std_imp_UNIF[i]:0.4f}"
        )

    # Plot: max coalition improvement factor over rho
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'font.serif': ['Linux Libertine', 'DejaVu Serif', 'Times New Roman']
    })
    fig, ax = plt.subplots(figsize=(3.33, 2.5))
    colors = plt.cm.tab10.colors
    ax.plot(rho_grid, max_imp_UTIL, color=colors[1], linewidth=3.5, label="Utilitarian Opt.")
    ax.fill_between(rho_grid, max_imp_UTIL - std_imp_UTIL, max_imp_UTIL + std_imp_UTIL,
                     color=colors[1], alpha=0.2)
    ax.plot(rho_grid, max_imp_UNIF, color=colors[4], linewidth=3.5, label="Uniform")
    ax.fill_between(rho_grid, max_imp_UNIF - std_imp_UNIF, max_imp_UNIF + std_imp_UNIF,
                     color=colors[4], alpha=0.2)
    ax.plot(rho_grid, max_imp_NW, color=colors[0], linewidth=3.5, label=r"$p^*$")
    ax.fill_between(rho_grid, max_imp_NW - std_imp_NW, max_imp_NW + std_imp_NW,
                     color=colors[0], alpha=0.2)
    ax.set_yscale('log')
    ax.set_xlabel(r"Polarization ($\rho$)")
    ax.set_ylabel(r"Max $\alpha$")
    ax.legend(frameon=False, loc='upper left', fontsize=10)
    ax.margins(0.08)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(width=2)
    plt.tight_layout()
    plt.savefig("core_violation_plot.png", dpi=600, bbox_inches="tight")
    print("Saved figure: core_violation_plot.png")

    # Optional: policy–lottery equivalence sanity check at a mid rho
    rho0 = float(rho_grid[len(rho_grid) // 2])
    U0, leaves0 = compute_utilities(v, w, rho0)
    p0 = FW_nash_welfare(U0, max_iters=1500, tol=1e-11)
    p_hat, tv = induced_policy_rollout(p0, leaves0, B, L, num_samples=200_000, seed=7)
    print(
        f"[Sanity] Induced-policy vs. target lottery at rho={rho0:0.2f}: TV distance = {tv:0.4f}"
    )


if __name__ == "__main__":
    main()
