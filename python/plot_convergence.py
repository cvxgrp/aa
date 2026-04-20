"""Generate the convergence plot shown in the README.

Runs the quickstart example with three variants and saves a PNG:
    - Vanilla gradient descent (no acceleration)
    - AA Type-I + GD
    - AA Type-II + GD

Usage: python plot_convergence.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np

import aa


def run(variant, dim, mem, N, Q, q, step, x0):
    f_star = -0.5 * np.linalg.solve(Q, q) @ q  # min of 1/2 x'Qx - q'x
    f = lambda x: 0.5 * x @ Q @ x - q @ x

    x = x0.copy()
    x_prev = x.copy()
    gaps = []

    if variant == "none":
        acc = None
    elif variant == "type1":
        acc = aa.AndersonAccelerator(dim, mem, type1=True, regularization=1e-8)
    elif variant == "type2":
        acc = aa.AndersonAccelerator(dim, mem, type1=False, regularization=1e-12)
    else:
        raise ValueError(variant)

    for i in range(N):
        if acc is not None and i > 0:
            acc.apply(x, x_prev)
        x_prev = x.copy()
        x = x - step * (Q @ x_prev - q)
        if acc is not None:
            acc.safeguard(x, x_prev)
        gaps.append(max(f(x) - f_star, 1e-16))

    return np.array(gaps)


def main():
    dim, mem, N = 100, 10, 1000
    rng = np.random.default_rng(0)
    Qh = rng.standard_normal((dim, dim)) / np.sqrt(dim)
    Q = Qh.T @ Qh + 1e-3 * np.eye(dim)
    q = rng.standard_normal(dim)
    eigs = np.linalg.eigvalsh(Q)
    step = 2.0 / (eigs.min() + eigs.max())  # optimal GD step for a quadratic
    x0 = rng.standard_normal(dim)

    curves = {
        "Vanilla GD": run("none", dim, mem, N, Q, q, step, x0),
        "AA Type-I + GD": run("type1", dim, mem, N, Q, q, step, x0),
        "AA Type-II + GD": run("type2", dim, mem, N, Q, q, step, x0),
    }

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    styles = {
        "Vanilla GD": {"color": "#888888", "linestyle": "--", "linewidth": 2.0},
        "AA Type-I + GD": {"color": "#d62728", "linewidth": 2.0},
        "AA Type-II + GD": {"color": "#1f77b4", "linewidth": 2.0},
    }
    for label, gaps in curves.items():
        ax.semilogy(gaps, label=label, **styles[label])

    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$f(x_k) - f^\star$")
    ax.set_title("Gradient descent on a convex quadratic")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = os.path.join(os.path.dirname(__file__), "convergence.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
