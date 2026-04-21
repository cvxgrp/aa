"""Latency and accuracy benchmarks for the aa Python bindings.

Prints a table of per-configuration numbers; wired into CI after the
pytest suite so every build leaves a baseline in the logs. Wall times
on CI runners jitter from run to run — the counters (applies, aa_rej,
sg_rej) and final_err are the stable regression signals.

Problem: gradient descent on a diagonal convex quadratic
    (1/2) x^T Q x,  Q = diag(eigs),  eigs uniform in [1/cond, 1],
    F(x) = x - step * (Qdiag * x),  step = 1 / max(eigs) = 1,
    optimum at x = 0,  initial x_i = sin((i+1) * 0.1).

The map is intentionally cheap (O(dim)) so AA overhead shows up as a
meaningful fraction of wall time.

Sweeps:
    scan:dim        fixed mem, varies dim      (AA scaling in dim)
    scan:mem        fixed dim, varies mem      (set_m/gemv scaling in mem)
    scan:type       type-I vs type-II
    scan:cond       varies conditioning        (convergence pressure)
    relaxation      exercises x_work path
    near-optimum    long run past machine-precision convergence — S,Y
                    columns are denormal noise, M is catastrophically
                    ill-conditioned. Exercises the aa_reset fallback.
    noisy-floor     deterministic iter-dependent perturbation in F.
                    Iterates never converge — they bounce in a ball of
                    radius ~ noise_scale. This is the realistic
                    near-optimum regime (stochastic gradients,
                    approximate prox ops, fixed inner tolerance): the
                    ~1e-6 iterate-difference world real solvers live in.

Usage:
    python tests/python/bench.py
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

import aa


@dataclass
class BenchCfg:
    label: str
    dim: int
    mem: int
    type1: bool
    relaxation: float
    cond: float
    iters: int
    regularization: float
    noise_scale: float = 0.0


def _run_one(cfg: BenchCfg):
    n = cfg.dim
    eig_lo = 1.0 / cfg.cond
    Qdiag = np.linspace(eig_lo, 1.0, n)
    x = np.sin(np.arange(1, n + 1) * 0.1)
    x_prev = x.copy()
    step = 1.0  # = 1 / max(Qdiag)

    acc = aa.AndersonAccelerator(
        n, cfg.mem,
        type1=cfg.type1,
        regularization=cfg.regularization,
        relaxation=cfg.relaxation,
        safeguard_factor=2.0,
        max_weight_norm=1e10,
        verbosity=0,
    )

    aa_s = 0.0
    applies = aa_rej = sg_rej = 0
    t_total = time.perf_counter()
    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(cfg.iters):
            if i > 0:
                t0 = time.perf_counter()
                w = acc.apply(x, x_prev)
                aa_s += time.perf_counter() - t0
                if w > 0:
                    applies += 1
                else:
                    aa_rej += 1
            x_prev = x.copy()
            # F(xprev) with optional iter-dependent noise.
            x -= step * Qdiag * x_prev
            if cfg.noise_scale > 0:
                noise = np.sin((i * 131 + np.arange(1, n + 1)) * 0.07)
                x += cfg.noise_scale * noise
            t0 = time.perf_counter()
            sg = acc.safeguard(x, x_prev)
            aa_s += time.perf_counter() - t0
            if sg != 0:
                sg_rej += 1
    total_s = time.perf_counter() - t_total

    err = float(np.linalg.norm(x))
    aa_calls = applies + aa_rej
    us_per_apply = (aa_s * 1e6 / aa_calls) if aa_calls else 0.0
    aa_frac = (aa_s / total_s) if total_s > 0 else 0.0
    type_str = "I" if cfg.type1 else "II"

    print(
        f"{cfg.label:<20s} | {n:6d} {cfg.mem:4d}   {type_str:<2s}  "
        f"{cfg.relaxation:4.2f} | {cfg.iters:6d} | {err:10.3e} | "
        f"{applies:6d} {aa_rej:5d} {sg_rej:5d} | {total_s * 1e3:9.2f} | "
        f"{us_per_apply:9.2f} | {aa_frac * 100:5.1f}%"
    )


def _print_header(section):
    print(f"\n-- {section} --")
    print(
        f"{'label':<20s} | {'dim':>6s} {'mem':>4s} {'type':>4s} {'rlx':>4s} | "
        f"{'iters':>6s} | {'final_err':>10s} | "
        f"{'applied':>6s} {'a_rej':>5s} {'s_rej':>5s} | "
        f"{'total_ms':>9s} | {'us/call':>9s} | {'aa%':>6s}"
    )
    print("-" * 136)


def _warmup():
    """Amortize dynamic-linker / kernel-fault costs that would otherwise
    get charged to the first config."""
    acc = aa.AndersonAccelerator(50, 5, type1=False, regularization=1e-12)
    x = np.ones(50)
    x_prev = x.copy()
    for i in range(50):
        if i > 0:
            acc.apply(x, x_prev)
        x_prev = x.copy()
        x *= 0.9
        acc.safeguard(x, x_prev)


def main():
    print("\n=== AA Python benchmarks (GD on diagonal quadratic) ===")
    print("optimum is x=0; final_err is ||x|| after `iters` iterations.")
    _warmup()

    # scan:dim — cond=1e6 + iters=300 keeps every size making progress.
    _print_header("scan:dim (fixed mem=10, type-II, cond=1e6)")
    for cfg in [
        BenchCfg("dim=10",      10,    10, False, 1.0, 1e6, 300, 1e-12),
        BenchCfg("dim=100",     100,   10, False, 1.0, 1e6, 300, 1e-12),
        BenchCfg("dim=1000",    1000,  10, False, 1.0, 1e6, 300, 1e-12),
        BenchCfg("dim=5000",    5000,  10, False, 1.0, 1e6, 300, 1e-12),
        BenchCfg("dim=20000",   20000, 10, False, 1.0, 1e6, 300, 1e-12),
    ]:
        _run_one(cfg)

    _print_header("scan:mem (fixed dim=500, type-II, cond=1e4)")
    for cfg in [
        BenchCfg("mem=1",       500, 1,   False, 1.0, 1e4, 500, 1e-12),
        BenchCfg("mem=5",       500, 5,   False, 1.0, 1e4, 500, 1e-12),
        BenchCfg("mem=10",      500, 10,  False, 1.0, 1e4, 500, 1e-12),
        BenchCfg("mem=20",      500, 20,  False, 1.0, 1e4, 500, 1e-12),
        BenchCfg("mem=50",      500, 50,  False, 1.0, 1e4, 500, 1e-12),
        BenchCfg("mem=100",     500, 100, False, 1.0, 1e4, 500, 1e-12),
    ]:
        _run_one(cfg)

    _print_header("scan:type (fixed dim=500, mem=10, cond=1e4)")
    for cfg in [
        BenchCfg("type-I  reg=1e-8",  500, 10, True,  1.0, 1e4, 1000, 1e-8),
        BenchCfg("type-I  reg=1e-12", 500, 10, True,  1.0, 1e4, 1000, 1e-12),
        BenchCfg("type-II reg=1e-12", 500, 10, False, 1.0, 1e4, 1000, 1e-12),
        BenchCfg("type-II reg=0",     500, 10, False, 1.0, 1e4, 1000, 0.0),
    ]:
        _run_one(cfg)

    _print_header("scan:cond (fixed dim=500, mem=10, type-II)")
    for cfg in [
        BenchCfg("cond=10",    500, 10, False, 1.0, 10,    500,  1e-12),
        BenchCfg("cond=100",   500, 10, False, 1.0, 100,   500,  1e-12),
        BenchCfg("cond=1e4",   500, 10, False, 1.0, 1e4,   500,  1e-12),
        BenchCfg("cond=1e6",   500, 10, False, 1.0, 1e6,   2000, 1e-12),
        BenchCfg("cond=1e8",   500, 10, False, 1.0, 1e8,   2000, 1e-12),
    ]:
        _run_one(cfg)

    _print_header("relaxation (fixed dim=500, mem=10, cond=1e4)")
    for cfg in [
        BenchCfg("relax=1.0 (none)",  500, 10, False, 1.0,  1e4, 1000, 1e-12),
        BenchCfg("relax=0.95",        500, 10, False, 0.95, 1e4, 1000, 1e-12),
        BenchCfg("relax=1.2 (over)",  500, 10, False, 1.2,  1e4, 1000, 1e-12),
    ]:
        _run_one(cfg)

    _print_header("near-optimum: long runs past machine-precision convergence")
    for cfg in [
        BenchCfg("saturate dim=50",   50,  5,  False, 1.0, 10, 2000, 1e-12),
        BenchCfg("saturate dim=200",  200, 10, False, 1.0, 10, 2000, 1e-12),
        BenchCfg("mem=20 tail",       200, 20, False, 1.0, 10, 2000, 1e-12),
        BenchCfg("mem=50 tail",       200, 50, False, 1.0, 10, 2000, 1e-12),
        BenchCfg("type-I saturate",   200, 10, True,  1.0, 10, 2000, 1e-8),
        BenchCfg("no-reg saturate",   200, 10, False, 1.0, 10, 2000, 0.0),
    ]:
        _run_one(cfg)

    _print_header(
        "noisy-floor: realistic near-optimum regime, iter diffs ~ noise_scale"
    )
    for cfg in [
        # noise_scale = 1e-4: moderate.
        BenchCfg("noise=1e-4 type-II", 200, 10, False, 1.0, 10, 2000, 1e-12, 1e-4),
        BenchCfg("noise=1e-4 type-I",  200, 10, True,  1.0, 10, 2000, 1e-8,  1e-4),
        # noise_scale = 1e-6: the regime the user pointed out — iterate
        # differences of O(1e-6) and Gram-matrix signal of the same order.
        BenchCfg("noise=1e-6 type-II", 200, 10, False, 1.0, 10, 2000, 1e-12, 1e-6),
        BenchCfg("noise=1e-6 type-I",  200, 10, True,  1.0, 10, 2000, 1e-8,  1e-6),
        BenchCfg("noise=1e-6 mem=20",  200, 20, False, 1.0, 10, 2000, 1e-12, 1e-6),
        # noise_scale = 1e-8: tight stress test.
        BenchCfg("noise=1e-8 type-II", 200, 10, False, 1.0, 10, 2000, 1e-12, 1e-8),
        BenchCfg("noise=1e-8 type-I",  200, 10, True,  1.0, 10, 2000, 1e-8,  1e-8),
        # Wider problems in the noisy regime.
        BenchCfg("noisy wide dim=2000",2000,10, False, 1.0, 10, 1000, 1e-12, 1e-6),
        # Hostile: no regularization + noise — many rejections expected.
        BenchCfg("noisy no-reg",       200, 10, False, 1.0, 10, 2000, 0.0,   1e-6),
    ]:
        _run_one(cfg)

    print()


if __name__ == "__main__":
    main()
