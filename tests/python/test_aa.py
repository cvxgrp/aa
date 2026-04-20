"""Tests for the aa Python bindings.

Install the extension in editable mode from the repo root
(``pip install -e .``) then run:

    pytest tests/python/test_aa.py
"""

import numpy as np
import pytest

import aa


DIM = 20
MEM = 5


def _make_quadratic(dim, seed=0):
    """Build a well-conditioned diagonal quadratic (1/2) x' Q x - q' x."""
    rng = np.random.default_rng(seed)
    eigs = np.linspace(0.1, 1.0, dim)
    Q = np.diag(eigs)
    q = rng.standard_normal(dim)
    x_star = q / eigs
    step = 1.0 / eigs.max()
    return Q, q, x_star, step


def _run_gd(Q, q, x0, step, steps, accelerator=None):
    x = x0.copy()
    x_prev = x.copy()
    for i in range(steps):
        if accelerator is not None and i > 0:
            accelerator.apply(x, x_prev)
        x_prev = x.copy()
        x -= step * (Q @ x - q)
        if accelerator is not None:
            accelerator.safeguard(x, x_prev)
    return x


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construct_defaults():
    w = aa.AndersonAccelerator(DIM, MEM)
    assert isinstance(w, aa.AndersonAccelerator)


def test_construct_type1():
    w = aa.AndersonAccelerator(DIM, MEM, type1=True, regularization=1e-8)
    assert isinstance(w, aa.AndersonAccelerator)


def test_construct_all_kwargs():
    w = aa.AndersonAccelerator(
        DIM, MEM,
        type1=True,
        regularization=1e-8,
        relaxation=0.9,
        safeguard_factor=1.5,
        max_weight_norm=1e4,
        verbosity=0,
    )
    assert isinstance(w, aa.AndersonAccelerator)


def test_construct_mem_zero():
    """mem=0 should be accepted and behave as a no-op accelerator."""
    w = aa.AndersonAccelerator(DIM, 0)
    f = np.ones(DIM)
    x = np.zeros(DIM)
    f_before = f.copy()
    w.apply(f, x)
    np.testing.assert_array_equal(f, f_before)


def test_construct_dim_one():
    w = aa.AndersonAccelerator(1, MEM)
    f = np.array([1.0])
    x = np.array([0.0])
    w.apply(f, x)
    w.safeguard(f, x)


def test_dim_one_accelerates():
    """End-to-end exercise at dim=1 — the 1-D quadratic f(x) = 0.5 x^2 - x."""
    w = aa.AndersonAccelerator(1, MEM, type1=True, regularization=1e-8)
    x = np.array([5.0])
    step = 1.0
    for i in range(20):
        if i > 0:
            w.apply(x, x_prev)
        x_prev = x.copy()
        x -= step * (x - 1.0)
        w.safeguard(x, x_prev)
    assert np.isfinite(x).all()


# ---------------------------------------------------------------------------
# apply / safeguard basic semantics
# ---------------------------------------------------------------------------


def test_first_apply_is_noop():
    w = aa.AndersonAccelerator(DIM, MEM)
    f = np.arange(DIM, dtype=np.float64)
    x = np.zeros(DIM)
    f_before = f.copy()
    w.apply(f, x)
    np.testing.assert_array_equal(f, f_before)


def test_apply_returns_float():
    w = aa.AndersonAccelerator(DIM, MEM)
    f = np.ones(DIM)
    x = np.zeros(DIM)
    result = w.apply(f, x)
    assert isinstance(result, float)


def test_safeguard_returns_int_like():
    w = aa.AndersonAccelerator(DIM, MEM)
    f = np.ones(DIM)
    x = np.zeros(DIM)
    result = w.safeguard(f, x)
    assert result in (0, -1)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_wrong_shape_rejected():
    w = aa.AndersonAccelerator(DIM, MEM)
    with pytest.raises(ValueError):
        w.apply(np.zeros(DIM + 1), np.zeros(DIM + 1))


def test_mismatched_shape_rejected():
    w = aa.AndersonAccelerator(DIM, MEM)
    with pytest.raises(ValueError):
        w.apply(np.zeros(DIM), np.zeros(DIM - 1))


def test_accepts_column_vector():
    """A (dim, 1) array should be squeezed and accepted."""
    w = aa.AndersonAccelerator(DIM, MEM)
    f = np.zeros((DIM, 1))
    x = np.zeros((DIM, 1))
    w.apply(f, x)


def test_accepts_row_vector():
    w = aa.AndersonAccelerator(DIM, MEM)
    f = np.zeros((1, DIM))
    x = np.zeros((1, DIM))
    w.apply(f, x)


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


def test_reset_restores_first_apply_noop():
    """After reset(), the next apply should again be a no-op."""
    Q, q, _, step = _make_quadratic(DIM)
    w = aa.AndersonAccelerator(DIM, MEM)
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal(DIM)
    _run_gd(Q, q, x0, step, steps=20, accelerator=w)

    w.reset()
    f = np.arange(DIM, dtype=np.float64)
    x = np.zeros(DIM)
    f_before = f.copy()
    w.apply(f, x)
    np.testing.assert_array_equal(f, f_before)


# ---------------------------------------------------------------------------
# Integration: AA should accelerate plain gradient descent.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("type1", [True, False])
def test_aa_accelerates_gd(type1):
    Q, q, x_star, step = _make_quadratic(DIM, seed=1)
    rng = np.random.default_rng(2)
    x0 = rng.standard_normal(DIM)

    steps = 200
    reg = 1e-8 if type1 else 1e-12
    w = aa.AndersonAccelerator(DIM, MEM, type1=type1, regularization=reg)

    with np.errstate(over="ignore", invalid="ignore"):
        x_aa = _run_gd(Q, q, x0, step, steps, accelerator=w)
    x_plain = _run_gd(Q, q, x0, step, steps, accelerator=None)

    err_aa = np.linalg.norm(x_aa - x_star)
    err_plain = np.linalg.norm(x_plain - x_star)
    assert err_aa < err_plain, (
        f"AA did not improve on plain GD (aa={err_aa}, plain={err_plain})"
    )
    # Loose absolute target — mostly to catch catastrophic regressions.
    assert err_aa < 1e-3


def test_mem_capped_to_dim():
    """mem > dim should be silently capped; apply still works."""
    Q, q, x_star, step = _make_quadratic(DIM, seed=3)
    rng = np.random.default_rng(4)
    x0 = rng.standard_normal(DIM)

    w = aa.AndersonAccelerator(DIM, DIM * 4, type1=True, regularization=1e-8)
    with np.errstate(over="ignore", invalid="ignore"):
        x = _run_gd(Q, q, x0, step, 200, accelerator=w)
    assert np.all(np.isfinite(x))
    assert np.linalg.norm(x - x_star) < 1e-2


# ---------------------------------------------------------------------------
# Many-workspace safety: aa_finish runs under __dealloc__.
# ---------------------------------------------------------------------------


def test_create_and_destroy_many():
    for _ in range(64):
        w = aa.AndersonAccelerator(DIM, MEM)
        del w

def test_readonly_arrays_rejected():
    accel = aa.AndersonAccelerator(dim=2, mem=5)
    f = np.array([1.0, 2.0])
    x = np.array([0.0, 0.0])
    
    f_ro = f.copy()
    f_ro.flags.writeable = False
    with pytest.raises(ValueError, match="writeable"):
        accel.apply(f_ro, x)

    x_ro = x.copy()
    x_ro.flags.writeable = False
    with pytest.raises(ValueError, match="writeable"):
        accel.apply(f, x_ro)

    with pytest.raises(ValueError, match="writeable"):
        accel.safeguard(f_ro, x)

    with pytest.raises(ValueError, match="writeable"):
        accel.safeguard(f, x_ro)

def test_safeguard_rejection_restores_arrays():
    accel = aa.AndersonAccelerator(dim=2, mem=2, safeguard_factor=0.01)
    
    # Simple linear contraction to ensure non-singular AA matrix
    x = np.array([1.0, 1.0])
    f = 0.5 * x
    accel.apply(f, x)
    
    x = f.copy()
    f = 0.5 * x
    accel.apply(f, x)
    
    x = f.copy()
    f = 0.5 * x
    accel.apply(f, x)

    # After a successful solve, we pass in a massive step
    f_new = np.array([100.0, 100.0])
    x_new = np.array([-100.0, -100.0])

    rejected = accel.safeguard(f_new, x_new)
    
    assert rejected == -1
    # Safeguard restores the last known good f and x.
    assert not np.allclose(f_new, [100.0, 100.0])
    assert not np.allclose(x_new, [-100.0, -100.0])
