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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(dim=0, mem=MEM), "dim must be positive"),
        (dict(dim=DIM, mem=-1), "mem must be non-negative"),
        (dict(dim=DIM, mem=MEM, regularization=float("nan")), "regularization must be finite"),
        (dict(dim=DIM, mem=MEM, regularization=float("inf")), "regularization must be finite"),
        (dict(dim=DIM, mem=MEM, relaxation=3.0), "relaxation must be in \\[0, 2\\]"),
        (dict(dim=DIM, mem=MEM, safeguard_factor=-1.0), "safeguard_factor must be non-negative"),
        (dict(dim=DIM, mem=MEM, max_weight_norm=0.0), "max_weight_norm must be positive"),
    ],
)
def test_construct_invalid_args_raise_value_error(kwargs, message):
    with pytest.raises(ValueError, match=message):
        aa.AndersonAccelerator(**kwargs)


def test_negative_regularization_pinned_mode():
    """Negative `regularization` selects the pinned branch: r is held fixed
    at |regularization| (no Frobenius scaling). Verify (a) construction and
    a convergence run succeed, and (b) the pinned branch is not a silent
    fallthrough to the scaled branch — same problem with reg=+R and reg=-R
    must produce different iterates (a sign-ignoring bug would give
    bit-identical output).
    """
    Q, q, x_star, step = _make_quadratic(DIM, seed=3)
    rng = np.random.default_rng(7)
    x0 = rng.standard_normal(DIM)

    acc_scaled = aa.AndersonAccelerator(DIM, MEM, type1=True, regularization=+1e-3)
    acc_pinned = aa.AndersonAccelerator(DIM, MEM, type1=True, regularization=-1e-3)
    x_scaled = _run_gd(Q, q, x0, step, steps=20, accelerator=acc_scaled)
    x_pinned = _run_gd(Q, q, x0, step, steps=20, accelerator=acc_pinned)

    assert np.isfinite(x_scaled).all()
    assert np.isfinite(x_pinned).all()
    diff = float(np.linalg.norm(x_scaled - x_pinned))
    # A bug that silently treated negative reg as scaled would make this 0.
    # 1e-10 is comfortably above rounding noise and far below the
    # 20-iter trajectory scale on this problem.
    assert diff > 1e-10, (
        f"pinned and scaled produced (near-)identical iterates (diff={diff:.2e}); "
        f"pinned branch may be a silent fallthrough"
    )

    # Smoke: pinned branch fully converges on a longer horizon.
    acc_long = aa.AndersonAccelerator(DIM, MEM, type1=True, regularization=-1e-8)
    x_long = _run_gd(Q, q, x0, step, steps=500, accelerator=acc_long)
    assert np.linalg.norm(x_long - x_star) < 1e-6


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


def test_python_sequences_rejected():
    w = aa.AndersonAccelerator(DIM, MEM)
    with pytest.raises(TypeError, match="numpy.ndarray"):
        w.apply([0.0] * DIM, np.zeros(DIM))
    with pytest.raises(TypeError, match="numpy.ndarray"):
        w.apply(np.zeros(DIM), tuple([0.0] * DIM))


def test_dim_one_scalar_array_rejected():
    w = aa.AndersonAccelerator(1, MEM)
    with pytest.raises(ValueError, match="Incorrect input dimension"):
        w.apply(np.array(1.0), np.array([0.0]))


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


def test_reset_clears_stale_safeguard_state():
    accel = aa.AndersonAccelerator(dim=2, mem=2, safeguard_factor=1.0)

    x = np.array([1.0, 1.0])
    f = 0.5 * x
    accel.apply(f, x)

    x = f.copy()
    f = 0.5 * x
    accel.apply(f, x)

    x = f.copy()
    f = 0.5 * x
    assert accel.apply(f, x) > 0

    accel.reset()

    f_new = np.array([3.0, 4.0])
    x_new = np.array([1.0, 2.0])
    accepted = accel.safeguard(f_new, x_new)

    assert accepted == 0
    np.testing.assert_array_equal(f_new, [3.0, 4.0])
    np.testing.assert_array_equal(x_new, [1.0, 2.0])
