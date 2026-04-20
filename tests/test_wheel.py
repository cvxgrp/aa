"""Smoke test run by cibuildwheel against each freshly built wheel.

Exercises the real API (not just import) so we catch BLAS/LAPACK linkage or
ABI bugs before a wheel is uploaded to PyPI.
"""
import warnings

import numpy as np

import aa

# AA extrapolation can transiently produce Inf/NaN iterates that the safeguard
# immediately rejects. That is expected behavior, but the intermediate numpy
# matmul emits RuntimeWarnings along the way — silence them so the test output
# is readable.
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _quadratic_gd(n=50, mem=5, iters=200, seed=0):
    # A deterministic, well-conditioned quadratic f(x) = 0.5 x'Qx with
    # eigenvalues in [0.1, 1]. Step = 1/lambda_max makes plain GD contract
    # at rate (1 - 0.1), and AA should accelerate noticeably on top of that.
    eigs = np.linspace(0.1, 1.0, n)
    Q = np.diag(eigs)
    step = 1.0 / eigs.max()

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    acc = aa.AndersonAccelerator(n, mem)

    xprev = x.copy()
    for i in range(iters):
        if i > 0:
            acc.apply(x, xprev)
        xprev = x.copy()
        x = x - step * (Q @ xprev)
        acc.safeguard(x, xprev)
    return float(np.linalg.norm(x))


def main():
    # Basic lifecycle.
    acc = aa.AndersonAccelerator(10, 3)
    f = np.zeros(10)
    x = np.zeros(10)
    acc.apply(f, x)
    acc.safeguard(f, x)
    acc.reset()

    # Input validation — should reject wrong dtype / shape.
    for bad in (np.zeros(10, dtype=np.float32), np.zeros(9), np.zeros((2, 5))):
        try:
            acc.apply(bad, np.zeros(10))
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"expected rejection for {bad.dtype} {bad.shape}")

    # Actually run a convergent problem — touches all BLAS paths.
    err = _quadratic_gd()
    assert np.isfinite(err), f"non-finite error: {err}"
    assert err < 1e-3, f"GD+AA did not converge: final err = {err}"

    print("wheel smoke test ok")


if __name__ == "__main__":
    main()
