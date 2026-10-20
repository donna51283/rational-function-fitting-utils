"""Exact rational-function fitting via the normal equations over ``Fraction``.

The public surface is deliberately small:
    fit_rational(xs, ys, numerator_degree, denominator_degree) -> RationalFit
    evaluate_rational(numerator, denominator, x) -> Fraction
    RationalFit  (returned by fit_rational)

No floats survive past the boundary: ``Fraction`` accepts ``float`` by converting
via the exact binary representation, and every internal computation stays in
``Fraction`` arithmetic.  The normal equations ``Aᵀ A c = Aᵀ y`` are solved by
Gaussian elimination with partial pivoting, again entirely over ``Fraction``.
"""

from __future__ import annotations

from fractions import Fraction

__all__ = ["fit_rational", "evaluate_rational", "RationalFit"]


def _to_fraction(value):
    """Convert *value* to an exact ``Fraction``.

    Accepts ``int``, ``Fraction``, ``str``, or ``float``.  ``float`` is converted
    via ``Fraction.from_float`` (the same path as ``Fraction(float_value)``),
    which uses the exact binary representation — no rounding.
    """
    if isinstance(value, Fraction):
        return value
    if isinstance(value, float):
        if not _is_finite(value):
            raise ValueError(f"non-finite float cannot be used in an exact fit: {value!r}")
        return Fraction.from_float(value)
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, str):
        return Fraction(value)
    # Allow other rational-ish objects by trying Fraction's constructor last.
    return Fraction(value)


def _is_finite(value: float) -> bool:
    """Return True iff *value* is neither NaN nor +/- infinity."""
    import math
    return math.isfinite(value)


def _validate_inputs(xs, ys, numerator_degree, denominator_degree):
    """Raise ``ValueError`` on any documented misuse."""
    if numerator_degree < 0 or denominator_degree < 0:
        raise ValueError(
            "numerator_degree and denominator_degree must be non-negative"
        )
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if len(xs) == 0:
        raise ValueError("cannot fit to zero data points")


def _evaluate_polynomial(coeffs, x_frac):
    """Evaluate ``sum(coeffs[i] * x**i)`` using Horner's rule over ``Fraction``."""
    result = Fraction(0)
    for c in reversed(coeffs):
        result = result * x_frac + c
    return result


def _build_row(x, numerator_degree, denominator_degree):
    """Build one row of the design matrix.

    The model is ``y = p(x) / q(x)`` where
        p(x) = a_0 + a_1 x + ... + a_m x^m
        q(x) = 1 + b_1 x + ... + b_n x^n   (b_0 fixed at 1)
    Multiplying through: ``p(x) = y * q(x)`` i.e.
        a_0 + a_1 x + ... + a_m x^m - y * (b_1 x + ... + b_n x^n) = y
    The unknowns are ``[a_0, ..., a_m, b_1, ..., b_n]`` and the RHS is ``y``.
    """
    x = _to_fraction(x)
    powers = [x ** k for k in range(max(numerator_degree, denominator_degree) + 1)]
    row = []
    for j in range(numerator_degree + 1):
        row.append(powers[j])
    return row


def _matmul_transpose(A, B):
    """Compute ``Aᵀ B`` over ``Fraction``."""
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])
    result = [[Fraction(0)] * cols_B for _ in range(cols_A)]
    for i in range(cols_A):
        for j in range(cols_B):
            s = Fraction(0)
            for k in range(rows_A):
                s += A[k][i] * B[k][j]
            result[i][j] = s
    return result


def _solve_linear_system(A, b):
    """Solve ``A x = b`` exactly over ``Fraction`` by Gaussian elimination.

    Raises ``ZeroDivisionError`` if the system is singular.
    """
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        pivot_row = None
        for r in range(col, n):
            if M[r][col] != 0:
                pivot_row = r
                break
        if pivot_row is None:
            raise ZeroDivisionError(
                "normal-equations matrix is singular; the fit is not unique"
            )
        M[col], M[pivot_row] = M[pivot_row], M[col]
        pivot = M[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = M[r][col] / pivot
            if factor == 0:
                continue
            for c in range(col, n + 1):
                M[r][c] -= factor * M[col][c]
    return [M[i][n] / M[i][i] for i in range(n)]


class RationalFit:
    """An exact rational fit ``p(x)/q(x)`` with ``Fraction`` coefficients."""

    __slots__ = ("_numerator", "_denominator")

    def __init__(self, numerator, denominator):
        self._numerator = [_to_fraction(c) for c in numerator]
        self._denominator = [_to_fraction(c) for c in denominator]

    @property
    def numerator(self):
        return list(self._numerator)

    @property
    def denominator(self):
        return list(self._denominator)

    def evaluate(self, x):
        x = _to_fraction(x)
        p = _evaluate_polynomial(self._numerator, x)
        q = _evaluate_polynomial(self._denominator, x)
        if q == 0:
            raise ZeroDivisionError("denominator evaluates to zero at x")
        return p / q

    def __repr__(self):
        num_repr = ", ".join(repr(c) for c in self._numerator)
        den_repr = ", ".join(repr(c) for c in self._denominator)
        return f"RationalFit(numerator=[{num_repr}], denominator=[{den_repr}])"


def evaluate_rational(numerator, denominator, x):
    """Evaluate ``p(x)/q(x)`` where ``p`` and ``q`` are coefficient lists."""
    num = [_to_fraction(c) for c in numerator]
    den = [_to_fraction(c) for c in denominator]
    x = _to_fraction(x)
    p = _evaluate_polynomial(num, x)
    q = _evaluate_polynomial(den, x)
    if q == 0:
        raise ZeroDivisionError("denominator evaluates to zero at x")
    return p / q


def fit_rational(xs, ys, numerator_degree, denominator_degree):
    """Fit a rational function to ``(xs, ys)`` exactly, over ``Fraction``."""
    _validate_inputs(xs, ys, numerator_degree, denominator_degree)

    xs_f = [_to_fraction(x) for x in xs]
    ys_f = [_to_fraction(y) for y in ys]

    n_params = numerator_degree + 1 + denominator_degree

    A = []
    b = []
    for x, y in zip(xs_f, ys_f):
        powers = [x ** k for k in range(max(numerator_degree, denominator_degree + 1) + 1)]
        row = []
        for j in range(numerator_degree + 1):
            row.append(powers[j])
        for j in range(1, denominator_degree + 1):
            row.append(-y * powers[j])
        A.append(row)
        b.append(y)

    if len(xs_f) < n_params:
        raise ValueError(
            "not enough data points for the requested degree; "
            f"need at least {n_params}, got {len(xs_f)}"
        )

    AtA = _matmul_transpose(A, A)
    Atb = _matmul_transpose(A, [[v] for v in b])
    Atb_flat = [row[0] for row in Atb]

    coeffs = _solve_linear_system(AtA, Atb_flat)

    numerator = coeffs[: numerator_degree + 1]
    denominator = [Fraction(1)] + coeffs[numerator_degree + 1 :]

    return RationalFit(numerator, denominator)
