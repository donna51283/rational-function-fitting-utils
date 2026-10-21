"""Tests for rational_function_fitting_utils.core.

Every test uses exact ``fractions.Fraction`` comparisons only — no floats, no
approximate comparisons.  This is the whole point of the library: the fit is
exact, so the tests can be exact too.
"""

import unittest
from fractions import Fraction

from rational_function_fitting_utils import (
    fit_rational,
    evaluate_rational,
    RationalFit,
)


class TestFitExactPolynomial(unittest.TestCase):
    """Data drawn from an actual rational function must be recovered exactly."""

    def test_constant_fit(self):
        # y = 7 for all x → a_0 = 7, denominator = [1].
        xs = [0, 1, 2, 3]
        ys = [7, 7, 7, 7]
        fit = fit_rational(xs, ys, numerator_degree=0, denominator_degree=0)
        self.assertEqual(fit.numerator, [Fraction(7)])
        self.assertEqual(fit.denominator, [Fraction(1)])

    def test_linear_polynomial_recovered(self):
        # y = 2x + 3 → a_0=3, a_1=2, denominator=[1].
        xs = [0, 1, 2, 3, 5]
        ys = [2 * x + 3 for x in xs]
        fit = fit_rational(xs, ys, numerator_degree=1, denominator_degree=0)
        self.assertEqual(fit.numerator, [Fraction(3), Fraction(2)])
        self.assertEqual(fit.denominator, [Fraction(1)])

    def test_quadratic_polynomial_recovered(self):
        # y = x^2 - 2x + 1
        xs = [-2, -1, 0, 1, 2, 3, 7]
        ys = [x * x - 2 * x + 1 for x in xs]
        fit = fit_rational(xs, ys, numerator_degree=2, denominator_degree=0)
        self.assertEqual(
            fit.numerator,
            [Fraction(1), Fraction(-2), Fraction(1)],
        )
        self.assertEqual(fit.denominator, [Fraction(1)])


class TestFitExactRational(unittest.TestCase):
    """Data drawn from a genuine rational (non-polynomial) function."""

    def test_simple_rational_recovered(self):
        # y = 1 / (x + 2): numerator=[1], denominator=[2, 1].
        # The parametrisation fixes b_0=1, so denominator=[1, 1/2] also works;
        # both evaluate to the same function.
        xs = [1, 2, 3, 4, 5, 10]
        ys = [Fraction(1, x + 2) for x in xs]
        fit = fit_rational(xs, ys, numerator_degree=0, denominator_degree=1)
        # Fit is exact: evaluation at each x equals the data y.
        for x, y in zip(xs, ys):
            self.assertEqual(fit.evaluate(x), y)

    def test_rational_with_nonunit_numerator_recovered(self):
        # y = (2x + 1) / (x + 3): numerator=[1, 2], denominator=[3, 1].
        xs = [1, 2, 4, 7, 11, 20]
        ys = [Fraction(2 * x + 1, x + 3) for x in xs]
        fit = fit_rational(xs, ys, numerator_degree=1, denominator_degree=1)
        for x, y in zip(xs, ys):
            self.assertEqual(fit.evaluate(x), y)

    def test_higher_degree_rational_recovered(self):
        # y = (x^2) / (x^2 + 1): numerator=[0, 0, 1], denominator=[1, 0, 1].
        xs = [-5, -3, -1, 0, 1, 3, 5, 8]
        ys = [Fraction(x * x, x * x + 1) for x in xs]
        fit = fit_rational(xs, ys, numerator_degree=2, denominator_degree=2)
        for x, y in zip(xs, ys):
            self.assertEqual(fit.evaluate(x), y)


class TestEvaluateRational(unittest.TestCase):
    """Direct tests of the standalone evaluator."""

    def test_evaluate_polynomial(self):
        # (2x + 3) / 1
        self.assertEqual(
            evaluate_rational([3, 2], [1], 4),
            Fraction(11),
        )

    def test_evaluate_constant_denominator(self):
        # 10 / 5 = 2
        self.assertEqual(evaluate_rational([10], [5], 999), Fraction(2))

    def test_evaluate_at_zero(self):
        # At x=0 only the constant terms survive.
        self.assertEqual(
            evaluate_rational([7, 3, 2], [1, 5, 6], 0),
            Fraction(7, 1),
        )

    def test_evaluate_denominator_zero_raises(self):
        # denominator = x - 1, which is zero at x = 1.
        with self.assertRaises(ZeroDivisionError):
            evaluate_rational([1], [-1, 1], 1)

    def test_evaluate_accepts_string_input(self):
        # String input exercises Fraction's string parser path.
        self.assertEqual(
            evaluate_rational(["1"], ["1", "1"], "1/2"),
            Fraction(2, 3),  # 1 / (1 + 1/2) = 1 / (3/2) = 2/3
        )
        self.assertEqual(
            evaluate_rational(["1"], ["1", "1"], "1/2"),
            Fraction(2, 3),
        )


class TestRationalFitRepr(unittest.TestCase):
    """The repr is part of the documented surface; pin it."""

    def test_repr_contains_coefficients(self):
        fit = RationalFit([Fraction(1), Fraction(2)], [Fraction(1)])
        r = repr(fit)
        self.assertIn("RationalFit", r)
        self.assertIn("numerator=[Fraction(1, 1), Fraction(2, 1)]", r)
        self.assertIn("denominator=[Fraction(1, 1)]", r)


class TestInputValidation(unittest.TestCase):
    """Bad inputs must raise, and must raise the *documented* exception."""

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            fit_rational([1, 2, 3], [1, 2], 1, 0)

    def test_negative_degree_raises(self):
        with self.assertRaises(ValueError):
            fit_rational([1, 2, 3], [1, 2, 3], -1, 0)
        with self.assertRaises(ValueError):
            fit_rational([1, 2, 3], [1, 2, 3], 0, -1)

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            fit_rational([], [], 0, 0)

    def test_non_finite_float_raises(self):
        # float('nan') is the one float Fraction() refuses.
        with self.assertRaises(ValueError):
            fit_rational([1.0, float("nan")], [1.0, 2.0], 1, 0)

    def test_string_xs_are_accepted(self):
        # String "3" should convert identically to int 3.
        xs = ["0", "1", "2", "3"]
        ys = [7, 7, 7, 7]
        fit = fit_rational(xs, ys, numerator_degree=0, denominator_degree=0)
        self.assertEqual(fit.numerator, [Fraction(7)])
        self.assertEqual(fit.denominator, [Fraction(1)])


class TestOverdeterminedExact(unittest.TestCase):
    """More data than the minimum should still recover the function exactly."""

    def test_many_points_linear(self):
        xs = list(range(-20, 21))
        ys = [5 * x - 2 for x in xs]
        fit = fit_rational(xs, ys, numerator_degree=1, denominator_degree=0)
        self.assertEqual(fit.numerator, [Fraction(-2), Fraction(5)])
        self.assertEqual(fit.denominator, [Fraction(1)])

    def test_many_points_rational(self):
        xs = list(range(1, 16))
        ys = [Fraction(3, x + 4) for x in xs]
        fit = fit_rational(xs, ys, numerator_degree=0, denominator_degree=1)
        for x, y in zip(xs, ys):
            self.assertEqual(fit.evaluate(x), y)


class TestExactlyDetermined(unittest.TestCase):
    """Number of parameters == number of distinct data points."""

    def test_minimal_linear_fit(self):
        # Two points, two unknowns (a_0, a_1), denominator trivial.
        xs = [0, 1]
        ys = [3, 5]
        fit = fit_rational(xs, ys, numerator_degree=1, denominator_degree=0)
        self.assertEqual(fit.numerator, [Fraction(3), Fraction(2)])
        self.assertEqual(fit.denominator, [Fraction(1)])

    def test_minimal_constant_fit(self):
        # One point, one unknown (a_0).
        fit = fit_rational([42], [7], numerator_degree=0, denominator_degree=0)
        self.assertEqual(fit.numerator, [Fraction(7)])
        self.assertEqual(fit.denominator, [Fraction(1)])


if __name__ == "__main__":
    unittest.main()
