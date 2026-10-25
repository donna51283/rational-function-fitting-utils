# rational-function-fitting-utils

Fits a rational function `p(x)/q(x)` to sample data using exact linear algebra over Python's `fractions` module — no floating-point rounding at any stage of the fit.

## Usage

```python
from rational_function_fitting_utils import fit_rational, evaluate_rational

# Data drawn from y = 1 / (x + 2), so p = [1], q = [2, 1].
xs = [1, 2, 3, 4, 5]
ys = [1/3, 1/4, 1/5, 1/6, 1/7]   # floats accepted; converted via exact binary repr.

fit = fit_rational(xs, ys, numerator_degree=0, denominator_degree=1)

# Coefficients are fractions.Fraction instances — exact.
print(fit.numerator)    # e.g. [Fraction(1, 1)]
print(fit.denominator) # e.g. [Fraction(2, 1), Fraction(1, 1)]

# Evaluate at a new point, exactly.
value = fit.evaluate(10)              # Fraction(1, 12)
value = evaluate_rational([1], [2, 1], 10)  # same thing, standalone
```

The library exports three names:

- `fit_rational(xs, ys, numerator_degree, denominator_degree) -> RationalFit`
- `evaluate_rational(numerator, denominator, x) -> Fraction`
- `RationalFit` — the object returned by `fit_rational`, with `.numerator`, `.denominator` (lists of `Fraction`) and `.evaluate(x)`.

## Why this exists

The specific problem: recovering the coefficients of a rational function from samples, where the fit must be *exactly* the function the data came from — not a floating-point approximation that is close-but-not-equal. This shows up when the coefficients themselves are the object of interest (e.g. constructing a Padé approximant by hand, or verifying a closed form) rather than just predicting `y` at new `x`.

The trade-off: we use the normal equations `Aᵀ A c = Aᵀ y` and invert `Aᵀ A` over `Fraction`. In floating point this would be the classic way to destroy conditioning; in exact arithmetic it introduces no error at all. The cost is speed — `O(N³)` in the number of parameters, with integer-arithmetic growth in the intermediate fractions. If you need to fit degree-50 rationals to 10000 points, this is the wrong library. If you need to fit degree-3 rationals to 10 points and recover the exact coefficients, it is the right one.

## The awkward edge

The denominator is parametrised with its constant term `b₀` fixed at `1`. This removes the scaling ambiguity (a rational function is only defined up to a common factor in numerator and denominator) but it also means the library **cannot represent** any rational function whose true denominator has `b₀ = 0`. For example, `y = x / (x + 1)` is representable (`q = [1, 1]`), but `y = 1 / x` is not — its denominator is `q = [0, 1]`, and with `b₀` forced to `1` the fit will be poor. If you know the denominator has no constant term, subtract the known root structure from your data or choose a different parametrisation before calling this library.

A secondary consequence: duplicate `x` values in the input, or an underdetermined degree choice, will produce an exactly singular normal-equations matrix and raise `ZeroDivisionError`. This is deliberate — the fit genuinely is not unique, and returning a silent least-squares guess would hide the problem.

## Design notes

The window stores values eagerly rather than keeping running aggregates. Running
sums drift with floating point over long streams, and recomputing from a small
buffer is cheap enough that the drift is not worth the speed.

