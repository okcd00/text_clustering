"""Independent, dense faded moments; not a DenStream implementation.

Only the Python standard library is used. See readme.md for the numerical
contract, including deliberately rejected partial-underflow states.
"""

from collections.abc import Mapping, Set
from dataclasses import dataclass
import math
from numbers import Integral, Real
import sys
from typing import Iterable, Optional, Tuple


# Relative to the two terms of each coordinate variance, with no absolute floor.
VARIANCE_RELATIVE_TOLERANCE = 64.0 * sys.float_info.epsilon


def _finite(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, not bool")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} is not representable as a finite float") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vector(values: Iterable[Real], name: str) -> Tuple[float, ...]:
    # In particular, do not silently interpret sparse mapping keys as coordinates.
    if isinstance(values, (Mapping, Set, str, bytes, bytearray)):
        raise ValueError(f"{name} must be a dense ordered vector")
    try:
        return tuple(_finite(value, name) for value in values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a dense ordered vector") from exc


@dataclass(frozen=True)
class FadedStatistics:
    """Immutable effective mass and coordinate moments at ``reference_time``.

    ``count`` counts raw insertions, including observations whose mass has faded
    away. ``creation_time`` is the earliest insertion time, not the last update.
    Direct construction is validated too; supplied vectors are copied to tuples.
    Invalid input or unrepresentable arithmetic raises ``ValueError``.
    """

    weight: float
    linear_sum: Tuple[float, ...]
    squared_sum: Tuple[float, ...]
    reference_time: float
    decay_rate: float
    count: int
    creation_time: Optional[float]

    def __post_init__(self) -> None:
        for name in ("weight", "reference_time", "decay_rate"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        for name in ("linear_sum", "squared_sum"):
            object.__setattr__(self, name, _vector(getattr(self, name), name))
        if not self.linear_sum or len(self.linear_sum) != len(self.squared_sum):
            raise ValueError("moments must have the same positive dimension")
        if self.weight < 0.0 or self.decay_rate < 0.0:
            raise ValueError("weight and decay_rate must be nonnegative")
        if (isinstance(self.count, bool) or not isinstance(self.count, Integral)
                or self.count < 0):
            raise ValueError("count must be a nonnegative integer, not bool")
        object.__setattr__(self, "count", int(self.count))
        if self.creation_time is not None:
            creation_time = _finite(self.creation_time, "creation_time")
            object.__setattr__(self, "creation_time", creation_time)
            if creation_time > self.reference_time:
                raise ValueError("creation_time must not exceed reference_time")
        if (self.count == 0) != (self.creation_time is None):
            raise ValueError("creation_time must be None exactly when count is zero")
        if self.count == 0 and self.weight != 0.0:
            raise ValueError("zero count requires zero weight")
        if any(value < 0.0 for value in self.squared_sum):
            raise ValueError("squared_sum must be nonnegative")
        if self.weight == 0.0:
            if any(self.linear_sum) or any(self.squared_sum):
                raise ValueError("zero weight requires zero moments")
        else:
            # Validate before publishing a new object, not only during readout.
            self._variances()

    @property
    def dimension(self) -> int:
        """Number of dense coordinates."""
        return len(self.linear_sum)

    @classmethod
    def empty(cls, dimension: int, decay_rate: float,
              time: float = 0.0) -> "FadedStatistics":
        """Create a zero-mass state, with a positive integer dimension."""
        if (isinstance(dimension, bool) or not isinstance(dimension, Integral)
                or dimension <= 0):
            raise ValueError("dimension must be a positive integer, not bool")
        zeros = (0.0,) * int(dimension)
        return cls(0.0, zeros, zeros, time, decay_rate, 0, None)

    def at(self, time: float) -> "FadedStatistics":
        """Return a fresh state aged by ``2 ** (-decay_rate * elapsed)``."""
        time = _finite(time, "time")
        if time < self.reference_time:
            raise ValueError("time must not precede reference_time")
        if self.decay_rate == 0.0 or time == self.reference_time:
            factor = 1.0
        else:
            elapsed = time - self.reference_time
            # Finite opposite-sign endpoints can overflow their subtraction,
            # even when multiplying the elapsed time by a small rate would not.
            exponent = (self.decay_rate * elapsed if math.isfinite(elapsed)
                        else self.decay_rate * time
                        - self.decay_rate * self.reference_time)
            factor = 2.0 ** (-exponent)
        weight = self.weight * factor
        if weight == 0.0:
            # Canonical complete underflow: never leave moments without mass.
            linear_sum = squared_sum = (0.0,) * self.dimension
        else:
            linear_sum = tuple(value * factor for value in self.linear_sum)
            squared_sum = tuple(value * factor for value in self.squared_sum)
            for old, new in ((self.linear_sum, linear_sum),
                             (self.squared_sum, squared_sum)):
                if any(a != 0.0 and b == 0.0 for a, b in zip(old, new)):
                    raise ValueError("partial moment underflow with positive weight")
        return type(self)(weight, linear_sum, squared_sum, time, self.decay_rate,
                          self.count, self.creation_time)

    def insert(self, point: Iterable[Real], time: float) -> "FadedStatistics":
        """Age old moments, then add one unit-weight dense observation."""
        point = _vector(point, "point")
        if len(point) != self.dimension:
            raise ValueError("point dimension does not match statistics")
        squares = tuple(_finite(value * value, "point squared") for value in point)
        if any(value != 0.0 and square == 0.0
               for value, square in zip(point, squares)):
            raise ValueError("point square underflow")
        aged = self.at(time)
        return type(self)(
            aged.weight + 1.0,
            tuple(a + b for a, b in zip(aged.linear_sum, point)),
            tuple(a + b for a, b in zip(aged.squared_sum, squares)),
            aged.reference_time, self.decay_rate, self.count + 1,
            self.creation_time if self.count else aged.reference_time,
        )

    # Candidate evaluation has exactly the same arithmetic and fresh-object
    # semantics as insertion, without committing anything to the original.
    candidate = insert

    def merge(self, other: "FadedStatistics", time: float) -> "FadedStatistics":
        """Add two compatible states after aging both to the requested time.

        No observation IDs are stored: callers must avoid unintended double
        counting. A deliberate self-merge doubles the effective and raw counts.
        """
        if not isinstance(other, FadedStatistics):
            raise ValueError("other must be FadedStatistics")
        if self.dimension != other.dimension or self.decay_rate != other.decay_rate:
            raise ValueError("merge requires identical dimension and decay_rate")
        left, right = self.at(time), other.at(time)
        creation_times = tuple(value for value in
                               (self.creation_time, other.creation_time)
                               if value is not None)
        return type(self)(
            left.weight + right.weight,
            tuple(a + b for a, b in zip(left.linear_sum, right.linear_sum)),
            tuple(a + b for a, b in zip(left.squared_sum, right.squared_sum)),
            left.reference_time, self.decay_rate, self.count + other.count,
            min(creation_times) if creation_times else None,
        )

    def center(self) -> Tuple[float, ...]:
        """Return the unnormalized weighted center; zero mass is undefined."""
        if self.weight == 0.0:
            raise ValueError("center is undefined for zero weight")
        return tuple(_finite(value / self.weight, "center")
                     for value in self.linear_sum)

    def _variances(self) -> Tuple[float, ...]:
        center = self.center()
        variances = []
        for linear, second, mean in zip(self.linear_sum, self.squared_sum, center):
            if second == 0.0 and linear != 0.0:
                raise ValueError("nonzero first moment requires a second moment")
            second_mean = _finite(second / self.weight, "normalized second moment")
            mean_squared = _finite(mean * mean, "squared center")
            variance = second_mean - mean_squared
            tolerance = VARIANCE_RELATIVE_TOLERANCE * max(second_mean, mean_squared)
            if variance < -tolerance:
                raise ValueError("negative variance exceeds scale-aware tolerance")
            variances.append(max(variance, 0.0))
        return tuple(variances)

    def radius(self) -> float:
        """Return Euclidean RMS radius, not maximum coordinate deviation.

        Clamp only negative coordinate variance within the explicit relative
        tolerance. ``hypot`` avoids overflowing the sum of valid variances.
        """
        return _finite(math.hypot(*(math.sqrt(value) for value in self._variances())),
                       "radius")
