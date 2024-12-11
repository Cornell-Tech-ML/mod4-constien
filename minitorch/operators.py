"""Collection of the core mathematical operators used throughout the code base."""

import math

from typing import Callable, Iterable, TypeVar

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


T = TypeVar("T")
U = TypeVar("U")


def id(x: float) -> float:
    """Identity operator. Returns the passed argument"""
    return x


def neg(x: float) -> float:
    """Negation operator. Returns the negative of the passed argument"""
    return -x  # type: ignore --- pyright bug - neither mypy nor pylint find an issue with this line


def inv(x: float) -> float:
    """Inverse operator. Returns the inverse of the passed argument"""
    return 1 / x


def inv_back(x: float, y: float) -> float:
    """Inverse back operator - calculates the derivative of inv() at the first argument then scales by the second

    Args:
    ----
        x: Any - the value to evaluate the derivative of the inverse at
        y: Any - the value by which to scale the derivative of the inverse

    Returns:
    -------
        float - the aforesaid value

    """
    return -y / (x**2)


def exp(x: float) -> float:
    """Exponential operator. Returns e raised to the power of the passed argument"""
    return math.exp(x)


def log(x: float) -> float:
    """Logarithmic operator. Returns the natural log of the passed argument"""
    return math.log(x + 1e-6)


def log_back(x: float, y: float) -> float:
    """Logarithmic back operator - calculates the derivative of log() at the first argument then scales by the second

    Args:
    ----
        x: Any - the value to evaluate the derivative of the log at
        y: Any - the value by which to scale the derivative of the log at

    Returns:
    -------
        float - the aforesaid value

    """
    return y / (x + 1e-6)


def relu(x: float) -> float:
    """ReLU (Rectified Linear Unit) operator. Returns the maximum of 0 and the passed argument"""
    return x if x > 0 else 0.0


def relu_back(x: float, y: float) -> float:
    """ReLU back operator - calculates the derivative of relu() at the first argument then scales by the second

    Args:
    ----
        x: Any - the value to evaluate the derivative of the relu at
        y: Any - the value by which to scale the derivative of the relu at
        ignore0: bool = True - a flag indicating whether x = 0 should produce 0 (as tensorflow does) or result
                               in an exception (as ReLU's derivative is not technically defined there)

    Returns:
    -------
        float - the aforesaid value

    """
    return y if x > 0 else 0.0


def sigmoid(x: float) -> float:
    """Sigmoid operator. Returns the result of the sigmoid function at the passed argument"""
    return (1 / (1 + math.exp(-x))) if x >= 0 else (math.exp(x) / (1 + math.exp(x)))


def add(x: float, y: float) -> float:
    """Addition operator. Returns the sum of the passed arguments"""
    return x + y


def mul(x: float, y: float) -> float:
    """Multiplication operator. Returns the product of the passed arguments"""
    return x * y


def eq(x: float, y: float) -> float:
    """Equality operator. Returns 1 only if the passed arguments are equal else 0"""
    return 1.0 if x == y else 0.0


def lt(x: float, y: float) -> float:
    """Less than operator. Returns 1 only if the first argument is less than the second else 0"""
    return 1.0 if x < y else 0.0


def is_close(x: float, y: float, /) -> bool:
    """Near-equality operator. Returns 1 only if the passed arguments are with .01 of each other else 0"""
    return (x - y < 1e-2) and (y - x < 1e-2)


def max(x: float, y: float) -> float:
    """Maximum operator. Returns the maximum of the passed arguments"""
    return x if x > y else y


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def map(fn: Callable[[T], U], iterable: Iterable[T], /) -> list[U]:
    """Applies a function to every element of an iterable"""
    return [fn(x) for x in iterable]


def reduce(
    fn: Callable[[T, T], T], iterable: Iterable[T], /, *, initial: T | None = None
) -> T:
    """Applies a two-argument function cumulatively to the passed iterable
    moving from left-to-right and optionally starting with a given initial value
    resulting in a single value
    """
    it = iter(iterable)

    if initial is None:
        try:
            initial = next(it)
        except StopIteration:
            raise AssertionError("Cannot find initial value")

    try:
        while True:
            initial = fn(initial, next(it))
    except StopIteration:
        pass

    return initial


def zipWith(fn: Callable[[T, T], T], /, *iterables: Iterable[T]) -> list[T]:
    """Element-wise reduction of a list of same-sized iterables, all of the same size, into a single list of accumulated values

    Args:
    ----
        fn: Callable[[T, T], T]: the function with which to reduce the ith elements of the iterables
        iterables: Iterable[T]: the iterables to reduce

    Returns:
    -------
        list[T]: the reduced iterables where the ith element is a reduction of the ith element of the iterables

    """
    if not iterables:
        return []

    iterators, sentinel = [iter(x) for x in iterables], object()

    zipped = []
    while True:
        iteration = [next(it, sentinel) for it in iterators]

        if sentinel in iteration:
            if any(val is not sentinel for val in iteration):
                raise AssertionError("Iterables not of same size")
            else:
                return zipped

        zipped.append(reduce(fn, iteration))  # type: ignore --- pyright cannot see that iteration is a list[T] at this point


def negList(iterable: Iterable[float], /) -> list[float]:
    """Returns an iterable with all elements of the passed iterable negated"""
    return map(neg, iterable)


def sum(iterable: Iterable[float], /) -> float:
    """Returns the sum of the elements of the passed iterable"""
    return reduce(add, iterable, initial=0)


def prod(iterable: Iterable[float], /) -> float:
    """Returns the product of the elements of the passed iterable"""
    return reduce(mul, iterable, initial=1)


def addLists(*iterable: Iterable[float]) -> list[float]:
    """Element-wise addition of an arbitrary number of lists all of the same size

    Args:
    ----
        iterable: (Iterable[T]) - an arbitrary number of iterable elements all of the same size

    Returns:
    -------
        list[T]: a list with the ith element referring to the sum of the ith elements of the passed iterables

    """
    return zipWith(add, *iterable)
