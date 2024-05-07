# This file is part of Minithesis, which may be found at
# https://github.com/DRMacIver/minithesis
#
# This work is copyright (C) 2020 David R. MacIver.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""
This file implements a simple property-based testing library called
minithesis. It's not really intended to be used as is, but is instead
a proof of concept that implements as much of the core ideas of
Hypothesis in a simple way that is designed for people who want to
implement a property-based testing library for non-Python languages.

minithesis is always going to be self-contained in a single file
and consist of < 1000 sloc (as measured by cloc). This doesn't
count comments and I intend to comment on it extensively.


=============
PORTING NOTES
=============

minithesis supports roughly the following features, more or less
in order of most to least important:

1. Test case generation.
2. Test case reduction ("shrinking")
3. A small library of primitive possibilities (generators) and combinators.


Anything that supports 1 and 2 is a reasonable good first porting
goal. You'll probably want to port most of the possibilities library
because it's easy and it helps you write tests, but don't worry
too much about the specifics.

"""

from __future__ import annotations


import hashlib
import os
from array import array
from enum import IntEnum
from random import Random
from typing import (
    cast,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    NoReturn,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)


T = TypeVar("T", covariant=True)
S = TypeVar("S", covariant=True)
U = TypeVar("U")  # Invariant

def run_test(test: Callable[[TestData],None],
    max_examples: int = 100,
    random: Optional[Random] = None,
    quiet: bool = False,
) -> None:
    """Decorator to run a test. Usage is:

    .. code-block: python

        @run_test()
        def _(test_case):
            n = test_case.choice(1000)
            ...

    The decorated function takes a ``TestCase`` argument,
    and should raise an exception to indicate a test failure.
    It will either run silently or print drawn values and then
    fail with an exception if minithesis finds some test case
    that fails.

    The test will be run immediately, unlike in Hypothesis where
    @given wraps a function to expose it to the the test runner.
    If you don't want it to be run immediately wrap it inside a
    test function yourself.

    Arguments:

    * max_examples: the maximum number of valid test cases to run for.
      Note that under some circumstances the test may run fewer test
      cases than this.
    * random: An instance of random.Random that will be used for all
      nondeterministic choices.
    * quiet: Will not print anything on failure if True.
    """

    def mark_failures_interesting(test_case: TestData) -> None:
        try:
            test(test_case)
        except Exception:
            if test_case.status is not None:
                raise
            test_case.mark_status(Status.INTERESTING)
    state = TestingState(
        random or Random(), mark_failures_interesting, max_examples
    )

    #Run random generation until either we have found an interesting
    #    test case or hit the limit of how many test cases we should
    #    evaluate.
    while (
            not state.test_is_trivial
            and state.result is None
            and state.valid_test_cases < state.max_examples
            and
            # We impose a limit on the maximum number of calls as
            # well as the maximum number of valid examples. This is
            # to avoid taking a prohibitively long time on tests which
            # have hard or impossible to satisfy preconditions.
            state.calls < state.max_examples * 10
        ) :
        test_case = TestData(prefix=(), random=state.random, max_size=BUFFER_SIZE)
        try:
            test(test_case)
        except StopTest:
            pass
        except Exception:
            if test_case.status is not None:
                raise
            test_case.status = Status.INTERESTING

        if test_case.status is None:
            test_case.status = Status.VALID

        state.calls += 1
        if test_case.status >= Status.INVALID and len(test_case.choices) == 0:
            state.test_is_trivial = True
        if test_case.status >= Status.VALID:
            state.valid_test_cases += 1

        if test_case.status == Status.INTERESTING and (
            state.result is None or sort_key(test_case.choices) < sort_key(state.result)
        ):
            state.result = test_case.choices


    state.shrink()

    if state.valid_test_cases == 0:
        raise Unsatisfiable()

    if state.result is not None:
        test(TestData.for_choices(state.result, print_results=not quiet))



class TestData(object):
    """Represents a single generated test case, which consists
    of an underlying set of choices that produce possibilities."""

    @classmethod
    def for_choices(
        cls,
        choices: Sequence[int],
        print_results: bool = False,
    ) -> TestData:
        """Returns a test case that makes this series of choices."""
        return TestData(
            prefix=choices,
            random=None,
            max_size=len(choices),
            print_results=print_results,
        )

    def __init__(
        self,
        prefix: Sequence[int],
        random: Optional[Random],
        max_size: float = float("inf"),
        print_results: bool = False,
    ):
        self.prefix = prefix
        # XXX Need a cast because below we assume self.random is not None;
        # it can only be None if max_size == len(prefix)
        self.random: Random = cast(Random, random)
        self.max_size = max_size
        self.choices: array[int] = array("Q")
        self.status: Optional[Status] = None
        self.print_results = print_results
        self.depth = 0

    def choice(self, n: int) -> int:
        """Returns a number in the range [0, n]"""
        result = self.__make_choice(n, lambda: self.random.randint(0, n))
        if self.__should_print():
            print(f"choice({n}): {result}")
        return result

    def weighted(self, p: float) -> int:
        """Return True with probability ``p``."""
        if p <= 0:
            result = self.forced_choice(0)
        elif p >= 1:
            result = self.forced_choice(1)
        else:
            result = bool(self.__make_choice(1, lambda: int(self.random.random() <= p)))
        if self.__should_print():
            print(f"weighted({p}): {result}")
        return result

    def forced_choice(self, n: int) -> int:
        """Inserts a fake choice into the choice sequence, as if
        some call to choice() had returned ``n``. You almost never
        need this, but sometimes it can be a useful hint to the
        shrinker."""
        if n.bit_length() > 64 or n < 0:
            raise ValueError(f"Invalid choice {n}")
        if self.status is not None:
            raise Frozen()
        if len(self.choices) >= self.max_size:
            self.mark_status(Status.OVERRUN)
        self.choices.append(n)
        return n

    def reject(self) -> NoReturn:
        """Mark this test case as invalid."""
        self.mark_status(Status.INVALID)

    def assume(self, precondition: bool) -> None:
        """If this precondition is not met, abort the test and
        mark this test case as invalid."""
        if not precondition:
            self.reject()

    def any(self, possibility: Possibility[U]) -> U:
        """Return a possible value from ``possibility``."""
        try:
            self.depth += 1
            result = possibility.produce(self)
        finally:
            self.depth -= 1

        if self.__should_print():
            print(f"any({possibility}): {result}")
        return result

    def mark_status(self, status: Status) -> NoReturn:
        """Set the status and raise StopTest."""
        if self.status is not None:
            raise Frozen()
        self.status = status
        raise StopTest()

    def __should_print(self) -> bool:
        return self.print_results and self.depth == 0

    def __make_choice(self, n: int, rnd_method: Callable[[], int]) -> int:
        """Make a choice in [0, n], by calling rnd_method if
        randomness is needed."""
        if n.bit_length() > 64 or n < 0:
            raise ValueError(f"Invalid choice {n}")
        if self.status is not None:
            raise Frozen()
        if len(self.choices) >= self.max_size:
            self.mark_status(Status.OVERRUN)
        if len(self.choices) < len(self.prefix):
            result = self.prefix[len(self.choices)]
        else:
            result = rnd_method()
        self.choices.append(result)
        if result > n:
            self.mark_status(Status.INVALID)
        return result


class Possibility(Generic[T]):
    """Represents some range of values that might be used in
    a test, that can be requested from a ``TestCase``.

    Pass one of these to TestCase.any to get a concrete value.
    """

    def __init__(self, produce: Callable[[TestData], T], name: Optional[str] = None):
        self.produce = produce
        self.name = produce.__name__ if name is None else name

    def __repr__(self) -> str:
        return self.name

    def map(self, f: Callable[[T], S]) -> Possibility[S]:
        """Returns a ``Possibility`` where values come from
        applying ``f`` to some possible value for ``self``."""
        return Possibility(
            lambda test_case: f(test_case.any(self)),
            name=f"{self.name}.map({f.__name__})",
        )

    def bind(self, f: Callable[[T], Possibility[S]]) -> Possibility[S]:
        """Returns a ``Possibility`` where values come from
        applying ``f`` (which should return a new ``Possibility``
        to some possible value for ``self`` then returning a possible
        value from that."""

        def produce(test_case: TestData) -> S:
            return test_case.any(f(test_case.any(self)))

        return Possibility[S](
            produce,
            name=f"{self.name}.bind({f.__name__})",
        )

    def satisfying(self, f: Callable[[T], bool]) -> Possibility[T]:
        """Returns a ``Possibility`` whose values are any possible
        value of ``self`` for which ``f`` returns True."""

        def produce(test_case: TestData) -> T:
            for _ in range(3):
                candidate = test_case.any(self)
                if f(candidate):
                    return candidate
            test_case.reject()

        return Possibility[T](produce, name=f"{self.name}.select({f.__name__})")


def integers(m: int, n: int) -> Possibility[int]:
    """Any integer in the range [m, n] is possible"""
    return Possibility(lambda tc: m + tc.choice(n - m), name=f"integers({m}, {n})")


def lists(
    elements: Possibility[U],
    min_size: int = 0,
    max_size: float = float("inf"),
) -> Possibility[List[U]]:
    """Any lists whose elements are possible values from ``elements`` are possible."""

    def produce(test_case: TestData) -> List[U]:
        result: List[U] = []
        while True:
            if len(result) < min_size:
                test_case.forced_choice(1)
            elif len(result) + 1 >= max_size:
                test_case.forced_choice(0)
                break
            elif not test_case.weighted(0.9):
                break
            result.append(test_case.any(elements))
        return result

    return Possibility[List[U]](produce, name=f"lists({elements.name})")


def just(value: U) -> Possibility[U]:
    """Only ``value`` is possible."""
    return Possibility[U](lambda tc: value, name=f"just({value})")


def nothing() -> Possibility[NoReturn]:
    """No possible values. i.e. Any call to ``any`` will reject
    the test case."""

    def produce(tc: TestData) -> NoReturn:
        tc.reject()

    return Possibility(produce)


def mix_of(*possibilities: Possibility[T]) -> Possibility[T]:
    """Possible values can be any value possible for one of ``possibilities``."""
    if not possibilities:
        # XXX Need a cast since NoReturn isn't a T (though perhaps it should be)
        return cast(Possibility[T], nothing())
    return Possibility(
        lambda tc: tc.any(possibilities[tc.choice(len(possibilities) - 1)]),
        name="mix_of({', '.join(p.name for p in possibilities)})",
    )


# XXX This signature requires PEP 646
def tuples(*possibilities: Possibility[Any]) -> Possibility[Any]:
    """Any tuple t of of length len(possibilities) such that t[i] is possible
    for possibilities[i] is possible."""
    return Possibility(
        lambda tc: tuple(tc.any(p) for p in possibilities),
        name="tuples({', '.join(p.name for p in possibilities)})",
    )


# We cap the maximum amount of entropy a test case can use.
# This prevents cases where the generated test case size explodes
# by effectively rejection
BUFFER_SIZE = 8 * 1024


def sort_key(choices: Sequence[int]) -> Tuple[int, Sequence[int]]:
    """Returns a key that can be used for the shrinking order
    of test cases."""
    return (len(choices), choices)


class TestingState(object):
    def __init__(
        self,
        random: Random,
        test_function: Callable[[TestData], None],
        max_examples: int,
    ):
        self.random = random
        self.max_examples = max_examples
        self.__test_function = test_function
        self.valid_test_cases = 0
        self.calls = 0
        self.result: Optional[array[int]] = None
        self.test_is_trivial = False

    def test_function(self, test_case: TestData) -> None:
        try:
            self.__test_function(test_case)
        except StopTest:
            pass
        if test_case.status is None:
            test_case.status = Status.VALID
        self.calls += 1
        if test_case.status >= Status.INVALID and len(test_case.choices) == 0:
            self.test_is_trivial = True
        if test_case.status >= Status.VALID:
            self.valid_test_cases += 1

        if test_case.status == Status.INTERESTING and (
            self.result is None or sort_key(test_case.choices) < sort_key(self.result)
        ):
            self.result = test_case.choices


    def shrink(self) -> None:
        """If we have found an interesting example, try shrinking it
        so that the choice sequence leading to our best example is
        shortlex smaller than the one we originally found. This improves
        the quality of the generated test case, as per our paper.

        https://drmaciver.github.io/papers/reduction-via-generation-preview.pdf
        """
        if not self.result:
            return

        # Shrinking will typically try the same choice sequences over
        # and over again, so we cache the test function in order to
        # not end up reevaluating it in those cases. This also allows
        # us to catch cases where we try something that is e.g. a prefix
        # of something we've previously tried, which is guaranteed
        # not to work.
        

        def consider(choices: array[int]) -> bool:
            if choices == self.result:
                return True
            test_case = TestData.for_choices(choices)
            self.test_function(test_case)
            assert test_case.status is not None
            return test_case.status == Status.INTERESTING

        assert consider(self.result)

        # We are going to perform a number of transformations to
        # the current result, iterating until none of them make any
        # progress - i.e. until we make it through an entire iteration
        # of the loop without changing the result.
        prev = None
        while prev != self.result:
            prev = self.result

            # A note on weird loop order: We iterate backwards
            # through the choice sequence rather than forwards,
            # because later bits tend to depend on earlier bits
            # so it's easier to make changes near the end and
            # deleting bits at the end may allow us to make
            # changes earlier on that we we'd have missed.
            #
            # Note that we do not restart the loop at the end
            # when we find a successful shrink. This is because
            # things we've already tried are less likely to work.
            #
            # If this guess is wrong, that's OK, this isn't a
            # correctness problem, because if we made a successful
            # reduction then we are not at a fixed point and
            # will restart the loop at the end the next time
            # round. In some cases this can result in performance
            # issues, but the end result should still be fine.

            # First try deleting each choice we made in chunks.
            # We try longer chunks because this allows us to
            # delete whole composite elements: e.g. deleting an
            # element from a generated list requires us to delete
            # both the choice of whether to include it and also
            # the element itself, which may involve more than one
            # choice. Some things will take more than 8 choices
            # in the sequence. That's too bad, we may not be
            # able to delete those. In Hypothesis proper we
            # record the boundaries corresponding to ``any``
            # calls so that we can try deleting those, but
            # that's pretty high overhead and also a bunch of
            # slightly annoying code that it's not worth porting.
            #
            # We could instead do a quadratic amount of work
            # to try all boundaries, but in general we don't
            # want to do that because even a shrunk test case
            # can involve a relatively large number of choices.
            k = 8
            while k > 0:
                i = len(self.result) - k - 1
                while i >= 0:
                    if i >= len(self.result):
                        # Can happen if we successfully lowered
                        # the value at i - 1
                        i -= 1
                        continue
                    attempt = self.result[:i] + self.result[i + k :]
                    assert len(attempt) < len(self.result)
                    if not consider(attempt):
                        # This fixes a common problem that occurs
                        # when you have dependencies on some
                        # length parameter. e.g. draw a number
                        # between 0 and 10 and then draw that
                        # many elements. This can't delete
                        # everything that occurs that way, but
                        # it can delete some things and often
                        # will get us unstuck when nothing else
                        # does.
                        if i > 0 and attempt[i - 1] > 0:
                            attempt[i - 1] -= 1
                            if consider(attempt):
                                i += 1
                        i -= 1
                k //= 2

            def replace(values: Mapping[int, int]) -> bool:
                """Attempts to replace some indices in the current
                result with new values. Useful for some purely lexicographic
                reductions that we are about to perform."""
                assert self.result is not None
                attempt = array("Q", self.result)
                for i, v in values.items():
                    # The size of self.result can change during shrinking.
                    # If that happens, stop attempting to make use of these
                    # replacements because some other shrink pass is better
                    # to run now.
                    if i >= len(attempt):
                        return False
                    attempt[i] = v
                return consider(attempt)

            # Now we try replacing blocks of choices with zeroes.
            # Note that unlike the above we skip k = 1 because we
            # handle that in the next step. Often (but not always)
            # a block of all zeroes is the shortlex smallest value
            # that a region can be.
            k = 8

            while k > 1:
                i = len(self.result) - k
                while i >= 0:
                    if replace({j: 0 for j in range(i, i + k)}):
                        # If we've succeeded then all of [i, i + k]
                        # is zero so we adjust i so that the next region
                        # does not overlap with this at all.
                        i -= k
                    else:
                        # Otherwise we might still be able to zero some
                        # of these values but not the last one, so we
                        # just go back one.
                        i -= 1
                k //= 2

            # Now try replacing each choice with a smaller value
            # by doing a binary search. This will replace n with 0 or n - 1
            # if possible, but will also more efficiently replace it with
            # a smaller number than doing multiple subtractions would.
            i = len(self.result) - 1
            while i >= 0:
                # Attempt to replace
                bin_search_down(0, self.result[i], lambda v: replace({i: v}))
                i -= 1

            # NB from here on this is just showing off cool shrinker tricks and
            # you probably don't need to worry about it and can skip these bits
            # unless they're easy and you want bragging rights for how much
            # better you are at shrinking than the local QuickCheck equivalent.

            # Try sorting out of order ranges of choices, as ``sort(x) <= x``,
            # so this is always a lexicographic reduction.
            k = 8
            while k > 1:
                for i in range(len(self.result) - k - 1, -1, -1):
                    consider(
                        self.result[:i]
                        + array("Q", sorted(self.result[i : i + k]))
                        + self.result[i + k :]
                    )
                k //= 2

            # Try adjusting nearby pairs of integers by redistributing value
            # between them. This is useful for tests that depend on the
            # sum of some generated values.
            for k in [2, 1]:
                for i in range(len(self.result) - 1 - k, -1, -1):
                    j = i + k
                    # This check is necessary because the previous changes
                    # might have shrunk the size of result, but also it's tedious
                    # to write tests for this so I didn't.
                    if j < len(self.result):  # pragma: no cover
                        # Try swapping out of order pairs
                        if self.result[i] > self.result[j]:
                            replace({j: self.result[i], i: self.result[j]})
                        # j could be out of range if the previous swap succeeded.
                        if j < len(self.result) and self.result[i] > 0:
                            previous_i = self.result[i]
                            previous_j = self.result[j]
                            bin_search_down(
                                0,
                                previous_i,
                                lambda v: replace(
                                    {i: v, j: previous_j + (previous_i - v)}
                                ),
                            )


def bin_search_down(lo: int, hi: int, f: Callable[[int], bool]) -> int:
    """Returns n in [lo, hi] such that f(n) is True,
    where it is assumed and will not be checked that
    f(hi) is True.

    Will return ``lo`` if ``f(lo)`` is True, otherwise
    the only guarantee that is made is that ``f(n - 1)``
    is False and ``f(n)`` is True. In particular this
    does *not* guarantee to find the smallest value,
    only a locally minimal one.
    """
    if f(lo):
        return lo
    while lo + 1 < hi:
        mid = lo + (hi - lo) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi


class Frozen(Exception):
    """Attempted to make choices on a test case that has been
    completed."""


class StopTest(Exception):
    """Raised when a test should stop executing early."""


class Unsatisfiable(Exception):
    """Raised when a test has no valid examples."""


class Status(IntEnum):
    # Test case didn't have enough data to complete
    OVERRUN = 0

    # Test case contained something that prevented completion
    INVALID = 1

    # Test case completed just fine but was boring
    VALID = 2

    # Test case completed and was interesting
    INTERESTING = 3
