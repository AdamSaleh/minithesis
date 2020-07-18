"""
This file implements a simple property-based testing library called
minithesis. It's not really intended to be used as is, but is instead
a proof of concept that implements as much of the core ideas of
Hypothesis in a simple way that is designed for people who want to
implement a property-based testing library for non-Python languages.

minithesis is always going to be self-contained in a single file
and consist of < 1000 sloc (as measured by cloc). This doesn't
count comments and I intend to comment on it extensively.

"""


import dbm
from array import array
from enum import IntEnum
from random import Random


def run_test(max_examples=100, random=None, database=None, quiet=False):
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
    * seed: A fixed seed to use for randomness.
    * dict: A dict-like object in which results will be cached and resumed
      from, ensuring that if a test is run twice it fails in the same way.
    * quiet: Will not print anything on failure if True.
    """

    def accept(test):
        def mark_failures_interesting(test_case):
            try:
                test(test_case)
            except Exception:
                if test_case.status is not None:
                    raise
                test_case.mark_status(Status.INTERESTING)

        state = TestingState(
            random or Random(), mark_failures_interesting, max_examples
        )

        if database is None:
            # We're using the DBM module because it's an easy default.
            # We don't use this in real Hypothesis - we've got a weird
            # thing there designed to be checked into git but honestly
            # nobody ever checks it into git - and I would encourage you
            # to use some more sensible key/value store here.
            db = dbm.open(".minithesis-cache", "c")
        else:
            db = database

        previous_failure = db.get(test.__name__)

        if previous_failure is not None:
            choices = [
                int.from_bytes(previous_failure[i : i + 8], "big")
                for i in range(0, len(previous_failure), 8)
            ]
            state.test_function(TestCase.for_choices(choices))

        if state.result is None:
            state.run()

        if state.valid_test_cases == 0:
            raise Unsatisfiable()

        if state.result is None:
            db.pop(test.__name__, None)
        else:
            db[test.__name__] = b"".join(i.to_bytes(8, "big") for i in state.result)

        if hasattr(db, "close"):
            db.close()

        if state.result is not None:
            test(TestCase.for_choices(state.result, print_results=not quiet))

    return accept


class Possibility(object):
    """Represents some range of values that might be used in
    a test, that can be requested from a ``TestCase``.

    Pass one of these to TestCase.any to get a concrete value.
    """

    def __init__(self, produce):
        self.produce = produce
        self.name = produce.__name__

    def __repr__(self):
        return self.name


class TestCase(object):
    """Represents a single generated test case, which consists
    of an underlying set of choices that produce possibilities."""

    @classmethod
    def for_choices(cls, choices, print_results=False):
        """Returns a test case that makes this series of choices."""
        return TestCase(
            prefix=choices,
            random=None,
            max_size=len(choices),
            print_results=print_results,
        )

    def __init__(self, prefix, random, max_size=float("inf"), print_results=False):
        self.prefix = prefix
        self.random = random
        self.max_size = max_size
        self.choices = array("I")
        self.status = None
        self.print_results = print_results
        self.depth = 0
        self.targeting_score = None

    def choice(self, n):
        """Returns a number in the range [0, n]"""
        result = self.__make_choice(n, lambda: self.random.randint(0, n))
        if self.__should_print():
            print(f"choice({n}): {result}")
        return result

    def weighted(self, p):
        """Return True with probability ``p``."""
        result = bool(self.__make_choice(1, lambda: int(self.random.random() <= p)))
        if self.__should_print():
            print(f"weighted({p}): {result}")
        return result

    def reject(self):
        """Mark this test case as invalid."""
        self.mark_status(Status.INVALID)

    def assume(self, precondition):
        """If this precondition is not met, abort the test and
        mark this test case as invalid."""
        if not precondition:
            self.reject()

    def target(self, score):
        """Set a score to maximize. Multiple calls to this function
        will override previous ones.

        The name and idea come from Löscher, Andreas, and Konstantinos
        Sagonas. "Targeted property-based testing." ISSTA. 2017, but
        the implementation is based on that found in Hypothesis,
        which is not that similar to anything described in the paper.
        """
        self.targeting_score = score

    def any(self, possibility):
        """Return a possible value from ``possibility``."""
        try:
            self.depth += 1
            result = possibility.produce(self)
        finally:
            self.depth -= 1

        if self.__should_print():
            print(f"any({possibility}): {result}")
        return result

    def mark_status(self, status):
        """Set the status and raise StopTest."""
        if self.status is not None:
            raise Frozen()
        self.status = status
        raise StopTest()

    def __should_print(self):
        return self.print_results and self.depth == 0

    def __make_choice(self, n, rnd_method):
        """Make a choice in [0, n], by calling rnd_method if
        randomness is needed."""
        if n.bit_length() >= 64 or n < 0:
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


# We cap the maximum amount of entropy a test case can use.
# This prevents cases where the generated test case size explodes
# by effectively rejection
BUFFER_SIZE = 8 * 1024


def sort_key(choices):
    """Returns a key that can be used for the shrinking order
    of test cases."""
    return (len(choices), choices)


class CachedTestFunction(object):
    """Returns a cached version of a function that maps
    a choice sequence to the status of calling a test function
    on a test case populated with it. Is able to take advantage
    of the structure of the test function to predict the result
    even if exact sequence of choices has not been seen
    previously.

    You can safely omit implementing this at the cost of
    somewhat increased shrinking time.
    """

    def __init__(self, test_function):
        self.test_function = test_function

        # Tree nodes are either a point at which a choice occurs
        # in which case they map the result of the choice to the
        # tree node we are in after, or a Status object indicating
        # mark_status was called at this point and all future
        # choices are irrelevant.
        #
        # Note that a better implementation of this would use
        # a Patricia trie, which implements long non-branching
        # paths as an array inline. For simplicity we don't
        # do that here.
        self.tree = {}

    def __call__(self, choices):
        node = self.tree
        try:
            for c in choices:
                node = node[c]
                # mark_status was called, thus future choices
                # will be ignored.
                if isinstance(node, Status):
                    assert node != Status.OVERRUN
                    return node
            # If we never entered an unknown region of the tree
            # or hit a Status value, then we know that another
            # choice will be made next and the result will overrun.
            return Status.OVERRUN
        except KeyError:
            pass

        # We now have to actually call the test function to find out
        # what happens.
        test_case = TestCase.for_choices(choices)
        self.test_function(test_case)
        assert test_case.status is not None

        # We enter the choices made in a tree.
        node = self.tree
        for i, c in enumerate(test_case.choices):
            if i + 1 < len(test_case.choices) or test_case.status == Status.OVERRUN:
                try:
                    node = node[c]
                except KeyError:
                    node = node.setdefault(c, {})
            else:
                node[c] = test_case.status
        return test_case.status


class TestingState(object):
    def __init__(self, random, test_function, max_examples):
        self.random = random
        self.max_examples = max_examples
        self.__test_function = test_function
        self.valid_test_cases = 0
        self.calls = 0
        self.result = None
        self.best_scoring = None
        self.test_is_trivial = False

    def test_function(self, test_case):
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

            if test_case.targeting_score is not None:
                relevant_info = (test_case.targeting_score, test_case.choices)
                if self.best_scoring is None:
                    self.best_scoring = relevant_info
                else:
                    best, existing_choices = self.best_scoring
                    if test_case.targeting_score > best:
                        self.best_scoring = relevant_info

        if test_case.status == Status.INTERESTING:
            if self.result is None or sort_key(test_case.choices) < sort_key(
                self.result
            ):
                self.result = test_case.choices

    def target(self):
        """If any test cases have had ``target()`` called on them, do a simple
        hill climbing algorithm to attempt to optimise that target score."""
        if self.result is not None or self.best_scoring is None:
            return

        def adjust(i, step):
            """Can we improve the score by changing choices[i] by step?"""
            score, choices = self.best_scoring
            if choices[i] + step < 0 or choices[i].bit_length() >= 64:
                return False
            attempt = array("I", choices)
            attempt[i] += step
            test_case = TestCase(
                prefix=attempt, random=self.random, max_size=BUFFER_SIZE
            )
            self.test_function(test_case)
            return (
                test_case.status >= Status.VALID
                and test_case.targeting_score is not None
                and test_case.targeting_score > score
            )

        while self.should_keep_generating():
            i = self.random.randrange(0, len(self.best_scoring[1]))
            sign = 0
            for k in [1, -1]:
                if adjust(i, k):
                    sign = k
                    break
            if sign == 0:
                continue

            k = 1
            while adjust(i, sign * k):
                k *= 2

            while k > 0:
                while adjust(i, sign * k):
                    pass
                k //= 2

    def run(self):
        self.generate()
        self.target()
        self.shrink()

    def should_keep_generating(self):
        return (
            not self.test_is_trivial
            and self.result is None
            and self.valid_test_cases < self.max_examples
            and
            # We impose a limit on the maximum number of calls as
            # well as the maximum number of valid examples. This is
            # to avoid taking a prohibitively long time on tests which
            # have hard or impossible to satisfy preconditions.
            self.calls < self.max_examples * 10
        )

    def generate(self):
        """Run random generation until either we have found an interesting
        test case or hit the limit of how many test cases we should
        evaluate."""
        while self.should_keep_generating() and (
            self.best_scoring is None or self.valid_test_cases <= self.max_examples // 2
        ):
            self.test_function(
                TestCase(prefix=(), random=self.random, max_size=BUFFER_SIZE)
            )

    def shrink(self):
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
        cached = CachedTestFunction(self.test_function)

        def consider(choices):
            return cached(choices) == Status.INTERESTING

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

            # First try deleting each choice we made in chunks
            k = 8
            while k > 0:
                i = len(self.result) - k - 1
                while i >= 0:
                    attempt = self.result[:i] + self.result[i + k :]
                    assert len(attempt) < len(self.result)
                    if not consider(attempt):
                        i -= 1
                k //= 2

            # Now try replacing blocks of choices with zeroes
            k = 8
            while k > 0:
                i = len(self.result) - k - 1
                while i >= 0:
                    attempt = (
                        self.result[:i] + array("I", [0] * k) + self.result[i + k :]
                    )
                    if consider(attempt):
                        i -= k
                    else:
                        i -= 1
                k //= 2

            # Now try replacing each choice with a smaller value
            # by doing a binary search.
            i = len(self.result) - 1
            while i >= 0:
                # We assume that if we could replace the choice with zero
                # then we would have on the previous step. Strictly
                # this needn't be true, but if it's not true then we're
                # not at a fixed point and so it will be tried again on
                # the next run through.
                lo = 0
                hi = self.result[i]
                while lo + 1 < hi:
                    mid = lo + (hi - lo) // 2
                    attempt = array("I", self.result)
                    attempt[i] = mid
                    if consider(attempt):
                        hi = mid
                    else:
                        lo = mid
                i -= 1


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
