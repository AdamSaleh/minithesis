from collections import defaultdict
from random import Random

import pytest
from hypothesis import HealthCheck, given, reject, settings
from hypothesis import strategies as st

from minithesis import (CachedTestFunction, DirectoryDB, Frozen, Possibility,
                        Status)
from minithesis import TestCase as TC
from minithesis import TestingState as State
from minithesis import (Unsatisfiable, integers, just, lists, mix_of, nothing,
                        run_test, tuples)


@Possibility
def list_of_integers(test_case):
    result = []
    while test_case.weighted(0.9):
        result.append(test_case.choice(10000))
    return result


def test_finds_small_list(capsys):

    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(test_case):
            ls = test_case.any(lists(integers(0, 10000)))
            assert sum(ls) <= 1000

    captured = capsys.readouterr()

    assert captured.out.strip() == "any(lists(integers(0, 10000))): [1001]"


def test_reduces_additive_pairs(capsys):

    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=10000)
        def _(test_case):
            m = test_case.choice(1000)
            n = test_case.choice(1000)
            assert m + n <= 1000

    captured = capsys.readouterr()

    assert [c.strip() for c in captured.out.splitlines()] == [
        "choice(1000): 1",
        "choice(1000): 1000",
    ]


def test_reuses_results_from_the_database(tmpdir):
    db = DirectoryDB(tmpdir)
    count = 0

    def run():
        with pytest.raises(AssertionError):

            @run_test(database=db)
            def _(test_case):
                nonlocal count
                count += 1
                assert test_case.choice(10000) < 10

    run()

    assert len(tmpdir.listdir()) == 1
    prev_count = count

    run()

    assert len(tmpdir.listdir()) == 1
    assert count == prev_count + 2


def test_test_cases_satisfy_preconditions():
    @run_test()
    def _(test_case):
        n = test_case.choice(10)
        test_case.assume(n != 0)
        assert n != 0


def test_error_on_too_strict_precondition():
    with pytest.raises(Unsatisfiable):

        @run_test()
        def _(test_case):
            n = test_case.choice(10)
            test_case.reject()


def test_error_on_unbounded_test_function():
    with pytest.raises(Unsatisfiable):

        @run_test(max_examples=5)
        def _(test_case):
            while True:
                test_case.choice(10)


def test_function_cache():
    def tf(tc):
        if tc.choice(1000) >= 200:
            tc.mark_status(Status.INTERESTING)
        if tc.choice(1) == 0:
            tc.reject()

    state = State(Random(0), tf, 100)

    cache = CachedTestFunction(state.test_function)

    assert cache([1, 1]) == Status.VALID
    assert cache([1]) == Status.OVERRUN
    assert cache([1000]) == Status.INTERESTING
    assert cache([1000]) == Status.INTERESTING
    assert cache([1000, 1]) == Status.INTERESTING

    assert state.calls == 2


def test_can_target_a_score_upwards(capsys):
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(test_case):
            n = test_case.choice(1000)
            m = test_case.choice(1000)
            score = n + m
            test_case.target(score)
            assert score < 2000

    captured = capsys.readouterr()

    assert [c.strip() for c in captured.out.splitlines()] == [
        "choice(1000): 1000",
        "choice(1000): 1000",
    ]


def test_targeting_when_most_do_not_benefit(capsys):
    with pytest.raises(AssertionError):
        big = 10000

        @run_test(database={}, max_examples=1000)
        def _(test_case):
            test_case.choice(1000)
            test_case.choice(1000)
            score = test_case.choice(big)
            test_case.target(score)
            assert score < big

    captured = capsys.readouterr()

    assert [c.strip() for c in captured.out.splitlines()] == [
        "choice(1000): 0",
        "choice(1000): 0",
        f"choice({big}): {big}",
    ]


def test_can_target_a_score_downwards(capsys):
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(test_case):
            n = test_case.choice(1000)
            m = test_case.choice(1000)
            score = n + m
            test_case.target(-score)
            assert score > 0

    captured = capsys.readouterr()

    assert [c.strip() for c in captured.out.splitlines()] == [
        "choice(1000): 0",
        "choice(1000): 0",
    ]


def test_prints_a_top_level_weighted(capsys):
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(test_case):
            assert test_case.weighted(0.5)

    captured = capsys.readouterr()
    assert captured.out.strip() == "weighted(0.5): False"


def test_errors_when_using_frozen():
    tc = TC.for_choices([0])
    tc.status = Status.VALID

    with pytest.raises(Frozen):
        tc.mark_status(Status.INTERESTING)

    with pytest.raises(Frozen):
        tc.choice(10)


def test_errors_on_too_large_choice():
    tc = TC.for_choices([0])
    with pytest.raises(ValueError):
        tc.choice(2 ** 64)


def test_can_choose_full_64_bits():
    @run_test()
    def _(tc):
        tc.choice(2 ** 64 - 1)


def test_mapped_possibility():
    @run_test()
    def _(tc):
        n = tc.any(integers(0, 5).map(lambda n: n * 2))
        assert n % 2 == 0


def test_selected_possibility():
    @run_test()
    def _(tc):
        n = tc.any(integers(0, 5).satisfying(lambda n: n % 2 == 0))
        assert n % 2 == 0


def test_bound_possibility():
    @run_test()
    def _(tc):
        m, n = tc.any(
            integers(0, 5).bind(lambda m: tuples(just(m), integers(m, m + 10),))
        )

        assert m <= n <= m + 10


def test_cannot_witness_nothing():
    with pytest.raises(Unsatisfiable):

        @run_test()
        def _(tc):
            tc.any(nothing())


def test_cannot_witness_empty_mix_of():
    with pytest.raises(Unsatisfiable):

        @run_test()
        def _(tc):
            tc.any(mix_of())


def test_can_draw_mixture():
    @run_test()
    def _(tc):
        m = tc.any(mix_of(integers(-5, 0), integers(2, 5)))
        assert -5 <= m <= 5
        assert m != 1


def test_target_and_reduce(capsys):
    """This test is very hard to trigger without targeting,
    and targeting will tend to overshoot the score, so we
    will see multiple interesting test cases before
    shrinking."""
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            m = tc.choice(100000)
            tc.target(m)
            assert m <= 99900

    captured = capsys.readouterr()

    assert captured.out.strip() == "choice(100000): 99901"


class ShouldFail(Exception):
    pass


@settings(
    suppress_health_check=HealthCheck.all(),
    deadline=None,
    report_multiple_bugs=False,
    max_examples=50,
)
@given(st.data())
def test_give_minithesis_a_workout(data):
    rnd = data.draw(st.randoms(use_true_random=False))
    max_examples = data.draw(st.integers(1, 500))

    method_call = st.one_of(
        st.tuples(
            st.just("mark_status"),
            st.sampled_from((Status.INVALID, Status.VALID, Status.INTERESTING)),
        ),
        st.tuples(st.just("target"), st.floats(0.0, 1.0)),
        st.tuples(st.just("weighted"), st.floats(0.0, 1.0)),
        st.tuples(st.just("choice"), st.integers(0, 1000)),
    )

    def new_node():
        return [None, defaultdict(new_node)]

    tree = new_node()

    database = {}

    try:

        @run_test(
            max_examples=max_examples, random=rnd, database=database, quiet=True,
        )
        def test_function(test_case):
            node = tree
            depth = 0

            while depth <= 5:
                depth += 1
                if node[0] is None:
                    node[0] = data.draw(method_call)
                if node[0] == ("mark_status", Status.INTERESTING):
                    raise ShouldFail()
                name, *rest = node[0]

                result = getattr(test_case, name)(*rest)
                node = node[1][result]

    except ShouldFail:
        pass
    except Unsatisfiable:
        reject()
