# This file is part of Minithesis, which may be found at
# https://github.com/DRMacIver/minithesis
#
# This work is copyright (C) 2020 David R. MacIver.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from collections import defaultdict
from random import Random

import pytest
from hypothesis import HealthCheck, Phase, given, note, reject, settings
from hypothesis import strategies as st

import minithesis as mt
from minithesis import Frozen, Possibility, Status
from minithesis import TestCase as TC
from minithesis import TestingState as State
from minithesis import (
    Unsatisfiable,
    integers,
    just,
    lists,
    mix_of,
    nothing,
    run_test,
    tuples,
)


@Possibility
def list_of_integers(test_case):
    result = []
    while test_case.weighted(0.9):
        result.append(test_case.choice(10000))
    return result


@pytest.mark.parametrize("seed", range(10))
def test_finds_small_list(capsys, seed):

    with pytest.raises(AssertionError):
        def t0(test_case):
            ls = test_case.any(lists(integers(0, 10000)))
            assert sum(ls) <= 1000
        run_test(t0, random=Random(seed))

    captured = capsys.readouterr()

    assert captured.out.strip() == "any(lists(integers(0, 10000))): [1001]"


@pytest.mark.parametrize("seed", range(10))
def test_finds_small_list_even_with_bad_lists(capsys, seed):
    """Minithesis can't really handle shrinking arbitrary
    monadic bind, but length parameters are a common case
    of monadic bind that it has a little bit of special
    casing for. This test ensures that that special casing
    works.

    The problem is that if you generate a list by drawing
    a length and then drawing that many elements, you can
    end up with something like ``[1001, 0, 0]`` then
    deleting those zeroes in the middle is a pain. minithesis
    will solve this by first sorting those elements, so that
    we have ``[0, 0, 1001]``, and then lowering the length
    by two, turning it into ``[1001]`` as desired.
    """

    with pytest.raises(AssertionError):

        @Possibility
        def bad_list(test_case):
            n = test_case.choice(10)
            return [test_case.choice(10000) for _ in range(n)]
        def t0(test_case):
            ls = test_case.any(bad_list)
            assert sum(ls) <= 1000

        run_test(t0, random=Random(seed))
        
    captured = capsys.readouterr()

    assert captured.out.strip() == "any(bad_list): [1001]"


def test_reduces_additive_pairs(capsys):

    with pytest.raises(AssertionError):

        def t0(test_case):
            m = test_case.choice(1000)
            n = test_case.choice(1000)
            assert m + n <= 1000
        run_test(t0, max_examples=10000)

    captured = capsys.readouterr()

    assert [c.strip() for c in captured.out.splitlines()] == [
        "choice(1000): 1",
        "choice(1000): 1000",
    ]


def test_test_cases_satisfy_preconditions():
    def t0(test_case):
        n = test_case.choice(10)
        test_case.assume(n != 0)
        assert n != 0
    run_test(t0)


def test_error_on_too_strict_precondition():
    with pytest.raises(Unsatisfiable):
        def t0(test_case):
            n = test_case.choice(10)
            test_case.reject()
        run_test(t0)


def test_error_on_unbounded_test_function(monkeypatch):
    monkeypatch.setattr(mt, "BUFFER_SIZE", 10)
    with pytest.raises(Unsatisfiable):

        def t0(test_case):
            while True:
                test_case.choice(10)
        run_test(t0, max_examples=5)


def test_prints_a_top_level_weighted(capsys):
    with pytest.raises(AssertionError):

        def t0(test_case):
            assert test_case.weighted(0.5)
        run_test(t0, max_examples=1000)

    captured = capsys.readouterr()
    assert captured.out.strip() == "weighted(0.5): False"


def test_errors_when_using_frozen():
    tc = TC.for_choices([0])
    tc.status = Status.VALID

    with pytest.raises(Frozen):
        tc.mark_status(Status.INTERESTING)

    with pytest.raises(Frozen):
        tc.choice(10)

    with pytest.raises(Frozen):
        tc.forced_choice(10)


def test_errors_on_too_large_choice():
    tc = TC.for_choices([0])
    with pytest.raises(ValueError):
        tc.choice(2 ** 64)


def test_can_choose_full_64_bits():
    def t0(tc):
        tc.choice(2 ** 64 - 1)

    run_test(t0)
    

def test_mapped_possibility():
    def t0(tc):
        n = tc.any(integers(0, 5).map(lambda n: n * 2))
        assert n % 2 == 0

    run_test(t0)
    

def test_selected_possibility():
    def t0(tc):
        n = tc.any(integers(0, 5).satisfying(lambda n: n % 2 == 0))
        assert n % 2 == 0
    run_test(t0)
    

def test_bound_possibility():
    def t0(tc):
        m, n = tc.any(
            integers(0, 5).bind(lambda m: tuples(just(m), integers(m, m + 10),))
        )

        assert m <= n <= m + 10
    run_test(t0)
    

def test_cannot_witness_nothing():
    with pytest.raises(Unsatisfiable):

        
        def t0(tc):
            tc.any(nothing())
        run_test(t0)

def test_cannot_witness_empty_mix_of():
    with pytest.raises(Unsatisfiable):

        def t0(tc):
            tc.any(mix_of())
        run_test(t0)

def test_can_draw_mixture():
    def t0(tc):
        m = tc.any(mix_of(integers(-5, 0), integers(2, 5)))
        assert -5 <= m <= 5
        assert m != 1
    run_test(t0)

def test_impossible_weighted():
    with pytest.raises(Failure):

        def t0(tc):
            tc.choice(1)
            for _ in range(10):
                if tc.weighted(0.0):
                    assert False
            if tc.choice(1):
                raise Failure()
        run_test(t0)

def test_guaranteed_weighted():
    with pytest.raises(Failure):

        def t0(tc):
            if tc.weighted(1.0):
                tc.choice(1)
                raise Failure()
            else:
                assert False
        run_test(t0)


def test_size_bounds_on_list():
    def t0(tc):
        ls = tc.any(lists(integers(0, 10), min_size=1, max_size=3))
        assert 1 <= len(ls) <= 3
    run_test(t0)


def test_forced_choice_bounds():
    with pytest.raises(ValueError):
        def t0(tc):
            tc.forced_choice(2 ** 64)
        run_test(t0)

class Failure(Exception):
    pass


@settings(
    suppress_health_check=HealthCheck.all(),
    deadline=None,
    report_multiple_bugs=False,
    max_examples=50,
)
@given(st.data())
def test_give_minithesis_a_workout(data):
    seed = data.draw(st.integers(0, 1000))
    rnd = Random(seed)
    max_examples = data.draw(st.integers(1, 100))

    method_call = st.one_of(
        st.tuples(
            st.just("mark_status"),
            st.sampled_from((Status.INVALID, Status.VALID, Status.INTERESTING)),
        ),
        st.tuples(st.just("choice"), st.integers(0, 1000)),
        st.tuples(st.just("weighted"), st.floats(0.0, 1.0)),
    )

    def new_node():
        return [None, defaultdict(new_node)]

    tree = new_node()

    failed = False
    call_count = 0
    valid_count = 0

    try:
        try:
            def test_function(test_case):
                node = tree
                depth = 0
                nonlocal call_count, valid_count, failed
                call_count += 1

                while True:
                    depth += 1
                    if node[0] is None:
                        node[0] = data.draw(method_call)
                    if node[0] == ("mark_status", Status.INTERESTING):
                        failed = True
                        raise Failure()
                    if node[0] == ("mark_status", Status.VALID):
                        valid_count += 1
                    name, *rest = node[0]

                    result = getattr(test_case, name)(*rest)
                    node = node[1][result]

            run_test(
                test_function, max_examples=max_examples, random=rnd, quiet=True,
            )
        except Failure:
            failed = True
        except Unsatisfiable:
            reject()

        if not failed:
            assert valid_count <= max_examples
            assert call_count <= max_examples * 10
    except Exception as e:

        @note
        def tree_as_code():
            """If the test fails, print out a test that will trigger that
            failure rather than making me hand-edit it into something useful."""

            i = 1
            while True:
                test_name = f"test_failure_from_hypothesis_{i}"
                if test_name not in globals():
                    break
                i += 1

            lines = [
                f"def {test_name}():",
                "    with pytest.raises(Failure):",
                f"        @run_test(max_examples=1000, database={{}}, random=Random({seed}))",
                "        def _(tc):",
            ]

            varcount = 0

            def recur(indent, node):
                nonlocal varcount

                if node[0] is None:
                    lines.append(" " * indent + "tc.reject()")
                    return

                method, *args = node[0]
                if method == "mark_status":
                    if args[0] == Status.INTERESTING:
                        lines.append(" " * indent + "raise Failure()")
                    elif args[0] == Status.VALID:
                        lines.append(" " * indent + "return")
                    elif args[0] == Status.INVALID:
                        lines.append(" " * indent + "tc.reject()")
                    else:
                        lines.append(
                            " " * indent + f"tc.mark_status(Status.{args[0].name})"
                        )
                elif method == "weighted":
                    cond = f"tc.weighted({args[0]})"
                    assert len(node[1]) > 0
                    if len(node[1]) == 2:
                        lines.append(" " * indent + "if {cond}:")
                        recur(indent + 4, node[1][True])
                        lines.append(" " * indent + "else:")
                        recur(indent + 4, node[1][False])
                    else:
                        if True in node[1]:
                            lines.append(" " * indent + f"if {cond}:")
                            recur(indent + 4, node[1][True])
                        else:
                            assert False in node[1]
                            lines.append(" " * indent + f"if not {cond}:")
                            recur(indent + 4, node[1][False])
                else:
                    varcount += 1
                    varname = f"n{varcount}"
                    lines.append(
                        " " * indent
                        + f"{varname} = tc.{method}({', '.join(map(repr, args))})"
                    )
                    first = True
                    for k, v in node[1].items():
                        if v[0] == ("mark_status", Status.INVALID):
                            continue
                        lines.append(
                            " " * indent
                            + ("if" if first else "elif")
                            + f" {varname} == {k}:"
                        )
                        first = False
                        recur(indent + 4, v)
                    lines.append(" " * indent + "else:")
                    lines.append(" " * (indent + 4) + "tc.reject()")

            recur(12, tree)
            return "\n".join(lines)

        raise e


def test_failure_from_hypothesis_1():
    with pytest.raises(Failure):

        def t0(tc):
            n1 = tc.weighted(0.0)
            if not n1:
                n2 = tc.choice(511)
                if n2 == 112:
                    n3 = tc.choice(511)
                    if n3 == 124:
                        raise Failure()
                    elif n3 == 93:
                        raise Failure()
                    else:
                        tc.mark_status(Status.INVALID)
                elif n2 == 93:
                    raise Failure()
                else:
                    tc.mark_status(Status.INVALID)
        run_test(t0, max_examples=1000,  random=Random(100))

def test_failure_from_hypothesis_2():
    with pytest.raises(Failure):

        def t0(tc):
            n1 = tc.choice(6)
            if n1 == 6:
                n2 = tc.weighted(0.0)
                if not n2:
                    raise Failure()
            elif n1 == 4:
                n3 = tc.choice(0)
                if n3 == 0:
                    raise Failure()
                else:
                    tc.mark_status(Status.INVALID)
            elif n1 == 2:
                raise Failure()
            else:
                tc.mark_status(Status.INVALID)
        run_test(t0, max_examples=1000,  random=Random(0))
