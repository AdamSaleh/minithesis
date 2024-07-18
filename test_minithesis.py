# This file is part of Minithesis, which may be found at
# https://github.com/DRMacIver/minithesis
#
# This work is copyright (C) 2020 David R. MacIver.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from random import Random

import pytest

import minithesis as mt
from minithesis import Frozen, Status
from minithesis import TestData as TC
from minithesis import TestingState as State
from minithesis import (
    Unsatisfiable,
    run_test,
)

def t_pass(tc):
    return True

def logC(choice):
    def f(test_case):
        test_case.depth += 1
        result = choice(test_case)
        test_case.depth -= 1
        if test_case.should_print():
            print(f"{choice.__name__}: {result}")
        return result
    return f

@logC
def list_of_int(test_case):
    result = []
    while test_case.weighted(0.9):
        result.append(test_case.choice(10000)) 
    return result


@pytest.mark.parametrize("seed", range(10))
def test_finds_small_list(capsys, seed):

    with pytest.raises(AssertionError):
        def t0(ls):
            return sum(ls) <= 1000
        run_test(
            list_of_int,
            t0,
            random=Random(seed))

    captured = capsys.readouterr()

    assert captured.out.strip() == "list_of_int: [1001]"


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
        @logC
        def bad_lst(test_case):
            n = test_case.choice(10)
            return[test_case.choice(10000) for _ in range(n)]

        def t0(ls):
            return sum(ls) <= 1000

        run_test(bad_lst, t0, random=Random(seed))
        
    captured = capsys.readouterr()

    assert captured.out.strip() == "bad_lst: [1001]"


def test_reduces_additive_pairs(capsys):

    with pytest.raises(AssertionError):
        @logC
        def tupl(test_case):
            m = test_case.choice(1000)
            n = test_case.choice(1000)
            return (m,n)

        def t0(t):
            m,n = t
            return m + n <= 1000
        run_test(tupl, t0, max_examples=10000)

    captured = capsys.readouterr()

    assert [c.strip() for c in captured.out.splitlines()] == ['tupl: (1, 1000)']


def test_test_cases_satisfy_preconditions():
    
    def non0(test_case):
        n = test_case.choice(10)
        test_case.assume(n != 0)
        return n
    
    def t0(n):
        return n != 0
    run_test(non0, t0)


def test_error_on_too_strict_precondition():
    with pytest.raises(Unsatisfiable):

        def r0(test_case):
            n = test_case.choice(10)
            test_case.reject()
            return n

        run_test(r0,t_pass)


def test_error_on_unbounded_test_function(monkeypatch):
    monkeypatch.setattr(mt, "BUFFER_SIZE", 10)
    with pytest.raises(Unsatisfiable):

        def r0(test_case):
            while True:
                n = test_case.choice(10)
            return n

        run_test(r0,t_pass, max_examples=5)


#def test_prints_a_top_level_weighted(capsys):
#    with pytest.raises(AssertionError):
#        def r0(test_case):
#            return test_case.weighted(0.5)

#        def t0(w):
#            assert w
#        run_test(r0, t0, max_examples=1000)

#    captured = capsys.readouterr()
#    assert captured.out.strip() == "weighted(0.5): False"


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
    def p0(tc):
        return tc.choice(2 ** 64 - 1)

    run_test(p0,t_pass)  


def test_impossible_weighted():
    with pytest.raises(Failure):
        def t0(tc):
            tc.choice(1)
            for _ in range(10):
                if tc.weighted(0.0):
                    assert False
            if tc.choice(1):
                raise Failure()
            return 0
        run_test(t0, t_pass)

def test_guaranteed_weighted():
    with pytest.raises(Failure):

        def t0(tc):
            if tc.weighted(1.0):
                tc.choice(1)
                raise Failure()
            else:
                assert False
        run_test(t0,t_pass)

def test_forced_choice_bounds():
    with pytest.raises(ValueError):
        def t0(tc):
            tc.forced_choice(2 ** 64)
        run_test(t0,t_pass)

class Failure(Exception):
    pass
