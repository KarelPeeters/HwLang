from typing import List

import z3
from z3 import BoolRef


def prove_all(known: List[BoolRef], to_prove: List[BoolRef]):
    s = z3.Solver()
    s.add(*known)

    if s.check() == z3.unsat:
        print("warning: known is conflicting, all derived expressions will be proven")

    # TODO solve constraint-per-constraint or all at once?
    #   all-at-once would only show one error, but maybe that's enough
    any_fail = False

    for p in to_prove:
        s.push()
        s.add(z3.Not(p))

        print(s)
        r = s.check()

        if r == z3.unsat:
            print(f"  success: {p} is true")
        elif r == z3.sat:
            print(f"  error: {p} is not true, counterexample: {s.model()}")
            any_fail = True
        elif r == z3.unknown:
            print(f"  error: {p} is unknown")
            any_fail = True
        else:
            assert False, f"unexpected check result {r}"

        s.pop()

    if any_fail:
        print("error")
    else:
        print("success")


# type derp(n: uint, x: int_range(0..n*(n+1))) = bits(0);

# type derp(a: uint, b: uint, c: int_range(a..a+b)) = ...

a = z3.Int("a")
b = z3.Int("b")

prove_all(
    known=[
        a >= 0,
        b >= 0,
    ],
    to_prove=[
        # a + b >= a,
        b >= a,
        a + 2 * b >= a,
    ]
)
