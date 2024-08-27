from typing import List

import z3  # type: ignore


z3.set_option(verbose = 20)

def prove_all(known: List[z3.BoolRef], to_prove: List[z3.BoolRef]):
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
# b = z3.Int("b")

# this function assumes that (exp >= 0 and not (base == 0 and exp == 0))
int_sort = z3.IntSort()
power = z3.RecFunction("power", int_sort, int_sort, int_sort)
power_base = z3.FreshConst(int_sort)
power_exp = z3.FreshConst(int_sort)
z3.RecAddDefinition(
    power,
    [power_base, power_exp],
    z3.If(
        (power_base <= 0) | (power_exp < 0),
        0,
        z3.If(
            power_exp == 0,
            1,
            power_base * power(power_base, power_exp-1),
        )
    )
)

prove_all(
    known=[
        a >= 1,
        # b >= 10,
    ],
    to_prove=[
        power(2, a) >= 0,
    ],
)
