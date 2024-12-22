from dataclasses import dataclass
from functools import reduce
from typing import Callable, List

import math
import sys
import time
import z3


@dataclass
class RangedValue:
    val: z3.ArithRef
    min: z3.ArithRef
    max: z3.ArithRef

    def eval(self, m):
        return f"{m.eval(self.val)} in {m.eval(self.min)}..={m.eval(self.max)}"


def check_op_range(name: str, f: Callable[[RangedValue, RangedValue, List[z3.BoolRef]], RangedValue]) -> bool:
    print(f"Checking {name}")

    a = RangedValue(val=z3.Int("a"), min=z3.Int("al"), max=z3.Int("ah"))
    b = RangedValue(val=z3.Int("b"), min=z3.Int("bl"), max=z3.Int("bh"))
    eq_req = []
    r = f(a, b, eq_req)

    reason_range_invalid = z3.Not(r.min <= r.max)
    reason_below_min = z3.Not(r.min <= r.val)
    reason_below_max = z3.Not(r.val <= r.max)

    eqs_input = [
        # inputs in ranges (this also asserts that the ranges are valid)
        a.min <= a.val,
        a.val <= a.max,
        b.min <= b.val,
        b.val <= b.max,
    ]

    # check for counterexample
    eq_counterexample = z3.Or(reason_range_invalid, reason_below_min, reason_below_max, )

    s = z3.Solver()
    s.add(eqs_input)
    s.add(eq_req)
    s.add(eq_counterexample)

    solved = s.check()
    if solved == z3.sat:
        m = s.model()
        print("  Failed: found case where result range is not correct")
        print(f"      m={m}")
        print(f"      a={a.eval(m)}")
        print(f"      b={b.eval(m)}")
        print(f"      r={r.eval(m)}")
        print("    failure reasons:")
        print(f"      range invalid: {m.eval(reason_range_invalid)}")
        print(f"      below min: {m.eval(reason_below_min)}")
        print(f"      above max: {m.eval(reason_below_max)}")
        return False
    elif solved == z3.unsat:
        print("  Success, proved that result range is always valid")
    else:
        print(f"  Failed: Unknown solver result {solved}")
        print(f"     {s.reason_unknown()}")

        return False

    # TODO check whether the result range is _tight_,
    #    whether for all possible input ranges there exist an input that results in
    #    the min/max value of the result range

    return True


def z3_min(values: List[z3.ArithRef]) -> z3.ArithRef:
    def _min(a, b):
        return z3.If(a < b, a, b)

    return reduce(_min, values)


def z3_max(values: List[z3.ArithRef]) -> z3.ArithRef:
    return -z3_min([-v for v in values])


def main():
    success = True

    def f_add(a, b, _):
        return RangedValue(
            val=a.val + b.val,
            min=a.min + b.min,
            # max=a.max + b.max,
            max=z3.If(a.max + b.max > 0, (a.max + b.max) * 2, a.max + b.max),
        )

    def f_sub(a, b, _):
        return RangedValue(
            val=a.val - b.val,
            min=a.min - b.max,
            max=a.max - b.min
        )

    def f_mul(a, b, _):
        extremes = [a.min * b.min, a.min * b.max, a.max * b.min, a.max * b.max]
        return RangedValue(
            val=a.val * b.val,
            min=z3_min(extremes),
            max=z3_max(extremes),
        )

    def f_pow(base, exp, req):
        # Z3 does not support exponentiation, so we have to implement it ourselves with a necessarily limited
        #   exponent range. An alternative would be using a recursive function, but Z3 fails to solve problems using it.
        max_exp = 7

        req.extend([
            # exp must be positive
            exp.min >= 0,
            exp.val >= 0,
            # base and cannot both be zero
            z3.Not(
                z3.And(base.min <= 0, 0 <= base.max),
                z3.And(exp.min <= 0, 0 <= exp.max),
            ),
            # limitation of z3: limit exp (the redundant extra constraint speeds up solving)
            exp.val <= max_exp,
            exp.max <= max_exp,
        ])

        def z3_pow(b, e):
            result = 0
            for e_const in reversed(range(max_exp + 1)):
                result = z3.If(e == e_const, math.prod([b] * e_const), result)
            return result

        return RangedValue(
            val=z3_pow(base.val, exp.val),
            min=z3_min([
                z3_pow(base.min, exp.min),
                z3_pow(base.min, exp.max),
                z3.If(exp.max > 0, z3_pow(base.min, exp.max - 1), z3_pow(base.min, exp.min)),
            ]),
            max=z3_max([
                z3_pow(base.min, exp.max),
                z3_pow(base.max, exp.max),
                z3.If(exp.max > 0, z3_pow(base.min, exp.max - 1), z3_pow(base.min, exp.max)),
            ]),
        )

    start = time.perf_counter()
    success &= check_op_range("add", f_add)
    success &= check_op_range("sub", f_sub)
    success &= check_op_range("mul", f_mul)
    success &= check_op_range("pow", f_pow)

    print(f"Took {time.perf_counter() - start:.2f}s")

    print("\n")
    sys.stdout.flush()
    assert success


if __name__ == "__main__":
    main()
