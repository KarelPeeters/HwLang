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
    z3.set_param(proof=True)

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
        print("  Success, proved that result range always contains the result")
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


def z3_floor_div(a: z3.ArithRef, b: z3.ArithRef) -> z3.ArithRef:
    return (a + z3.If(b > 0, 0, (-b - 1))) / b


def z3_floor_mod(a: z3.ArithRef, b: z3.ArithRef) -> z3.ArithRef:
    return a - b * z3_floor_div(a, b)


def z3_eval_binary_f(f, *args):
    solver = z3.Solver()

    v_args = [z3.Int(f"x{i}") for i in range(len(args))]
    for i in range(len(args)):
        solver.add(v_args[i] == args[i])
    v_result = z3.Int("y")
    solver.add(v_result == f(*v_args))

    assert solver.check() == z3.sat
    result = solver.model()[v_result]

    solver.add(v_result != result)
    assert solver.check() == z3.unsat

    return result.as_long()


def check_z3_floor_div_mod():
    print("Checking floor_div impl")
    for a in reversed(range(-5, 5 + 1)):
        for b in reversed(range(-5, 5 + 1)):
            if b == 0:
                continue
            d = z3_eval_binary_f(z3_floor_div, a, b)
            m = z3_eval_binary_f(z3_floor_mod, a, b)
            assert d == a // b and m == a % b, f"expected ({a}, {b}) -> /={a // b}, %={a % b}, got /={d} %={m}"


def interactive_div():
    x_range = range(-10, 9)
    y_range = range(-6, 4)

    possible = set()
    for x in x_range:
        for y in y_range:
            if y != 0:
                possible.add(x / y)


def main():
    success = True

    def f_add(a, b, _):
        return RangedValue(
            val=a.val + b.val,
            min=a.min + b.min,
            max=a.max + b.max,
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
            # base and exp cannot both include zero
            z3.Or(
                base.min > 0,
                base.max < 0,
                exp.min > 0,
                exp.max < 0,
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

    def f_div(a, b, req):
        # b's range cannot contain zero
        # (this also allows the if conditions below to be valid, the range is either entirely positive or negative)
        req.append(z3.Not(z3.And(b.min <= 0, 0 <= b.max)))

        return RangedValue(
            val=z3_floor_div(a.val, b.val),
            min=z3.If(
                b.min > 0,
                z3_min([z3_floor_div(a.min, b.max), z3_floor_div(a.min, b.min)]),
                z3_min([z3_floor_div(a.max, b.max), z3_floor_div(a.max, b.min)]),
            ),
            max=z3.If(
                b.min > 0,
                z3_max([z3_floor_div(a.max, b.max), z3_floor_div(a.max, b.min)]),
                z3_max([z3_floor_div(a.min, b.max), z3_floor_div(a.min, b.min)]),
            ),
        )

    def f_mod(a, b, req):
        # b's range cannot contain zero
        # (this also allows the if conditions below to be valid, the range is either entirely positive or negative)
        req.append(z3.Not(z3.And(b.min <= 0, 0 <= b.max)))

        # TODO this could be tighter, eg. if b's range does not cover the entire mod interval
        return RangedValue(
            val=z3_floor_mod(a.val, b.val),
            min=z3.If(b.min > 0, 0, b.min + 1),
            max=z3.If(b.min > 0, b.max - 1, 0),
        )

    start = time.perf_counter()

    # success &= check_op_range("add", f_add)
    # success &= check_op_range("sub", f_sub)
    # success &= check_op_range("mul", f_mul)
    # success &= check_op_range("pow", f_pow)
    check_z3_floor_div_mod()
    success &= check_op_range("div", f_div)
    success &= check_op_range("mod", f_mod)

    print(f"Took {time.perf_counter() - start:.2f}s")

    print("\n")
    sys.stdout.flush()
    assert success


if __name__ == "__main__":
    main()
