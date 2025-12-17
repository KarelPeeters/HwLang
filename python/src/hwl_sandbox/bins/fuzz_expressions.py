import os
import random
import re
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import hwl

from hwl_sandbox.common.compare import compare_compile, compare_get_type
from hwl_sandbox.common.util_no_hwl import enable_rust_backtraces


def sample_range_edge(rng: random.Random, max_abs: Optional[int]) -> int:
    t = rng.random()
    if t < 1 / 3:
        return rng.randint(-4, 4)
    if rng.random() < 2 / 3:
        return rng.randint(-256, 256)
    else:
        bits = int(rng.expovariate(0.1))
        mag = rng.randint(0, 2 ** bits)
        sign = rng.randint(0, 1) * 2 - 1

        if max_abs is None or mag <= max_abs:
            return sign * mag
        else:
            # failed, just sample uniformly
            return rng.randint(-max_abs, max_abs)


def sample_range(rng: random.Random, must_contain: Optional[hwl.Range] = None,
                 max_abs: Optional[int] = None) -> hwl.Range:
    # TODO include multi-ranges
    tries = 0
    while True:
        if must_contain is not None and tries > 40:
            return must_contain
        tries += 1

        start = sample_range_edge(rng, max_abs=max_abs)
        if rng.random() < 0.1:
            end = start + 1
        else:
            end = sample_range_edge(rng, max_abs=max_abs)

        if start < end:
            r = hwl.Range(start=start, end=end)
            if must_contain is None or r.contains_range(must_contain):
                return r


def sample_from_range(rng: random.Random, r: Optional[hwl.Range]) -> int:
    if r is None:
        return rng.random() < .5

    if rng.random() < 0.3:
        choices = []
        for a in [0, r.start, r.end - 1]:
            for d in [-2, -1, 0, 1, 2]:
                v = a + d
                if r.contains(v):
                    choices.append(v)
        if choices:
            return rng.choice(choices)

    return rng.randrange(r.start, r.end)


@dataclass
class SampledCode:
    # "None" means this is a boolean input
    # TODO replace this with hwl Type instances once those are convenient enough
    input_ranges: List[hwl.Range | None]
    input_tys: List[str]
    res_ty: str
    body: str


def try_sample_code(rng: random.Random) -> Optional[SampledCode]:
    input_ranges: List[hwl.Range | None] = []
    next_var_index = 0
    available_values_int = []
    available_values_bool = []
    body = ""

    def sample_value(ty_int_not_bool: bool, depth: int) -> str:
        nonlocal next_var_index, body

        if depth > 8 or (depth > 0 and rng.random() < 0.5):
            # reuse an existing value (if possible)
            if rng.random() < .5:
                if ty_int_not_bool and available_values_int:
                    return rng.choice(available_values_int)
                if not ty_int_not_bool and available_values_bool:
                    return rng.choice(available_values_bool)

            # create a new input
            input_str = f"a{len(input_ranges)}"
            if ty_int_not_bool:
                r = sample_range(rng, max_abs=None)
                input_ranges.append(r)
                available_values_int.append(input_str)
            else:
                input_ranges.append(None)
                available_values_bool.append(input_str)

            return input_str

        # create a new expression
        # pick operator
        if ty_int_not_bool:
            # int result
            # TODO include power, unary minus
            operators = ["+", "-", "*", "/", "%"]
            operand_int_not_bool = True
        else:
            # bool result
            if rng.random() < 0.8:
                # int operands
                operators = ["==", "!=", "<", "<=", ">", ">="]
                operand_int_not_bool = True
            else:
                # bool operands
                # TODO add all binary bool operators
                operators = ["&&", "||", "^^", "&", "|", "^"]
                operand_int_not_bool = False

        operator = rng.choice(operators)

        # pick operands
        operand_a = sample_value(ty_int_not_bool=operand_int_not_bool, depth=depth + 1)
        if rng.random() < 0.1:
            operand_b = operand_a
        else:
            operand_b = sample_value(ty_int_not_bool=operand_int_not_bool, depth=depth + 1)

        # build expression
        expr = f"{operand_a} {operator} {operand_b}"

        # maybe store in variable
        if rng.random() < 0.8:
            var_index = next_var_index
            next_var_index += 1

            val_str = f"v_{var_index}"
            body += f"val {val_str} = {expr};\n"
            return val_str
        else:
            return f"({expr})"

    # sample the final return value
    return_value = sample_value(ty_int_not_bool=rng.random() < 0.8, depth=0)
    body += f"return {return_value};"

    # check expression validness and extract the return type
    input_types = [f"int({r.start}..{r.end})" if r is not None else "bool" for r in input_ranges]
    try:
        ty_res_min = compare_get_type(ty_inputs=input_types, body=body, prefix="")
    except hwl.DiagnosticException as e:
        # check that this is once of the expected failure modes
        allowed_messages = [
            "division by zero is not allowed",
            "modulo by zero is not allowed",
            "invalid power operation",
        ]
        if all(any(a in m for a in allowed_messages) for m in e.messages):
            return None

        # unexpected error
        raise e

    # success, we've generated a valid expression
    # parse return type and generate a random range that contains it
    if ty_res_min == "bool":
        res_ty = "bool"
    else:
        m = re.fullmatch(r"int\((-?\d+)\.\.(-?\d+)\)", ty_res_min)
        assert m, f"failed to parse return type `{ty_res_min}`"
        range_res_min = hwl.Range(start=int(m[1]), end=int(m[2]))
        range_res = sample_range(rng, must_contain=range_res_min)
        res_ty = f"int({range_res.start}..{range_res.end})"

    return SampledCode(input_ranges=input_ranges, input_tys=input_types, res_ty=res_ty, body=body)


def sample_code(rng: random.Random) -> SampledCode:
    iter_count = 0
    while True:
        iter_count += 1
        code = try_sample_code(rng=rng)
        if code is not None:
            print(f"Found valid code after {iter_count} attempt(s)")
            return code


def fuzz_step(build_dir: Path, sample_count: int, rng: random.Random):
    # TODO allow multiple args and returns to increase fuzzing throughput
    # TODO expand this for multiple expressions, more operators, mix of ints and non-ints,
    #    arrays, conditional statements, variable assignments, ...
    # TODO add power, add shifts, add bitwise, add binary

    sampled_code = sample_code(rng)

    # generate and compile code
    compiled = compare_compile(
        ty_inputs=sampled_code.input_tys,
        ty_res=sampled_code.res_ty,
        body=sampled_code.body,
        build_dir=build_dir
    )

    # put through some random values
    for _ in range(sample_count):
        values = [sample_from_range(rng, r) for r in sampled_code.input_ranges]
        res_func, res_mod = compiled.eval(values)
        assert res_func == res_mod, f"Mismatch for code {sampled_code}, values `{values}`: function {res_func} != module {res_mod}"


def main_iteration(build_dir_base: Path, sample_count: int, seed_base: int, i: int):
    # TODO move this print into a lock
    # TODO log current expression and number of trials
    print(f"Starting fuzz iteration: {i}")

    rng = random.Random(str((seed_base, i)))

    # create a new build dir for each iteration to avoid issues with dlopen caching old versions
    build_dir = build_dir_base / f"iter_{i}"
    shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True, exist_ok=False)

    fuzz_step(build_dir=build_dir, sample_count=sample_count, rng=rng)

    shutil.rmtree(build_dir, ignore_errors=True)


@dataclass
class Common:
    stopped: bool

    counter_lock: threading.Lock
    counter_next: int
    counter_max: Optional[int]


def main_thread(common: Common, build_dir_base: Path, sample_count: int, seed_base: int):
    try:
        while not common.stopped:
            if common.counter_max is not None and common.counter_next >= common.counter_max:
                break

            with common.counter_lock:
                i = common.counter_next
                common.counter_next += 1

            main_iteration(build_dir_base=build_dir_base, sample_count=sample_count, seed_base=seed_base, i=i)
    finally:
        common.stopped = True


def main():
    # settings
    # TODO use multiprocessing, with ccache python itself becomes the bottleneck
    sample_count = 1024
    thread_count = 16
    build_dir_base = Path(__file__).parent / "../../../build/" / Path(__file__).stem
    os.environ["OBJCACHE"] = "ccache"
    max_iter_count = None

    # random seed
    seed = 42 + 3
    start_iter = 0
    print(f"Using random seed: {seed}")

    # create threads
    common = Common(
        stopped=False,
        counter_lock=threading.Lock(),
        counter_next=start_iter,
        counter_max=start_iter + max_iter_count if max_iter_count is not None else None,
    )
    threads = [
        threading.Thread(target=main_thread, args=(common, build_dir_base, sample_count, seed,))
        for _ in range(thread_count)
    ]

    # start and wait for threads
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    enable_rust_backtraces()
    main()
