import random
import re
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, List

import hwl

from hwl_sandbox.common.compare import compare_compile, compare_get_type
from hwl_sandbox.common.util_no_hwl import enable_rust_backtraces


# TODO use hwl.Range and hwl.Types once those are convenient enough
@dataclass(frozen=True)
class Range:
    start: int
    end_inc: int

    def __contains__(self, item: Union[int, "Range"]) -> bool:
        if isinstance(item, Range):
            return self.start <= item.start and item.end_inc <= self.end_inc
        else:
            return self.start <= item <= self.end_inc


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


def sample_range(rng: random.Random, must_contain: Optional[Range] = None, max_abs: Optional[int] = None) -> Range:
    # TODO allow empty ranges?
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

        if start <= end:
            r = Range(start=start, end_inc=end)
            if must_contain is None or must_contain in r:
                return r


def sample_from_range(rng: random.Random, r: Range) -> int:
    if rng.random() < 0.3:
        choices = []
        for a in [0, r.start, r.end_inc]:
            for d in [-2, -1, 0, 1, 2]:
                v = a + d
                if v in r:
                    choices.append(v)
        if choices:
            return rng.choice(choices)

    return rng.randint(r.start, r.end_inc)


@dataclass
class SampledCode:
    input_ranges: List[Range]
    input_tys: List[str]
    res_ty: str
    body: str


def try_sample_code(rng: random.Random) -> Optional[SampledCode]:
    # decide operator
    operators = ["+", "-", "*", "/", "%", "**", "==", "!=", "<", "<=", ">", ">="]
    operator = rng.choice(operators)

    # decide types
    max_abs = None if operator != "**" else 1024
    ra = sample_range(rng, max_abs=max_abs)
    if rng.random() < 0.1:
        rb = ra
    else:
        rb = sample_range(rng, max_abs=max_abs)

    # check that power ranges are not too large
    max_range_abs = max(abs(ra.start), abs(rb.start), abs(ra.end_inc), abs(rb.end_inc))
    if operator == "**" and max_range_abs ** max_range_abs > 2 ** 1024:
        return None

    input_ranges = [ra, rb]
    input_tys = [f"int({r.start}..={r.end_inc})" for r in input_ranges]

    # check expression validness and extract the return type
    expression = f"a0 {operator} a1"
    body = f"return {expression};"
    try:
        ty_res_min = compare_get_type(ty_inputs=input_tys, body=body)
    except hwl.DiagnosticException as e:
        # check that this is once of the expected failure modes
        allowed_messages = [
            "division by zero is not allowed",
            "modulo by zero is not allowed",
            "invalid power operation",
        ]
        if len(e.messages) == 1 and any(m in e.messages[0] for m in allowed_messages):
            return None

        # unexpected error
        raise e

    # success, we've generated a valid expression
    # parse return type and generate a random range that contains it
    if ty_res_min == "bool":
        res_ty = "bool"
    else:
        m = re.fullmatch(r"int\((-?\d+)\.\.=(-?\d+)\)", ty_res_min)
        assert m
        range_res_min = Range(start=int(m[1]), end_inc=int(m[2]))
        range_res = sample_range(rng, must_contain=range_res_min)
        res_ty = f"int({range_res.start}..={range_res.end_inc})"

    return SampledCode(input_ranges=input_ranges, input_tys=input_tys, res_ty=res_ty, body=body)


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


def main_thread(common: Common, build_dir_base: Path, sample_count: int, seed_base: int):
    try:
        while not common.stopped:
            with common.counter_lock:
                i = common.counter_next
                common.counter_next += 1

            main_iteration(build_dir_base=build_dir_base, sample_count=sample_count, seed_base=seed_base, i=i)
    finally:
        common.stopped = True


def main():
    # settings
    sample_count = 1024
    thread_count = 8
    build_dir_base = Path(__file__).parent / "../../../build/" / Path(__file__).stem

    # random seed
    seed = 42 + 3
    start_iter = 0
    print(f"Using random seed: {seed}")

    # create threads
    common = Common(
        stopped=False,
        counter_lock=threading.Lock(),
        counter_next=start_iter,
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
