import random
import re
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional

import hwl

from hwl_sandbox.common.expression import expression_compile, expression_get_type
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


def sample_range_edge(rng: random.Random) -> int:
    t = rng.random()
    if t < 1 / 3:
        return rng.randint(-4, 4)
    if rng.random() < 2 / 3:
        return rng.randint(-256, 256)
    else:
        bits = int(rng.expovariate(0.1))
        mag = rng.randint(0, 2 ** bits)
        sign = rng.randint(0, 1) * 2 - 1
        return sign * mag


def sample_range(rng: random.Random, must_contain: Optional[Range] = None) -> Range:
    # TODO allow empty ranges?
    while True:
        start = sample_range_edge(rng)
        if rng.random() < 0.1:
            end = start + 1
        else:
            end = sample_range_edge(rng)

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


def fuzz_step(build_dir: Path, sample_count: int, rng: random.Random):
    while True:
        # decide types and operations
        # TODO add "expansion" tests, where the output range is too large, to see if the value is properly re-encoded
        # TODO allow multiple args and returns to increase fuzzing throughput
        # TODO expand this for multiple expressions, more operators, mix of ints and non-ints,
        #    arrays, conditional statements, variable assignments, ...
        ra = sample_range(rng)
        if rng.random() < 0.1:
            rb = ra
        else:
            rb = sample_range(rng)

        ty_a0 = f"int({ra.start}..={ra.end_inc})"
        ty_a1 = f"int({rb.start}..={rb.end_inc})"

        ty_inputs = [ty_a0, ty_a1]
        operator = rng.choice(["+", "-", "*", "/", "%"])  # , "**"])
        expression = f"a0 {operator} a1"

        # check expression validness and extract the return type
        try:
            ty_res_min = expression_get_type(ty_inputs=ty_inputs, expr=expression)

            # TODO check that result type is small enough to avoid power overflows

            # success, we've generated a valid expression
            break
        except hwl.DiagnosticException as e:
            # check that this is once of the expected failure modes
            allowed_messages = [
                "division by zero is not allowed",
                "modulo by zero is not allowed",
                "invalid power operation",
            ]
            if len(e.messages) == 1 and any(m in e.messages[0] for m in allowed_messages):
                continue

            # unexpected error
            raise e

    # parse return type and generate a random range that contains it
    m = re.fullmatch(r"int\((-?\d+)\.\.=(-?\d+)\)", ty_res_min)
    assert m
    range_res_min = Range(start=int(m[1]), end_inc=int(m[2]))
    range_res = sample_range(rng, must_contain=range_res_min)
    ty_res = f"int({range_res.start}..={range_res.end_inc})"

    # generate and compile code
    compiled = expression_compile(ty_inputs=ty_inputs, ty_res=ty_res, expr=expression, build_dir=build_dir)

    # put through some random values
    for _ in range(sample_count):
        val_a = sample_from_range(rng, ra)
        val_b = sample_from_range(rng, rb)

        res_func, res_mod = compiled.eval([val_a, val_b])
        assert res_func == res_mod, f"Mismatch for types `{ty_a0}`, `{ty_a1}`, `{ty_res}`, expression `{expression}`, values `{val_a}` `{val_b}`: function {res_func} != module {res_mod}"


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
    seed = 42
    print(f"Using random seed: {seed}")

    # create threads
    common = Common(
        stopped=False,
        counter_lock=threading.Lock(),
        counter_next=0,
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
