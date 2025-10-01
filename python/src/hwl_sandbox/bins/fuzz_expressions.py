import random
import shutil
from pathlib import Path

import itertools

from hwl_sandbox.common.expression import compile_expression


def sample_range():
    # TODO speed up by uniformly sampling from triangle
    # TODO increase range
    # TODO allow empty ranges

    while True:
        start = random.randint(-1024, 1024)
        end = random.randint(-1024, 1024)

        if start <= end:
            return range(start, end + 1)


def fuzz_step(build_dir: Path, sample_count: int):
    # decide types and operations
    ra = sample_range()
    rb = sample_range()

    # TODO somehow compute output range? can we ask the compiler?
    #   or just randomly generate one and get rejected if it doesn't fit
    # TODO add "expansion" tests, where the output range is too large, to see if the value is properly re-encoded
    # TODO allow multiple args and returns to increase fuzzing throughput

    ty_a = f"int({ra.start}..{ra.stop})"
    ty_b = f"int({rb.start}..{rb.stop})"
    ty_res = f"int(-4096..=4096)"
    expression = "a0 + a1"

    # generate and compile code
    compiled = compile_expression(ty_inputs=[ty_a, ty_b], ty_res=ty_res, expr=expression, build_dir=build_dir)

    # put through some random values
    for _ in range(sample_count):
        val_a = random.choice(ra)
        val_b = random.choice(rb)

        res_func, res_mod = compiled.eval([val_a, val_b])
        assert res_func == res_mod, f"Mismatch for types `{ty_a}`, `{ty_b}`, `{ty_res}`, expression `{expression}`, values `{val_a}` `{val_b}`: function {res_func} != module {res_mod}"


def main():
    random.seed(42)

    sample_count = 100
    build_dir_base = Path(__file__).parent / "../../../build/" / Path(__file__).stem

    for i in itertools.count():
        print(f"Starting fuzz iteration: {i}")

        # create a new build dir for each iteration to avoid issues with dlopen caching old versions
        build_dir = build_dir_base / f"iter_{i}"
        shutil.rmtree(build_dir, ignore_errors=True)
        build_dir.mkdir(parents=True, exist_ok=False)

        fuzz_step(build_dir=build_dir, sample_count=sample_count)

        shutil.rmtree(build_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
