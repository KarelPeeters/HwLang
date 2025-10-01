import random
import shutil
from pathlib import Path

import hwl
import itertools

from hwl_sandbox.common.util import compile_custom


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
    op = "+"

    # TODO somehow compute output range? can we ask the compiler?
    #   or just randomly generate one and get rejected if it doesn't fit
    # TODO add "expansion" tests, where the output range is too large, to see if the value is properly re-encoded
    ty_a = f"int({ra.start}..{ra.stop})"
    ty_b = f"int({rb.start}..{rb.stop})"
    ty_res = f"int(-4096..=4096)"

    # generate code
    src = f"""
    import std.types.int;
    fn compute_func(a: {ty_a}, b: {ty_b}) -> {ty_res} {{
        return a {op} b;
    }}
    module compute_mod ports(a: in async {ty_a}, b: in async {ty_b}, res: out async {ty_res}) {{
        comb {{
            res = compute_func(a, b);
        }}
    }}
    """

    # compile code
    # TODO also include C++ backend in this comparison
    com = compile_custom(src)
    compute_func: hwl.Function = com.resolve("top.compute_func")
    compute_mod: hwl.Module = com.resolve("top.compute_mod")
    compile_ver = compute_mod.as_verilated(build_dir=str(build_dir)).instance()

    # put through some random values
    for _ in range(sample_count):
        val_a = random.choice(ra)
        val_b = random.choice(rb)

        val_res_func = compute_func(val_a, val_b)

        compile_ver.ports.a.value = val_a
        compile_ver.ports.b.value = val_b
        compile_ver.step(1)
        val_res_ver = compile_ver.ports.res.value

        print(f"{val_a} {op} {val_b} = {val_res_func} (func) vs {val_res_ver} (verilog)")


def main():
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
