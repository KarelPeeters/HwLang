import random
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path

from hwl_sandbox.common.expression import compile_expression
from hwl_sandbox.common.util import enable_rust_backtraces


def sample_range(rng: random.Random) -> range:
    # TODO speed up by uniformly sampling from triangle
    # TODO increase range
    # TODO allow empty ranges

    while True:
        start = rng.randint(-1024, 1024)
        end = rng.randint(-1024, 1024)

        if start <= end:
            return range(start, end + 1)


def fuzz_step(build_dir: Path, sample_count: int, rng: random.Random):
    # decide types and operations
    ra = sample_range(rng)
    rb = sample_range(rng)

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
        val_a = rng.choice(ra)
        val_b = rng.choice(rb)

        res_func, res_mod = compiled.eval([val_a, val_b])
        assert res_func == res_mod, f"Mismatch for types `{ty_a}`, `{ty_b}`, `{ty_res}`, expression `{expression}`, values `{val_a}` `{val_b}`: function {res_func} != module {res_mod}"


def main_iteration(build_dir_base: Path, sample_count: int, seed_base: int, i: int):
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
    thread_count = 4
    build_dir_base = Path(__file__).parent / "../../../build/" / Path(__file__).stem

    # random seed
    seed = random.randint(0, 2 ** 64 - 1)
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
