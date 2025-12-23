import tempfile
from pathlib import Path

import hwl

EXAMPLE_INPUT = """
L68
L30
R48
L5
R60
L55
L1
L99
R14
L82
""".strip()


def run(m: hwl.VerilatedInstance, input_data: bytes) -> bytes:
    ports = m.ports

    # initial inputs
    ports.clk.value = False
    ports.data_in_valid.value = False
    ports.data_in_data.value = 0
    ports.data_out_ready.value = False

    # reset
    ports.rst.value = False
    m.step(1)
    ports.rst.value = True
    m.step(1)
    ports.rst.value = False
    m.step(1)

    input_data_left = input_data
    result_data = []
    ports.data_out_ready.value = True

    while True:
        ports.clk.value = True
        m.step(1)
        ports.clk.value = False
        m.step(1)

        if ports.error.value:
            raise RuntimeError("Module signaled error")

        # input handshake
        if ports.data_in_valid.value and ports.data_in_ready.value:
            ports.data_in_valid.value = False

        # send new input
        if not ports.data_in_valid.value and input_data_left is not None:
            ports.data_in_valid.value = True
            if len(input_data_left) == 0:
                ports.data_in_data.value = 0
                input_data_left = None
            else:
                ports.data_in_data.value = input_data_left[0]
                input_data_left = input_data_left[1:]

        # receive output
        if ports.data_out_valid.value and ports.data_out_ready.value:
            byte = ports.data_out_data.value
            if byte == 0:
                return bytes(result_data)
            result_data.append(byte)


def main():
    parent = Path(__file__).parent

    input_data = EXAMPLE_INPUT
    # input_data = (parent / "d01.txt").read_text().strip()

    c = hwl.Source.new_from_manifest_path(str(parent / "hwl.toml")).compile()
    m: hwl.Module = c.resolve("d01.d01")
    (parent / "d01.v").write_text(m.as_verilog().source)

    with tempfile.TemporaryDirectory() as build_dir:
        inst = m.as_verilated(Path(build_dir)).instance()
        result = run(inst, input_data.encode("utf-8")).decode("utf-8")

    print(f"Result: {result}")


if __name__ == '__main__':
    main()
