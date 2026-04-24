import json
import os
from pathlib import Path

from hwl_sandbox.common.util import compile_custom


def build_wave_gui_example_json(build_dir: Path, output_path: Path) -> Path:
    src = """
    struct Packet {
        tag: uint(4),
        payload: [2]uint(8),
        valid: bool,
    }

    module leaf ports(
        clk: in clock,
        rst: in async bool,
        enable: in sync(clk, async rst) bool,
        tag_in: in sync(clk, async rst) uint(4),
        sample: in sync(clk, async rst) uint(8),
        packet: out sync(clk, async rst) Packet,
        parity: out sync(clk, async rst) bool,
        debug: out sync(clk, async rst) Tuple(uint(8), bool),
    ) {
        wire mixed: uint(8);
        wire flag: bool;
        comb {
            mixed = 255 - sample;
            flag = enable ^ (sample == 0);
        }

        clocked(clk, async rst) {
            reg wire packet = Packet.new(tag=0, payload=[0, 0], valid=false);
            reg wire parity = false;
            reg wire debug = (0, false);

            packet = Packet.new(tag=tag_in, payload=[sample, mixed], valid=enable);
            parity = flag;
            debug = (mixed, flag);
        }
    }

    module top ports(
        clk: in clock,
        rst: in async bool,
        mode: in sync(clk, async rst) Tuple(uint(4), bool),
        x: in sync(clk, async rst) uint(8),
        y: out sync(clk, async rst) Packet,
        summary: out sync(clk, async rst) Tuple(Packet, [2]uint(8), bool),
    ) {
        wire enable_left: bool;
        wire enable_right: bool;
        wire tag_left: uint(4);
        wire tag_right: uint(4);
        wire left_packet: Packet;
        wire right_packet: Packet;
        wire left_parity: bool;
        wire right_parity: bool;
        wire left_debug: Tuple(uint(8), bool);
        wire right_debug: Tuple(uint(8), bool);
        wire right_sample: uint(8);

        comb {
            enable_left = mode.1;
            enable_right = !mode.1;
            tag_left = mode.0;
            tag_right = 15 - mode.0;
            right_sample = 255 - x;
        }

        instance left = leaf ports(
            clk=clk,
            rst=rst,
            enable=enable_left,
            tag_in=tag_left,
            sample=x,
            packet=left_packet,
            parity=left_parity,
            debug=left_debug,
        );
        instance right = leaf ports(
            clk=clk,
            rst=rst,
            enable=enable_right,
            tag_in=tag_right,
            sample=right_sample,
            packet=right_packet,
            parity=right_parity,
            debug=right_debug,
        );

        clocked(clk, async rst) {
            reg wire y = Packet.new(tag=0, payload=[0, 0], valid=false);
            reg wire summary = (Packet.new(tag=0, payload=[0, 0], valid=false), [0, 0], false);

            if (mode.1) {
                y = left_packet;
                summary = (left_packet, [left_debug.0, right_debug.0], left_parity);
            } else {
                y = right_packet;
                summary = (right_packet, [right_debug.0, left_debug.0], right_parity);
            }
        }
    }
    """

    top = compile_custom(src).resolve("top.top")
    build_dir.mkdir(parents=True, exist_ok=True)
    inst = top.as_cpp(build_dir).instance()
    rec = inst.start_recording()
    ports = inst.ports

    def settle(time: int, mode: tuple[int, bool], x: int):
        ports.mode.value = mode
        ports.x.value = x
        inst.step(1)
        rec.sample(time)

    def tick(time: int):
        ports.clk.value = True
        inst.step(1)
        rec.sample(time)
        ports.clk.value = False
        inst.step(1)
        rec.sample(time + 1)

    ports.clk.value = False
    ports.rst.value = False
    settle(0, (0, False), 0)
    ports.rst.value = True
    inst.step(1)
    rec.sample(1)
    ports.rst.value = False
    inst.step(1)
    rec.sample(2)

    for i, x in enumerate([0x12, 0x34, 0x56, 0x78, 0x9A]):
        time = 3 + i * 3
        settle(time, (i & 0xF, i % 2 == 0), x)
        tick(time + 1)

    rec.save_json(output_path)
    return output_path


def test_wave_gui_example_json_export(tmp_dir: Path):
    default_output_path =  "wave_gui_example.json"
    output_path = Path(os.environ.get("HWL_WAVE_GUI_EXAMPLE_OUT", default_output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    path = build_wave_gui_example_json(tmp_dir / "build", output_path)
    store = json.loads(path.read_text())

    signal_names = {(".".join(signal["path"]), signal["name"]) for signal in store["signals"]}
    assert ("top.left", "packet") in signal_names
    assert ("top.right", "debug") in signal_names
    assert ("top", "summary") in signal_names
    assert any(len(changes) > 1 for changes in store["changes"])

    print(f"wave_gui example JSON written to {path}")
