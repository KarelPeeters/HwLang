import shutil
import textwrap
from pathlib import Path


def generate_source(depth: int, width: int):
    assert depth >= 1
    assert width >= 1
    result = ""

    for level in range(depth):
        result += f"""
module passthrough_{level}(w: int, dummy: int) ports(
    clk: in clock,
    rst: in async bool,
    sync(clk, async rst) {{
        select: in bool,
        data_a: in [w]bool,
        data_b: in [w]bool,
        data_out: out [w]bool,
    }}
) {{"""
        if level == 0:
            result += f"""
    clocked(clk, async rst) {{
        reg wire data_out = undef;
        data_out = select_{level}([w]bool, select, data_a, data_b);
    }}
"""
        else:
            result += f"""
    instance passthrough_{level - 1}(w=w, dummy=dummy) ports(
        clk,
        rst,
        select,
        data_a,
        data_b,
        data_out,
    );"""
        result += f"""
}}
fn select_{level}(T: type, select: bool, a: T, b: T) -> T {{
    var result: T;
    if (select) {{
        result = a;
    }} else {{
        result = b;
    }}
    return result;
}}"""

    ports: str = ""
    instances: str = ""
    for lane in range(width):
        ports += f"        data_a_{lane}: in [4]bool,\n"
        ports += f"        data_b_{lane}: in [4]bool,\n"
        ports += f"        data_out_{lane}: out [4]bool,\n"

        instances += f"    instance passthrough_{depth - 1}(w=4, dummy={lane}) ports(\n"
        instances += f"        clk,\n"
        instances += f"        rst,\n"
        instances += f"        select,\n"
        instances += f"        data_a=data_a_{lane},\n"
        instances += f"        data_b=data_b_{lane},\n"
        instances += f"        data_out=data_out_{lane},\n"
        instances += f"    );\n"

    result += f"""
pub module top ports(
    clk: in clock,
    rst: in async bool,
    sync(clk, async rst) {{
{ports}    }}
) {{
    wire select: sync(clk, async rst) bool = true;
{instances}}}"""

    return result


def write_project(output_path: Path, depth: int, width: int):
    source = generate_source(depth=depth, width=width)

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    manifest = """
    [source]
    _ = "."
    """

    (output_path / "hwl.toml").write_text(textwrap.dedent(manifest).lstrip())
    (output_path / "top.kh").write_text(source)

    print(f"Wrote depth={depth} width={width} lines={source.count("\n")} to {output_path}")


def main():
    curr_path = Path(__file__).parent
    base_output_path = curr_path / "../../../../ignored/profile_test"

    write_project(base_output_path / "deep", depth=1024 * 32, width=1)
    write_project(base_output_path / "wide", depth=1024 * 32, width=32)


if __name__ == "__main__":
    main()
