import shutil
from pathlib import Path


def generate_source(n: int):
    assert n >= 0
    result = """
import std.types.[
    bool, int_range, int, uint, natural,
    int_bits, uint_bits, any,
];
"""

    for i in range(n):
        result += f"""
module passthrough_{i} generics(w: int) ports(
    clk: in clock,
    rst: in async bool,
    sync(clk, rst) {{
        select: in bool,
        data_a: in [w]bool,
        data_b: in [w]bool,
        data_out: out [w]bool,
    }}
) body {{"""
        if i == 0:
            result += f"""
    reg out data_out = undef;
    clocked(clk, rst) {{
        data_out = select_{i}([w]bool, select, data_a, data_b);
    }}
"""
        else:
            result += f"""
    instance passthrough_{i - 1} generics(w=w) ports(
        .clk(clk),
        .rst(rst),
        .select(select),
        .data_a(data_a),
        .data_b(data_b),
        .data_out(data_out)
    );"""
        result += f"""
}}
fn select_{i}(T: type, select: bool, a: T, b: T) -> T {{
    var result: T;
    if (select) {{
        result = a;
    }} else {{
        result = b;
    }}
    return result;
}}"""

    result += f"""
pub module top ports(
    clk: in clock,
    rst: in async bool,
    sync(clk, rst) {{
        data_a: in [4]bool,
        data_b: in [4]bool,
        data_out: out [4]bool,
    }}
) body {{
    wire select: sync(clk, rst) bool = true;
    instance passthrough_{n - 1} generics(w=4) ports(
        .clk(clk),
        .rst(rst),
        .select(select),
        .data_a(data_a),
        .data_b(data_b),
        .data_out(data_out)
    );
}}"""

    return result


def main():
    n = 1024 * 32

    curr_path = Path(__file__).parent
    output_path = curr_path / "profile_test"

    source = generate_source(n=n)
    print(f"Generated {len(source.splitlines())} loc")

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir()
    shutil.copytree(curr_path / "../design/project/std", output_path / "std")

    with open(output_path / "top.kh", "w") as f:
        f.write(source)


if __name__ == "__main__":
    main()
