from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import hwl

from hwl_sandbox.common.util import compile_custom


@dataclass
class CompiledCompare:
    input_count: int

    compile: hwl.Compile

    eval_func: hwl.Value
    eval_mod: hwl.Module
    eval_mod_inst_ver: hwl.VerilatedInstance
    eval_mod_inst_sim: hwl.SimulatorInstance

    def eval(self, values: List[object]) -> Tuple[object, object, object]:
        assert len(values) == self.input_count, \
            f"Input value count mismatch, expected {self.input_count}, got {len(values)}"

        # eval func
        result_func = self.eval_func(*values)

        # eval verilator
        ports_ver = self.eval_mod_inst_ver.ports
        for p_name, v in zip(ports_ver, values):
            ports_ver[p_name].value = v
        self.eval_mod_inst_ver.step(1)
        result_mod_ver = ports_ver.p_res.value

        # eval llvm sim
        ports_sim = self.eval_mod_inst_sim.ports
        for p_name, v in zip(ports_sim, values):
            ports_sim[p_name].value = v
        self.eval_mod_inst_sim.step(1)
        result_mod_sim = ports_sim.p_res.value

        return result_func, result_mod_ver, result_mod_sim

    def eval_assert(self, values: List[object], expected: object):
        result_func, result_mod_ver, result_mod_sim = self.eval(values)
        assert result_func == expected, f"Function result {result_func} != expected {expected}"
        assert result_mod_ver == expected, f"Verilator result {result_mod_ver} != expected {expected}"
        assert result_mod_sim == expected, f"LLVM sim result {result_mod_sim} != expected {expected}"


def compare_codegen(ty_inputs: List[str], ty_res: str, body: str, prefix: str) -> hwl.Compile:
    args = ", ".join(f"a{i}: {t}" for i, t in enumerate(ty_inputs))
    ports_in = ", ".join(f"p{i}: in async {t}" for i, t in enumerate(ty_inputs))
    params = ", ".join(f"a{i}=p{i}" for i in range(len(ty_inputs)))
    ports_comma = ", " if ports_in else ""

    body_indented = "\n".join("    " + s for s in body.splitlines())

    src = f"""
{prefix}
fn eval_func({args}) -> any {{
{body_indented}
}}
module print_type_mod ports({ports_in}) {{
    wire w_res = eval_func({params});
    const {{
        print(type(w_res), end="");        
    }}
}}
module eval_mod ports({ports_in}{ports_comma}p_res: out async {ty_res}) {{
    comb {{
        p_res = eval_func({params});
    }}
}}
    """

    # print(src)

    return compile_custom(src)


def compare_get_type(ty_inputs: List[str], body: str, prefix: str) -> str:
    c = compare_codegen(ty_inputs, "dummy", body, prefix)
    with c.capture_prints() as capture:
        c.resolve("top.print_type_mod")

    assert len(capture.prints) == 1
    return capture.prints[0]


def compare_body(
        ty_inputs: List[str],
        ty_res: str,
        body: str,
        build_dir: Path,
        prefix: str = ""
) -> CompiledCompare:
    c = compare_codegen(ty_inputs, ty_res, body, prefix)
    eval_func = c.resolve("top.eval_func")
    eval_mod = c.resolve_module("top.eval_mod")

    print(eval_mod.as_verilog().source)

    build_dir.mkdir(parents=True, exist_ok=True)
    eval_mod_inst_ver = eval_mod.as_verilated(build_dir).instance()
    eval_mod_inst_sim = eval_mod.as_simulator().instance()

    return CompiledCompare(
        input_count=len(ty_inputs),
        compile=c,
        eval_func=eval_func,
        eval_mod=eval_mod,
        eval_mod_inst_ver=eval_mod_inst_ver,
        eval_mod_inst_sim=eval_mod_inst_sim,
    )


def compare_expression(
        ty_inputs: List[str], ty_res: str, expr: str, build_dir: Path, prefix: str = ""
) -> CompiledCompare:
    body = f"return {expr};"
    return compare_body(ty_inputs=ty_inputs, ty_res=ty_res, body=body, build_dir=build_dir, prefix=prefix)
