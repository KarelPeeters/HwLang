from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import hwl

from hwl_sandbox.common.util import compile_custom


# TODO also include the result of the C++ backend once that's properly integrated
@dataclass
class CompiledExpression:
    input_count: int

    compile: hwl.Compile

    eval_func: hwl.Function
    eval_mod: hwl.Module
    eval_mod_inst: hwl.VerilatedInstance

    def eval(self, values: List[int]) -> Tuple[int, int]:
        assert len(values) == self.input_count

        # eval func
        val_res_func = self.eval_func(*values)

        # eval module
        ports = self.eval_mod_inst.ports
        for p_name, v in zip(ports, values):
            ports[p_name].value = v
        self.eval_mod_inst.step(1)
        val_res_mod = ports.p_res.value

        return val_res_func, val_res_mod

    def eval_assert(self, values: List[int], expected: int):
        val_res_func, val_res_mod = self.eval(values)
        assert val_res_func == expected, f"Function result {val_res_func} != expected {expected}"
        assert val_res_mod == expected, f"Module result {val_res_mod} != expected {expected}"


def expression_codegen(ty_inputs: List[str], ty_res: str, expr: str) -> hwl.Compile:
    args = ", ".join(f"a{i}: {t}" for i, t in enumerate(ty_inputs))
    ports_in = ", ".join(f"p{i}: in async {t}" for i, t in enumerate(ty_inputs))
    params = ", ".join(f"a{i}=p{i}" for i in range(len(ty_inputs)))

    src = f"""
    import std.types.[any, int];
    import std.util.print;
    fn eval_func({args}) -> int {{
        return {expr};
    }}
    module print_type_mod ports({ports_in}) {{
        wire w_res = eval_func({params});
        const {{
            print(type(w_res));        
        }}
    }}
    module eval_mod ports({ports_in}, p_res: out async {ty_res}) {{
        comb {{
            p_res = eval_func({params});
        }}
    }}
    """

    print(src)

    return compile_custom(src)


def expression_get_type(ty_inputs: List[str], expr: str) -> str:
    c = expression_codegen(ty_inputs, "dummy", expr)
    with c.capture_prints() as capture:
        c.resolve("top.print_type_mod")

    assert len(capture.prints) == 1
    return capture.prints[0]


def expression_compile(ty_inputs: List[str], ty_res: str, expr: str, build_dir: Path) -> CompiledExpression:
    c = expression_codegen(ty_inputs, ty_res, expr)
    eval_func: hwl.Function = c.resolve("top.eval_func")
    eval_mod: hwl.Module = c.resolve("top.eval_mod")

    print(eval_mod.as_verilog().source)
    eval_mod_inst = eval_mod.as_verilated(str(build_dir)).instance()

    return CompiledExpression(
        input_count=len(ty_inputs),
        compile=c,
        eval_func=eval_func,
        eval_mod=eval_mod,
        eval_mod_inst=eval_mod_inst
    )
