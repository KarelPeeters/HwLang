from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import hwl

from hwl_sandbox.common.util import compile_custom


# TODO also include the result of the C++ backend once that's properly integrated
@dataclass
class CompiledExpression:
    compile: hwl.Compile

    eval_func: hwl.Function
    eval_mod: hwl.Module
    eval_mod_inst: hwl.VerilatedInstance

    def eval(self, val_a: int, val_b: int) -> Tuple[int, int]:
        val_res_func = self.eval_func(val_a, val_b)

        self.eval_mod_inst.ports.a.value = val_a
        self.eval_mod_inst.ports.b.value = val_b
        self.eval_mod_inst.step(1)
        val_res_mod = self.eval_mod_inst.ports.res.value

        return val_res_func, val_res_mod

    def eval_assert(self, val_a: int, val_b: int, expected: int):
        val_res_func, val_res_mod = self.eval(val_a, val_b)
        assert val_res_func == expected, f"Function result {val_res_func} != expected {expected}"
        assert val_res_mod == expected, f"Module result {val_res_mod} != expected {expected}"


def compile_expression(ty_a: str, ty_b: str, ty_res: str, expr: str, build_dir: Path) -> CompiledExpression:
    src = f"""
    import std.types.int;
    fn eval_func(a: {ty_a}, b: {ty_b}) -> {ty_res} {{
        return {expr};
    }}
    module eval_mod ports(a: in async {ty_a}, b: in async {ty_b}, res: out async {ty_res}) {{
        comb {{
            res = eval_func(a, b);
        }}
    }}
    """

    com = compile_custom(src)
    eval_func: hwl.Function = com.resolve("top.eval_func")
    eval_mod: hwl.Module = com.resolve("top.eval_mod")
    eval_mod_inst = eval_mod.as_verilated(str(build_dir)).instance()

    return CompiledExpression(com, eval_func, eval_mod, eval_mod_inst)
