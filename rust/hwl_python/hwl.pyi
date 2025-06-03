from typing import List, Optional, Any, Iterator


class SourceBuilder:
    def __init__(self) -> None: ...

    def add_file(self, path: List[str], path_raw: str, source: str) -> None: ...

    def add_tree(self, prefix: List[str], path: str) -> None: ...

    def finish(self) -> 'Source': ...


class Source:
    @property
    def files(self) -> List[str]: ...

    @staticmethod
    def simple(path: str) -> 'Source': ...

    def parse(self) -> 'Parsed': ...

    def compile(self) -> 'Compile': ...


class Parsed:
    source: Source

    def compile(self) -> 'Compile': ...


class Compile:
    parsed: Parsed

    def resolve(self, path: str) -> Any: ...


class UnsupportedValue:
    def __repr__(self) -> str: ...


class Undefined: ...


class Type:
    def __str__(self) -> str: ...


class IncRange:
    start_inc: Optional[int]
    end_inc: Optional[int]

    def __init__(self, start_inc: Optional[int], end_inc: Optional[int]) -> None: ...

    def __str__(self) -> str: ...


class Function:
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class Module:
    def as_verilog(self) -> 'ModuleVerilog': ...

    def as_verilated(self, build_dir: str) -> 'ModuleVerilated': ...


class ModuleVerilog:
    module_name: str
    source: str


class ModuleVerilated:
    def instance(self, trace_path: Optional[str] = None) -> 'VerilatedInstance': ...


class VerilatedInstance:
    @property
    def ports(self) -> 'VerilatedPorts': ...

    def step(self, increment_time: int) -> None: ...

    def save_trace(self) -> None: ...


class VerilatedPorts:
    def __getattr__(self, name: str) -> 'VerilatedPort': ...

    def __getitem__(self, key: str) -> 'VerilatedPort': ...

    def __setattr__(self, name: str, value: Any) -> None: ...

    def __iter__(self) -> Iterator[str]: ...


class VerilatedPort:
    @property
    def value(self) -> Any: ...

    @value.setter
    def value(self, value: Any) -> None: ...

    @property
    def type(self) -> Type: ...

    @property
    def name(self) -> str: ...

    @property
    def direction(self) -> str: ...

    def __bool__(self) -> bool: ...


class HwlException(Exception): ...


class SourceSetException(HwlException): ...


class DiagnosticException(HwlException): ...


class ResolveException(HwlException): ...


class GenerateVerilogException(HwlException): ...


class VerilationException(HwlException): ...
