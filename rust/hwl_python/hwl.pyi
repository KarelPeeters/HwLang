from typing import List, Optional, Any, Iterator


def format_file(source: str) -> str: ...


class Source:
    def __init__(self) -> None: ...

    @property
    def files(self) -> List[str]: ...

    @staticmethod
    def new_from_manifest_path(manifest_path: str) -> 'Source': ...

    def add_file_content(self, steps: List[str], debug_info_path: str, content: str) -> None: ...

    def add_tree(self, steps: List[str], path: str) -> None: ...

    def parse(self) -> 'Parsed': ...

    def compile(self) -> 'Compile': ...


class Parsed:
    source: Source

    def compile(self) -> 'Compile': ...


class Compile:
    parsed: Parsed

    def resolve(self, path: str) -> Any: ...

    def capture_prints(self, capture: Optional['CapturePrints'] = None) -> 'CapturePrintsContext': ...


class CapturePrints:
    prints: List[str]

    def __init__(self) -> None: ...


class CapturePrintsContext:
    def __enter__(self) -> CapturePrints: ...

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool: ...


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


class DiagnosticException(HwlException):
    messages: List[str]
    messages_colored: List[str]


class ResolveException(HwlException): ...


class GenerateVerilogException(HwlException): ...


class VerilationException(HwlException): ...
