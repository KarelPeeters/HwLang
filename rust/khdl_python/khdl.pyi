from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator, Literal, Union, overload


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

    def resolve_module(self, path: str) -> 'Module': ...

    def capture_prints(self, capture: Optional['CapturePrints'] = None) -> 'CapturePrintsContext': ...


class CapturePrints:
    prints: List[str]

    def __init__(self) -> None: ...


class CapturePrintsContext:
    def __enter__(self) -> CapturePrints: ...

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool: ...


class Value:
    def __repr__(self) -> str: ...

    def __eq__(self, other: Any) -> bool: ...

    def __ne__(self, other: Any) -> bool: ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    def __getattr__(self, name: str) -> Any: ...


class StructValue(Value):
    @property
    def fields(self) -> Dict[str, Any]: ...


class EnumValue(Value):
    @property
    def tag(self) -> str: ...

    @property
    def payload(self) -> Optional[Any]: ...


class Range:
    def __init__(self, start: Optional[int], end: Optional[int]) -> None: ...

    @property
    def start(self) -> Optional[int]: ...

    @property
    def end(self) -> Optional[int]: ...

    def contains(self, value: int) -> bool: ...

    def contains_range(self, other: 'Range') -> bool: ...

    def __str__(self) -> str: ...


class Module(Value):
    def as_verilog(self) -> 'ModuleVerilog': ...

    def as_verilated(self, build_dir: Path, *, extra_verilog_files: Optional[List[Path]] = None, optimization: Optional[int] = None, compiler: Optional[Literal['gcc', 'clang']] = None) -> 'ModuleVerilated': ...


class ModuleVerilog:
    module_name: str
    source: str


class ModuleVerilated:
    def instance(self, trace_path: Optional[str] = None) -> 'VerilatedInstance': ...


class VerilatedInstance:
    @property
    def ports(self) -> 'VerilatedPorts': ...

    @property
    def ports_flat(self) -> 'VerilatedPortsFlat': ...

    def step(self, increment_time: int) -> None: ...

    def save_trace(self) -> None: ...


class VerilatedPorts:
    """The ports as declared in the original module, with interface and interface arrays kept intact"""

    def __getattr__(self, name: str) -> Union['VerilatedPort', 'VerilatedInterface', 'VerilatedInterfaceArray']: ...

    def __getitem__(self, key: str) -> Union['VerilatedPort', 'VerilatedInterface', 'VerilatedInterfaceArray']: ...

    def __setattr__(self, name: str, value: Any) -> None: ...

    def __iter__(self) -> Iterator[str]: ...


class VerilatedPortsFlat:
    """The ports as a flattened list"""

    def __getattr__(self, name: str) -> 'VerilatedPort': ...

    def __getitem__(self, key: str) -> 'VerilatedPort': ...

    def __setattr__(self, name: str, value: Any) -> None: ...

    def __iter__(self) -> Iterator[str]: ...


class VerilatedInterface:
    def __getattr__(self, name: str) -> 'VerilatedPort': ...

    def __getitem__(self, key: str) -> 'VerilatedPort': ...

    def __setattr__(self, name: str, value: Any) -> None: ...

    def __iter__(self) -> Iterator[str]: ...


class VerilatedInterfaceArray:
    @property
    def shape(self) -> tuple[int, ...]: ...

    @overload
    def __getitem__(self, index: int) -> Union['VerilatedInterface', 'VerilatedInterfaceArray']: ...

    @overload
    def __getitem__(self, index: slice) -> 'VerilatedInterfaceArray': ...

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[Union['VerilatedInterface', 'VerilatedInterfaceArray']]: ...


class VerilatedPort:
    @property
    def value(self) -> Any: ...

    @value.setter
    def value(self, value: Any) -> None: ...

    @property
    def type(self) -> Value: ...

    @property
    def name(self) -> str: ...

    @property
    def direction(self) -> str: ...

    def __bool__(self) -> bool: ...


class KhdlException(Exception): ...


class DiagnosticException(KhdlException):
    diagnostics: List[Diagnostic]
    combined_string: str
    combined_string_colored: str


class Diagnostic:
    level: Literal["error", "warning"]
    title: str
    messages: list[str]
    infos: list[str]
    full_string: str
    full_string_colored: str


class SourceSetException(KhdlException): ...


class DiagnosticPreviouslyReportedException(KhdlException): ...


class GenerateVerilogException(KhdlException): ...


class VerilationException(KhdlException): ...


class SimulationFinishedException(KhdlException): ...
