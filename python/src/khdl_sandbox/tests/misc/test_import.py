import khdl
import pytest

from khdl_sandbox.common.util import compile_custom_files, diag_error

LIB = """
pub fn f() -> int { return 1; }
fn private() -> int { return 2; }
"""


def test_basic():
    files = {
        "a.b": "pub const c = 4;",
        "top": "import a.b; const d = b.c;",
    }
    c = compile_custom_files(files)
    assert c.resolve("top.d") == 4


def test_import_package():
    for files in [
        {"lib": LIB, "top": "import lib; fn g() -> int { return lib.f(); }"},
        {"a.b.lib": LIB, "top": "import a.b.lib; fn g() -> int { return lib.f(); }"},
        {"lib": LIB, "top": "import lib as renamed; fn g() -> int { return renamed.f(); }"},
        {"lib": LIB, "top": "import [lib]; fn g() -> int { return lib.f(); }"},
    ]:
        assert compile_custom_files(files).resolve("top.g")() == 1


def test_import_package_and_item():
    files = {
        "a.lib": LIB,
        "a.other": "pub const x = 5;",
        "top": "import a.[lib, other]; import a.lib.f; fn g() -> int { return lib.f() + other.x + f(); }",
    }
    assert compile_custom_files(files).resolve("top.g")() == 7


def test_package_is_first_class_value():
    files = {
        "lib": LIB,
        "other": "pub fn f() -> int { return 2; }",
        "top": """
        import [lib, other];
        fn call(package: any) -> int { return package.f(); }
        fn g() -> any { return (call(lib), call(other), lib == lib, lib == other); }
        fn get() -> any { return lib; }
        """,
    }
    c = compile_custom_files(files)
    assert c.resolve("top.g")() == (1, 2, True, False)

    # package values can also be used from python
    assert c.resolve("top.get")().f() == 1


def test_package_does_not_expose_imports():
    files = {
        "other": "pub const x = 5;",
        "lib": "import other.x; pub const y = x;",
        "top": "import lib; fn g() -> int { return lib.y; } fn h() -> int { return lib.x; }",
    }
    c = compile_custom_files(files)
    assert c.resolve("top.g")() == 5
    with diag_error("import not found"):
        c.resolve("top.h")()


def test_package_private_item():
    files = {"lib": LIB, "top": "import lib; fn g() -> int { return lib.private(); }"}
    with diag_error("cannot import private item `private`"):
        compile_custom_files(files).resolve("top.g")()


def test_item_child_conflict():
    # items with the same name as a child are reported at the declaration, exactly once
    with diag_error("item `b` has the same name as a child package"):
        compile_custom_files({"a": "pub const b = 1;", "a.b": ""})
    with diag_error("item `b` has the same name as a child package"):
        compile_custom_files({"a": "pub const b = 1;", "a.b": "", "top": "import a.b;"})
    with diag_error("item `b` has the same name as a child package"):
        compile_custom_files({"a": "pub const b = 1;", "a.b.c": ""})


def test_import_not_found():
    for top in ["import lib.missing;", "import missing.lib;", "import lib.f.x;"]:
        with diag_error("import not found"):
            compile_custom_files({"lib": LIB, "top": top})


def test_private_through_own_file():
    # private items are blocked through package paths, even from within the declaring file
    files = {"lib": "import lib; fn p() -> int { return 1; } pub fn g() -> int { return lib.p(); }"}
    with diag_error("cannot import private item `p`"):
        compile_custom_files(files).resolve("lib.g")()


def test_resolve_module_wrong_type():
    c = compile_custom_files({"lib": LIB})
    with pytest.raises(TypeError, match="does not resolve to a module, got value with type `Function`"):
        c.resolve_module("lib.f")
    with pytest.raises(TypeError, match="does not resolve to a module, got value with type `Package`"):
        c.resolve_module("lib")


def test_package_missing_item():
    files = {"lib": LIB, "top": "import lib; fn g() -> int { return lib.missing; }"}
    with diag_error("import not found"):
        compile_custom_files(files).resolve("top.g")()


def test_resolve_does_not_find_imports():
    c = compile_custom_files({"lib": LIB, "top": "import lib.f; fn g() -> int { return f(); }"})
    assert c.resolve("top.g")() == 1

    # resolve only sees items declared in the file itself, not imported or prelude items
    with diag_error("import not found"):
        c.resolve("top.f")
    with diag_error("import not found"):
        c.resolve("top.int")


def test_import_private_item():
    with diag_error("cannot import private item `private`"):
        compile_custom_files({"lib": LIB, "top": "import lib.private;"})


def test_import_not_found_options():
    files = {
        "a": "pub const x = 1; const hidden = 2;",
        "a.child": "",
        "a.dir.deep": "",
        "top": "import a.missing;",
    }
    with pytest.raises(khdl.DiagnosticException) as e:
        compile_custom_files(files)

    # child nodes and public items are suggested, private items are not
    (diag,) = e.value.diagnostics
    assert diag.title == "import not found"
    assert 'possible options: ["child", "dir", "x"]' in diag.full_string


def test_lookups_share_messages():
    # imports, dot-indexing and python all use the same lookup, so they report the same error
    title = "import not found"
    with diag_error(title):
        compile_custom_files({"lib": LIB, "top": "import lib.missing;"})
    with diag_error(title):
        compile_custom_files({"lib": LIB, "top": "import lib; fn g() -> int { return lib.missing; }"}).resolve(
            "top.g")()
    with diag_error(title):
        compile_custom_files({"lib": LIB}).resolve("lib.missing")


def test_resolve_package_and_private():
    c = compile_custom_files({"a.lib": LIB})

    # python can resolve packages, and can access private items for now
    assert c.resolve("a.lib").f() == 1
    assert c.resolve("a.lib.private")() == 2
    assert repr(c.resolve("a.lib")) == "<package a.lib>"
    assert repr(c.resolve("a")) == "<package a>"

    # attribute access on packages uses the normal rules, so private items are not accessible
    with diag_error("cannot import private item `private`"):
        _ = c.resolve("a.lib").private
    with pytest.raises(AttributeError):
        _ = c.resolve("a.lib").missing


def test_resolve_empty_path():
    with diag_error("path cannot be empty"):
        compile_custom_files({"lib": LIB}).resolve("")


def test_resolve_empty_step():
    c = compile_custom_files({"lib": LIB})
    for path in ["lib.", ".lib", "lib..f"]:
        with diag_error(f"path `{path}` contains an empty step"):
            c.resolve(path)


def test_dot_index_child_nodes():
    # dot-indexing into a package can also access child packages
    for files in [
        {"lib": LIB, "lib.child": "pub const x = 1;", "top": "import lib; fn g() -> any { return lib.child.x; }"},
        {"a": "", "a.b.c": "pub const x = 1;", "top": "import a; fn g() -> any { return a.b.c.x; }"},
    ]:
        assert compile_custom_files(files).resolve("top.g")() == 1


def test_package_type():
    files = {
        "lib": LIB,
        "other": "pub fn f() -> int { return 2; }",
        "top": """
        import [lib, other];
        fn pick(c: bool) -> Package {
            var package: Package = lib;
            if (c) { package = other; }
            return package;
        }
        fn g(c: bool) -> int { return pick(c).f(); }
        fn types() -> any { return (Package, type(lib)); }
        """,
    }
    c = compile_custom_files(files)
    assert c.resolve("top.g")(False) == 1
    assert c.resolve("top.g")(True) == 2
    assert str(c.resolve("top.types")()) == "(Package, Package)"


def test_package_in_module():
    files = {
        "lib": "pub fn f() -> int { return 1; }",
        "top": """
        import lib;
        module with_param(F: Package) ports(y: out async int(0..4)) { comb { y = F.f(); } }
        module with_import ports(y: out async int(0..4)) { comb { y = lib.f(); } }
        """,
    }
    c = compile_custom_files(files)
    c.resolve("top.with_param")(c.resolve("lib")).as_verilog()
    c.resolve_module("top.with_import").as_verilog()


def test_prelude_shadowing():
    # declared items, imported items and imported packages all shadow prelude items
    files = {
        "int": "pub const x = 1;",
        "lib": "pub const bool = 2;",
        "top": """
        import int;
        import lib.bool;
        const clog2 = 3;
        fn g() -> any { return (int.x, bool, clog2); }
        """,
    }
    assert compile_custom_files(files).resolve("top.g")() == (1, 2, 3)


def test_import_duplicate():
    with diag_error("identifier `lib` declared multiple times"):
        compile_custom_files({"lib": LIB, "top": "import lib; import [lib];"})
    with diag_error("identifier `lib` declared multiple times"):
        compile_custom_files({"lib": LIB, "top": "import lib; fn lib() {}"})
