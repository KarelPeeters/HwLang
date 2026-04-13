#!/usr/bin/env python3
"""
Generate Rust test cases for the hwl_vhdl parser from external NVC and GHDL test suites.

Only includes files with valid VHDL syntax (no intentional parse errors).

Usage:
    python3 scripts/generate_external_parse_tests.py

Outputs:
    rust/hwl_vhdl/src/tests/test_cases_nvc.rs
    rust/hwl_vhdl/src/tests/test_cases_ghdl.rs
"""

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NVC_DIR = ROOT / "external" / "nvc"
GHDL_DIR = ROOT / "external" / "ghdl"
RUST_TESTS = ROOT / "rust" / "hwl_vhdl" / "src" / "tests"

# Relative path prefix from the Rust crate root to the repo root
REL_PREFIX = "../../external"

# Directories/files to exclude (non-VHDL-2019 or broken test code)
# bug030: uses VHDL-87 reserved words as identifiers (context, protected, xnor)
# ashenden ch_18_fg_18_09: incomplete textbook excerpt with elided procedure bodies
# billowitch tc2862: uses % as string delimiter (VHDL-87 only, removed in VHDL-93)
# bug096: uses VHDL-87 file open syntax (is in "filename")
GHDL_EXCLUDE_DIRS = {"bug030", "bug096"}
GHDL_EXCLUDE_FILES = {
    "ch_18_fg_18_09.vhd",  # ashenden: incomplete textbook excerpt
    "tc2853.vhd",          # billowitch: ! as | replacement (VHDL-87)
    "tc2860.vhd",          # billowitch: % as " replacement (VHDL-87)
    "tc2862.vhd",          # billowitch: % as " replacement (VHDL-87)
}

# NVC files to exclude (non-VHDL-2019 compliant syntax)
# range4.vhd: chains relational operators (a > b = false) which is not valid per LRM §9.2
NVC_EXCLUDE_FILES = {
    "range4.vhd",
}


def find_nvc_error_files_from_c(c_file, dirname):
    """Parse a NVC test_*.c file to find files that have expected errors."""
    c_path = NVC_DIR / "test" / c_file
    if not c_path.exists():
        return set()

    content = c_path.read_text()
    pattern = r"START_TEST\(\w+\)\s*\{(.*?)\}\s*END_TEST"
    matches = re.findall(pattern, content, re.DOTALL)

    error_files = set()
    for body in matches:
        if "expect_errors" in body:
            files = re.findall(
                rf'input_from_file\(TESTDIR\s*"/{re.escape(dirname)}/([^"]+)"',
                body,
            )
            error_files.update(files)

    return error_files


# Map NVC test directories to their C test files
NVC_TEST_C_FILES = {
    "parse": "test_parse.c",
    "sem": "test_sem.c",
    "elab": "test_elab.c",
    "lower": "test_lower.c",
    "bounds": "test_bounds.c",
}


def is_psl_file(path):
    """Check if a file is a PSL test file (which we skip)."""
    name = path.name.lower()
    return "psl" in name


def has_preprocessor_directives(path):
    """Check if a file uses GHDL preprocessor directives (--V87/--!V87)."""
    try:
        data = path.read_bytes()
        return b"--V87" in data or b"--!V87" in data
    except OSError:
        return True


def has_guillemets(path):
    """Check if a file contains guillemet characters (« » = 0xAB/0xBB)."""
    try:
        data = path.read_bytes()
        return b"\xab" in data or b"\xbb" in data
    except OSError:
        return True


def is_valid_vhdl_file(path):
    """Combined filter: returns True if the file should be included."""
    if is_psl_file(path):
        return False
    if has_guillemets(path):
        return False
    if has_preprocessor_directives(path):
        return False
    return True


def is_vhdl_ams_file(path):
    """Check if a file uses VHDL-AMS constructs (a separate standard we skip)."""
    try:
        data = path.read_bytes().lower()
        # Quick byte check for common AMS keywords
        return b"nature " in data or b"terminal " in data or b"quantity " in data
    except OSError:
        return True


def has_psl_content(path):
    """Check if a file contains PSL constructs (not just PSL in name)."""
    try:
        data = path.read_bytes().lower()
        return (b"-- psl" in data or b"vunit " in data or b"default clock" in data
                or b"async_abort" in data or b"sync_abort" in data
                or b"\nsequence " in data or b"\n  sequence " in data)
    except OSError:
        return True


def collect_nvc_files():
    """Collect valid VHDL files from the NVC test suite."""
    test_dir = NVC_DIR / "test"

    # Gather error files from each test_*.c file
    error_files_by_dir = {}
    for dirname, c_file in NVC_TEST_C_FILES.items():
        error_files_by_dir[dirname] = find_nvc_error_files_from_c(c_file, dirname)

    # Directories to scan
    dirs_to_scan = [
        "bounds", "cover", "dump", "elab", "lower", "misc",
        "parse", "sem", "regress",
    ]
    # Note: "charset" excluded — encoding test files, not standard VHDL syntax tests

    groups = {}

    for dirname in dirs_to_scan:
        dirpath = test_dir / dirname
        if not dirpath.is_dir():
            continue

        error_files = error_files_by_dir.get(dirname, set())
        files = sorted(dirpath.glob("*.vhd"))
        valid = []
        for f in files:
            relname = f.name

            # Skip files with expected errors
            if relname in error_files:
                continue

            # Skip files explicitly excluded (non-VHDL-2019 compliant)
            if relname in NVC_EXCLUDE_FILES:
                continue

            # Skip PSL files (by name or content)
            if is_psl_file(f) or has_psl_content(f):
                continue

            # Skip guillemet files and preprocessed files
            if has_guillemets(f) or has_preprocessor_directives(f):
                continue

            # Skip VHDL-AMS files
            if is_vhdl_ams_file(f):
                continue

            valid.append(f)

        if valid:
            groups[dirname] = valid

    # Also check for standalone files in test/ root
    root_files = sorted(test_dir.glob("*.vhd"))
    root_valid = [
        f for f in root_files
        if is_valid_vhdl_file(f)
        and not has_psl_content(f) and not is_vhdl_ams_file(f)
    ]
    if root_valid:
        groups["test"] = root_valid

    return groups


def parse_ghdl_testsuite_sh(sh_path):
    """Parse a GHDL testsuite.sh to extract valid and invalid files.

    Returns (valid_files, failure_files, has_any_success) where has_any_success
    indicates whether any successful analyze/synth/elab_simulate was found.
    """
    try:
        content = sh_path.read_text(errors="replace")
    except OSError:
        return set(), set(), False

    valid = set()
    failure = set()

    # Detect whether the script has any successful (non-failure) analysis commands
    # Strip comments first
    stripped = re.sub(r"#[^\n]*", "", content)
    has_success = bool(
        re.search(r"\b(?:synth_tb|synth_analyze|elab_simulate|elab_simulate_failure)\b", stripped)
        or re.search(r"^\s*analyze\s+(?!.*failure)", stripped, re.MULTILINE)
        or re.search(r"\bsynth\b", stripped)
        or re.search(r"\$GHDL\s+-[asi]\s+(?!.*--expect-failure)", stripped)
    )

    # Expand shell variables with file lists
    shell_vars = {}
    for m in re.finditer(r'(\w+)="([^"]*)"', content):
        varname = m.group(1)
        value = m.group(2)
        filenames = re.findall(r"[\w./-]+\.vhdl?\b", value)
        if filenames:
            shell_vars[varname] = filenames

    # Also capture variable assignments without quotes: var=file.vhdl
    for m in re.finditer(r"^(\w+)=([\w./-]+\.vhdl?)\s*$", content, re.MULTILINE):
        shell_vars.setdefault(m.group(1), []).append(m.group(2))

    # Check for --expect-failure with explicit filenames
    for m in re.finditer(
        r"\$GHDL\s+(?:-[si]|-a)\s+[^;\n]*--expect-failure[^;\n]*?([\w./-]+\.vhdl?)\b",
        content,
    ):
        failure.add(m.group(1))

    # Also check reverse order: --expect-failure before filename
    for m in re.finditer(
        r"--expect-failure[^;\n]*?([\w./-]+\.vhdl?)\b", content
    ):
        failure.add(m.group(1))

    # run $GHDL ... --expect-failure ... file.vhdl
    for m in re.finditer(
        r"run\s+\$GHDL\s+[^;\n]*--expect-failure[^;\n]*?([\w./-]+\.vhdl?)\b",
        content,
    ):
        failure.add(m.group(1))

    # Look for analyze_failure lines with explicit filenames
    for m in re.finditer(
        r"analyze_failure\b[^;\n]*?([\w./-]+\.vhdl?)\b", content
    ):
        failure.add(m.group(1))

    # Look for analyze lines with explicit filenames
    for m in re.finditer(r"^\s*analyze\s+([^;\n#]+)", content, re.MULTILINE):
        args = m.group(1)
        if "failure" in args:
            continue
        for word in args.split():
            if re.match(r"[\w./-]+\.vhdl?$", word):
                valid.add(word)

    # $GHDL -a / -s / -i with explicit filenames (not --expect-failure)
    for m in re.finditer(
        r"\$GHDL\s+(?:-[asi])\s+(?:\$GHDL_STD_FLAGS\s+)?([^;\n#]+)", content
    ):
        args = m.group(1)
        if "--expect-failure" in args:
            continue
        for word in args.split():
            if re.match(r"[\w./-]+\.vhdl?$", word):
                valid.add(word)

    # Handle for loops with $var; do analyze_failure/analyze $loopvar
    for m in re.finditer(
        r"for\s+(\w+)\s+in\s+\$(\w+)\s*;\s*do\s*\n\s*(analyze_failure|analyze)\b",
        content,
    ):
        listvar = m.group(2)
        cmd = m.group(3)
        if listvar in shell_vars:
            target = failure if cmd == "analyze_failure" else valid
            target.update(shell_vars[listvar])

    # Handle for loops with literal file list; do analyze/analyze_failure
    for m in re.finditer(
        r"for\s+\w+\s+in\s+((?:[\w./-]+\.vhdl?\s*)+);\s*do\s*\n\s*(analyze_failure|analyze)\b",
        content,
    ):
        filenames = re.findall(r"[\w./-]+\.vhdl?\b", m.group(1))
        cmd = m.group(2)
        target = failure if cmd == "analyze_failure" else valid
        target.update(filenames)

    # Handle for loops where extension is appended in the body:
    # for f in name1 name2; do analyze_failure $f.vhdl
    for m in re.finditer(
        r"for\s+(\w+)\s+in\s+([^$;]+?);\s*do\s*\n\s*(analyze_failure|analyze)\s+\$\1\.(\w+)",
        content,
    ):
        names = m.group(2).split()
        ext = m.group(4)
        cmd = m.group(3)
        target = failure if cmd == "analyze_failure" else valid
        for name in names:
            name = name.strip()
            if re.match(r"^[\w-]+$", name):
                target.add(f"{name}.{ext}")

    # Handle for loops with $var where extension is appended:
    # for f in $files; do analyze $f.vhdl
    for m in re.finditer(
        r"for\s+(\w+)\s+in\s+\$(\w+)\s*;\s*do\s*\n\s*(analyze_failure|analyze)\s+\$\1\.(\w+)",
        content,
    ):
        listvar = m.group(2)
        cmd = m.group(3)
        ext = m.group(4)
        if listvar in shell_vars:
            # Variable already contains .vhdl extensions — use as-is
            target = failure if cmd == "analyze_failure" else valid
            target.update(shell_vars[listvar])
        else:
            # Variable contains basenames without extension
            # Try to find basenames in the variable definition
            for vm in re.finditer(
                rf'{listvar}="([^"]*)"', content
            ):
                names = vm.group(1).split()
                target = failure if cmd == "analyze_failure" else valid
                for name in names:
                    name = name.strip()
                    if name:
                        target.add(f"{name}.{ext}")

    # Handle "for i in *.vhdl; do analyze_failure $i"
    for m in re.finditer(
        r"for\s+\w+\s+in\s+\*\.vhdl?\s*;\s*do\s*\n\s*(analyze_failure)\b",
        content,
    ):
        failure.add("*")

    # Handle "if $GHDL -i $f; then echo 'error expected'" pattern (issue1823 style)
    for m in re.finditer(
        r"for\s+\w+\s+in\s+([\w./\s-]+?);\s*do.*?(?:error\s+expected|should\s+fail)",
        content, re.DOTALL,
    ):
        filenames = re.findall(r"[\w./-]+\.vhdl?\b", m.group(1))
        failure.update(filenames)

    # Handle for loops with --expect-failure in body
    # for f in file1.vhdl file2.vhdl; do ... --expect-failure ...
    for m in re.finditer(
        r"for\s+\w+\s+in\s+([\w./\s-]+?);\s*do.*?--expect-failure",
        content, re.DOTALL,
    ):
        filenames = re.findall(r"[\w./-]+\.vhdl?\b", m.group(1))
        failure.update(filenames)

    # Handle elab_simulate / synth (implies analyze succeeded)
    for m in re.finditer(
        r"(?:elab_simulate|synth)\s+([^;\n#]+)", content
    ):
        args = m.group(1)
        for word in args.split():
            if re.match(r"[\w./-]+\.vhdl?$", word):
                valid.add(word)

    return valid, failure, has_success


def collect_ghdl_files():
    """Collect valid VHDL files from the GHDL test suite."""
    groups = {}

    # 1. Libraries - all valid
    lib_dir = GHDL_DIR / "libraries"
    if lib_dir.is_dir():
        for subdir in sorted(lib_dir.iterdir()):
            if not subdir.is_dir():
                continue
            files = sorted(
                list(subdir.glob("*.vhd")) + list(subdir.glob("*.vhdl"))
            )
            files = [
                f for f in files
                if is_valid_vhdl_file(f) and not is_vhdl_ams_file(f)
            ]
            if files:
                groups[f"libraries/{subdir.name}"] = files

    # 2. Vests - standards compliance tests
    vests_dir = GHDL_DIR / "testsuite" / "vests"
    if vests_dir.is_dir():
        for root, dirs, fnames in os.walk(vests_dir):
            root_path = Path(root)
            rel = root_path.relative_to(GHDL_DIR)
            rel_str = str(rel)

            # Skip non_compliant directories (intentional failures)
            # and vhdl-ams/vhdl_ams directories (different standard)
            if "non_compliant" in rel_str or "vhdl-ams" in rel_str or "vhdl_ams" in rel_str:
                continue

            vhd_files = sorted(
                [root_path / f for f in fnames if f.endswith((".vhd", ".vhdl"))]
            )
            vhd_files = [
                f for f in vhd_files
                if is_valid_vhdl_file(f) and not is_vhdl_ams_file(f)
                   and f.name not in GHDL_EXCLUDE_FILES
            ]
            if vhd_files:
                groups[str(rel)] = vhd_files

    # 3. Synth tests - parse testsuite.sh per directory
    synth_dir = GHDL_DIR / "testsuite" / "synth"
    if synth_dir.is_dir():
        _collect_ghdl_testdir(synth_dir, "testsuite/synth", groups)

    # 4. GNA tests - parse testsuite.sh per directory
    gna_dir = GHDL_DIR / "testsuite" / "gna"
    if gna_dir.is_dir():
        _collect_ghdl_testdir(gna_dir, "testsuite/gna", groups)

    # 5. Sanity tests
    sanity_dir = GHDL_DIR / "testsuite" / "sanity"
    if sanity_dir.is_dir():
        _collect_ghdl_testdir(sanity_dir, "testsuite/sanity", groups)

    return groups


def _collect_ghdl_testdir(parent_dir, group_prefix, groups):
    """Collect valid VHDL files from a GHDL test directory by parsing testsuite.sh."""
    for subdir in sorted(parent_dir.iterdir()):
        if not subdir.is_dir():
            continue

        if subdir.name in GHDL_EXCLUDE_DIRS:
            continue

        sh_path = subdir / "testsuite.sh"
        all_vhdl = set(
            f.name for f in subdir.iterdir()
            if f.suffix in (".vhd", ".vhdl") and f.is_file()
        )

        if not all_vhdl:
            continue

        if sh_path.exists():
            valid_names, failure_names, _ = parse_ghdl_testsuite_sh(sh_path)

            # If "*" in failure, all files are failures
            if "*" in failure_names:
                continue

            # If the script only has failure commands and no explicitly valid files,
            # skip the entire directory (extra files are likely also invalid)
            if not valid_names and failure_names:
                continue

            # Files explicitly marked as valid
            # Files NOT mentioned in analyze_failure are potentially valid too
            if valid_names:
                # Use explicitly valid files
                selected = valid_names - failure_names
            else:
                # No explicit analyze calls found; include all except failures
                selected = all_vhdl - failure_names

            valid_files = []
            for name in sorted(selected):
                if name in GHDL_EXCLUDE_FILES:
                    continue
                fpath = subdir / name
                if not fpath.exists():
                    continue
                if not is_valid_vhdl_file(fpath):
                    continue
                if has_psl_content(fpath):
                    continue
                if is_vhdl_ams_file(fpath):
                    continue
                valid_files.append(fpath)

            if valid_files:
                key = f"{group_prefix}/{subdir.name}"
                groups[key] = valid_files
        else:
            # No testsuite.sh — include all VHDL files
            valid_files = sorted(
                [subdir / n for n in all_vhdl
                 if is_valid_vhdl_file(subdir / n) and not is_vhdl_ams_file(subdir / n)]
            )
            if valid_files:
                key = f"{group_prefix}/{subdir.name}"
                groups[key] = valid_files


def make_test_name(group_key, prefix):
    """Convert a group key to a valid Rust test function name."""
    name = group_key.replace("/", "_").replace("-", "_").replace(".", "_")
    # Remove leading/trailing underscores, collapse multiples
    name = re.sub(r"_+", "_", name).strip("_")
    return f"{prefix}_{name}"


def generate_rust_test_file(groups, source_name, external_subdir):
    """Generate a Rust test file from grouped files."""
    lines = []
    lines.append(f"// Auto-generated by scripts/generate_external_parse_tests.py")

    total_dirs = len(groups)
    total_files = sum(len(fs) for fs in groups.values())
    lines.append(f"// Source: {source_name} ({total_dirs} groups, {total_files} files)")
    lines.append(f"// Do not edit manually - regenerate with the script above.")
    lines.append("")
    lines.append("use crate::tests::test_parse_files;")

    for group_key in sorted(groups.keys()):
        files = groups[group_key]
        test_name = make_test_name(group_key, source_name)

        lines.append("")
        lines.append("#[test]")
        lines.append(f"fn {test_name}() {{")

        if len(files) == 1:
            rel = file_to_rel_path(files[0], external_subdir)
            lines.append(f'    test_parse_files(&["{rel}"]);')
        else:
            lines.append("    test_parse_files(&[")
            for f in files:
                rel = file_to_rel_path(f, external_subdir)
                lines.append(f'        "{rel}",')
            lines.append("    ]);")

        lines.append("}")

    return "\n".join(lines) + "\n"


def file_to_rel_path(filepath, external_subdir):
    """Convert an absolute file path to a relative path from the Rust crate."""
    rel = filepath.relative_to(ROOT / "external" / external_subdir)
    return f"{REL_PREFIX}/{external_subdir}/{rel}"


def main():
    print("Collecting NVC test files...")
    nvc_groups = collect_nvc_files()
    nvc_total = sum(len(fs) for fs in nvc_groups.values())
    print(f"  {len(nvc_groups)} groups, {nvc_total} files")

    print("Collecting GHDL test files...")
    ghdl_groups = collect_ghdl_files()
    ghdl_total = sum(len(fs) for fs in ghdl_groups.values())
    print(f"  {len(ghdl_groups)} groups, {ghdl_total} files")

    print("Generating test_cases_nvc.rs...")
    nvc_content = generate_rust_test_file(nvc_groups, "nvc", "nvc")
    (RUST_TESTS / "test_cases_nvc.rs").write_text(nvc_content)

    print("Generating test_cases_ghdl.rs...")
    ghdl_content = generate_rust_test_file(ghdl_groups, "ghdl", "ghdl")
    (RUST_TESTS / "test_cases_ghdl.rs").write_text(ghdl_content)

    print(f"Done! {nvc_total + ghdl_total} total files across {len(nvc_groups) + len(ghdl_groups)} test functions.")


if __name__ == "__main__":
    main()
