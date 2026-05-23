from contextlib import contextmanager

import hwl


def compile_custom(top: str) -> hwl.Compile:
    # TODO compile/elaborate everything by default
    source = hwl.Source()
    source.add_file_content(["top"], "python.kh", top)
    return source.compile()


def diag_error(title: str, has_message: str | None = None, has_info: str | None = None):
    return diag_general(level="error", title=title, has_message=has_message, has_info=has_info)


def diag_warning(title: str, has_message: str | None = None, has_info: str | None = None):
    return diag_general(level="warning", title=title, has_message=has_message, has_info=has_info)


@contextmanager
def diag_general(level: str, title: str, has_message: str | None, has_info: str | None):
    raised = False

    try:
        yield
    except hwl.DiagnosticException as e:
        raised = True

        diags = e.diagnostics
        assert len(diags) > 0, "Diagnostic exception should not be empty"

        assert len(diags) == 1, \
            f"Expected exactly one diagnostic, got {len(diags)} in\n{e.combined_string_colored}"
        diag = diags[0]

        assert diag.title == title, f"Diagnostic title mismatch, expected {title}"
        assert diag.level == level, f"Diagnostic level mismatch, expected {level}"
        if has_message is not None:
            assert has_message in diag.messages, f"Diagnostic message check failed, expected {has_message}, got {diag.messages}"
        if has_info is not None:
            assert has_info in diag.infos, f"Diagnostic info check failed, expected {has_info}, got {diag.infos}"

    assert raised, "Expected diagnostic, got no exception"
