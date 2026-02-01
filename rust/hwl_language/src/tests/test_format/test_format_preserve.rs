//! Test that all combinations of items, comments and blank lines are correctly preserved/discarded.
//! This is important so temporarily commenting out some items in a list does not mess with formatting.

use crate::tests::test_format::util::assert_formats_to;
use crate::util::exhaust::exhaust;
use hwl_util::{swrite, swriteln};

#[test]
fn preserve_blanks_file_items() {
    exhaust(|ex| {
        println!("Iteration {}", ex.iteration());

        let mut src = String::new();
        let mut expected = String::new();

        for i in 0..ex.choose_range(1, 5) {
            // maybe add blank before item
            if ex.choose_bool() {
                swriteln!(src);
                if i > 0 {
                    swriteln!(expected);
                }
            }

            // maybe make the item a comment
            if ex.choose_bool() {
                swrite!(src, "// ");
                swrite!(expected, "// ");
            }

            // add the item
            swriteln!(src, "const c_{i} = 0;");
            swriteln!(expected, "const c_{i} = 0;");
        }

        // maybe add a final blank
        if ex.choose_bool() {
            swriteln!(src);
        }

        println!("Source:");
        println!("{src}");
        println!("Expected:");
        println!("{expected}");

        assert_formats_to(&src, &expected);
    });
}

#[test]
fn preserve_blanks_extra_list() {
    exhaust(|ex| {
        println!("Iteration {}", ex.iteration());

        let mut src = String::new();
        let mut expected = String::new();

        swriteln!(src, "fn foo(");
        swriteln!(expected, "fn foo(");

        for i in 0..ex.choose_range(1, 5) {
            // maybe add blank before item
            if ex.choose_bool() {
                swriteln!(src);
                if i > 0 {
                    swriteln!(expected);
                }
            }

            // indent
            swrite!(src, "    ");
            swrite!(expected, "    ");

            // maybe make the item a comment
            if ex.choose_bool() {
                swrite!(src, "// ");
                swrite!(expected, "// ");
            }

            swriteln!(src, "p_{i}: uint,");
            swriteln!(expected, "p_{i}: uint,");
        }

        // maybe add a final blank
        if ex.choose_bool() {
            swriteln!(src);
        }

        swriteln!(src, ") {{}}");
        swriteln!(expected, ") {{}}");

        println!("Source:");
        println!("{src}");
        println!("Expected:");
        println!("{expected}");

        assert_formats_to(&src, &expected);
    });
}

#[test]
fn preserve_blanks_block() {
    exhaust(|ex| {
        println!("Iteration {}", ex.iteration());

        let mut src = String::new();
        let mut expected = String::new();

        swriteln!(src, "fn foo() {{");
        swriteln!(expected, "fn foo() {{");

        for i in 0..ex.choose_range(1, 5) {
            // maybe add blank before item
            if ex.choose_bool() {
                swriteln!(src);
                if i > 0 {
                    swriteln!(expected);
                }
            }

            // indent
            swrite!(src, "    ");
            swrite!(expected, "    ");

            // maybe make the item a comment
            if ex.choose_bool() {
                swrite!(src, "// ");
                swrite!(expected, "// ");
            }

            swriteln!(src, "val v_{i} = 0;");
            swriteln!(expected, "val v_{i} = 0;");
        }

        // maybe add a final blank
        if ex.choose_bool() {
            swriteln!(src);
        }

        swriteln!(src, "}}");
        swriteln!(expected, "}}");

        println!("Source:");
        println!("{src}");
        println!("Expected:");
        println!("{expected}");

        assert_formats_to(&src, &expected);
    });
}

#[test]
fn preserve_blanks_block_with_expression() {
    exhaust(|ex| {
        println!("Iteration {}", ex.iteration());

        let mut src = String::new();
        let mut expected = String::new();

        let prefix = "fn foo() {\n    val r = {";
        swriteln!(src, "{}", prefix);
        swriteln!(expected, "{}", prefix);

        let item_count = ex.choose_range(1, 5);
        for i in 0..item_count {
            // maybe add blank before item
            if ex.choose_bool() {
                swriteln!(src);
                if i > 0 {
                    swriteln!(expected);
                }
            }

            // indent
            swrite!(src, "        ");
            swrite!(expected, "        ");

            // maybe make the item a comment
            if ex.choose_bool() {
                swrite!(src, "// ");
                swrite!(expected, "// ");
            }

            swriteln!(src, "val v_{i} = 0;");
            swriteln!(expected, "val v_{i} = 0;");
        }

        // maybe add blank before expression
        if ex.choose_bool() {
            swriteln!(src);
            swriteln!(expected);
        }

        // indent
        swriteln!(src, "        e_{item_count}");
        swriteln!(expected, "        e_{item_count}");

        // maybe add a final blank
        if ex.choose_bool() {
            swriteln!(src);
        }

        swriteln!(src, "    }};\n}}");
        swriteln!(expected, "    }};\n}}");

        println!("Source:");
        println!("{src}");
        println!("Expected:");
        println!("{expected}");

        assert_formats_to(&src, &expected);
    });
}

#[test]
fn preserve_blanks_match_cases() {
    exhaust(|ex| {
        println!("Iteration {}", ex.iteration());

        let mut src = String::new();
        let mut expected = String::new();

        let prefix = "const {\n    match (v) {\n";
        swrite!(src, "{}", prefix);
        swrite!(expected, "{}", prefix);

        for i in 0..ex.choose_range(1, 5) {
            // maybe add blank before item
            if ex.choose_bool() {
                swriteln!(src);
                if i > 0 {
                    swriteln!(expected);
                }
            }

            // indent
            swrite!(src, "        ");
            swrite!(expected, "        ");

            // maybe make the item a comment
            if ex.choose_bool() {
                swrite!(src, "// ");
                swrite!(expected, "// ");
            }

            // add the item
            swriteln!(src, "0 => {{}}");
            swriteln!(expected, "0 => {{}}");
        }

        // maybe add a final blank
        if ex.choose_bool() {
            swriteln!(src);
        }

        let suffix = "    }\n}\n";
        swrite!(src, "{}", suffix);
        swrite!(expected, "{}", suffix);

        println!("Source:");
        println!("{src}");
        println!("Expected:");
        println!("{expected}");

        assert_formats_to(&src, &expected);
    });
}

#[test]
fn preserve_blanks_comma_list() {
    exhaust(|ex| {
        println!("Iteration {}", ex.iteration());

        let mut src = String::new();
        let mut expected = String::new();

        let prefix = "const {\n    f(\n";
        swrite!(src, "{}", prefix);
        swrite!(expected, "{}", prefix);

        for i in 0..ex.choose_range(1, 5) {
            // maybe add blank before item
            if ex.choose_bool() {
                swriteln!(src);
                if i > 0 {
                    swriteln!(expected);
                }
            }

            // indent
            swrite!(src, "        ");
            swrite!(expected, "        ");

            // maybe make the item a comment
            if ex.choose_bool() {
                swrite!(src, "// ");
                swrite!(expected, "// ");
            }

            // add the item
            swriteln!(src, "arg,");
            swriteln!(expected, "arg,");
        }

        // maybe add a final blank
        if ex.choose_bool() {
            swriteln!(src);
        }

        let suffix = "    );\n}\n";
        swrite!(src, "{}", suffix);
        swrite!(expected, "{}", suffix);

        println!("Source:");
        println!("{src}");
        println!("Expected:");
        println!("{expected}");

        assert_formats_to(&src, &expected);
    });
}

#[test]
fn preserve_blanks_interface() {
    exhaust(|ex| {
        println!("Iteration {}", ex.iteration());

        let mut src = String::new();
        let mut expected = String::new();

        let prefix = "interface Foo {\n";
        swrite!(src, "{}", prefix);
        swrite!(expected, "{}", prefix);

        // ports
        let port_count = ex.choose_range(1, 3);
        for i in 0..port_count {
            // maybe add blank before item
            if ex.choose_bool() {
                swriteln!(src);
                if i > 0 {
                    swriteln!(expected);
                }
            }

            // indent
            swrite!(src, "    ");
            swrite!(expected, "    ");

            // maybe make the item a comment
            if ex.choose_bool() {
                swrite!(src, "// ");
                swrite!(expected, "// ");
            }

            // add the item
            swriteln!(src, "x: bool,");
            swriteln!(expected, "x: bool,");
        }

        // views
        for i in 0..ex.choose_range(1, 3) {
            // maybe add blank before item
            if ex.choose_bool() {
                swriteln!(src);
                if i > 0 || port_count > 0 {
                    swriteln!(expected);
                }
            }

            // indent
            swrite!(src, "    ");
            swrite!(expected, "    ");

            // maybe make the item a comment
            if ex.choose_bool() {
                swrite!(src, "// ");
                swrite!(expected, "// ");
            }

            // add the item
            swriteln!(src, "interface Bar {{}}");
            swriteln!(expected, "interface Bar {{}}");
        }

        // maybe add a final blank
        if ex.choose_bool() {
            swriteln!(src);
        }

        swriteln!(src, "}}");
        swriteln!(expected, "}}");

        println!("Source:\n{src}");
        println!("Expected:\n{expected}");

        assert_formats_to(&src, &expected);
    });
}
