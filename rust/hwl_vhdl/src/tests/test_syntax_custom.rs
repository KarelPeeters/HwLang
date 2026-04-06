use crate::tests::test_parse;

#[test]
fn empty() {
    test_parse("")
}

#[test]
fn passthrough() {
    test_parse(
        "
        entity pass is
            port(x: in std_logic; y: out std_logic);
        end;
        architecture rtl of pass is
        begin
            y <= x;
        end rtl;
        ",
    )
}

#[test]
fn simple_type_int() {
    test_parse(
        "
        entity top is
            type A is range 1 to 10;
            type B is range 1 to 10;
        end;
        ",
    )
}

#[test]
fn simple_type_enum() {
    test_parse(
        "
        entity top is
            type E is (A, B, C);
            type F is ('0', '1', Z);
        end;
        ",
    )
}
