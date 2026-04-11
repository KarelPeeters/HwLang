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

#[test]
fn subtype_range_and_array_constraints() {
    test_parse(
        "
        entity top is
            subtype nibble_t is integer range 0 to 15;
            subtype word_t is std_logic_vector(15 downto 0);
            subtype matrix_col_t is matrix_t(0 to 1)(7 downto 0);
        end;
        ",
    )
}

#[test]
fn array_type_definitions_with_constraints() {
    test_parse(
        "
        entity top is
            type vec_t is array (integer range <>) of integer;
            type mat_t is array (0 to 1, 7 downto 0) of std_logic;
            subtype row_t is vec_t(7 downto 0);
        end;
        ",
    )
}

#[test]
fn architecture_signal_and_subprogram_declarations() {
    test_parse(
        "
        entity top is
        end;
        architecture rtl of top is
            signal a, b: std_logic;
            signal c: std_logic bus := a;
            signal d: std_logic register;
            procedure clear;
            function id_bit return std_logic;
        begin
            a <= b;
        end rtl;
        ",
    )
}

#[test]
fn concurrent_block_statement_basic() {
    test_parse(
        "
        entity top is
        end;
        architecture rtl of top is
            signal a, b: std_logic;
        begin
            guard_blk: block (a = b)
            is
                signal c: std_logic;
            begin
                c <= a;
                a <= c;
            end block guard_blk;
        end rtl;
        ",
    )
}

#[test]
fn concurrent_process_statement_basic() {
    test_parse(
        "
        entity top is
        end;
        architecture rtl of top is
        begin
            p1: process (all)
            is
                constant one: integer := 1;
            begin
                null;
            end process p1;

            postponed process (clk, rst)
            begin
                null;
            end postponed process;
        end rtl;
        ",
    )
}

#[test]
fn concurrent_procedure_call_statement_basic() {
    test_parse(
        "
        entity top is
        end;
        architecture rtl of top is
        begin
            tick;
            call1: postponed work_pkg.step;
        end rtl;
        ",
    )
}
