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

#[test]
fn broad_declarative_items_and_subprogram_bodies() {
    test_parse(
        "
        entity top is
            component child
            end component;
            attribute keep : std_logic;
            signal es: std_logic;
            variable ev: integer := 0;
            file ef: text;
            alias ea is es;
            procedure ep;
            function efn return integer;
            procedure ep2 is
            begin
                null;
            end procedure ep2;
            function efn2 return integer is
            begin
                return 0;
            end function efn2;
        end;

        architecture rtl of top is
            component child
            end component;
            signal s: std_logic;
            variable v: integer := 0;
            file f: text;
            alias a is s;
            attribute mark : std_logic;
            procedure p;
            function fn return integer;
            procedure p_body is
            begin
                null;
            end procedure p_body;
            function f_body return integer is
            begin
                return 0;
            end function f_body;
        begin
            assert v = 0 report s severity v;
            u1: component child;
            g_for: for i in 0 to 3 generate
            begin
                s <= s;
            end generate g_for;
            g_if: if v = 0 generate
            begin
                s <= s;
            end generate g_if;
        end rtl;
        ",
    )
}

#[test]
fn sequential_statements_coverage() {
    test_parse(
        "
        entity top is
        end;
        architecture rtl of top is
            signal s: std_logic;
            signal t: std_logic;
            variable v: integer := 0;
        begin
            process (all)
            begin
                wait on s, t until v = 0 for 1;
                assert v = 0 report s severity v;
                report s severity v;
                s <= t;
                v := v + 1;
                do_work;
                if v = 0 then
                    null;
                elsif v = 1 then
                    report s;
                else
                    return;
                end if;
                case v is
                    when 0 => null;
                    when others => null;
                end case;
                loop
                    exit when v = 4;
                end loop;
                while v = 0 loop
                    next when v = 1;
                    exit;
                end loop;
                for i in 0 to 2 loop
                    null;
                end loop;
                next;
                return 0;
                block is
                    constant c: integer := 1;
                begin
                    null;
                end block;
                null;
            end process;
        end rtl;
        ",
    )
}

#[test]
fn subprogram_parameter_lists_and_process_decl_without_is() {
    test_parse(
        "
        entity top is
            procedure p_decl(one: out integer);
            function f_decl(constant x: integer; y: in integer) return integer;
        end;
        architecture rtl of top is
        begin
            process
                function returns_last(p: bit_vector) return bit is
                begin
                    return p;
                end function returns_last;
            begin
                null;
            end process;
        end rtl;
        ",
    )
}

#[test]
fn record_types_named_associations_and_entity_instantiation() {
    test_parse(
        "
        entity child is
            port (x: in std_logic; y: out std_logic);
        end;
        architecture rtl of child is
        begin
            y <= x;
        end rtl;

        entity top is
            type pair_t is record
                a: integer;
                b: integer;
            end record;
            subtype pair_word_t is pair_t;
        end;
        architecture rtl of top is
            signal s, t: std_logic;
        begin
            u0: entity work.child(rtl) port map (x => s, y => t);
            process (all)
                variable v: integer := 0;
            begin
                v := add_one(x => v);
                report pair_t'image((a => 1, b => 2));
            end process;
        end rtl;
        ",
    )
}

#[test]
fn library_clause_and_impure_function() {
    test_parse(
        "
        library ieee, work;
        use ieee.std_logic_1164.all;

        package p is
            impure function f(x: integer) return integer;
        end;

        package body p is
            impure function f(x: integer) return integer is
            begin
                return x;
            end function;
        end;
        ",
    )
}

#[test]
fn ast_captures_function_purity_and_subtype_resolution() {
    test_parse(
        "
        package p is
            impure function f return integer;
            pure function g return integer;
            function h return integer;
            subtype bounded_int is resolve_fn integer range 0 to 3;
        end;
    ",
    );
}

#[test]
fn aggregate_choices_operator_symbol_and_select_suffix() {
    test_parse(
        "
        entity top is
            type vec_t is array (natural range <>) of bit_vector(3 downto 0);
        end;
        architecture rtl of top is
            constant c: vec_t := (1 => \"0000\", 2 => \"0001\", others => \"0011\");
        begin
            process
                variable i: integer := 1;
                variable v: bit;
            begin
                v := \"not\"(v);
                report integer'image(i) & c(i)(0);
                wait;
            end process;
        end rtl;
        ",
    )
}

#[test]
fn generic_type_package_instantiation_and_generate_alt_labels() {
    test_parse(
        "
        package p_gen is
            generic (type el_t);
            type arr_t is array (integer range <>) of el_t;
        end;
        package p is new work.p_gen generic map (el_t => bit);

        entity child is
            port (x: in bit; y: out bit);
        end;
        architecture rtl of child is begin y <= x; end;

        entity top is
            generic (type t);
            port (x: in bit; y: out bit);
        end;
        architecture rtl of top is
            component child is
                port (x: in bit; y: out bit);
            end component;
        begin
            g: if x = x generate
                begin
                a1: entity work.child(rtl) port map (x => x, y => y);
            else generate
                begin
                a2: component child port map (x => x, y => y);
            end generate;
        end rtl;
        ",
    )
}

#[test]
fn attribute_specifications_for_subprograms_and_signals() {
    test_parse(
        "
        package p is
            attribute foreign : string;
            procedure house(reg: in integer);
            attribute foreign of house : procedure is \"VHPIDIRECT house\";
        end;

        package body p is
            procedure house(reg: in integer) is
            begin
                null;
            end procedure;
            attribute foreign of house : procedure is \"VHPIDIRECT house_body\";
        end;

        entity top is
            port (s, t: in bit);
            attribute keep : string;
        end;

        architecture rtl of top is
        begin
        end;

        architecture attr of top is
            signal x, y: bit;
            attribute keep of x, y : signal is \"true\";
        begin
        end;
        ",
    )
}

#[test]
fn access_and_protected_type_definitions() {
    test_parse(
        "
        package p is
            type line is access string(1 to 7);
            procedure rep1(variable msg: line := new string(1 to 7));
            type prot is protected
                procedure get(a: integer);
            end protected prot;
        end;

        package body p is
            type prot is protected body
                variable v: integer;

                function inc(a: integer) return integer is
                begin
                    return a + 1;
                end function;

                procedure get(a: integer) is
                begin
                    v := inc(a);
                end procedure;
            end protected body prot;
        end;
        ",
    )
}

#[test]
fn aggregate_attribute_choice() {
    test_parse(
        "
        entity top is
        end;

        architecture rtl of top is
        begin
            process
                variable msg: string(1 to 7);
            begin
                msg := (msg'range => ' ');
                wait;
            end process;
        end;
        ",
    )
}
