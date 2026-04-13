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
            attribute keep : std_logic;
            signal es: std_logic;
            shared variable ev: integer := 0;
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
            shared variable v: integer := 0;
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
            shared variable v: integer := 0;
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
                private variable state: integer := 0;
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
fn shared_and_private_variable_declarations() {
    test_parse(
        "
        entity top is
            shared variable ev: integer := 0;
        end;

        architecture rtl of top is
            shared variable av: integer := 0;
        begin
            process
                variable pv: integer := 0;
            begin
                pv := av + ev;
                wait;
            end process;
        end;

        package p is
            type prot is protected
                private variable state: integer := 0;
            end protected prot;
        end;
        ",
    )
}

#[test]
fn declarative_mode_view_group_and_specifications() {
    test_parse(
        "
        package p is
            type rec_t is record
                a: bit;
                b: bit;
            end record;

            view nested_view of rec_t is
                a : in;
                b : out;
            end view nested_view;

            view rec_view of rec_t is
                a : in;
                b : view nested_view;
            end view rec_view;

            group signal_pair is (signal, signal);
            group g1 : signal_pair (a, b);
        end;

        entity child is
        end;

        entity top is
            signal es : bit bus;
            disconnect es : bit after 1 ns;
            group entity_group is (signal <>);
            group eg : entity_group (es);
        end;

        architecture rtl of top is
            component child
            end component;
            signal s : bit bus;
            for all : child use entity work.child;
            disconnect s : bit after 1 ns;
            group arch_group is (signal <>);
            group ag : arch_group (s);
        begin
            u1: component child;
        end;
        ",
    )
}

#[test]
fn subprogram_instantiation_declarations() {
    test_parse(
        "
        package p is
            procedure base_p(x: integer);
            function base_f(x: integer) return integer;
            procedure p_inst is new base_p;
            function f_inst is new base_f;

            type prot is protected
                procedure get(x: integer);
                procedure get_inst is new get;
                function inc(x: integer) return integer;
                function inc_inst is new inc;
                private variable state: integer := 0;
            end protected prot;
        end;

        package body p is
            procedure body_base is
            begin
                null;
            end procedure;

            procedure body_inst is new body_base;

            function body_fun return integer is
            begin
                return 0;
            end function;

            function body_fun_inst is new body_fun;

            type prot is protected body
                procedure get(x: integer) is
                begin
                    null;
                end procedure;

                procedure get_alias is new get;
            end protected body prot;
        end;

        entity top is
            procedure ent_base;
            procedure ent_inst is new ent_base;
        end;

        architecture rtl of top is
            procedure arch_base;
            procedure arch_inst is new arch_base;
        begin
            process
                procedure proc_base;
                procedure proc_inst is new proc_base;
            begin
                null;
            end process;
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

#[test]
fn selected_waveform_multi_element() {
    test_parse(
        "
        entity top is end;
        architecture rtl of top is
            signal x, y : integer;
            signal sel : boolean;
        begin
            with sel select
                x <= 1, 2 after 5 ns when true,
                     3 when false;
        end;
        ",
    )
}

#[test]
fn nested_resolution_indication() {
    test_parse(
        "
        entity top is end;
        architecture rtl of top is
            subtype my_type is (resolved) my_utype;
        begin
        end;
        ",
    )
}

#[test]
fn default_as_identifier() {
    test_parse(
        "
        entity top is
            port (default : in integer := 22);
        end;
        architecture default of top is
        begin
        end;
        ",
    )
}

#[test]
fn character_literal_star() {
    test_parse(
        "
        entity top is end;
        architecture rtl of top is
            type t is array (natural range <>) of character;
            constant c : t := ('/', '*', 'X', '0');
        begin
        end;
        ",
    )
}

#[test]
fn empty_bit_string() {
    test_parse(
        "
        entity top is end;
        architecture rtl of top is
            constant c1 : bit_vector := x\"\";
            constant c2 : bit_vector := 4x\"\";
        begin
        end;
        ",
    )
}

#[test]
fn based_literal_colon_delimiter() {
    test_parse(
        "
        entity top is end;
        architecture rtl of top is
            constant c1 : integer := 16:E:E1;
            constant c2 : real := 5:1234.4321:E-10;
        begin
        end;
        ",
    )
}

#[test]
fn access_incomplete_type() {
    test_parse(
        "
        entity top is
            generic (
                type t1 is access type is private;
                type t2 is file of type is private
            );
        end;
        ",
    )
}

#[test]
fn physical_incomplete_type_units() {
    test_parse(
        "
        entity top is
            generic (
                type t is units <>
            );
        end;
        ",
    )
}

#[test]
fn physical_literal_without_value() {
    test_parse(
        "
        entity top is end;
        architecture rtl of top is
            type my_time is range 0 to 100
                units
                    fs;
                    ps = 1000 fs;
                    foo = fs;
                end units;
        begin
        end;
        ",
    )
}

#[test]
fn postfix_chain_call_select_attr() {
    test_parse(
        "
        entity top is end;
        architecture rtl of top is
            signal s : integer;
        begin
            process
            begin
                s <= foo(1).bar(2).baz;
                s <= foo(1)'length;
                wait;
            end process;
        end;
        ",
    )
}

#[test]
fn empty_record() {
    test_parse(
        "
        entity top is end;
        architecture rtl of top is
            type empty_rec is record
            end record;
        begin
        end;
        ",
    )
}

#[test]
fn generate_inner_end_for() {
    test_parse(
        "
        entity top is end;
        architecture rtl of top is
        begin
            gen: for i in 0 to 3 generate
            begin
            end gen;
            end generate gen;
        end;
        ",
    )
}

#[test]
fn generate_inner_end_if() {
    test_parse(
        "
        entity top is end;
        architecture rtl of top is
            signal test : natural := 3;
        begin
            ll: if test = 10 generate
            begin
            end;
            elsif test = 5 generate
            begin
            end;
            end generate;
        end;
        ",
    )
}

#[test]
fn generate_inner_end_if_else() {
    test_parse(
        "
        entity top is end;
        architecture rtl of top is
            signal test : boolean := true;
        begin
            g: if test generate
            begin
            end g_true;
            else generate
            begin
            end g_false;
            end generate g;
        end;
        ",
    )
}

#[test]
fn generate_inner_end_case() {
    test_parse(
        "
        entity top is end;
        architecture rtl of top is
            constant c : integer := 1;
        begin
            gen: case c generate
                when 0 =>
                begin
                end case0;
                when 1 =>
                begin
                end case1;
            end generate gen;
        end;
        ",
    )
}

// UseClause with simple name (no selection)
#[test]
fn use_clause_simple_name() {
    test_parse(
        "
        library ieee;
        use d;
        entity e is end;
        ",
    )
}

// Package instantiation with attribute on uninstantiated name (in generic interface)
#[test]
fn package_instantiation_attribute_name() {
    test_parse(
        "
        package gen0 is
            generic (type t);
            function get return natural;
        end gen0;
        entity e is
            generic (package p is new k'g generic map (<>));
        end;
        ",
    )
}

// Architecture body with attribute on entity name
#[test]
fn architecture_attribute_entity_name() {
    test_parse(
        "
        entity e is end;
        architecture a of e't is
        begin
        end;
        ",
    )
}

// Subprogram instantiation with attribute on uninstantiated name
#[test]
fn subprogram_instantiation_attribute_name() {
    test_parse(
        "
        package p is
            function f is new k'g;
            procedure p is new k'g;
        end;
        ",
    )
}

// External name with postfix select
#[test]
fn external_name_postfix_select() {
    test_parse(
        "
        entity e is end;
        architecture a of e is
            signal s : integer;
        begin
            s <= <<signal .tb.top_i.sigb : integer>>.field;
        end;
        ",
    )
}

// Interface type declaration in subprogram parameters
#[test]
fn subprogram_type_parameter() {
    test_parse(
        "
        package p is
            function f (type t) return integer;
        end;
        ",
    )
}

// Port interface with constant and file declarations
#[test]
fn port_interface_constant_file() {
    test_parse(
        "
        entity e is
            port (
                constant c : integer := 0;
                file f : integer
            );
        end;
        ",
    )
}

// Generic interface with signal, variable, file declarations
#[test]
fn generic_interface_signal_variable_file() {
    test_parse(
        "
        entity e is
            generic (
                signal s : integer;
                variable v : integer;
                file f : integer
            );
        end;
        ",
    )
}

// Procedure call with build_name_call (Name + CallArgs)
#[test]
fn procedure_call_with_args() {
    test_parse(
        "
        entity e is end;
        architecture a of e is
        begin
            process
            begin
                proc(1, 2, 3);
                pkg.proc;
                pkg.proc(x);
            end process;
        end;
        ",
    )
}

// LRM 5.3.3 Unspecified type indication in interface declarations
#[test]
fn unspecified_type_indication() {
    test_parse(
        "
        entity e is
            port ( p : type is private );
        end entity;
        ",
    )
}

#[test]
fn unspecified_type_indication_variants() {
    test_parse(
        "
        entity e is
            generic (
                type t is private;
                type u;
                type v is (<>);
                type w is range <>;
                type x is range <>.<>;
                type y is access type is private
            );
        end entity;
        ",
    )
}

// LRM 9.3.4 Function/procedure call with generic_map_aspect
#[test]
fn generic_map_function_call() {
    test_parse(
        "
        architecture a of e is
            signal x : integer_vector(1 to 8);
            signal y : x'subtype;
        begin
            process
                variable v : integer_vector := reverse generic map(x'subtype)(x);
            begin
                display generic map (x'subtype)(x);
            end process;
        end;
        ",
    )
}

// LRM 8.6 Character literal with signature and attribute
#[test]
fn char_literal_with_signature_attribute() {
    test_parse(
        "
        architecture a of e is
        begin
            process
            begin
                report '1' [return my_enum]'test;
            end process;
        end;
        ",
    )
}

// LRM 15.11 Tool directives (backtick lines)
#[test]
fn tool_directive() {
    test_parse(
        "
        `protect begin
        entity e is
        end entity;
        `protect end
        ",
    )
}

// LRM 8.6 Chained attributes
#[test]
fn chained_attributes() {
    test_parse(
        "
        architecture a of e is
            type ranges_t is array(natural range <>) of integer'range'record;
            signal s : integer;
        begin
            process
            begin
                assert s'range'value = (1, 2, ascending);
                report x'element'element'image;
            end process;
        end;
        ",
    )
}

// LRM 8.6 Attribute after indexed name
#[test]
fn attribute_after_indexed_name() {
    test_parse(
        "
        architecture a of e is
        begin
            process
            begin
                report x(0)'subtype'image;
                report x'range(1)'value;
            end process;
        end;
        ",
    )
}

// LRM 16.2.6 Record attribute
#[test]
fn record_attribute() {
    test_parse(
        "
        architecture a of e is
            type ranges_t is array(natural range <>) of integer'range'record;
        begin
        end;
        ",
    )
}

#[test]
fn record_constraint() {
    test_parse(
        "
        architecture a of e is
            subtype my_sub is my_rec(field1(0 to 7), field2(0 to 3));
            signal s : my_rec(field1(0 to 7));
        begin
        end;
        ",
    )
}

#[test]
fn alias_with_signature() {
    test_parse(
        "
        package p is
            alias my_add is \"+\" [integer, integer return integer];
            alias 'a' is my_char;
            alias \"and\" : std_logic is my_signal;
        end;
        ",
    )
}

#[test]
fn concat_operator() {
    test_parse(
        "
        architecture a of e is
        begin
            s <= a & b & c;
        end;
        ",
    )
}

#[test]
fn guarded_concurrent_selected_signal() {
    test_parse(
        "
        architecture a of e is
        begin
            with sel select
                target <= guarded transport
                    val1 when \"00\",
                    val2 when others;
        end;
        ",
    )
}

#[test]
fn aggregate_target() {
    test_parse(
        "
        architecture a of e is
        begin
            (a, b, c) <= d;
        end;
        ",
    )
}
