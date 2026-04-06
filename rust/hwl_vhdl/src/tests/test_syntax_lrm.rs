use crate::tests::test_parse;

#[test]
fn comments() {
    test_parse("-- The last sentence above echoes the Algol 68 report.\n");
    test_parse("----------- The first two hyphens start the comment.\n");
    test_parse("/* A long comment may be written\non several consecutive lines */");
    test_parse("/* Comments /* do not nest */");
}

#[test]
fn entity_header() {
    test_parse("entity Full_Adder is port(X, Y, Cin: in Bit; Sum: out Bit); end Full_Adder;");
    test_parse(
        "entity AndGate is generic(N: Natural := 2); port(inputs: in Bit_Vector(1 to N); result: out Bit); end entity AndGate;",
    );
    test_parse("entity TestBench is end TestBench;");
}

#[test]
fn architecture_statement() {
    test_parse(
        "
        architecture DataFlow of Full_Adder is
            signal A,B: Bit;
        begin
            A <= X xor Y;
            B <= A and Cin;
            Sum <= A xor Cin;
            Cout <= B or (X and Y);
        end architecture DataFlow;
        ",
    );

    test_parse(
        "
        library Test;
        use Test.Components.all;

        architecture Structure of TestBench is
            component Full_Adder
                port (X, Y, Cin: Bit; Cout, Sum: out Bit);
            end component;

            signal A,B,C,D,E,F,G: Bit;
            signal OK: Boolean;
        begin
            UUT: Full_Adder port map (A,B,C,D,E);
            Generator: AdderTest port map (A,B,C,F,G);
            Comparator: AdderCheck port map (D,E,F,G,OK);
        end Structure;
        ",
    );

    test_parse(
        "
        architecture Behavior of AndGate is
        begin
            process (Inputs)
                variable Temp: Bit;
            begin
                Temp := '1';
                for i in Inputs'Range loop
                    if Inputs(i) = '0' then
                        Temp := '0';
                        exit;
                    end if;
                end loop;
                Result <= Temp after 10 ns;
            end process;
        end Behavior;
        ",
    )
}
