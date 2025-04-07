-- Copyright (c) 2023 Maarten Baert <info@maartenbaert.be>
-- Available under the MIT License - see LICENSE.txt for details.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library convcode;
use convcode.convcode.all;

entity convcode_decoder is
    generic(
        depth: natural := 32
    );
    port(

        -- clock and synchronous reset
        clk: in std_logic;
        rst: in std_logic;

        -- input data
        input_data0: in unsigned(convcode_analog_bits - 1 downto 0);
        input_data1: in unsigned(convcode_analog_bits - 1 downto 0);
        input_valid: in std_logic;

        -- output data
        output_data: out std_logic;
        output_valid: out std_logic

    );
end convcode_decoder;

architecture rtl of convcode_decoder is

    function f_clog2(a: natural) return natural is
        variable n: natural;
    begin
        n := 0;
        while a > 2 ** n loop
            n := n + 1;
        end loop;
        return n;
    end f_clog2;

    constant c_num_states: natural := 2 ** convcode_generator_length;
    constant c_prob_bits: natural := f_clog2(convcode_generator_length) + convcode_analog_bits + 2;

    type codebook_t is array(0 to c_num_states - 1) of unsigned(1 downto 0);

    function f_codebook return codebook_t is
        variable v_code: std_logic_vector(convcode_generator_length - 1 downto 0);
        variable v_output: unsigned(1 downto 0);
        variable v_codebook: codebook_t;
    begin
        for i in 0 to c_num_states - 1 loop
            v_code := std_logic_vector(to_unsigned(i, convcode_generator_length));
            v_output := (others => '0');
            for j in 0 to convcode_generator_length - 1 loop
                v_output(0) := v_output(0) xor (
                    v_code(j) and convcode_generator0(convcode_generator_length - 1 - j));
                v_output(1) := v_output(1) xor (
                    v_code(j) and convcode_generator1(convcode_generator_length - 1 - j));
            end loop;
            v_codebook(i) := v_output;
        end loop;
        return v_codebook;
    end f_codebook;

    constant c_codebook: codebook_t := f_codebook;

    type probs_t is array(natural range <>) of unsigned(c_prob_bits - 1 downto 0);
    type paths_t is array(natural range <>) of std_logic_vector(depth - 1 downto 0);

    subtype halfprobs_t is probs_t(0 to c_num_states / 2 - 1);
    subtype halfpaths_t is paths_t(0 to c_num_states / 2 - 1);
    subtype fullprobs_t is probs_t(0 to c_num_states - 1);
    subtype fullpaths_t is paths_t(0 to c_num_states - 1);

    signal r_halfprobs: halfprobs_t;
    signal r_halfpaths: halfpaths_t;

begin

    process(clk)
        variable v_outprobs_max: halfprobs_t;
        variable v_outputs_max: std_logic_vector(0 to c_num_states / 2 - 1);
        variable v_bookprobs: probs_t(0 to 3);
        variable v_probs_new: fullprobs_t;
        variable v_paths_new: fullpaths_t;
        variable v_halfprobs_0: halfprobs_t;
        variable v_halfprobs_1: halfprobs_t;
        variable v_halfpaths_0: halfpaths_t;
        variable v_halfpaths_1: halfpaths_t;
        variable v_delta: unsigned(c_prob_bits - 1 downto 0);
    begin
        if rising_edge(clk) then
            if rst = '1' then

                -- reset tables
                r_halfprobs <= (others => (others => '0'));
                r_halfpaths <= (others => (others => '0'));

                -- reset output
                output_data <= '0';

            else

                if input_valid = '1' then

                    -- calculate output
                    v_outprobs_max := r_halfprobs;
                    for i in 0 to c_num_states / 2 - 1 loop
                        v_outputs_max(i) := r_halfpaths(i)(0);
                    end loop;
                    for j in convcode_generator_length - 2 downto 0 loop
                        for i in 0 to 2 ** j - 1 loop
                            v_delta := v_outprobs_max(2 * i + 0) - v_outprobs_max(2 * i + 1);
                            if v_delta(v_delta'high) = '0' then
                                v_outprobs_max(i) := v_outprobs_max(2 * i + 0);
                                v_outputs_max(i) := v_outputs_max(2 * i + 0);
                            else
                                v_outprobs_max(i) := v_outprobs_max(2 * i + 1);
                                v_outputs_max(i) := v_outputs_max(2 * i + 1);
                            end if;
                        end loop;
                    end loop;
                    output_data <= v_outputs_max(0);

                    -- calculate new probabilities and paths
                    v_bookprobs(0) := resize(not input_data0, c_prob_bits) + resize(not input_data1,
                        c_prob_bits);
                    v_bookprobs(1) := resize(input_data0, c_prob_bits) + resize(not input_data1,
                        c_prob_bits);
                    v_bookprobs(2) := resize(not input_data0, c_prob_bits) + resize(input_data1,
                        c_prob_bits);
                    v_bookprobs(3) := resize(input_data0, c_prob_bits) + resize(input_data1,
                        c_prob_bits);
                    v_probs_new := r_halfprobs & r_halfprobs;
                    v_paths_new := r_halfpaths & r_halfpaths;
                    for i in 0 to c_num_states - 1 loop
                        v_probs_new(i) := v_probs_new(i) + v_bookprobs(to_integer(c_codebook(i)));
                    end loop;

                    -- update probabilities and paths
                    for i in 0 to c_num_states / 2 - 1 loop
                        v_halfprobs_0(i) := v_probs_new(2 * i + 0);
                        v_halfprobs_1(i) := v_probs_new(2 * i + 1);
                    end loop;
                    for i in 0 to c_num_states / 2 - 1 loop
                        v_halfpaths_0(i) := '0' & v_paths_new(2 * i + 0)(depth - 1 downto 1);
                        v_halfpaths_1(i) := '1' & v_paths_new(2 * i + 1)(depth - 1 downto 1);
                    end loop;

                    -- select best probabilities and paths
                    for i in 0 to c_num_states / 2 - 1 loop
                        v_delta := v_halfprobs_0(i) - v_halfprobs_1(i);
                        if v_delta(v_delta'high) = '0' then
                            r_halfprobs(i) <= v_halfprobs_0(i);
                            r_halfpaths(i) <= v_halfpaths_0(i);
                        else
                            r_halfprobs(i) <= v_halfprobs_1(i);
                            r_halfpaths(i) <= v_halfpaths_1(i);
                        end if;
                    end loop;

                end if;
                output_valid <= input_valid;

            end if;
        end if;
    end process;

end rtl;
