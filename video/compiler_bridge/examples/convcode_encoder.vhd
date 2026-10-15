-- Copyright (c) 2023 Maarten Baert <info@maartenbaert.be>
-- Available under the MIT License - see LICENSE.txt for details.

library ieee;
use ieee.std_logic_1164.all;

library convcode;
use convcode.convcode.all;

entity convcode_encoder is
    port(

        -- clock and synchronous reset
        clk: in std_logic;
        rst: in std_logic;

        -- input data
        input_data: in std_logic;
        input_valid: in std_logic;

        -- output data
        output_data0: out std_logic;
        output_data1: out std_logic;
        output_valid: out std_logic

    );
end convcode_encoder;

architecture rtl of convcode_encoder is

    signal r_delay_line: std_logic_vector(convcode_generator_length - 2 downto 0);

begin

    process(clk)
        variable v_delay_line: std_logic_vector(convcode_generator_length - 1 downto 0);
        variable v_output0: std_logic;
        variable v_output1: std_logic;
    begin
        if rising_edge(clk) then
            if rst = '1' then

                -- reset delay line
                r_delay_line <= (others => '0');

                -- reset output
                output_data0 <= '0';
                output_data1 <= '0';
                output_valid <= '0';

            else

                if input_valid = '1' then

                    -- update delay line
                    v_delay_line := input_data & r_delay_line;
                    r_delay_line <= v_delay_line(convcode_generator_length - 1 downto 1);

                    -- do convolution
                    v_output0 := '0';
                    v_output1 := '0';
                    for i in 0 to convcode_generator_length - 1 loop
                        v_output0 := v_output0 xor (v_delay_line(i) and convcode_generator0(
                            convcode_generator_length - 1 - i));
                        v_output1 := v_output1 xor (v_delay_line(i) and convcode_generator1(
                            convcode_generator_length - 1 - i));
                    end loop;
                    output_data0 <= v_output0;
                    output_data1 <= v_output1;

                end if;
                output_valid <= input_valid;

            end if;
        end if;
    end process;

end rtl;
