-- Copyright (c) 2022 Maarten Baert <info@maartenbaert.be>
-- Available under the MIT License - see LICENSE.txt for details.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity axi_gearbox_v2 is
    generic (
        input_width: natural := 48;
        output_width: natural := 64
    );
    port (

        -- clock and synchronous reset
        clk: in std_logic;
        rst: in std_logic;

        -- input side
        input_data: in  std_logic_vector(input_width - 1 downto 0);
        input_valid: in  std_logic;
        input_ready: out std_logic;

        -- output side
        output_data: out std_logic_vector(output_width - 1 downto 0);
        output_valid: out std_logic;
        output_ready: in  std_logic

    );
end axi_gearbox_v2;

architecture rtl of axi_gearbox_v2 is

    function f_min (
        a: natural;
        b: natural
        ) return natural is
    begin
        if a < b then
            return a;
        else
            return b;
        end if;
    end f_min;

    function f_gcd (
        a: natural;
        b: natural
        ) return natural is
        variable v_a: natural;
        variable v_b: natural;
    begin
        v_a := a;
        v_b := b;
        while true loop
            v_a := v_a mod v_b;
            if v_a = 0 then
                return v_b;
            end if;
            v_b := v_b mod v_a;
            if v_b = 0 then
                return v_a;
            end if;
        end loop;
    end f_gcd;

    function f_clog2 (
        a: natural
        ) return natural is
        variable v_res: natural;
        variable v_pow: natural;
    begin
        v_res := 0;
        v_pow := 1;
        while true loop
            if v_pow >= a then
                return v_res;
            end if;
            v_res := v_res + 1;
            v_pow := v_pow * 2;
        end loop;
    end f_clog2;

    -- constants
    constant c_step: natural := f_gcd(input_width, output_width);
    constant c_bufsize: natural := input_width + output_width + f_min(input_width,
        output_width) - c_step;
    constant c_shiftbits: natural := f_clog2((c_bufsize - input_width) / c_step + 1);

    -- flow control
    signal r_buffer: std_logic_vector(c_bufsize - 1 downto 0);
    signal r_level: natural range 0 to c_bufsize / c_step;

begin

    -- generate output data
    output_data <= r_buffer(output_width - 1 downto 0);

    -- generate output flags
    output_valid <= '1' when r_level >= output_width / c_step else '0';
    input_ready <= '1' when r_level <= (c_bufsize - input_width) / c_step else '0';

    process(clk)
        variable v_level: natural range 0 to c_bufsize / c_step;
        variable v_data: std_logic_vector(
            input_width + c_step * (2 ** c_shiftbits - 1) - 1 downto 0);
        variable v_mask: std_logic_vector(
            input_width / c_step + (2 ** c_shiftbits - 1) - 1 downto 0);
        variable v_shift: unsigned(c_shiftbits - 1 downto 0);
        variable v_buffer: std_logic_vector(c_bufsize - 1 downto 0);
    begin
        if rising_edge(clk) then

            if rst = '1' then

                -- reset current level
                r_level <= 0;

            else

                -- copy registers into variables
                v_level := r_level;
                v_buffer := r_buffer;

                -- handle input
                if input_valid = '1' and r_level <= (c_bufsize - input_width) / c_step then
                    v_data := (others => 'X');
                    v_mask := (others => 'X');
                    v_data(input_width - 1 downto 0) := input_data;
                    v_mask(input_width / c_step - 1 downto 0) := (others => '1');
                    v_shift := to_unsigned(v_level, c_shiftbits);
                    for i in 0 to  c_shiftbits - 1 loop
                        if v_shift(i) = '1' then
                            if input_width + 2 ** (i + 1) > c_bufsize then
                                v_data(c_bufsize - 1 downto c_step * 2 ** i) := v_data(
                                    c_bufsize - c_step * 2 ** i - 1 downto 0);
                                v_mask(c_bufsize / c_step - 1 downto 2 ** i) := v_mask(
                                    c_bufsize / c_step - 2 ** i - 1 downto 0);
                            else
                                v_data(input_width + c_step * (
                                    2 ** (i + 1) - 1) - 1 downto c_step * 2 ** i) := v_data(
                                    input_width + c_step * (2 ** i - 1) - 1 downto 0);
                                v_mask(input_width / c_step + (
                                    2 ** (i + 1) - 1) - 1 downto 2 ** i) := v_mask(
                                    input_width / c_step + (2 ** i - 1) - 1 downto 0);
                            end if;
                            v_mask(2 ** i - 1 downto 0) := (others => '0');
                        else
                            if input_width + 2 ** (i + 1) > c_bufsize then
                                v_mask(
                                    c_bufsize / c_step - 1 downto input_width / c_step + 2 ** i - 1) := (
                                    others => '0');
                            else
                                v_mask(input_width / c_step + (2 ** (
                                    i + 1) - 1) - 1 downto input_width / c_step + 2 ** i - 1) := (
                                    others => '0');
                            end if;
                        end if;
                    end loop;
                    for i in 0 to c_bufsize / c_step - 1 loop
                        if v_mask(i) = '1' then
                            v_buffer((i + 1) * c_step - 1 downto i * c_step) := v_data(
                                (i + 1) * c_step - 1 downto i * c_step);
                        end if;
                    end loop;
                    v_level := v_level + input_width / c_step;
                end if;

                -- handle output
                if output_ready = '1' and r_level >= output_width / c_step then
                    v_buffer(c_bufsize - output_width - 1 downto 0) := v_buffer(
                        c_bufsize - 1 downto output_width);
                    v_level := v_level - output_width / c_step;
                end if;

                -- copy variables back into registers
                r_level <= v_level;
                r_buffer <= v_buffer;

            end if;

        end if;
    end process;

end rtl;
