-- Copyright (c) 2023 Maarten Baert <info@maartenbaert.be>
-- Available under the MIT License - see LICENSE.txt for details.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package convcode is

constant convcode_generator_length: natural := 7;
constant convcode_generator0: std_logic_vector(0 to convcode_generator_length - 1) := "1111001";
constant convcode_generator1: std_logic_vector(0 to convcode_generator_length - 1) := "1011011";

constant convcode_analog_bits: natural := 8;

component convcode_encoder is
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
end component;

component convcode_decoder is
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
end component;

end convcode;
