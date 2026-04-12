// TODO worry about utf-8 vs iec-8859
//   VHDL standard says source is iec-8859, but we'll accept utf-8 too
//   internally in the compiler we'll convert everything to utf-8 first
// TODO tokenizer newlines do not agree with newlines for the purposes of LineOffsets, is that a problem?

pub mod syntax;
pub mod util;

#[cfg(test)]
mod tests;
