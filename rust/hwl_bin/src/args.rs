use clap::{Parser, Subcommand};
use hwl_util::constants::HWL_VERSION;
use std::num::NonZeroUsize;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(version=HWL_VERSION)]
pub struct Args {
    #[command(subcommand)]
    pub command: ArgsCommand,
}

#[derive(Subcommand, Debug)]
pub enum ArgsCommand {
    Build(ArgsBuild),
    Fmt(ArgsFormat),
}

#[derive(Parser, Debug)]
pub struct ArgsBuild {
    // input
    #[arg(long)]
    pub manifest: Option<PathBuf>,

    // TODO add positional command to only check certain items
    // TODO allow separately selecting which items to elaborate
    // TODO allow specifying output verilog and cpp files

    // performance
    // TODO maybe make this a common option?
    #[arg(long, short = 'j')]
    pub thread_count: Option<NonZeroUsize>,

    // debug
    // TODO some of these have major effects and are not really debug options
    #[arg(long)]
    pub profile: bool,
    #[arg(long)]
    pub print_files: bool,
    #[arg(long)]
    pub print_ir: bool,
    #[arg(long)]
    pub only_top: bool,
    #[arg(long)]
    pub skip_lower: bool,
    #[arg(long)]
    pub keep_main_stack: bool,
}

#[derive(Parser, Debug)]
pub struct ArgsFormat {
    #[arg(long)]
    pub manifest: Option<PathBuf>,
    pub file: Option<PathBuf>,
}
