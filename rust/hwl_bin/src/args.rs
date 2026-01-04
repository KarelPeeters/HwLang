use clap::{Parser, Subcommand};
use hwl_util::constants::HWL_VERSION;
use hwl_util::hwl_manifest_file_name_macro;
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

// TODO add positional command to only check certain items
// TODO allow separately selecting which items to elaborate
// TODO allow specifying output verilog and cpp files

/// Build a project.
#[derive(Parser, Debug)]
pub struct ArgsBuild {
    #[arg(long, help=MANIFEST_DOC)]
    pub manifest: Option<PathBuf>,

    // TODO maybe make this a common option?
    /// The number of threads to use, defaults to the number of hardware threads.
    #[arg(long, short = 'j')]
    pub thread_count: Option<NonZeroUsize>,

    // debug options
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

/// Format source files.
#[derive(Parser, Debug)]
pub struct ArgsFormat {
    #[arg(long, help=MANIFEST_DOC)]
    pub manifest: Option<PathBuf>,

    /// Check if files are formatted correctly without making any changes,
    /// exiting with a non-zero code if any files would be changed by formatting.
    #[arg(long)]
    pub check: bool,

    /// File or directory to format. If not provided all files in the project manifest will be formatted.
    #[arg(conflicts_with = "manifest")]
    pub path: Option<PathBuf>,

    /// Verbose output, set to print some more information about the formatting process.
    #[arg(long, short)]
    pub verbose: bool,
    /// Debug output file path. Set to get debug output of the internal formatting IRs and the final output.
    /// When this is used, the source files are not modified.
    #[arg(long, short)]
    pub debug: Option<PathBuf>,
}

const MANIFEST_DOC: &str = concat!(
    "Path to the project manifest file.",
    "If not provided, the current directory and its parents will be searched for a file named `",
    hwl_manifest_file_name_macro!(),
    "`.",
);

#[test]
fn args() {
    use clap::CommandFactory;
    Args::command().debug_assert();
}
