//! CLI front-end for the three-body problem visualization generator.

mod cli;

use three_body_problem::{Result, pipeline};

fn main() -> Result<()> {
    let args = cli::Args::parse_args();
    cli::setup_logging(&args.log_level)?;
    pipeline::run_generation(&args.to_generation_request())?;
    Ok(())
}
