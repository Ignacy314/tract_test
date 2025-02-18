use std::error::Error;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;

use clap::{Parser, Subcommand};
use tract_onnx::prelude::*;

use self::spectrogram::{Stft, HOP_LENGTH, N_FFT};

mod spectrogram;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate spectrograms to csv
    Generate(GenerateArgs),
    /// Test model inference
    Infer(InferArgs),
}

#[derive(clap::Args)]
struct GenerateArgs {
    /// Path to output file
    #[arg(short, long)]
    output: String,
    /// Path to input file
    #[arg(short, long)]
    input: String,
}

#[derive(clap::Parser)]
struct InferArgs {
    /// Number of sample to start at
    #[arg(short, long)]
    start_sample: u32,
    //#[arg(short, long)]
    //output_file: String,
    /// Path to the input wav file
    #[arg(short, long)]
    input_file: String,
    /// Number of frames to process
    #[arg(short, long)]
    frames: usize,
    /// Path to the onnx model
    #[arg(short, long)]
    model: String,
}

fn min_max_scale(x_vec: &[f64]) -> Vec<f64> {
    let mut x_max = f64::MIN;
    let mut x_min = f64::MAX;
    for &x in x_vec {
        if x > x_max {
            x_max = x;
        }
        if x < x_min {
            x_min = x;
        }
    }
    let x_std = x_vec
        .iter()
        .map(|x| (x - x_min) / (x_max - x_min))
        .collect::<Vec<_>>();
    x_std
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate(args) => {
            let mut reader = hound::WavReader::open(args.input).unwrap();
            let mut w = BufWriter::new(File::create(args.output)?);
            // TODO: header?
            // writeln!(w, "")?;

            let mut stft = Stft::new(N_FFT, HOP_LENGTH);
            let samples = reader.samples::<i32>();
            for s in samples {
                let sample = s.unwrap();
                if let Some((mut col, harm, perc)) = stft.process_samples(&[sample as f64]) {
                    stft.hpss_one(&mut col, &harm, &perc);
                    assert_eq!(col.len(), 4097);
                    let scaled = min_max_scale(&col);
                    for s in &scaled[..4096] {
                        write!(w, "{s},")?;
                    }
                    writeln!(w, "{}", scaled[4096])?;
                }
            }
        }
        Commands::Infer(args) => {
            let model = tract_onnx::onnx().model_for_path(args.model)?;
            let model = model.with_input_fact(0, f64::fact([4097]).into())?;
            let model = model.into_optimized()?;
            let model = model.into_runnable()?;

            let mut reader = hound::WavReader::open(args.input_file).unwrap();
            reader.seek(args.start_sample).unwrap();

            let mut stft = Stft::new(N_FFT, HOP_LENGTH);
            let samples = reader.samples::<i32>();
            let mut f = 0;
            for s in samples {
                let sample = s.unwrap();
                if let Some((mut col, harm, perc)) = stft.process_samples(&[sample as f64]) {
                    stft.hpss_one(&mut col, &harm, &perc);
                    assert_eq!(col.len(), 4097);
                    let scaled = min_max_scale(&col);
                    let input: Tensor = tract_ndarray::Array1::from_vec(scaled).into();
                    let result = model.run(tvec!(input.into()))?;
                    println!("{f}: result: {:?}", result[0].to_array_view::<TDim>());

                    f += 1;
                    if f >= args.frames {
                        break;
                    }
                }
            }
        }
    }

    Ok(())
}
