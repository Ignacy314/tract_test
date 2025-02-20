use std::error::Error;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::time::Instant;

use circular_buffer::CircularBuffer;
use clap::{Parser, Subcommand};
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use rand::rng;
use rand::Rng;
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
    /// Generate images for resnet
    ImgGen(ImgGenArgs),
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
    /// Path to the input wav file
    #[arg(short, long)]
    input: String,
    /// Number of frames to process
    #[arg(short, long)]
    frames: usize,
    /// Path to the MLP onnx model
    #[arg(short, long)]
    mlp: String,
    /// Path to the ResNet onnx model
    #[arg(short, long)]
    resnet: String,
}

#[derive(clap::Args)]
struct ImgGenArgs {
    /// Path to input wav file
    #[arg(short, long)]
    input: String,
    /// Path to output file
    #[arg(short, long)]
    output: String,
}

fn amplitude_to_db(x_vec: &mut [f64]) {
    let mut x_max = f64::MIN;
    for x in x_vec.iter_mut() {
        *x *= *x;
        if *x > x_max {
            x_max = *x;
        }
    }
    let sub = 10.0 * x_max.max(1e-10).log10();
    for x in x_vec.iter_mut() {
        *x = (10.0 * x.max(1e-10).log10() - sub).max(-80.0);
    }
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
            let width = reader.duration() / 4096;
            // TODO: header?
            // writeln!(w, "")?;
            let pb = ProgressBar::new(u64::from(width));
            let t = f64::from(width).log10().ceil() as u64;
            pb.set_style(
                ProgressStyle::with_template(&format!(
                "[{{elapsed_precise}}] {{bar:40.cyan/blue}} {{pos:>{t}}}/{{len:{t}}} ({{percent}}%) {{msg}}"
            ))
                .unwrap()
                .progress_chars("##-"),
            );
            let mut stft = Stft::new(N_FFT, HOP_LENGTH);
            let samples = reader.samples::<i32>();
            for s in samples {
                let sample = s.unwrap();
                if let Some(mut col) = stft.process_samples(&[sample as f64]) {
                    Stft::hpss_one(&mut col, &stft.harm, &stft.perc);
                    amplitude_to_db(&mut col);
                    assert_eq!(col.len(), 4097);
                    let scaled = min_max_scale(&col);
                    for s in &scaled[..4096] {
                        write!(w, "{s},")?;
                    }
                    writeln!(w, "{}", scaled[4096])?;
                    pb.inc(1);
                }
            }
            pb.finish();
        }
        Commands::Infer(args) => {
            let mlp = tract_onnx::onnx().model_for_path(args.mlp)?;
            let mlp = mlp.with_input_fact(0, f64::fact([4097]).into())?;
            let mlp = mlp.into_optimized()?;
            let mlp = mlp.into_runnable()?;

            let resnet = tract_onnx::onnx().model_for_path(args.resnet)?;
            let resnet = resnet.with_input_fact(0, f32::fact([1, 224, 224, 3]).into())?;
            let resnet = resnet.into_optimized()?;
            let resnet = resnet.into_runnable()?;

            let mut rng = rng();

            let mut buffer = CircularBuffer::<224, [f32; 4097]>::new();

            let mut reader = hound::WavReader::open(args.input).unwrap();
            reader.seek(args.start_sample).unwrap();

            let mut stft = Stft::new(N_FFT, HOP_LENGTH);
            let samples = reader.samples::<i32>();
            let mut f = 0;
            for s in samples {
                let sample = s.unwrap();
                if let Some(mut col) = stft.process_samples(&[sample as f64]) {
                    f += 1;

                    Stft::hpss_one(&mut col, &stft.harm, &stft.perc);
                    amplitude_to_db(&mut col);
                    assert_eq!(col.len(), 4097);
                    let scaled = min_max_scale(&col);

                    let mut image_col = [0f32; 4097];
                    for (i, s) in scaled.iter().enumerate() {
                        image_col[i] = *s as f32;
                    }
                    buffer.push_back(image_col);

                    if buffer.is_full() {
                        let start_row = rng.random_range(1500..1900);
                        //let start_row = 1600;
                        let resnet_input: Tensor = {
                            tract_ndarray::Array4::from_shape_fn(
                                (1, 224, 224, 3),
                                |(_, x, y, _)| buffer.get(x).unwrap()[y + start_row],
                            )
                            .into()
                        };
                        let start = Instant::now();
                        let resnet_result = resnet.run(tvec!(resnet_input.into()))?;
                        let elapsed = start.elapsed();

                        let mlp_input: Tensor = tract_ndarray::Array1::from_vec(scaled).into();
                        let mlp_result = mlp.run(tvec!(mlp_input.into()))?;

                        let resnet_val = &resnet_result[0].to_array_view::<f32>().unwrap();
                        let resnet_best = resnet_val
                            .iter()
                            .zip(0..)
                            .max_by(|a, b| a.0.total_cmp(b.0))
                            .unwrap();

                        println!(
                            "{f:6}: {:2} | {} | {} | {} | {start_row} | {resnet_val:?}",
                            mlp_result[0]
                                .to_array_view::<TDim>()
                                .unwrap()
                                .get(0)
                                .unwrap(),
                            resnet_best.1,
                            resnet_best.0,
                            elapsed.as_millis()
                        );
                    }

                    if f >= args.frames {
                        break;
                    }
                }
            }
        }
        Commands::ImgGen(args) => {
            let mut reader = hound::WavReader::open(args.input).unwrap();
            const HEIGHT: u32 = 4097;
            let width = reader.duration() / 4096;
            let pb = ProgressBar::new(u64::from(width));
            let t = f64::from(width).log10().ceil() as u64;
            pb.set_style(
                ProgressStyle::with_template(&format!(
                "[{{elapsed_precise}}] {{bar:40.cyan/blue}} {{pos:>{t}}}/{{len:{t}}} ({{percent}}%) {{msg}}"
            ))
                .unwrap()
                .progress_chars("##-"),
            );
            let mut image = image::GrayImage::new(width, HEIGHT);
            let mut x: u32 = 0;

            let mut stft = Stft::new(N_FFT, HOP_LENGTH);
            let samples = reader.samples::<i32>();
            for s in samples {
                let sample = s.unwrap();
                if let Some(mut col) = stft.process_samples(&[sample as f64]) {
                    Stft::hpss_one(&mut col, &stft.harm, &stft.perc);
                    amplitude_to_db(&mut col);
                    assert_eq!(
                        col.len(),
                        4097,
                        "STFT column length different from N_FFT / 2 + 1 (4097 by default)"
                    );
                    let scaled = min_max_scale(&col);
                    for (y, s) in scaled.iter().enumerate() {
                        image.get_pixel_mut(x, y as u32).0 = [((s * 255.0).round() as u8)];
                    }
                    x += 1;
                    pb.inc(1);
                    assert!(x < width, "Generated more STFT frames than expected");
                }
            }
            image.save(args.output)?;
            pb.finish();
        }
    }
    Ok(())
}
