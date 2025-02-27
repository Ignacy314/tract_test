use std::error::Error;
use std::fmt::Display;
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

//use self::spectrogram::FILTER_WIDTH;
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
    /// Test MLP accuracy
    TestMlp(TestMlpArgs),
}

#[derive(clap::Args)]
struct GenerateArgs {
    /// Path to output file
    #[arg(short, long)]
    output: String,
    /// Path to input file
    #[arg(short, long)]
    input: String,
    /// Width of the median filter
    #[arg(short, long)]
    width: usize,
    /// Power of the softmask
    #[arg(short, long)]
    power: i32,
    /// Amplitude to dB reference value
    #[arg(short, long)]
    ref_db: f64,
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
    resnet: Option<String>,
    /// Width of the median filter
    #[arg(short, long)]
    width: usize,
    /// Power of the softmask
    #[arg(short, long)]
    power: i32,
    /// Amplitude to dB reference value
    #[arg(short, long)]
    ref_db: f64,
}

#[derive(clap::Args)]
struct ImgGenArgs {
    /// Path to input wav file
    #[arg(short, long)]
    input: String,
    /// Path to output file
    #[arg(short, long)]
    output: String,
    /// Width of the median filter
    #[arg(short, long)]
    width: usize,
    /// Power of the softmask
    #[arg(short, long)]
    power: i32,
    /// Amplitude to dB reference value
    #[arg(short, long)]
    ref_db: f64,
}

#[derive(clap::Args)]
struct TestMlpArgs {
    /// Path to input wav file
    #[arg(short, long)]
    input: String,
    /// Path to the MLP onnx model
    #[arg(short, long)]
    mlp: String,
    /// Path to the drone distance class csv corresponding to the wav file
    #[arg(short, long)]
    drone_csv: String,
    /// Width of the median filter
    #[arg(short, long)]
    width: usize,
    /// Power of the softmask
    #[arg(short, long)]
    power: i32,
    /// Amplitude to dB reference value
    #[arg(short, long)]
    ref_db: f64,
}

fn amplitude_to_db(x_vec: &mut [f64], ref_db: f64) {
    //let mut x_max = f64::MIN;
    //for x in x_vec.iter_mut() {
    //    *x *= *x;
    //    if *x > x_max {
    //        x_max = *x;
    //    }
    //}
    let sub = 10.0 * ref_db.powi(2).max(1e-10).log10();
    for x in x_vec.iter_mut() {
        *x = (10.0 * x.max(1e-10).log10() - sub).max(-80.0);
    }
}

fn min_max_scale(x_vec: &mut [f64]) {
    let mut x_max = f64::MIN;
    let mut x_min = f64::MAX;
    for &x in x_vec.iter() {
        if x > x_max {
            x_max = x;
        }
        if x < x_min {
            x_min = x;
        }
    }
    for x in x_vec {
        *x = (*x - x_min) / (x_max - x_min);
    }
    //let x_std = x_vec
    //    .iter()
    //    .map(|x| (x - x_min) / (x_max - x_min))
    //    .collect::<Vec<_>>();
    //x_std
}

fn generate(args: GenerateArgs) -> Result<(), Box<dyn Error>> {
    let mut reader = hound::WavReader::open(args.input)?;
    let mut w = BufWriter::new(File::create(args.output)?);
    let width = (reader.duration() - 4096) / 4096;

    let pb = ProgressBar::new(u64::from(width));
    let t = f64::from(width).log10().ceil() as u64;
    pb.set_style(
        ProgressStyle::with_template(&format!(
        "[{{elapsed_precise}}] {{bar:40.cyan/blue}} {{pos:>{t}}}/{{len:{t}}} ({{percent}}%) {{msg}}"
    ))
        .unwrap()
        .progress_chars("##-"),
    );

    let mut i = 0;
    let mut stft = Stft::new(N_FFT, HOP_LENGTH, args.width);
    let samples = reader.samples::<i32>();
    for s in samples {
        let sample = s?;
        if let Some(mut col) = stft.process_samples(&mut [sample as f64]) {
            assert_eq!(col.len(), 4097);
            i += 1;
            assert!(i <= width);

            stft.hpss_one(&mut col, args.power);
            amplitude_to_db(&mut col, args.ref_db);
            min_max_scale(&mut col);

            for s in &col[..4096] {
                write!(w, "{s},")?;
            }
            writeln!(w, "{}", col[4096])?;
            pb.inc(1);
        }
    }
    pb.finish();
    Ok(())
}

#[derive(Debug)]
struct InferError(String);
impl Display for InferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl Error for InferError {}

fn infer(args: InferArgs) -> Result<(), Box<dyn Error>> {
    let mlp = tract_onnx::onnx().model_for_path(args.mlp)?;
    let mlp = mlp.with_input_fact(0, f64::fact([4097]).into())?;
    let mlp = mlp.into_optimized()?;
    let mlp = mlp.into_runnable()?;

    let resnet = if let Some(resnet_path) = args.resnet {
        let resnet = tract_onnx::onnx().model_for_path(resnet_path)?;
        let resnet = resnet.with_input_fact(0, f32::fact([1, 224, 224, 3]).into())?;
        let resnet = resnet.into_optimized()?;
        let resnet = resnet.into_runnable()?;
        Some(resnet)
    } else {
        None
    };

    let mut rng = rng();

    let mut buffer = CircularBuffer::<224, [f32; 4097]>::new();

    let mut reader = hound::WavReader::open(args.input)?;

    if resnet.is_some() {
        if args.start_sample < 224 {
            return Err(Box::new(InferError(
                "When using resnet need to start at least at sample 224".to_string(),
            )));
        } else {
            reader.seek(args.start_sample - 224)?;
        }
    } else {
        reader.seek(args.start_sample)?;
    }

    let mut stft = Stft::new(N_FFT, HOP_LENGTH, args.width);
    let samples = reader.samples::<i32>();
    let mut f = 0;
    for s in samples {
        let sample = s?;
        if let Some(mut col) = stft.process_samples(&mut [sample as f64]) {
            assert_eq!(col.len(), 4097);
            f += 1;

            stft.hpss_one(&mut col, args.power);
            amplitude_to_db(&mut col, args.ref_db);
            min_max_scale(&mut col);

            if let Some(resnet) = resnet.as_ref() {
                let mut image_col = [0f32; 4097];
                for (i, s) in col.iter().enumerate() {
                    image_col[i] = *s as f32;
                }
                buffer.push_back(image_col);

                if buffer.is_full() {
                    let start_row = rng.random_range(1800..2500);
                    let resnet_input: Tensor = {
                        tract_ndarray::Array4::from_shape_fn((1, 224, 224, 3), |(_, x, y, _)| {
                            buffer.get(x).unwrap()[y + start_row]
                        })
                        .into()
                    };
                    let start = Instant::now();
                    let resnet_result = resnet.run(tvec!(resnet_input.into()))?;
                    let elapsed = start.elapsed();

                    let mlp_input: Tensor = tract_ndarray::Array1::from_vec(col).into();
                    let mlp_result = mlp.run(tvec!(mlp_input.into()))?;

                    let resnet_val = &resnet_result[0].to_array_view::<f32>()?;
                    let resnet_best = resnet_val
                        .iter()
                        .zip(0..)
                        .max_by(|a, b| a.0.total_cmp(b.0))
                        .unwrap();

                    println!(
                        "{f:6}: {:2} | {} | {:.3} | {:4} | {start_row}",
                        mlp_result[0].to_array_view::<TDim>()?.get(0).unwrap(),
                        resnet_best.1,
                        resnet_best.0,
                        elapsed.as_millis()
                    );
                }
            } else {
                let mlp_input: Tensor = tract_ndarray::Array1::from_vec(col).into();
                let mlp_result = mlp.run(tvec!(mlp_input.into()))?;
                println!("{f:6}: {:2}", mlp_result[0].to_array_view::<TDim>()?.get(0).unwrap(),);
            }

            if f >= args.frames {
                break;
            }
        }
    }
    Ok(())
}

fn img_gen(args: ImgGenArgs) -> Result<(), Box<dyn Error>> {
    let mut reader = hound::WavReader::open(args.input)?;
    const HEIGHT: u32 = 4097;
    let width = (reader.duration() - 4096) / 4096;
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

    let mut stft = Stft::new(N_FFT, HOP_LENGTH, args.width);
    let samples = reader.samples::<i32>();
    for s in samples {
        let sample = s?;
        if let Some(mut col) = stft.process_samples(&mut [sample as f64]) {
            assert_eq!(col.len(), 4097);

            stft.hpss_one(&mut col, args.power);
            amplitude_to_db(&mut col, args.ref_db);
            min_max_scale(&mut col);

            for (y, s) in col.iter().enumerate() {
                image.get_pixel_mut(x, y as u32).0 = [((s * 255.0).round() as u8)];
            }
            x += 1;
            pb.inc(1);
            assert!(x <= width);
        }
    }
    image.save(args.output)?;
    pb.finish();
    Ok(())
}

fn test_mlp(args: TestMlpArgs) -> Result<(), Box<dyn Error>> {
    let mlp = tract_onnx::onnx().model_for_path(args.mlp)?;
    let mlp = mlp.with_input_fact(0, f64::fact([4097]).into())?;
    let mlp = mlp.into_optimized()?;
    let mlp = mlp.into_runnable()?;

    let mut reader = hound::WavReader::open(args.input)?;

    let mut csv = csv::Reader::from_path(args.drone_csv)?;
    let mut csv = csv.deserialize();
    //let mut csv = csv.deserialize().skip((FILTER_WIDTH + 1) / 2);

    let n = (reader.duration() - 4096) / 4096;
    let pb = ProgressBar::new(u64::from(n));
    let t = f64::from(n).log10().ceil() as u64;
    pb.set_style(
        ProgressStyle::with_template(&format!(
        "[{{elapsed_precise}}] {{bar:40.cyan/blue}} {{pos:>{t}}}/{{len:{t}}} ({{percent}}%) {{msg}}"
    ))
        .unwrap()
        .progress_chars("##-"),
    );
    let mut sum_diff = 0i64;
    let mut count_ok = 0u32;

    let mut stft = Stft::new(N_FFT, HOP_LENGTH, args.width);
    let samples = reader.samples::<i32>();
    for s in samples {
        let sample = s?;
        if let Some(mut col) = stft.process_samples(&mut [sample as f64]) {
            assert_eq!(col.len(), 4097);
            if let Some(csv_result) = csv.next() {
                stft.hpss_one(&mut col, args.power);
                amplitude_to_db(&mut col, args.ref_db);
                min_max_scale(&mut col);

                let mlp_input: Tensor = tract_ndarray::Array1::from_vec(col).into();
                let mlp_result = mlp.run(tvec!(mlp_input.into()))?;
                let mlp_class = mlp_result[0]
                    .to_array_view::<TDim>()?
                    .get(0)
                    .unwrap()
                    .to_i64()?;

                let record: i64 = csv_result?;

                //println!("{record} | {mlp_class}");

                let diff = (mlp_class - record).abs();
                if diff == 0 {
                    count_ok += 1;
                }
                sum_diff += diff;

                pb.inc(1);
                pb.set_message(format!("Acc: {}", count_ok as f32 / pb.position() as f32));
            } else {
                break;
            }
        }
    }
    let acc = count_ok as f32 / n as f32;
    let avg_diff = sum_diff as f32 / n as f32;
    pb.finish_with_message(format!(
        "Acc: {acc} | Avg diff: {avg_diff} | Frames processed: {}",
        pb.position()
    ));
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate(args) => generate(args)?,
        Commands::Infer(args) => infer(args)?,
        Commands::ImgGen(args) => img_gen(args)?,
        Commands::TestMlp(args) => test_mlp(args)?,
    }
    Ok(())
}
