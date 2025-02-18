use std::error::Error;
use std::fs::File;
use std::io::BufWriter;

use clap::Parser;
use tract_onnx::prelude::*;

use self::spectrogram::{Stft, HOP_LENGTH, N_FFT};

mod spectrogram;

#[derive(clap::Parser)]
struct Args {
    #[arg(short, long)]
    start_sample: u32,
    #[arg(short, long)]
    output_file: String,
    #[arg(short, long)]
    input_file: String,
    #[arg(short, long)]
    frames: usize,
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
    let x_std = x_vec.iter().map(|x| (x - x_min) / (x_max - x_min)).collect::<Vec<_>>();
    x_std
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let model = tract_onnx::onnx().model_for_path(args.model)?;
    let model = model.with_input_fact(0, f64::fact([4097]).into())?;
    let model = model.into_optimized()?;
    let model = model.into_runnable()?;

    let writer = BufWriter::new(File::create(args.output_file).unwrap());

    let mut reader = hound::WavReader::open(args.input_file).unwrap();
    reader.seek(args.start_sample).unwrap();

    let mut stft = Stft::new(N_FFT, HOP_LENGTH);
    let samples = reader.samples::<i32>();
    let mut f = 0;
    let mut data = Vec::new();
    for s in samples {
        let sample = s.unwrap();
        if let Some((harm, perc)) = stft.process_samples(&[sample as f64]) {
            let col = stft.hpss_one(harm, &perc);
            let scaled = min_max_scale(&col);
            let input: Tensor = tract_ndarray::Array1::from_vec(scaled).into();
            let result = model.run(tvec!(input.into()))?;
            println!("{f}: result: {:?}", result[0].to_array_view::<TDim>());

            data.push(col);
            f += 1;
            if f >= args.frames {
                break;
            }
        }
    }
    serde_json::to_writer(writer, &data).unwrap();
    Ok(())
}
