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

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let model = tract_onnx::onnx().model_for_path(args.model)?;
    let model = model.with_input_fact(0, f64::fact([4097]).into())?;
    let model = model.into_optimized()?;
    let model = model.into_runnable()?;

    //let model = tract_onnx::onnx()
    //    // load the model
    //    .model_for_path(args.model)?
    //    // specify input type and shape
    //    .with_input_fact(0, f64::fact([4097]).into())?
    //    // optimize the model
    //    .into_optimized()?
    //    // make the model runnable and fix its inputs and outputs
    //    .into_runnable()?;

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
            let input: Tensor = tract_ndarray::Array1::from_vec(col.clone()).into();
            let result = model.run(tvec!(input.into()))?;
            println!("{f}: result: {:?}", result[0].to_array_view::<f32>());
            //let best = result[0]
            //    .to_array_view::<f32>()?
            //    .iter()
            //    .copied()
            //    .zip(1..)
            //    .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

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
