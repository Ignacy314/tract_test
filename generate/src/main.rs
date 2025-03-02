use std::error::Error;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;

use clap::Parser;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;

use spectrogram::amplitude_to_db;
use spectrogram::min_max_scale;
use spectrogram::{Stft, HOP_LENGTH, N_FFT};

#[derive(Parser)]
struct GenerateArgs {
    /// Path to output file
    #[arg(short, long)]
    output: String,
    /// Path to input file
    #[arg(short, long)]
    input: String,
    /// Background pattern csv to subtract
    #[arg(short, long)]
    bg_pattern: Option<String>,
    ///// Width of the median filter
    //#[arg(short, long)]
    //width: usize,
    ///// Power of the softmask
    //#[arg(short, long)]
    //power: i32,
    ///// Amplitude to dB reference value
    //#[arg(short = 'b', long)]
    //ref_db: f64,
    //#[arg(short, long)]
    //hpss: bool,
    //#[arg(short, long)]
    //amp_to_db: bool,
    //#[arg(short, long)]
    //min_max_scale: bool,
    #[arg(short, long)]
    skip: Option<usize>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = GenerateArgs::parse();
    let mut reader = hound::WavReader::open(args.input)?;
    let mut w = BufWriter::new(File::create(args.output)?);
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

    let pattern = if let Some(bg_pattern) = args.bg_pattern {
        let mut csv = csv::Reader::from_path(bg_pattern)?;
        let mut records = csv.deserialize();
        let pattern: Vec<f64> = records.next().unwrap()?;
        Some(pattern)
    } else {
        None
    };

    let skip = args.skip.unwrap_or(1);

    //let mut i = 0;
    let mut stft = Stft::new(N_FFT, HOP_LENGTH, pattern);
    let samples = reader.samples::<i32>().step_by(skip);
    for s in samples {
        let sample = s?;
        if let Some(mut col) = stft.process_samples(&mut [sample as f64]) {
            //assert_eq!(col.len(), 4097);
            //i += 1;
            //assert!(i <= width);

            stft.hpss_one(&mut col);
            amplitude_to_db(&mut col);
            min_max_scale(&mut col);

            for s in &col[..4096] {
                write!(w, "{s},")?;
            }
            writeln!(w, "{}", col[4096])?;
            pb.inc(1);
        }
    }
    for mut col in stft.process_tail() {
        amplitude_to_db(&mut col);
        min_max_scale(&mut col);
        for s in &col[..4096] {
            write!(w, "{s},")?;
        }
        writeln!(w, "{}", col[4096])?;
        pb.inc(1);
    }
    pb.finish_with_message(format!("Frames processed: {}", pb.position()));
    Ok(())
}
