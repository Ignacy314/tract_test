use std::fs::File;
use std::io::{BufWriter, Write};

use clap::Parser;

use self::spectrogram::Spectro;

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
}

fn main() {
    let args = Args::parse();

    let mut writer = BufWriter::new(File::create(args.output_file).unwrap());

    let mut reader = hound::WavReader::open(args.input_file).unwrap();
    reader.seek(args.start_sample).unwrap();

    let mut spectro = Spectro::new();
    let samples = reader.samples::<i32>();
    let mut buf = [0f64; 4096];
    let mut j = 0;
    let mut f = 0;
    write!(writer, "[").unwrap();
    for (i, s) in samples.enumerate() {
        let sample = s.unwrap();
        buf[j] = sample as f64;
        j += 1;
        if j == 4096 {
            j = 0;
            let harm = spectro.process_samples(&buf);
            if i > 4096 {
                let harm = harm.unwrap();
                let col = spectro.hpss_last(harm);
                write!(writer, "[").unwrap();
                col.iter().for_each(|c| write!(writer, "{c},").unwrap());
                writeln!(writer, "],").unwrap();
                f += 1;
            }
            if f >= args.frames {
                break;
            }
        }
    }
    write!(writer, "]").unwrap();
}
