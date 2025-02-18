use std::fs::File;
use std::io::BufWriter;

use clap::Parser;

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
}

fn main() {
    let args = Args::parse();

    let writer = BufWriter::new(File::create(args.output_file).unwrap());

    let mut reader = hound::WavReader::open(args.input_file).unwrap();
    reader.seek(args.start_sample).unwrap();

    let mut stft = Stft::new(N_FFT, HOP_LENGTH);
    let samples = reader.samples::<i32>();
    //let mut buf = [0f64; 4096];
    //let mut j = 0;
    let mut f = 0;
    //write!(writer, "[").unwrap();
    //let mut spectro = Spectro { data: vec![] };
    let mut data = Vec::new();
    for s in samples {
        let sample = s.unwrap();
        if let Some((harm, perc)) = stft.process_samples(&[sample as f64]) {
            //let harm = harm.unwrap();
            let col = stft.hpss_one(harm, &perc);
            //println!("{}", col.len());
            //write!(writer, "[").unwrap();
            data.push(col);
            //col.iter().for_each(|c| write!(writer, "{c},").unwrap());
            //writeln!(writer, "],").unwrap();
            f += 1;
            println!("{f}");
            if f >= args.frames {
                break;
            }
        }
        //buf[j] = sample as f64;
        //j += 1;
        //if j == 4096 {
        //    j = 0;
        //    let harm = stft.process_samples(&buf);
        //    if let Some(harm) = harm {
        //        //let harm = harm.unwrap();
        //        let col = stft.hpss_one(harm);
        //        //println!("{}", col.len());
        //        //write!(writer, "[").unwrap();
        //        data.push(col);
        //        //col.iter().for_each(|c| write!(writer, "{c},").unwrap());
        //        //writeln!(writer, "],").unwrap();
        //        f += 1;
        //        println!("{f}");
        //    }
        //    if f >= args.frames {
        //        break;
        //    }
        //}
    }
    serde_json::to_writer(writer, &data).unwrap();
    //write!(writer, "]").unwrap();
}
