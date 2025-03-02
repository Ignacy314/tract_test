use std::error::Error;

use clap::Parser;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;

use rand::random_range;
use rand::rng;
use rand::rngs::ThreadRng;
use rand::seq::IteratorRandom;
use spectrogram::amplitude_to_db;
use spectrogram::min_max_scale;
use spectrogram::{Stft, HOP_LENGTH, N_FFT};

#[derive(Parser)]
struct ImgGenArgs {
    /// Path to input wav file
    #[arg(short, long)]
    input: String,
    /// Path to output file
    #[arg(short, long)]
    output: Option<String>,
    /// Background pattern csv to subtract
    #[arg(short, long)]
    bg_pattern: Option<String>,
    /// Output directory of cut images
    #[arg(short, long, requires("prefix_for_split"))]
    split_output_dir: Option<String>,
    /// Prefix for split file name
    #[arg(short, long)]
    prefix_for_split: Option<String>,
}

#[allow(clippy::too_many_arguments)]
fn split(
    col_count: u32,
    next_col_split: &mut u32,
    n: u32,
    col: &[f64],
    j: &mut u32,
    rng: &mut ThreadRng,
    dir: &str,
    prefix_for_split: &str,
) -> Result<(), Box<dyn Error>> {
    if col_count >= *next_col_split && col_count + 224 < n {
        let mut row_count = 0;
        let mut train_i = 0;
        let mut test_i = 0;

        let mut images = Vec::new();
        while row_count + 224 < 2048 {
            let mut image = image::RgbImage::new(224, 224);
            for x in 0..224 {
                for (y, s) in col.iter().skip(row_count).take(224).enumerate() {
                    let pixel = (s * 255.0).round() as u8;
                    image.get_pixel_mut(x, 224 - 1 - y as u32).0 = [pixel, pixel, pixel];
                }
            }

            images.push(image);

            row_count += random_range(56..112);
        }
        let mut test_indices =
            (0..images.len()).choose_multiple(rng, (images.len() as f32 / 5.0).round() as usize);
        test_indices.sort_unstable();
        let mut test_iter = test_indices.iter();
        let mut next_test = test_iter.next();
        for (u, image) in images.iter().enumerate() {
            if let Some(test_index) = next_test {
                if u == *test_index {
                    image.save(format!("{}/test/{}_{j}_{test_i}.png", dir, prefix_for_split))?;
                    test_i += 1;
                    next_test = test_iter.next();
                    continue;
                }
            }
            image.save(format!("{}/train/{}_{j}_{train_i}.png", dir, prefix_for_split))?;
            train_i += 1;
        }
        *next_col_split += random_range(56..112);
        *j += 1;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = ImgGenArgs::parse();
    let mut reader = hound::WavReader::open(args.input)?;
    const HEIGHT: u32 = 4097;
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
    //let mut col_count: u32 = 0;

    let pattern = if let Some(bg_pattern) = args.bg_pattern {
        let mut csv = csv::Reader::from_path(bg_pattern)?;
        let mut records = csv.deserialize();
        let pattern: Vec<f64> = records.next().unwrap()?;
        Some(pattern)
    } else {
        None
    };

    let mut spectrogram = Vec::new();

    let mut stft = Stft::new(N_FFT, HOP_LENGTH, pattern);
    let samples = reader.samples::<i32>();
    for s in samples {
        let sample = s?;
        if let Some(mut col) = stft.process_samples(&mut [sample as f64]) {
            stft.hpss_one(&mut col);
            //let col_max = col.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
            //stft.set_ref_db(*col_max);
            amplitude_to_db(&mut col);
            min_max_scale(&mut col);

            spectrogram.push(col);

            //let mut col_img = image::GrayImage::new(1, HEIGHT);

            //if args.output.is_some() {
            //    for (y, s) in col.iter().enumerate() {
            //        image.get_pixel_mut(col_count, HEIGHT - 1 - y as u32).0 =
            //            [((s * 255.0).round() as u8)];
            //        //col_img.get_pixel_mut(1, HEIGHT - 1 - y as u32).0 = [((s * 255.0).round() as u8)];
            //    }
            //}
            ////let jpeg_buf = Cursor::new(Vec::new());
            ////let mut jpeg_writer = BufWriter::new(jpeg_buf);
            ////col_img.write_to(&mut jpeg_writer, image::ImageFormat::Jpeg)?;
            ////let jpeg_data = jpeg_writer.buffer();
            //
            //if let Some(dir) = args.split_output_dir.as_ref() {
            //    split(
            //        col_count,
            //        &mut next_col_split,
            //        n,
            //        &col,
            //        &mut j,
            //        &mut rng,
            //        dir,
            //        args.prefix_for_split.as_ref().unwrap(),
            //    )?
            //}
            //
            //col_count += 1;
            pb.inc(1);
        }
    }
    for mut col in stft.process_tail() {
        //let col_max = col.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
        //stft.set_ref_db(*col_max);
        amplitude_to_db(&mut col);
        min_max_scale(&mut col);

        spectrogram.push(col);

        //for (y, s) in col.iter().enumerate() {
        //    image.get_pixel_mut(col_count, HEIGHT - 1 - y as u32).0 = [((s * 255.0).round() as u8)];
        //}
        //
        //if let Some(dir) = args.split_output_dir.as_ref() {
        //    split(
        //        col_count,
        //        &mut next_col_split,
        //        n,
        //        &col,
        //        &mut j,
        //        &mut rng,
        //        dir,
        //        args.prefix_for_split.as_ref().unwrap(),
        //    )?
        //}
        //
        //col_count += 1;
        pb.inc(1);
    }
    if let Some(output) = args.output {
        let mut image = image::GrayImage::new(spectrogram.len() as u32, HEIGHT);
        for (col_count, col) in spectrogram.iter().enumerate() {
            for (y, s) in col.iter().enumerate() {
                image
                    .get_pixel_mut(col_count as u32, HEIGHT - 1 - y as u32)
                    .0 = [((s * 255.0).round() as u8)];
                //col_img.get_pixel_mut(1, HEIGHT - 1 - y as u32).0 = [((s * 255.0).round() as u8)];
            }
        }
        image.save(output)?;
    }
    if let Some(dir) = args.split_output_dir.as_ref() {
        let mut next_col_split = 0;
        let mut j = 0;
        let mut rng = rng();
        for (col_count, col) in spectrogram.iter().enumerate() {
            split(
                col_count as u32,
                &mut next_col_split,
                n,
                col,
                &mut j,
                &mut rng,
                dir,
                args.prefix_for_split.as_ref().unwrap(),
            )?
        }
    }

    pb.finish_with_message(format!("Frames processed: {}", pb.position()));
    Ok(())
}
