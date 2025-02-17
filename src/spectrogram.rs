use circular_buffer::CircularBuffer;
use librustosa::stft::STFT;
use median::Filter;

const WINDOW_SIZE: usize = 8192;
const STEP_SIZE: usize = 4096;
const FILTER_WIDTH: usize = 31;
const COLS: usize = FILTER_WIDTH * 2 + 1;
const ROWS: usize = 4097;

pub struct Spectro {
    stft: STFT<f64>,
    columns: CircularBuffer<COLS, Vec<f64>>,
    //filtered_columns: CircularBuffer<COLS, Vec<f64>>,
}

impl Spectro {
    pub fn new() -> Self {
        Self {
            stft: STFT::new(WINDOW_SIZE, STEP_SIZE).unwrap(),
            columns: CircularBuffer::new(),
            //filtered_columns: CircularBuffer::new(),
        }
    }

    pub fn process_samples(&mut self, samples: &[f64]) -> Option<Vec<f64>> {
        self.stft.append_samples(samples);
        let mut res = None;
        while self.stft.contains_enough_to_compute() {
            let col = self.stft.compute_column().unwrap();

            let mut filter = Filter::new(FILTER_WIDTH);
            let mut filtered = Vec::new();
            col.iter().for_each(|&s| {
                filtered.push(filter.consume(s));
            });
            //self.filtered_columns.push_back(filtered);
            res = Some(filtered);
            self.columns.push_back(col);

            self.stft.move_to_next_column();
        }
        res
    }

    pub fn hpss_last(&mut self, mut harm: Vec<f64>) -> Vec<f64> {
        //let mut harm = self.filtered_columns.back().unwrap();

        let perc = {
            let mut perc = Vec::new();
            let iter =
                (0..ROWS).map(|row_idx| self.columns.iter().flatten().skip(row_idx).step_by(COLS));
            for row in iter {
                let mut filtered = Vec::new();
                let mut filter = Filter::new(FILTER_WIDTH);
                row.for_each(|&s| {
                    filtered.push(filter.consume(s));
                });
                perc.push(*filtered.last().unwrap());
            }
            perc
        };

        let mask = Self::softmask_one(&harm, &perc);
        for (h, m) in harm.iter_mut().zip(mask) {
            *h *= m
        };
        harm
    }

    //pub fn hpss_full(&mut self) -> Vec<Vec<f64>> {
    //    let mut harm = self.filtered_columns.to_vec();
    //
    //    let perc = {
    //        let mut perc = Vec::new();
    //        let iter =
    //            (0..ROWS).map(|row_idx| self.columns.iter().flatten().skip(row_idx).step_by(COLS));
    //        for row in iter {
    //            let mut filtered = Vec::new();
    //            let mut filter = Filter::new(FILTER_WIDTH);
    //            row.for_each(|&s| {
    //                filtered.push(filter.consume(s));
    //            });
    //            perc.push(filtered);
    //        }
    //        perc
    //    };
    //
    //    let mask = Self::softmask(&harm, &perc);
    //    for (hh, mm) in harm.iter_mut().zip(mask) {
    //        for (h, m) in hh.iter_mut().zip(mm) {
    //            *h *= m;
    //        }
    //    }
    //    harm
    //}

    fn softmask_one(x: &[f64], y: &[f64]) -> Vec<f64> {
        let z = x.iter().zip(y).map(|(x, y)| {
            let maxi = x.max(*y);
            if maxi < f64::MIN_POSITIVE  {
                (1.0, false)
            } else {
                (maxi, true)
            }
        });

        let mut mask = x.iter().zip(z.clone()).map(|(x, z)| (x / z.0).powi(2)).collect::<Vec<f64>>();

        let ref_mask = y.iter().zip(z.clone()).map(|(y, z)| (y / z.0).powi(2));

        for ((m, r), z) in (mask.iter_mut().zip(ref_mask)).zip(z) {
            if z.1 {
                *m /= *m + r
            } else {
                *m = 0.5
            }
        }

        mask
    }

    //fn softmask(x: &[Vec<f64>], y: &[Vec<f64>]) -> Vec<Vec<f64>> {
    //    let z = x.iter().zip(y).map(|(xx, yy)| {
    //        xx.iter().zip(yy).map(|(x, y)| {
    //            let maxi = x.max(*y);
    //            if maxi < f64::MIN_POSITIVE {
    //                (1.0, false)
    //            } else {
    //                (maxi, true)
    //            }
    //        })
    //    });
    //
    //    let mut mask = x
    //        .iter()
    //        .zip(z.clone())
    //        .map(|(xx, zz)| xx.iter().zip(zz).map(|(x, z)| (x / z.0).powi(2)).collect::<Vec<f64>>()).collect::<Vec<Vec<f64>>>();
    //
    //    let ref_mask = y
    //        .iter()
    //        .zip(z.clone())
    //        .map(|(yy, zz)| yy.iter().zip(zz).map(|(y, z)| (y / z.0).powi(2)));
    //
    //    for ((mm, rr), zz) in (mask.iter_mut().zip(ref_mask)).zip(z) {
    //        for ((m, r), z) in (mm.iter_mut().zip(rr)).zip(zz) {
    //            if z.1 {
    //                *m /= *m + r
    //            } else {
    //                *m = 0.5
    //            }
    //        }
    //    }
    //
    //    mask
    //}
}
