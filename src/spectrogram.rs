use std::f64::consts::PI;
use std::sync::Arc;

use circular_buffer::CircularBuffer;
use median::Filter;
use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
use strider::{SliceRing, SliceRingImpl};

pub const N_FFT: usize = 8192;
pub const HOP_LENGTH: usize = 4096;
const FILTER_WIDTH: usize = 31;
const COLS: usize = FILTER_WIDTH * 2 + 1;
const ROWS: usize = 4097;

fn new_hann_window(size: usize) -> Vec<f64> {
    let mut window = Vec::with_capacity(size);

    for n in 0..size {
        let x = n as f64 / (size - 1) as f64;
        let value = 0.5 * (1.0 - (2.0 * PI * x).cos());
        window.push(value);
    }

    window
}

pub struct Stft {
    n_fft: usize,
    hop_length: usize,
    window: Vec<f64>,
    sample_ring: SliceRingImpl<f64>,
    //planner: RealFftPlanner<f64>,
    forward: Arc<dyn RealToComplex<f64>>,
    indata: Vec<f64>,
    outdata: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
    columns: CircularBuffer<COLS, Vec<Complex<f64>>>,
}

impl Stft {
    pub fn new(n_fft: usize, hop_length: usize) -> Self {
        let mut planner = RealFftPlanner::new();
        let forward = planner.plan_fft_forward(n_fft);
        let indata = forward.make_input_vec();
        let outdata = forward.make_output_vec();
        let scratch = forward.make_scratch_vec();
        Self {
            n_fft,
            hop_length,
            window: new_hann_window(n_fft),
            sample_ring: SliceRingImpl::new(),
            //planner,
            forward,
            indata,
            outdata,
            scratch,
            columns: CircularBuffer::new(),
        }
    }

    pub fn contains_enough_to_compute(&self) -> bool {
        self.n_fft <= self.sample_ring.len()
    }

    pub fn process_samples(&mut self, samples: &[f64]) -> Option<Vec<f64>> {
        self.sample_ring.push_many_back(samples);

        let mut out = None;
        while self.contains_enough_to_compute() {
            self.compute_into_outdata();
            let col = self.outdata.clone();

            let mut filter = Filter::new(FILTER_WIDTH);
            let mut filtered = Vec::new();
            col.iter().for_each(|&s| {
                filtered.push(filter.consume(s.norm()));
            });
            self.columns.push_back(col);

            out = Some(filtered);
            self.sample_ring.drop_many_front(self.hop_length);
        }
        out
    }

    fn compute_into_outdata(&mut self) {
        self.sample_ring.read_many_front(&mut self.indata[..]);
        print!("{}, ", self.indata.len());

        for (r, w) in self.indata.iter_mut().zip(self.window.iter()) {
            *r *= *w;
        }

        self.forward
            .process_with_scratch(&mut self.indata, &mut self.outdata, &mut self.scratch)
            .unwrap();

        println!("{}", self.outdata.len());
    }

    pub fn hpss_one(&mut self, mut harm: Vec<f64>) -> Vec<f64> {
        let perc = {
            let mut perc = Vec::new();
            let iter =
                (0..ROWS).map(|row_idx| self.columns.iter().flatten().skip(row_idx).step_by(COLS));
            for row in iter {
                let mut filtered = Vec::new();
                let mut filter = Filter::new(FILTER_WIDTH);
                row.for_each(|&s| {
                    filtered.push(filter.consume(s.norm()));
                });
                perc.push(*filtered.last().unwrap());
            }
            perc
        };

        let mask = Self::softmask_one(&harm, &perc);
        for (h, m) in harm.iter_mut().zip(mask) {
            *h *= m
        }
        harm
    }

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
}
