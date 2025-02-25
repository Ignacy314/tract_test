use std::array;
use std::f64::consts::PI;
use std::sync::Arc;

use circular_buffer::CircularBuffer;
use median::Filter;
use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
use strider::{SliceRing, SliceRingImpl};

pub const N_FFT: usize = 8192;
pub const HOP_LENGTH: usize = 4096;
pub const FILTER_WIDTH: usize = 31;
pub const HALF_FILTER_WIDTH: usize = 31 / 2;
const COLS: usize = HALF_FILTER_WIDTH;
const ROWS: usize = N_FFT / 2 + 1;

fn new_hann_window(size: usize) -> Vec<f64> {
    let mut window = Vec::with_capacity(size);

    for n in 0..size {
        let x = n as f64 / size as f64;
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
    forward: Arc<dyn RealToComplex<f64>>,
    indata: Vec<f64>,
    outdata: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
    row_filters: [Filter<f64>; ROWS],
    //cols: CircularBuffer<COLS, [f64; ROWS]>,
    cols: CircularBuffer<COLS, ([f64; ROWS], Vec<f64>)>,
    pub harm: [f64; ROWS],
    pub perc: [f64; ROWS],
    //norm: [f64; ROWS],
    ready_counter: usize,
    filter_width: usize,
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
            forward,
            indata,
            outdata,
            scratch,
            row_filters: array::from_fn(|_| Filter::new(FILTER_WIDTH)),
            cols: CircularBuffer::new(),
            harm: [0f64; ROWS],
            perc: [0f64; ROWS],
            //norm: [0f64; ROWS],
            ready_counter: 0,
            filter_width: FILTER_WIDTH,
        }
    }

    pub fn contains_enough_to_compute(&self) -> bool {
        self.n_fft <= self.sample_ring.len()
    }

    /// Pushes the samples into a ring buffer and while there is enough samples for another FFT
    /// computes it, then computes the median filtered harmonic and the newest elements of the
    /// median filtered percussives. Returns the newest median filtered harmonic column vector and a vector
    /// consisting of the newest element of the median filtered percussives
    pub fn process_samples(&mut self, samples: &mut [f64]) -> Option<Vec<f64>> {
        for s in samples.iter_mut() {
            *s /= i32::MAX as f64;
        }
        self.sample_ring.push_many_back(samples);

        let mut out = None;
        while self.contains_enough_to_compute() {
            self.compute_into_outdata();

            let mut filter = Filter::new(self.filter_width);
            let norm_col = self.outdata.iter().map(|s| s.norm()).collect::<Vec<f64>>();
            let relfect = norm_col
                .iter()
                .take(self.filter_width / 2)
                .chain(norm_col.iter())
                .chain(norm_col.iter().rev().take(self.filter_width / 2));

            for (i, sn) in relfect.enumerate() {
                let f = filter.consume(*sn);
                if i >= self.filter_width {
                    self.perc[i - self.filter_width] = f;
                }
            }

            match self.ready_counter {
                0..HALF_FILTER_WIDTH => {}
                HALF_FILTER_WIDTH => {
                    for (_, col) in self.cols.iter().rev().chain(self.cols.iter()) {
                        self.row_filters
                            .iter_mut()
                            .zip(col.iter())
                            .for_each(|(r, c)| {
                                r.consume(*c);
                            });
                    }
                    self.row_filters
                        .iter_mut()
                        .zip(norm_col.iter())
                        .enumerate()
                        .for_each(|(i, (r, &sn))| {
                            self.harm[i] = r.consume(sn);
                        });
                }
                _ => {
                    self.row_filters
                        .iter_mut()
                        .zip(norm_col.iter())
                        .enumerate()
                        .for_each(|(i, (r, &sn))| {
                            self.harm[i] = r.consume(sn);
                        });
                }
            }

            if let Some((perc, norm_col)) = self.cols.push_back((self.perc, norm_col)) {
                self.perc = perc;
                out = Some(norm_col);
            }

            self.ready_counter += 1;

            self.sample_ring.drop_many_front(self.hop_length);
        }
        out
    }

    pub fn process_tail(&mut self) -> Vec<Vec<f64>> {
        let mut tail = Vec::new();
        for ((perc, norm_col), (_, rev_col)) in self.cols.iter().zip(self.cols.iter().rev()) {
            self.row_filters
                .iter_mut()
                .zip(rev_col.iter())
                .enumerate()
                .for_each(|(i, (r, &sn))| {
                    self.harm[i] = r.consume(sn);
                });
            self.perc = *perc;
            let mut norm_col = norm_col.clone();
            self.hpss_one(norm_col.as_mut());
            tail.push(norm_col);
        }
        tail
    }

    /// Computes the next FFT of the STFT
    /// The user in responsible for ensuring there is enough samples in the ring buffer (by a
    /// previous call to self.contains_enought_to_compute())
    fn compute_into_outdata(&mut self) {
        self.sample_ring.read_many_front(&mut self.indata[..]);

        for (r, w) in self.indata.iter_mut().zip(self.window.iter()) {
            *r *= *w;
        }

        self.forward
            .process_with_scratch(&mut self.indata, &mut self.outdata, &mut self.scratch)
            .unwrap();
    }

    /// Computes hpss for one column of median filtered harmonics, and a vector of the last
    /// elements of the corresponding median filtered percussives
    pub fn hpss_one(&self, x: &mut [f64]) {
        let mask = Self::softmask_one(&self.harm, &self.perc, 2);
        for (h, m) in x.iter_mut().zip(mask) {
            *h *= m
        }
    }

    /// Computes softmask for the vector x with vector y as reference
    fn softmask_one(x: &[f64], y: &[f64], power: i32) -> Vec<f64> {
        let z = x.iter().zip(y).map(|(x, y)| {
            let maxi = x.max(*y);
            if maxi < f64::MIN_POSITIVE {
                (1.0, false)
            } else {
                (maxi, true)
            }
        });

        let mut mask = x
            .iter()
            .zip(z.clone())
            .map(|(x, z)| (x / z.0).powi(power))
            .collect::<Vec<f64>>();

        let ref_mask = y.iter().zip(z.clone()).map(|(y, z)| (y / z.0).powi(power));

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
