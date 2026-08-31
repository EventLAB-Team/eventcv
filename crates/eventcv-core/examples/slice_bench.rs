//! Per-slice cost on the lazy path: open once, then pull fixed windows.
use std::time::Instant;
use eventcv_core::io::{open_raw_slice, LoadOptions, SliceSource};
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let o = LoadOptions::default();
    let t0 = Instant::now();
    let src = open_raw_slice(&path, &o).unwrap();
    println!("open (index pass)      {:8.2} ms  ({} events)", t0.elapsed().as_secs_f64()*1e3, src.n_events());
    let (lo, hi) = src.time_span();
    let dt = 33_000i64;
    let mut total = 0usize;
    let t1 = Instant::now();
    let mut n = 0;
    let mut t = lo;
    while t + dt <= hi { total += src.slice_time(t, t + dt).unwrap().len(); t += dt; n += 1; }
    let ms = t1.elapsed().as_secs_f64()*1e3;
    println!("{n} slices of 33 ms     {ms:8.2} ms total, {:.3} ms/slice ({total} events)", ms / n as f64);
    let t2 = Instant::now();
    let whole = src.slice_index(0, src.n_events()).unwrap();
    println!("slice_index(0, all)    {:8.2} ms  ({} events)", t2.elapsed().as_secs_f64()*1e3, whole.len());
}
