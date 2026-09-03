//! The parallel packet path must equal the sequential one, event for event.
use eventcv_core::io::{open_aedat4_slice, read_aedat4, LoadOptions, SliceSource};
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let o = LoadOptions::default();
    let src = open_aedat4_slice(&path, &o).unwrap();
    let (lo, hi) = src.time_span();
    let par = read_aedat4(&path, &o).unwrap();          // parallel slice_index
    let seq = src.slice_time(lo, hi + 1).unwrap();      // sequential, packet by packet
    println!("parallel {} events, sequential {} events", par.len(), seq.len());
    assert_eq!(par.len(), seq.len(), "count");
    assert_eq!(par.xs(), seq.xs(), "x");
    assert_eq!(par.ys(), seq.ys(), "y");
    assert_eq!(par.ts(), seq.ts(), "t");
    assert_eq!(par.ps(), seq.ps(), "p");
    let sorted = par.ts().windows(2).all(|w| w[0] <= w[1]);
    println!("timestamps non-decreasing: {sorted}");
    assert!(sorted);
    println!("IDENTICAL");
}
