//! Correctness check: the single-pass eager read must equal the indexed slice, event for event.
use eventcv_core::io::{open_raw_slice, read_raw, LoadOptions};
use eventcv_core::io::SliceSource;
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let o = LoadOptions::default();
    let fast = read_raw(&path, &o).unwrap();
    let src = open_raw_slice(&path, &o).unwrap();
    let slow = src.slice_index(0, src.n_events()).unwrap();
    println!("fast {} events, slow {} events", fast.len(), slow.len());
    assert_eq!(fast.len(), slow.len(), "event count differs");
    assert_eq!(fast.xs(), slow.xs(), "x differs");
    assert_eq!(fast.ys(), slow.ys(), "y differs");
    assert_eq!(fast.ts(), slow.ts(), "t differs");
    assert_eq!(fast.ps(), slow.ps(), "p differs");
    assert_eq!(fast.sensor_size(), slow.sensor_size(), "sensor size differs");
    println!("IDENTICAL");
}
