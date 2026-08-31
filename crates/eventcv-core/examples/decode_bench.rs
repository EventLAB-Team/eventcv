//! Decode-throughput microbenchmark: `cargo run --release --example decode_bench -- <files...>`
use std::time::Instant;
use eventcv_core::io::{read_aedat4, read_raw, open_raw_slice, LoadOptions};

fn bench<T>(label: &str, reps: u32, mut f: impl FnMut() -> T) -> f64 {
    f();
    let mut best = f64::MAX;
    let mut all = Vec::new();
    for _ in 0..reps {
        let t0 = Instant::now();
        let _ = f();
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        best = best.min(ms);
        all.push(ms);
    }
    all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = all[all.len() / 2];
    println!("  {label:<28} median {med:8.2} ms   min {best:8.2} ms");
    med
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let reps: u32 = std::env::var("REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(7);
    let opts = LoadOptions::default();
    for path in &args {
        println!("{path}");
        if path.ends_with(".raw") {
            let n = read_raw(path, &opts).unwrap().len();
            let idx = bench("open_raw_slice (index pass)", reps, || open_raw_slice(path, &opts).unwrap());
            let all = bench("read_raw (index + build)", reps, || read_raw(path, &opts).unwrap());
            println!("  {n} events -> {:.1} Mev/s eager, materialise = {:.2} ms",
                     n as f64 / all * 1e-3, all - idx);
        } else if path.ends_with(".aedat4") {
            let n = read_aedat4(path, &opts).unwrap().len();
            let all = bench("read_aedat4", reps, || read_aedat4(path, &opts).unwrap());
            println!("  {n} events -> {:.1} Mev/s", n as f64 / all * 1e-3);
        }
    }
}
