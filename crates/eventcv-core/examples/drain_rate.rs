// Probe: measure the RAW USB buffer delivery cadence, decoupled from any render/present/vsync.
// A tight loop consumes events as fast as the driver hands them over (non-blocking drain), and we
// record every drain that returned events. If lit-drains/s here is far above the ~22 fps the live
// viewer manages, the viewer's single-threaded coupling (drain+render+present serialized) is the
// bottleneck and a decoupled capture thread would help. If it's also ~22/s, the cadence is a
// driver/USB-transport limit that threading can't fix.
//
// Run (wave something in front of the lens for the whole duration):
//   cargo run --release --features camera --example drain_rate -- 12
use eventcv_core::device::{Capture, Window};
use std::time::{Duration, Instant};

fn main() {
    let secs: f64 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(12.0);

    let mut capture = match Capture::open(None, Window::Duration(30.0)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("open failed: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "opened {} ({}x{}) — consuming for {secs}s, wave now...",
        capture.name(),
        capture.width(),
        capture.height()
    );

    let start = Instant::now();
    let mut iters: u64 = 0; // total drain calls (spin rate)
    let mut lit_times_ms: Vec<f64> = Vec::new(); // elapsed-ms of each drain that had events
    let mut lit_counts: Vec<u64> = Vec::new(); // events per lit drain
    let mut backlogs: Vec<usize> = Vec::new();
    let mut total_events: u64 = 0;
    let mut overflows: u64 = 0;

    while start.elapsed() < Duration::from_secs_f64(secs) {
        iters += 1;
        let mut n: u64 = 0;
        let overflow = capture
            .drain_events(|_x, _y, _t, _p| n += 1)
            .expect("drain failed");
        if overflow {
            overflows += 1;
        }
        if n > 0 {
            lit_times_ms.push(start.elapsed().as_secs_f64() * 1000.0);
            lit_counts.push(n);
            backlogs.push(capture.backlog());
            total_events += n;
        }
    }
    let dur = start.elapsed().as_secs_f64();

    // Inter-arrival gaps between consecutive lit drains.
    let mut gaps: Vec<f64> = lit_times_ms.windows(2).map(|w| w[1] - w[0]).collect();
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut counts_sorted = lit_counts.clone();
    counts_sorted.sort_unstable();
    let pct = |v: &[f64], p: f64| -> f64 {
        if v.is_empty() {
            return f64::NAN;
        }
        v[((p / 100.0 * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)]
    };
    let pct_u = |v: &[u64], p: f64| -> u64 {
        if v.is_empty() {
            return 0;
        }
        v[((p / 100.0 * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)]
    };

    println!("\n=== raw delivery cadence over {dur:.2}s ===");
    println!(
        "  drain calls (spin): {iters}  ({:.0}/s)",
        iters as f64 / dur
    );
    println!(
        "  lit drains: {}  -> {:.1} deliveries/s   <<< compare to viewer's ~22 fps",
        lit_times_ms.len(),
        lit_times_ms.len() as f64 / dur
    );
    println!(
        "  total events: {total_events}  ({:.2} M ev/s)",
        total_events as f64 / dur / 1e6
    );
    println!(
        "  events/delivery: med={} p90={} max={}",
        pct_u(&counts_sorted, 50.0),
        pct_u(&counts_sorted, 90.0),
        counts_sorted.last().copied().unwrap_or(0)
    );
    println!(
        "  gap between deliveries (ms): med={:.2} p90={:.2} max={:.2}",
        pct(&gaps, 50.0),
        pct(&gaps, 90.0),
        gaps.last().copied().unwrap_or(f64::NAN)
    );
    println!(
        "  backlog at delivery (buffers): med={} max={}",
        {
            let mut b = backlogs.clone();
            b.sort_unstable();
            b.get(b.len() / 2).copied().unwrap_or(0)
        },
        backlogs.iter().max().copied().unwrap_or(0)
    );
    println!("  overflows: {overflows}");
}
