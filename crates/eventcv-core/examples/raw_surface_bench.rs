use eventcv_core::viz::RawSurface;
use std::time::Instant;

fn main() {
    let (width, height) = (1280usize, 720usize);
    let mut surface = RawSurface::new(width, height, 30.0);
    // Worst case for the render hot loop: every pixel lit at a range of ages, so the
    // early-continue (age >= cutoff) almost never triggers and every pixel goes through the
    // full intensity computation + colour write.
    for y in 0..height {
        for x in 0..width {
            let age = ((x + y) % 100) as f64; // spread of ages, all well under the 195ms cutoff
            surface.stamp(x, y, age, (x + y) % 2 == 0);
        }
    }
    // warm up
    for _ in 0..5 {
        std::hint::black_box(surface.render());
    }
    let iters = 100;
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(surface.render());
    }
    let elapsed = start.elapsed();
    println!(
        "render(): {:?}/call over {} calls ({} total)",
        elapsed / iters,
        iters,
        width * height
    );
}
