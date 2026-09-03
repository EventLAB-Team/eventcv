//! Compares two recordings event for event. Used to check the ROS 2 round-trip.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let a = eventcv_core::io::load(args.next().unwrap(), Default::default())?;
    let b = eventcv_core::io::load(args.next().unwrap(), Default::default())?;
    println!("a: {} events, b: {} events", a.len(), b.len());
    println!("sensor: {:?} vs {:?}", a.sensor_size(), b.sensor_size());
    if a.len() != b.len() {
        println!("MISMATCH: lengths differ");
        return Ok(());
    }
    let mut bad = 0usize;
    for i in 0..a.len() {
        if a.xs()[i] != b.xs()[i]
            || a.ys()[i] != b.ys()[i]
            || a.ts()[i] != b.ts()[i]
            || a.ps()[i] != b.ps()[i]
        {
            if bad < 5 {
                println!(
                    "  [{i}] ({},{},{},{}) vs ({},{},{},{})",
                    a.xs()[i],
                    a.ys()[i],
                    a.ts()[i],
                    a.ps()[i],
                    b.xs()[i],
                    b.ys()[i],
                    b.ts()[i],
                    b.ps()[i]
                );
            }
            bad += 1;
        }
    }
    println!(
        "{}",
        if bad == 0 {
            "IDENTICAL".to_owned()
        } else {
            format!("{bad} events differ")
        }
    );
    Ok(())
}
