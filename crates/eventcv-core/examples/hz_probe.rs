//! Reproducer for the one direction of ROS 2 interop that does not work.
//!
//! `rmw_zenoh_cpp` 0.12.0 publishes any message carrying a `uint8[]` field through its new buffer
//! endpoints (its liveliness token gains a `backends:cpu` segment), and a subscriber that does not
//! advertise a compatible backend receives nothing. `hiroz` 0.2.0 does not advertise one.
//!
//! With a router and `ros2 run demo_nodes_cpp talker` running:
//!
//! ```text
//! cargo run --example hz_probe --features ros2 -- string /chatter     # receives
//! ```
//!
//! With a router and `ros2 topic pub -r 2 /probe_image sensor_msgs/msg/Image '{...}'`:
//!
//! ```text
//! cargo run --example hz_probe --features ros2 -- image /probe_image  # times out
//! ```
//!
//! The difference between the two is the `uint8[] data` field. Everything published *from* this
//! library is received by ROS 2 in both cases.
use hiroz::Builder;
use std::time::Duration;

fn main() -> hiroz::Result<()> {
    if std::env::var("RUST_LOG").is_ok() {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .try_init();
    }
    let kind = std::env::args().nth(1).unwrap_or_else(|| "string".into());
    let topic = std::env::args().nth(2).unwrap_or_else(|| "/chatter".into());
    let ctx = hiroz::context::ZContextBuilder::default()
        .connect_to_local_zenohd()
        .with_mode("client")
        .build()?;
    let node = ctx.create_node("hz_probe").build()?;
    match kind.as_str() {
        "image" => {
            let sub = node
                .create_sub::<hiroz_msgs::sensor_msgs::Image>(&topic)
                .build()?;
            for _ in 0..4 {
                match sub.recv_timeout(Duration::from_secs(8)) {
                    Ok(m) => println!("got image {}x{} {} bytes", m.width, m.height, m.step),
                    Err(e) => println!("timeout/err: {e}"),
                }
            }
        }
        _ => {
            let sub = node
                .create_sub::<hiroz_msgs::example_interfaces::String>(&topic)
                .build()?;
            for _ in 0..4 {
                match sub.recv_timeout(Duration::from_secs(8)) {
                    Ok(m) => println!("got: {}", m.data),
                    Err(e) => println!("timeout/err: {e}"),
                }
            }
        }
    }
    Ok(())
}
