//! The ROS 2 nodes, as a runnable program: `cargo run --example ros2_node --features ros2 -- <cmd>`.
//!
//! Nothing here links a ROS 2 library. It talks to a `rmw_zenoh_cpp` deployment over Zenoh, which
//! is what makes the whole integration a cargo feature rather than a second distribution.

use std::time::{Duration, Instant};

use eventcv_core::io::EvtVersion;
use eventcv_core::representation::{Representation, VoxelGrid};
use eventcv_core::ros2::{record_topic, replay_file, Ros2Context};

fn arg(name: &str, default: &str) -> String {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == name {
            return args.next().unwrap_or_else(|| default.to_owned());
        }
    }
    default.to_owned()
}

fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // hiroz and zenoh both log through `tracing`; without a subscriber installed the logs go
    // nowhere, which is unhelpful the first time a topic does not match.
    if std::env::var("RUST_LOG").is_ok() {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .try_init();
    }
    let command = std::env::args().nth(1).unwrap_or_else(|| "help".to_owned());
    let topic = arg("--topic", "/event_camera/events");
    let endpoint = arg("--router", "");
    let mode = arg("--mode", "");
    let context = Ros2Context::with_endpoint_and_mode(
        (!endpoint.is_empty()).then_some(endpoint.as_str()),
        (!mode.is_empty()).then_some(mode.as_str()),
    )?;

    match command.as_str() {
        // Replays a recording onto a topic. The publisher a real driver would be.
        "publish" => {
            let path = arg("--file", "/tmp/data/evt3.raw");
            let window: f64 = arg("--window-ms", "33").parse()?;
            let mut publisher = context.event_publisher("eventcv_publisher", &topic)?;
            if arg("--encoding", "evt3") == "evt2" {
                publisher = publisher.with_encoding(EvtVersion::Evt2);
            }
            // Give discovery a moment, or the first packets go out before anyone is listening.
            std::thread::sleep(Duration::from_millis(
                arg("--settle-ms", "1500").parse::<u64>()?,
            ));
            let loops: u32 = arg("--loops", "1").parse()?;
            let started = Instant::now();
            let (mut packets, mut events, mut empty) = (0u64, 0usize, 0u64);
            for _ in 0..loops.max(1) {
                let report = replay_file(&mut publisher, &path, window, flag("--wall-clock"))?;
                packets += report.packets;
                events += report.events;
                empty += report.empty_windows;
            }
            println!(
                "published {packets} packets, {events} events, {empty} empty windows in {:.2}s",
                started.elapsed().as_secs_f64()
            );
        }

        // Records a topic back to a file, in any format the library writes.
        "record" => {
            let out = arg("--out", "/tmp/recorded.raw");
            let idle: u64 = arg("--idle-ms", "5000").parse()?;
            let mut subscriber = context.event_subscriber("eventcv_recorder", &topic)?;
            let report = record_topic(
                &mut subscriber,
                &out,
                Duration::from_millis(idle),
                arg("--max", "0").parse::<u64>().ok().filter(|&n| n > 0),
            )?;
            println!(
                "recorded {} packets, {} events, {} dropped -> {out}",
                report.packets, report.events, report.dropped
            );
        }

        // Subscribes, builds a representation, publishes it as sensor_msgs/Image.
        "represent" => {
            let out_topic = arg("--image-topic", "/event_camera/voxel");
            let bins: usize = arg("--bins", "3").parse()?;
            let window: f64 = arg("--window-ms", "33").parse()?;
            let idle: u64 = arg("--idle-ms", "5000").parse()?;
            let mut subscriber = context.event_subscriber("eventcv_representation", &topic)?;
            let images = context.image_publisher("eventcv_representation_out", &out_topic)?;
            let voxel = VoxelGrid::new(bins, window);
            let mut made = 0u64;
            while let Some(packet) = subscriber.recv_packet(Duration::from_millis(idle))? {
                let stream = packet.decode()?;
                let frame = voxel.generate(&stream)?;
                images.publish(&frame, packet.time_base_ns())?;
                made += 1;
            }
            println!(
                "published {made} images on {out_topic} ({} packets in, {} dropped)",
                subscriber.received(),
                subscriber.dropped()
            );
        }

        // Subscribes, builds the model's input, runs the graph, publishes the output tensor.
        #[cfg(feature = "onnx")]
        "infer" => {
            let model_path = arg("--model", "");
            let out_topic = arg("--tensor-topic", "/event_camera/inference");
            let bins: usize = arg("--bins", "5").parse()?;
            let window: f64 = arg("--window-ms", "33").parse()?;
            let idle: u64 = arg("--idle-ms", "5000").parse()?;
            let model = eventcv_core::model::Model::load(&model_path)?;
            let subscriber = context.event_subscriber("eventcv_inference", &topic)?;
            let tensors = context.tensor_publisher("eventcv_inference_out", &out_topic)?;
            let mut node = eventcv_core::ros2::InferenceNode::with_voxel(
                subscriber, model, tensors, bins, window,
            );
            let (inputs, outputs) = node.ports();
            println!(
                "graph: {:?} -> {:?}",
                inputs.iter().map(|p| &p.shape).collect::<Vec<_>>(),
                outputs.iter().map(|p| &p.shape).collect::<Vec<_>>()
            );
            let n = node.run(Duration::from_millis(idle), None)?;
            println!("ran {n} inferences, published on {out_topic}");
        }

        // One process, one context: a publisher and a subscriber that should see each other.
        // The first thing to check when nothing arrives — it separates a hiroz problem from a
        // discovery problem.
        "loopback" => {
            let mut subscriber = context.event_subscriber("eventcv_loopback_sub", &topic)?;
            let mut publisher = context.event_publisher("eventcv_loopback_pub", &topic)?;
            std::thread::sleep(Duration::from_millis(
                arg("--settle-ms", "3000").parse::<u64>()?,
            ));
            let reader = eventcv_core::io::open(
                arg("--file", "/tmp/data/evt3.raw"),
                eventcv_core::io::LoadOptions::default(),
            )?;
            let (t0, _) = reader.time_span();
            let slice = reader.slice_time(t0, t0 + 33_000)?;
            println!("publishing {} events", slice.len());
            publisher.publish(&slice)?;
            match subscriber.recv(Duration::from_secs(10))? {
                Some(back) => println!(
                    "received {} events, first t {} (sent {})",
                    back.len(),
                    back.ts().first().copied().unwrap_or(-1),
                    slice.ts().first().copied().unwrap_or(-1)
                ),
                None => println!("received nothing in 10s"),
            }
        }

        // Publishes preloaded windows as fast as the topic will take them, so the number is the
        // cost of encode + serialize + put, not of reading the file.
        "bench-pub" => {
            let path = arg("--file", "/tmp/data/evt3.raw");
            let window: f64 = arg("--window-ms", "33").parse()?;
            let reader = eventcv_core::io::open(&path, eventcv_core::io::LoadOptions::default())?;
            let (t0, t1) = reader.time_span();
            let step = (window * 1000.0).round() as i64;
            let mut slices = Vec::new();
            let mut t = t0;
            while t <= t1 {
                let slice = reader.slice_time(t, t + step)?;
                if !slice.is_empty() {
                    slices.push(slice);
                }
                t += step;
            }
            let events: usize = slices.iter().map(|s| s.len()).sum();
            let mut publisher = context.event_publisher("eventcv_bench_pub", &topic)?;
            // Pacing, so a run can be made lossless: the default subscriber queue is KeepLast(10),
            // and an unpaced publisher outruns any consumer by two orders of magnitude.
            let pace = arg("--pace-us", "0").parse::<u64>()?;
            std::thread::sleep(Duration::from_millis(
                arg("--settle-ms", "3000").parse::<u64>()?,
            ));
            // After the settle, not before: the offset has to be taken at the moment publishing
            // starts or every stamp is late by however long discovery took.
            if flag("--wall-clock") {
                let now_ns = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_nanos() as i64;
                publisher = publisher.with_clock_offset(now_ns - t0 * 1_000);
            }
            // Encode alone, so the split between codec and transport is visible.
            let encode_start = Instant::now();
            let mut bytes = 0usize;
            for (i, slice) in slices.iter().enumerate() {
                let packet =
                    eventcv_core::ros2::EventPacket::encode(slice, i as u64, EvtVersion::Evt3)?;
                bytes += packet.events.len();
            }
            let encode = encode_start.elapsed().as_secs_f64();
            let start = Instant::now();
            for slice in &slices {
                publisher.publish(slice)?;
                if pace > 0 {
                    std::thread::sleep(Duration::from_micros(pace));
                }
            }
            let total = start.elapsed().as_secs_f64();
            println!(
                "packets {} events {} payload {:.2} MB\n\
                 encode {:.3}s = {:.2} Mev/s\n\
                 encode+publish {:.3}s = {:.2} Mev/s, {:.0} packets/s, {:.1} MB/s",
                slices.len(),
                events,
                bytes as f64 / 1e6,
                encode,
                events as f64 / encode / 1e6,
                total,
                events as f64 / total / 1e6,
                slices.len() as f64 / total,
                bytes as f64 / total / 1e6,
            );
        }

        // Receives packets and reports decode throughput and the wire latency the header implies.
        "bench-sub" => {
            let idle: u64 = arg("--idle-ms", "15000").parse()?;
            let mut subscriber = context.event_subscriber("eventcv_bench_sub", &topic)?;
            let mut events = 0usize;
            let mut bytes = 0usize;
            let mut decode = Duration::ZERO;
            let mut latencies = Vec::new();
            let mut first: Option<Instant> = None;
            let mut last = Instant::now();
            while let Some(packet) = subscriber.recv_packet(Duration::from_millis(idle))? {
                let arrived = Instant::now();
                first.get_or_insert(arrived);
                let sent_ns = packet.time_base_ns();
                let now_ns = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_nanos() as u64;
                if sent_ns > 0 && now_ns > sent_ns && now_ns - sent_ns < 10_000_000_000 {
                    latencies.push((now_ns - sent_ns) as f64 / 1e6);
                }
                bytes += packet.events.len();
                let at = Instant::now();
                let stream = packet.decode()?;
                decode += at.elapsed();
                events += stream.len();
                last = arrived;
            }
            let span = first.map_or(0.0, |f| (last - f).as_secs_f64()).max(1e-9);
            latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "packets {} events {} payload {:.2} MB\n\
                 decode {:.3}s = {:.2} Mev/s\n\
                 wall {:.3}s = {:.2} Mev/s end to end, {:.1} MB/s, {} dropped",
                subscriber.received(),
                events,
                bytes as f64 / 1e6,
                decode.as_secs_f64(),
                events as f64 / decode.as_secs_f64() / 1e6,
                span,
                events as f64 / span / 1e6,
                bytes as f64 / span / 1e6,
                subscriber.dropped(),
            );
            if !latencies.is_empty() {
                println!(
                    "latency ms: median {:.2}, p95 {:.2}, max {:.2} (n={})",
                    latencies[latencies.len() / 2],
                    latencies[latencies.len() * 95 / 100],
                    latencies[latencies.len() - 1],
                    latencies.len()
                );
            }
        }

        other => {
            eprintln!(
                "usage: ros2_node <publish|record|represent|infer> [--topic T] [--file F] \
                 [--out F] [--window-ms N] [--bins N] [--idle-ms N] [--wall-clock] [--router E]\n\
                 unknown command: {other}"
            );
            std::process::exit(2);
        }
    }
    Ok(())
}
