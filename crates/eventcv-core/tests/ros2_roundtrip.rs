//! The ROS 2 round-trip, against a live Zenoh router.
//!
//! Ignored by default because it needs one running — `ros2 run rmw_zenoh_cpp rmw_zenohd`, or any
//! `zenohd` on the default port. Run it with:
//!
//! ```text
//! cargo test -p eventcv-core --features ros2 --test ros2_roundtrip -- --ignored --nocapture
//! ```
//!
//! What it pins down is the thing the unit tests cannot: that a stream published as
//! `event_camera_msgs/msg/EventPacket` and received back through a real router is the same stream,
//! event for event, and that the sequence numbers account for every packet.

#![cfg(feature = "ros2")]

use std::time::Duration;

use eventcv_core::ros2::{record_topic, Ros2Context};
use eventcv_core::{EventStream, EventStreamBuilder};

fn synthetic() -> EventStream {
    let mut builder = EventStreamBuilder::new(640, 480, 0.001);
    for i in 0..120_000i64 {
        builder.push(
            (i % 640) as u16,
            ((i / 640) % 480) as u16,
            1_000_000 + i * 3,
            i % 3 == 0,
        );
    }
    builder.build()
}

#[test]
#[ignore = "needs a running Zenoh router (rmw_zenohd)"]
fn a_stream_survives_a_round_trip_through_ros2() {
    let dir = std::env::temp_dir().join(format!("eventcv_ros2_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let source = dir.join("source.raw");
    let recorded = dir.join("recorded.raw");
    eventcv_core::io::save_stream(&source, &synthetic(), &Default::default())
        .expect("write source");

    let topic = format!("/eventcv_test/{}", std::process::id());
    let recorder_topic = topic.clone();
    let recorded_path = recorded.clone();

    // The subscriber has to exist before the publisher starts, or the first packets go out to
    // nobody — a property of publish/subscribe, not a flake to paper over with a retry.
    let recorder = std::thread::spawn(move || {
        let context = Ros2Context::new().expect("context");
        let mut subscriber = context
            .event_subscriber("eventcv_test_sub", &recorder_topic)
            .expect("subscriber");
        record_topic(
            &mut subscriber,
            &recorded_path,
            Duration::from_secs(10),
            None,
        )
        .expect("record")
    });

    std::thread::sleep(Duration::from_secs(3));
    let context = Ros2Context::new().expect("context");
    let mut publisher = context
        .event_publisher("eventcv_test_pub", &topic)
        .expect("publisher");
    std::thread::sleep(Duration::from_secs(2));

    // Published window by window rather than through `replay_file`, for two reasons: no clock
    // offset, so the timestamps that come back should be the ones that went out; and a small pace,
    // because an unpaced publisher outruns the default KeepLast(10) queue by two orders of
    // magnitude and this test is about fidelity, not back-pressure.
    let reader = eventcv_core::io::open(&source, Default::default()).expect("open source");
    let (t0, t1) = reader.time_span();
    let step = 33_000i64;
    let (mut packets, mut events) = (0u64, 0usize);
    let mut t = t0;
    while t <= t1 {
        let slice = reader.slice_time(t, t + step).expect("slice");
        if !slice.is_empty() {
            publisher.publish(&slice).expect("publish");
            packets += 1;
            events += slice.len();
            std::thread::sleep(Duration::from_millis(2));
        }
        t += step;
    }

    let report = recorder.join().expect("recorder thread");
    assert_eq!(report.packets, packets, "packet count");
    assert_eq!(report.events, events, "event count");
    assert_eq!(report.dropped, 0, "packets dropped");

    let before = eventcv_core::io::load(&source, Default::default()).expect("read source");
    let after = eventcv_core::io::load(&recorded, Default::default()).expect("read recorded");
    assert_eq!(
        after.len(),
        before.len(),
        "event count after the round trip"
    );
    assert_eq!(after.xs(), before.xs(), "x");
    assert_eq!(after.ys(), before.ys(), "y");
    assert_eq!(after.ts(), before.ts(), "timestamps");
    assert_eq!(after.ps(), before.ps(), "polarity");

    let _ = std::fs::remove_dir_all(&dir);
}
