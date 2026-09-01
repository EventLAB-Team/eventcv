//! ROS 2 publish and subscribe, with no ROS 2 linked in.
//!
//! Every other way of speaking ROS 2 from Rust — `rclrs`, `r2r` — links the ROS C libraries,
//! which means a ROS installation at build time, a second build system, and a package that
//! cannot ship in a wheel. This module goes over [`hiroz`], a Zenoh-native ROS 2 stack in pure
//! Rust, so the integration is a cargo feature rather than a separate distribution: `pip install
//! eventcv` and you have a node.
//!
//! The trade is stated plainly: Zenoh on the wire means interoperating with ROS 2 deployments
//! running `rmw_zenoh_cpp`. A deployment on the default DDS needs `zenoh-bridge-ros2dds` in
//! between. That is a real constraint, not a detail.
//!
//! # What was checked, and one thing that does not work
//!
//! Against ROS 2 Rolling with `rmw_zenoh_cpp` 0.12.0: a topic published from here appears in
//! `ros2 topic list` as `event_camera_msgs/msg/EventPacket`, `ros2 topic echo` deserialises it
//! against the real `.msg`, and the RIHS01 type hash derived from the struct below matches the
//! one ROS computes from the `.msg` — byte for byte, which is why no shared IDL is needed. A
//! recording replayed to a topic and recorded back comes out event-identical.
//!
//! The gap is in one direction. `rmw_zenoh_cpp` 0.12.0 publishes any message carrying a `uint8[]`
//! field through its new *buffer endpoints* path — its liveliness token gains a `backends:cpu`
//! segment — and a subscriber that does not advertise a compatible backend receives nothing at
//! all. `hiroz` 0.2.0 does not advertise one, so a C++ node publishing `EventPacket` (or
//! `sensor_msgs/msg/Image`, or anything else with a byte array) is not received here, while a
//! `String` is, and everything published from here is received there. Until `hiroz` speaks that
//! part of the protocol, subscribing to a C++ event-camera driver on that `rmw_zenoh_cpp` version
//! needs `zenoh-bridge-ros2dds` in front of it; publishing needs nothing.
//!
//! # Encoding
//!
//! [`EventPacket`] is `event_camera_msgs/msg/EventPacket`, the message the ROS 2 event-camera
//! stack uses. Its `encoding` field takes `"evt3"`, which is a format this library already reads
//! and writes, so a packet carries the sensor's own words rather than one message field per
//! event: [`RawEventSink`](crate::io::RawEventSink) encodes and
//! [`decode_words`](crate::io::decode_words) decodes, and no separate wire format exists to keep
//! in step.

use std::sync::Arc;
use std::time::Duration;

use hiroz::{context::ZContextBuilder, Builder, MessageTypeInfo};
use hiroz_msgs::builtin_interfaces::Time;
use hiroz_msgs::sensor_msgs::Image;
use hiroz_msgs::std_msgs::Header;
use hiroz_msgs::std_msgs::{Float32MultiArray, MultiArrayDimension, MultiArrayLayout};
use serde::{Deserialize, Serialize};

use crate::io::{decode_words, EvtVersion, IoError, RawEventSink};
use crate::representation::{EventFrame, EventFrameData};
use crate::{EventStream, EventStreamBuilder};

/// EVT2 and EVT3 words carry microseconds, and so does everything this module encodes. A stream
/// on another tick is rescaled on the way in rather than silently mis-timed.
const MICROSECOND_SCALE_MS: f64 = 0.001;

/// ROS time is nanoseconds; `EventPacket.time_base` is a ROS time.
const NS_PER_US: i64 = 1_000;

/// Rescales `stream` to microsecond ticks, which is what the EVT encoders write.
fn in_microseconds(stream: &EventStream) -> std::borrow::Cow<'_, EventStream> {
    let scale = stream.timestamp_scale_ms();
    if (scale - MICROSECOND_SCALE_MS).abs() < f64::EPSILON {
        return std::borrow::Cow::Borrowed(stream);
    }
    let factor = scale / MICROSECOND_SCALE_MS;
    let (width, height) = stream.sensor_size();
    let mut builder =
        EventStreamBuilder::with_capacity(width, height, MICROSECOND_SCALE_MS, stream.len());
    let (xs, ys, ts, ps) = (stream.xs(), stream.ys(), stream.ts(), stream.ps());
    for i in 0..stream.len() {
        builder.push(xs[i], ys[i], (ts[i] as f64 * factor).round() as i64, ps[i]);
    }
    std::borrow::Cow::Owned(builder.build())
}

/// What went wrong talking to ROS 2.
#[derive(Debug)]
pub enum Ros2Error {
    /// The Zenoh session, node or endpoint could not be created.
    Session(String),
    /// A message could not be published or received.
    Transport(String),
    /// The payload did not decode: a truncated packet, or an `encoding` this build cannot read.
    Payload(String),
    /// Encoding or decoding events failed.
    Io(IoError),
}

impl std::fmt::Display for Ros2Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Session(m) => write!(f, "ROS 2 session: {m}"),
            Self::Transport(m) => write!(f, "ROS 2 transport: {m}"),
            Self::Payload(m) => write!(f, "ROS 2 payload: {m}"),
            Self::Io(e) => write!(f, "{e}"),
        }
    }
}
impl std::error::Error for Ros2Error {}
impl From<IoError> for Ros2Error {
    fn from(e: IoError) -> Self {
        Self::Io(e)
    }
}

type Result<T> = std::result::Result<T, Ros2Error>;

// ---------------------------------------------------------------------------- messages ----

/// `event_camera_msgs/msg/EventPacket`, field for field and in order.
///
/// CDR is positional, so the order below is the wire format and must match
/// `msg_ros2/EventPacket.msg` in `ros-event-camera/event_camera_msgs` exactly. It is checked
/// against a real ROS 2 node rather than against itself — see the interop test.
#[derive(Debug, Clone, Serialize, Deserialize, Default, MessageTypeInfo)]
#[ros_msg(type_name = "event_camera_msgs/msg/EventPacket")]
pub struct EventPacket {
    pub header: Header,
    pub height: u32,
    pub width: u32,
    /// Sequence number, for spotting a dropped packet.
    pub seq: u64,
    /// Event time is `time_base` plus the decoded per-event time.
    pub time_base: u64,
    /// `"evt3"`, `"evt2"`, `"libcaer"`, `"mono"`, ... — this build reads the first two.
    pub encoding: String,
    pub is_bigendian: bool,
    /// The encoded words. Not one field per event.
    pub events: Vec<u8>,
}

impl hiroz::msg::ZMessage for EventPacket {
    type Serdes = hiroz::msg::SerdeCdrSerdes<EventPacket>;
}

impl EventPacket {
    /// Encodes a window of events into a packet.
    ///
    /// `time_base` is subtracted from every timestamp before encoding, which is what keeps the
    /// per-event field narrow; the receiver adds it back.
    pub fn encode(stream: &EventStream, seq: u64, encoding: EvtVersion) -> Result<Self> {
        Self::encode_with(stream, seq, encoding, 0, "event_camera")
    }

    /// As [`encode`](Self::encode), with `clock_offset_ns` added to the sensor clock and a
    /// `frame_id` on the header.
    ///
    /// The offset is what puts a recording's sensor time on the wall clock: a replay node sets it
    /// once at start, a live camera leaves it at zero.
    pub fn encode_with(
        stream: &EventStream,
        seq: u64,
        encoding: EvtVersion,
        clock_offset_ns: i64,
        frame_id: &str,
    ) -> Result<Self> {
        let stream = in_microseconds(stream);
        let (width, height) = stream.sensor_size();
        // The message defines `event ros time = time_base + decoded event_time`, and a ROS time is
        // nanoseconds; the words carry microseconds. So the base goes out in nanoseconds and the
        // decoded remainder is scaled on the way back, which is also what `event_camera_codecs`
        // does — a consumer that never heard of this library gets the right times.
        let first_us = stream.ts().first().copied().unwrap_or(0);
        let time_base_ns = (first_us * NS_PER_US)
            .saturating_add(clock_offset_ns)
            .max(0) as u64;
        let shifted = stream.time_shift(-first_us);
        let mut sink = RawEventSink::to_writer(Vec::new(), encoding, false);
        {
            use crate::io::EventSink;
            sink.append(&shifted)?;
        }
        Ok(Self {
            header: Header {
                stamp: stamp_from_nanos(time_base_ns),
                frame_id: frame_id.to_owned(),
            },
            height: height as u32,
            width: width as u32,
            seq,
            time_base: time_base_ns,
            encoding: match encoding {
                EvtVersion::Evt2 => "evt2".to_owned(),
                EvtVersion::Evt3 => "evt3".to_owned(),
            },
            is_bigendian: cfg!(target_endian = "big"),
            events: sink.into_inner()?,
        })
    }

    /// Decodes a packet back into a stream, with `time_base` added back on.
    pub fn decode(&self) -> Result<EventStream> {
        let version = match self.encoding.as_str() {
            "evt3" => EvtVersion::Evt3,
            "evt2" => EvtVersion::Evt2,
            other => {
                return Err(Ros2Error::Payload(format!(
                    "this build decodes the \"evt2\" and \"evt3\" encodings; the packet says \
                     \"{other}\". The ROS 2 event-camera stack also defines libcaer, \
                     libcaer_cmp, mono and trigger."
                )))
            }
        };
        let stream = decode_words(
            &self.events,
            version,
            self.width as usize,
            self.height as usize,
        )?;
        Ok(stream.time_shift(self.time_base as i64 / NS_PER_US))
    }

    /// The packet's ROS time, in nanoseconds.
    pub fn time_base_ns(&self) -> u64 {
        self.time_base
    }
}

/// Splits nanoseconds into the `sec`/`nanosec` pair a ROS header carries.
fn stamp_from_nanos(ns: u64) -> Time {
    Time {
        sec: (ns / 1_000_000_000) as i32,
        nanosec: (ns % 1_000_000_000) as u32,
    }
}

// ----------------------------------------------------------------------------- context ----

/// A ROS 2 context: one Zenoh session, one tokio runtime, shared by every node built from it.
///
/// The runtime lives here rather than in the caller because everything else in this library is
/// synchronous, and a Python caller has no runtime to lend.
pub struct Ros2Context {
    ctx: hiroz::context::ZContext,
    runtime: Arc<tokio::runtime::Runtime>,
}

impl Ros2Context {
    /// Connects to a local `rmw_zenohd` router, which is what a ROS 2 system running
    /// `rmw_zenoh_cpp` provides.
    pub fn new() -> Result<Self> {
        Self::with_endpoint(None)
    }

    /// The Zenoh session mode this integration uses unless told otherwise.
    ///
    /// `rmw_zenoh_cpp` runs every ROS 2 node as a Zenoh *client* of the local router, and a node
    /// that joins the same deployment as a *peer* is not reliably routed to: it publishes into the
    /// graph fine, but a subscription declared from peer mode did not receive from router-side
    /// publishers in testing. Matching what the C++ side does is both the working configuration
    /// and the least surprising one.
    pub const DEFAULT_MODE: &'static str = "client";

    /// As [`new`](Self::new), against a named router endpoint (`tcp/host:7447`).
    pub fn with_endpoint(endpoint: Option<&str>) -> Result<Self> {
        Self::with_endpoint_and_mode(endpoint, None)
    }

    /// As [`with_endpoint`](Self::with_endpoint), in an explicit Zenoh session mode
    /// (`"client"`, `"peer"`, `"router"`).
    pub fn with_endpoint_and_mode(endpoint: Option<&str>, mode: Option<&str>) -> Result<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| Ros2Error::Session(e.to_string()))?;
        let ctx = {
            let _guard = runtime.enter();
            let builder = match endpoint {
                Some(e) => ZContextBuilder::default()
                    .with_router_endpoint(e)
                    .map_err(|e| Ros2Error::Session(e.to_string()))?,
                None => ZContextBuilder::default().connect_to_local_zenohd(),
            };
            let builder = builder.with_mode(mode.unwrap_or(Self::DEFAULT_MODE));
            builder
                .build()
                .map_err(|e| Ros2Error::Session(e.to_string()))?
        };
        Ok(Self {
            ctx,
            runtime: Arc::new(runtime),
        })
    }

    /// A publisher of [`EventPacket`] on `topic`, from a node called `node_name`.
    pub fn event_publisher(&self, node_name: &str, topic: &str) -> Result<EventPublisher> {
        let _guard = self.runtime.enter();
        let node = self
            .ctx
            .create_node(node_name)
            .build()
            .map_err(|e| Ros2Error::Session(e.to_string()))?;
        let publisher = node
            .create_pub::<EventPacket>(topic)
            .build()
            .map_err(|e| Ros2Error::Session(e.to_string()))?;
        Ok(EventPublisher {
            publisher,
            runtime: Arc::clone(&self.runtime),
            _node: node,
            seq: 0,
            encoding: EvtVersion::Evt3,
            clock_offset_ns: 0,
            frame_id: "event_camera".to_owned(),
            bytes: 0,
        })
    }

    /// A subscriber to [`EventPacket`] on `topic`.
    pub fn event_subscriber(&self, node_name: &str, topic: &str) -> Result<EventSubscriber> {
        let _guard = self.runtime.enter();
        let node = self
            .ctx
            .create_node(node_name)
            .build()
            .map_err(|e| Ros2Error::Session(e.to_string()))?;
        let subscriber = node
            .create_sub::<EventPacket>(topic)
            .build()
            .map_err(|e| Ros2Error::Session(e.to_string()))?;
        Ok(EventSubscriber {
            subscriber,
            _node: node,
            received: 0,
            last_seq: None,
            last_time_base_ns: 0,
            dropped: 0,
        })
    }

    /// A publisher of `sensor_msgs/msg/Image` on `topic`, for sending a representation on to the
    /// rest of a ROS 2 system — rviz, a recorder, a classifier that wants pixels.
    pub fn image_publisher(&self, node_name: &str, topic: &str) -> Result<ImagePublisher> {
        let _guard = self.runtime.enter();
        let node = self
            .ctx
            .create_node(node_name)
            .build()
            .map_err(|e| Ros2Error::Session(e.to_string()))?;
        let publisher = node
            .create_pub::<Image>(topic)
            .build()
            .map_err(|e| Ros2Error::Session(e.to_string()))?;
        Ok(ImagePublisher {
            publisher,
            _node: node,
            frame_id: "event_camera".to_owned(),
        })
    }

    /// A publisher of `std_msgs/msg/Float32MultiArray` on `topic` — where a model's output goes.
    pub fn tensor_publisher(&self, node_name: &str, topic: &str) -> Result<TensorPublisher> {
        let _guard = self.runtime.enter();
        let node = self
            .ctx
            .create_node(node_name)
            .build()
            .map_err(|e| Ros2Error::Session(e.to_string()))?;
        let publisher = node
            .create_pub::<Float32MultiArray>(topic)
            .build()
            .map_err(|e| Ros2Error::Session(e.to_string()))?;
        Ok(TensorPublisher {
            publisher,
            _node: node,
            labels: Vec::new(),
        })
    }
}

/// Publishes windows of events as `event_camera_msgs/msg/EventPacket`.
pub struct EventPublisher {
    publisher: hiroz::pubsub::ZPub<EventPacket, hiroz::msg::SerdeCdrSerdes<EventPacket>>,
    runtime: Arc<tokio::runtime::Runtime>,
    _node: hiroz::node::ZNode,
    seq: u64,
    encoding: EvtVersion,
    clock_offset_ns: i64,
    frame_id: String,
    bytes: usize,
}

impl EventPublisher {
    /// Encode as EVT2 instead of the EVT3 default.
    pub fn with_encoding(mut self, encoding: EvtVersion) -> Self {
        self.encoding = encoding;
        self
    }

    /// Adds `ns` to every packet's `time_base`, which is how a replay of a recording that starts
    /// at sensor time zero is stamped with a plausible wall clock.
    pub fn with_clock_offset(mut self, ns: i64) -> Self {
        self.clock_offset_ns = ns;
        self
    }

    /// The `frame_id` on every header. Defaults to `event_camera`.
    pub fn with_frame_id(mut self, frame_id: &str) -> Self {
        self.frame_id = frame_id.to_owned();
        self
    }

    /// Encodes and publishes one window. The sequence number advances per call, so a subscriber
    /// can see a gap.
    pub fn publish(&mut self, stream: &EventStream) -> Result<u64> {
        self.seq += 1;
        let packet = EventPacket::encode_with(
            stream,
            self.seq,
            self.encoding,
            self.clock_offset_ns,
            &self.frame_id,
        )?;
        self.bytes += packet.events.len();
        self.runtime
            .block_on(self.publisher.async_publish(&packet))
            .map_err(|e| Ros2Error::Transport(e.to_string()))?;
        Ok(self.seq)
    }

    /// How many packets have gone out.
    pub fn published(&self) -> u64 {
        self.seq
    }

    /// Bytes of encoded events put on the wire, not counting the rest of each message.
    ///
    /// Worth having outside a benchmark: it is what sizing a radio link comes down to, and the
    /// answer depends on the recording — EVT3's vector words pay off on dense rows and cost on
    /// sparse ones, so a number from someone else's data is not yours.
    pub fn bytes_published(&self) -> usize {
        self.bytes
    }
}

// -------------------------------------------------------------------------- subscriber ----

/// Receives `event_camera_msgs/msg/EventPacket` and decodes it back into a stream.
///
/// Nothing here is event-camera-specific beyond the decode: what comes out is an
/// [`EventStream`], so every representation, filter and model in this library applies to a live
/// topic exactly as it does to a file. That is the whole point of the integration.
/// Named through the builder's associated output rather than spelled out, because the queue's
/// element type is a Zenoh type and this crate does not depend on Zenoh directly — hiroz does.
type EventSub = <hiroz::pubsub::ZSubBuilder<EventPacket, hiroz::msg::SerdeCdrSerdes<EventPacket>> as Builder>::Output;

pub struct EventSubscriber {
    subscriber: EventSub,
    _node: hiroz::node::ZNode,
    received: u64,
    last_seq: Option<u64>,
    last_time_base_ns: u64,
    dropped: u64,
}

impl EventSubscriber {
    /// The next packet, or `Ok(None)` if none arrived inside `timeout`.
    ///
    /// A timeout is not an error: a subscriber on a quiet topic is doing its job. Only a
    /// malformed payload is.
    pub fn recv_packet(&mut self, timeout: Duration) -> Result<Option<EventPacket>> {
        let packet = match self.subscriber.recv_timeout(timeout) {
            Ok(packet) => packet,
            // hiroz reports a timeout as an error; everything else about the queue is fatal, but
            // there is no typed variant to match on, so the message is all there is to go by.
            Err(e) => {
                let text = e.to_string();
                if is_timeout(&text) {
                    return Ok(None);
                }
                return Err(Ros2Error::Transport(text));
            }
        };
        self.received += 1;
        if let Some(previous) = self.last_seq {
            self.dropped += packet.seq.saturating_sub(previous + 1);
        }
        self.last_seq = Some(packet.seq);
        self.last_time_base_ns = packet.time_base;
        Ok(Some(packet))
    }

    /// The next packet, already decoded.
    pub fn recv(&mut self, timeout: Duration) -> Result<Option<EventStream>> {
        match self.recv_packet(timeout)? {
            Some(packet) => Ok(Some(packet.decode()?)),
            None => Ok(None),
        }
    }

    /// How many packets have arrived.
    pub fn received(&self) -> u64 {
        self.received
    }

    /// The ROS time on the most recent packet, in nanoseconds; zero before the first one.
    ///
    /// Kept because it is the only thing a decoded [`EventStream`] does not carry: the stream has
    /// the sensor's clock, and this is where that clock sat in ROS time. Comparing it with the
    /// clock now is how a consumer knows how old the events in its hands are.
    pub fn last_time_base_ns(&self) -> u64 {
        self.last_time_base_ns
    }

    /// How many packets the sequence numbers say went missing on the way. Reported rather than
    /// hidden: a representation built from a topic that quietly dropped a third of its packets is
    /// not the representation the caller thinks it is.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }
}

fn is_timeout(message: &str) -> bool {
    let text = message.to_ascii_lowercase();
    text.contains("timeout") || text.contains("timed out")
}

// --------------------------------------------------------------------- image publisher ----

/// Publishes an [`EventFrame`] as `sensor_msgs/msg/Image`.
///
/// The channel count picks the encoding — `mono8`, `mono16`, `32FC3` for a three-bin voxel grid —
/// so a representation this library already computes shows up in rviz without a conversion node
/// in between.
pub struct ImagePublisher {
    publisher: hiroz::pubsub::ZPub<Image, <Image as hiroz::msg::ZMessage>::Serdes>,
    _node: hiroz::node::ZNode,
    frame_id: String,
}

impl ImagePublisher {
    /// The `frame_id` on every header. Defaults to `event_camera`.
    pub fn with_frame_id(mut self, frame_id: &str) -> Self {
        self.frame_id = frame_id.to_owned();
        self
    }

    /// Publishes `frame`, stamped `stamp_ns`.
    pub fn publish(&self, frame: &EventFrame, stamp_ns: u64) -> Result<()> {
        let image = self.to_image(frame, stamp_ns)?;
        self.publisher
            .publish(&image)
            .map_err(|e| Ros2Error::Transport(e.to_string()))
    }

    /// The `sensor_msgs/msg/Image` for `frame`, without publishing it.
    pub fn to_image(&self, frame: &EventFrame, stamp_ns: u64) -> Result<Image> {
        image_message(frame, stamp_ns, &self.frame_id)
    }
}

/// The `sensor_msgs/msg/Image` for `frame`. Free-standing so it can be checked without a Zenoh
/// session — the shaping is the part that can be wrong, and it should not need a router to test.
pub fn image_message(frame: &EventFrame, stamp_ns: u64, frame_id: &str) -> Result<Image> {
    let (channels, height, width) = frame.shape();
    if channels == 0 || width == 0 || height == 0 {
        return Err(Ros2Error::Payload("frame has a zero dimension".to_owned()));
    }
    let (encoding, sample_bytes, bytes) = encode_frame(frame.data(), channels, width, height)?;
    Ok(Image {
        header: Header {
            stamp: stamp_from_nanos(stamp_ns),
            frame_id: frame_id.to_owned(),
        },
        height: height as u32,
        width: width as u32,
        encoding,
        is_bigendian: u8::from(cfg!(target_endian = "big")),
        step: (width * channels * sample_bytes) as u32,
        data: bytes.into(),
    })
}

/// Interleaves a channel-major `[C, H, W]` buffer into the pixel-major order a ROS image wants.
///
/// Representations here are planar because that is what a model wants; `sensor_msgs/Image` is
/// interleaved because that is what a display wants. One of the two has to give, and it is not
/// the one with a published spec.
fn interleave<T: Copy, const N: usize>(
    values: &[T],
    channels: usize,
    pixels: usize,
    to_bytes: impl Fn(T) -> [u8; N],
) -> Vec<u8> {
    if channels == 1 {
        return values.iter().flat_map(|v| to_bytes(*v)).collect();
    }
    let mut out = Vec::with_capacity(pixels * channels * N);
    for pixel in 0..pixels {
        for channel in 0..channels {
            out.extend_from_slice(&to_bytes(values[channel * pixels + pixel]));
        }
    }
    out
}

/// Maps a frame's samples onto a ROS image encoding, its sample size, and interleaved bytes.
///
/// `u64` has no ROS encoding — an unnormalised event count is the common case — so it is narrowed
/// to 16 bits with saturation rather than refused. A pixel that genuinely saw more than 65535
/// events in one window saturates, which is the honest outcome for a display format.
fn encode_frame(
    data: &EventFrameData,
    channels: usize,
    width: usize,
    height: usize,
) -> Result<(String, usize, Vec<u8>)> {
    if channels > 4 {
        return Err(Ros2Error::Payload(format!(
            "sensor_msgs/Image tops out at 4 channels; this frame has {channels}. Publish one \
             topic per channel, or reduce the representation first."
        )));
    }
    let pixels = width * height;
    let expect = |len: usize| -> Result<()> {
        if len == pixels * channels {
            Ok(())
        } else {
            Err(Ros2Error::Payload(format!(
                "frame says {channels}x{height}x{width} but carries {len} samples"
            )))
        }
    };
    Ok(match data {
        EventFrameData::U8(values) => {
            expect(values.len())?;
            let name = if channels == 1 {
                "mono8".to_owned()
            } else {
                format!("8UC{channels}")
            };
            (name, 1, interleave(values, channels, pixels, |v| [v]))
        }
        EventFrameData::U16(values) => {
            expect(values.len())?;
            let name = if channels == 1 {
                "mono16".to_owned()
            } else {
                format!("16UC{channels}")
            };
            (
                name,
                2,
                interleave(values, channels, pixels, u16::to_ne_bytes),
            )
        }
        EventFrameData::U64(values) => {
            expect(values.len())?;
            let name = if channels == 1 {
                "mono16".to_owned()
            } else {
                format!("16UC{channels}")
            };
            (
                name,
                2,
                interleave(values, channels, pixels, |v: u64| {
                    (v.min(u16::MAX as u64) as u16).to_ne_bytes()
                }),
            )
        }
        EventFrameData::F32(values) => {
            expect(values.len())?;
            (
                format!("32FC{channels}"),
                4,
                interleave(values, channels, pixels, f32::to_ne_bytes),
            )
        }
    })
}

// -------------------------------------------------------------------------- inference ----

/// Publishes an arbitrary float tensor as `std_msgs/msg/Float32MultiArray`.
///
/// A model's output is a tensor of some shape the message cannot name, so the shape travels in the
/// layout's dimension labels and the consumer reshapes. This is what `Float32MultiArray` is for,
/// and it beats inventing a message that only this library speaks.
pub struct TensorPublisher {
    publisher:
        hiroz::pubsub::ZPub<Float32MultiArray, <Float32MultiArray as hiroz::msg::ZMessage>::Serdes>,
    _node: hiroz::node::ZNode,
    labels: Vec<String>,
}

impl TensorPublisher {
    /// Names the axes, which is how a consumer knows what it is looking at. Defaults to
    /// `dim0`, `dim1`, …
    pub fn with_labels<I, T>(mut self, labels: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.labels = labels.into_iter().map(Into::into).collect();
        self
    }

    /// Publishes `values` laid out as `shape`, row-major.
    pub fn publish(&self, shape: &[usize], values: Vec<f32>) -> Result<()> {
        self.publisher
            .publish(&self.to_message(shape, values)?)
            .map_err(|e| Ros2Error::Transport(e.to_string()))
    }

    /// The message for `values`, without publishing it.
    pub fn to_message(&self, shape: &[usize], values: Vec<f32>) -> Result<Float32MultiArray> {
        tensor_message(shape, values, &self.labels)
    }
}

/// The `std_msgs/msg/Float32MultiArray` for `values` laid out as `shape`, row-major. Free-standing
/// for the same reason [`image_message`] is.
pub fn tensor_message(
    shape: &[usize],
    values: Vec<f32>,
    labels: &[String],
) -> Result<Float32MultiArray> {
    {
        let expected: usize = shape.iter().product();
        if expected != values.len() {
            return Err(Ros2Error::Payload(format!(
                "shape {shape:?} wants {expected} values, got {}",
                values.len()
            )));
        }
        // ROS defines `stride` as the number of elements spanned by one step along that axis,
        // counting the axis itself: the first is the whole tensor, the last is its own size.
        let mut dim = Vec::with_capacity(shape.len());
        for (axis, &size) in shape.iter().enumerate() {
            let stride: usize = shape[axis..].iter().product();
            dim.push(MultiArrayDimension {
                label: labels
                    .get(axis)
                    .cloned()
                    .unwrap_or_else(|| format!("dim{axis}")),
                size: size as u32,
                stride: stride as u32,
            });
        }
        Ok(Float32MultiArray {
            layout: MultiArrayLayout {
                dim,
                data_offset: 0,
            },
            data: values,
        })
    }
}

/// A representation as a model input: the shape, and the samples as `f32`.
///
/// Every representation this library computes is a dense `[C, H, W]` block in one of four sample
/// types; a graph wants `f32` and usually a leading batch axis. Doing the widening here means the
/// inference node does not care which representation it was handed.
pub fn frame_as_input(frame: &EventFrame) -> (Vec<usize>, Vec<f32>) {
    let (channels, height, width) = frame.shape();
    let values = match frame.data() {
        EventFrameData::U8(v) => v.iter().map(|&x| x as f32).collect(),
        EventFrameData::U16(v) => v.iter().map(|&x| x as f32).collect(),
        EventFrameData::U64(v) => v.iter().map(|&x| x as f32).collect(),
        EventFrameData::F32(v) => v.clone(),
    };
    (vec![1, channels, height, width], values)
}

/// Subscribes to a topic, builds a representation, runs a graph, publishes the result.
///
/// The preprocessing step is a closure rather than a representation name, because the core has no
/// name-to-representation table and should not grow one for this: the caller already chose a
/// representation, and the bindings already parse names. [`frame_as_input`] is the bridge.
#[cfg(feature = "onnx")]
pub struct InferenceNode {
    subscriber: EventSubscriber,
    model: crate::model::Model,
    publisher: TensorPublisher,
    #[allow(clippy::type_complexity)]
    preprocess: Box<dyn FnMut(&EventStream) -> Result<(Vec<usize>, Vec<f32>)> + Send>,
    inferred: u64,
}

#[cfg(feature = "onnx")]
impl InferenceNode {
    pub fn new(
        subscriber: EventSubscriber,
        model: crate::model::Model,
        publisher: TensorPublisher,
        preprocess: impl FnMut(&EventStream) -> Result<(Vec<usize>, Vec<f32>)> + Send + 'static,
    ) -> Self {
        Self {
            subscriber,
            model,
            publisher,
            preprocess: Box::new(preprocess),
            inferred: 0,
        }
    }

    /// The common case: a voxel grid in, the graph's first output out.
    pub fn with_voxel(
        subscriber: EventSubscriber,
        model: crate::model::Model,
        publisher: TensorPublisher,
        bins: usize,
        window_ms: f64,
    ) -> Self {
        use crate::representation::{Representation, VoxelGrid};
        let voxel = VoxelGrid::new(bins, window_ms);
        Self::new(subscriber, model, publisher, move |stream| {
            let frame = voxel
                .generate(stream)
                .map_err(|e| Ros2Error::Payload(e.to_string()))?;
            Ok(frame_as_input(&frame))
        })
    }

    /// One packet in, one inference out. `Ok(None)` when nothing arrived inside `timeout`.
    ///
    /// Returns the published output's shape, so a caller can log what the graph actually produced
    /// rather than what it was supposed to.
    pub fn step(&mut self, timeout: Duration) -> Result<Option<Vec<usize>>> {
        let Some(stream) = self.subscriber.recv(timeout)? else {
            return Ok(None);
        };
        let (shape, values) = (self.preprocess)(&stream)?;
        let input = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), values)
            .map_err(|e| Ros2Error::Payload(e.to_string()))?;
        let outputs = self
            .model
            .run(input)
            .map_err(|e| Ros2Error::Payload(e.to_string()))?;
        let first = outputs
            .into_iter()
            .next()
            .ok_or_else(|| Ros2Error::Payload("the graph produced no output".to_owned()))?;
        let shape = first.shape().to_vec();
        let (values, _) = first.into_raw_vec_and_offset();
        self.publisher.publish(&shape, values)?;
        self.inferred += 1;
        Ok(Some(shape))
    }

    /// Runs until the topic goes quiet for `idle`, or `max_packets` have been through.
    pub fn run(&mut self, idle: Duration, max_packets: Option<u64>) -> Result<u64> {
        while !max_packets.is_some_and(|cap| self.inferred >= cap) {
            if self.step(idle)?.is_none() {
                break;
            }
        }
        Ok(self.inferred)
    }

    /// How many packets have been through the graph.
    pub fn inferred(&self) -> u64 {
        self.inferred
    }

    /// What the loaded graph expects and produces — for a caller that wants to check the model
    /// matches the representation before starting.
    pub fn ports(&self) -> (&[crate::model::Port], &[crate::model::Port]) {
        (self.model.inputs(), self.model.outputs())
    }
}

// ------------------------------------------------------------------- replay and record ----

/// What a replay did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayReport {
    /// Packets published — one per non-empty window.
    pub packets: u64,
    /// Events published.
    pub events: usize,
    /// Windows that held no events and so were not published.
    pub empty_windows: u64,
}

/// Publishes a recording onto a topic, one `window_ms` slice at a time.
///
/// This is the same windowing the rest of the library uses, so a topic replayed from a file and a
/// file read directly give the same slices — which is what makes the round-trip test below worth
/// anything.
///
/// `wall_clock` stamps the packets against the current time instead of the sensor's own clock, and
/// paces the publishing to match: what you want when something downstream is timing itself against
/// the header. Without it the replay runs as fast as the topic will take it, which is what you
/// want when something downstream is being tested.
pub fn replay_file(
    publisher: &mut EventPublisher,
    path: impl AsRef<std::path::Path>,
    window_ms: f64,
    wall_clock: bool,
) -> Result<ReplayReport> {
    // `<= 0.0` rather than `!(> 0.0)` so a NaN window is caught by the `is_finite` half rather
    // than by the negation, which clippy rightly says is hard to read.
    if !window_ms.is_finite() || window_ms <= 0.0 {
        return Err(Ros2Error::Payload(
            "the replay window must be a positive number of milliseconds".to_owned(),
        ));
    }
    let reader = crate::io::open(path, crate::io::LoadOptions::default())?;
    let (t0, t1) = reader.time_span();
    let step = (window_ms * 1_000.0).round() as i64;
    let started = std::time::Instant::now();
    if wall_clock {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| Ros2Error::Session(e.to_string()))?
            .as_nanos() as i64;
        publisher.clock_offset_ns = now_ns - t0.saturating_mul(NS_PER_US);
    }
    let mut report = ReplayReport {
        packets: 0,
        events: 0,
        empty_windows: 0,
    };
    let mut t = t0;
    while t <= t1 {
        let slice = reader.slice_time(t, t + step)?;
        if slice.is_empty() {
            report.empty_windows += 1;
        } else {
            report.events += slice.len();
            report.packets += 1;
            publisher.publish(&slice)?;
        }
        if wall_clock {
            // Pace against the recording's own clock rather than sleeping a fixed step, so a slow
            // window does not push every window after it late.
            let due = Duration::from_micros((t + step - t0).max(0) as u64);
            if let Some(wait) = due.checked_sub(started.elapsed()) {
                std::thread::sleep(wait);
            }
        }
        t += step;
    }
    Ok(report)
}

/// What a recording run did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordReport {
    pub packets: u64,
    pub events: usize,
    /// Packets the sequence numbers say never arrived.
    pub dropped: u64,
}

/// Writes everything arriving on a topic into a file, in any format this library writes.
///
/// Stops after `idle` with nothing on the topic, or after `max_packets` — whichever comes first.
/// `None` for the cap means "until the topic goes quiet", which is how you record a replay without
/// knowing in advance how long it is.
pub fn record_topic(
    subscriber: &mut EventSubscriber,
    path: impl AsRef<std::path::Path>,
    idle: Duration,
    max_packets: Option<u64>,
) -> Result<RecordReport> {
    let mut sink = crate::io::open_sink(path, &crate::io::SaveOptions::default())?;
    let mut report = RecordReport {
        packets: 0,
        events: 0,
        dropped: 0,
    };
    loop {
        if max_packets.is_some_and(|cap| report.packets >= cap) {
            break;
        }
        match subscriber.recv(idle)? {
            Some(stream) => {
                report.events += stream.len();
                report.packets += 1;
                sink.append(&stream)?;
            }
            None => break,
        }
    }
    report.dropped = subscriber.dropped();
    sink.finish()?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::representation::Representation;
    use crate::EventStreamBuilder;

    fn stream() -> EventStream {
        let mut b = EventStreamBuilder::new(1280, 720, 0.001);
        for i in 0..500i64 {
            b.push(
                (i % 1280) as u16,
                (i % 720) as u16,
                1_000_000 + i * 37,
                i % 2 == 0,
            );
        }
        b.build()
    }

    #[test]
    fn a_packet_round_trips_through_evt3() {
        let original = stream();
        let packet = EventPacket::encode(&original, 1, EvtVersion::Evt3).unwrap();
        assert_eq!(packet.encoding, "evt3");
        assert_eq!(packet.width, 1280);
        assert_eq!(packet.height, 720);
        // The payload is encoded words, not one field per event.
        assert!(packet.events.len() < original.len() * 8);
        let back = packet.decode().unwrap();
        assert_eq!(back.len(), original.len(), "event count changed");
        assert_eq!(back.xs(), original.xs(), "x changed");
        assert_eq!(back.ys(), original.ys(), "y changed");
        assert_eq!(back.ts(), original.ts(), "timestamps changed");
        assert_eq!(back.ps(), original.ps(), "polarity changed");
    }

    #[test]
    fn evt2_round_trips_too() {
        let original = stream();
        let packet = EventPacket::encode(&original, 7, EvtVersion::Evt2).unwrap();
        assert_eq!(packet.encoding, "evt2");
        assert_eq!(packet.seq, 7);
        let back = packet.decode().unwrap();
        assert_eq!(back.ts(), original.ts());
        assert_eq!(back.xs(), original.xs());
    }

    #[test]
    fn an_unknown_encoding_names_the_ones_that_work() {
        let mut packet = EventPacket::encode(&stream(), 1, EvtVersion::Evt3).unwrap();
        packet.encoding = "libcaer".to_owned();
        let message = packet.decode().unwrap_err().to_string();
        assert!(message.contains("libcaer"), "{message}");
        assert!(message.contains("evt3"), "{message}");
    }

    #[test]
    fn the_time_base_is_a_ros_time_in_nanoseconds() {
        // The message says "event ros time = time_base + decoded event_time", and a decoder that
        // never heard of this library reads the words as microseconds. So the base has to be
        // nanoseconds, or every consumer downstream is out by a factor of a thousand.
        let original = stream();
        let first_us = original.ts()[0];
        let packet = EventPacket::encode(&original, 1, EvtVersion::Evt3).unwrap();
        assert_eq!(packet.time_base_ns(), (first_us * 1_000) as u64);
        assert_eq!(packet.header.stamp.sec, (first_us / 1_000_000) as i32);
        assert_eq!(packet.decode().unwrap().ts()[0], first_us);
    }

    #[test]
    fn a_clock_offset_moves_the_base_and_nothing_else() {
        let original = stream();
        let hour_ns = 3_600 * 1_000_000_000i64;
        let packet =
            EventPacket::encode_with(&original, 1, EvtVersion::Evt3, hour_ns, "cam0").unwrap();
        assert_eq!(packet.header.frame_id, "cam0");
        assert_eq!(
            packet.time_base_ns(),
            (original.ts()[0] * 1_000 + hour_ns) as u64
        );
        let back = packet.decode().unwrap();
        // Every event moved by the same hour; the intervals are untouched.
        let shift = back.ts()[0] - original.ts()[0];
        assert_eq!(shift, hour_ns / 1_000);
        for (a, b) in back.ts().iter().zip(original.ts()) {
            assert_eq!(a - b, shift);
        }
    }

    #[test]
    fn a_stream_on_another_tick_is_rescaled_not_mis_timed() {
        // A stream in milliseconds, not microseconds: the encoder writes EVT words, which are
        // microseconds by definition, so the conversion has to happen somewhere.
        let mut b = EventStreamBuilder::new(64, 64, 1.0);
        for i in 0..50i64 {
            b.push(i as u16, i as u16, 100 + i, i % 2 == 0);
        }
        let millis = b.build();
        let packet = EventPacket::encode(&millis, 1, EvtVersion::Evt3).unwrap();
        // 100 ms is 100_000 us is 100_000_000 ns.
        assert_eq!(packet.time_base_ns(), 100_000_000);
        let back = packet.decode().unwrap();
        assert_eq!(back.len(), millis.len());
        assert_eq!(back.ts()[0], 100_000);
        assert_eq!(back.ts()[49], 149_000);
    }

    #[test]
    fn a_count_frame_becomes_a_mono16_image() {
        let frame = crate::representation::EventCount::new(false)
            .generate(&stream())
            .unwrap();
        let (encoding, sample, bytes) = encode_frame(frame.data(), 1, 1280, 720).unwrap();
        assert_eq!(encoding, "mono16");
        assert_eq!(sample, 2);
        assert_eq!(bytes.len(), 1280 * 720 * 2);
    }

    #[test]
    fn a_voxel_grid_is_interleaved_into_32fc3() {
        let frame = crate::representation::VoxelGrid::new(3, 30.0)
            .generate(&stream())
            .unwrap();
        let (channels, height, width) = frame.shape();
        assert_eq!(channels, 3);
        let (encoding, sample, bytes) =
            encode_frame(frame.data(), channels, width, height).unwrap();
        assert_eq!(encoding, "32FC3");
        assert_eq!(sample, 4);
        assert_eq!(bytes.len(), width * height * 3 * 4);

        // Pixel-major, not plane-major: pixel 0's three channels come first.
        let planar = match frame.data() {
            EventFrameData::F32(v) => v.clone(),
            other => panic!("voxel grid is f32, got {other:?}"),
        };
        let pixels = width * height;
        for channel in 0..3 {
            let at = channel * 4;
            let read = f32::from_ne_bytes(bytes[at..at + 4].try_into().unwrap());
            assert_eq!(read, planar[channel * pixels]);
        }
    }

    #[test]
    fn an_image_message_carries_the_layout_ros_expects() {
        let frame = crate::representation::VoxelGrid::new(3, 30.0)
            .generate(&stream())
            .unwrap();
        let image = image_message(&frame, 1_500_000_042, "cam0").unwrap();
        assert_eq!(image.encoding, "32FC3");
        assert_eq!((image.width, image.height), (1280, 720));
        // step is one row of interleaved pixels, in bytes.
        assert_eq!(image.step, 1280 * 3 * 4);
        assert_eq!(image.header.frame_id, "cam0");
        assert_eq!(image.header.stamp.sec, 1);
        assert_eq!(image.header.stamp.nanosec, 500_000_042);
    }

    #[test]
    fn a_tensor_message_strides_the_ros_way() {
        // ROS strides count the axis itself: dim[i].stride is the product of shape[i..].
        let message = tensor_message(&[2, 3, 4], vec![0.0; 24], &[]).unwrap();
        let dims = &message.layout.dim;
        assert_eq!(dims.len(), 3);
        assert_eq!(
            dims.iter().map(|d| d.size).collect::<Vec<_>>(),
            vec![2, 3, 4]
        );
        assert_eq!(
            dims.iter().map(|d| d.stride).collect::<Vec<_>>(),
            vec![24, 12, 4]
        );
        assert_eq!(
            dims.iter().map(|d| d.label.as_str()).collect::<Vec<_>>(),
            vec!["dim0", "dim1", "dim2"]
        );
        assert_eq!(message.data.len(), 24);
    }

    #[test]
    fn a_tensor_message_checks_the_shape_against_the_data() {
        let message = tensor_message(&[2, 3], vec![0.0; 5], &[])
            .unwrap_err()
            .to_string();
        assert!(message.contains('6'), "{message}");
        assert!(message.contains('5'), "{message}");
    }

    #[test]
    fn a_frame_becomes_a_batched_float_input() {
        let frame = crate::representation::EventCount::new(false)
            .generate(&stream())
            .unwrap();
        let (shape, values) = frame_as_input(&frame);
        assert_eq!(shape, vec![1, 1, 720, 1280]);
        assert_eq!(values.len(), 720 * 1280);
        // The counts widened rather than being reinterpreted.
        let counted: f32 = values.iter().sum();
        assert_eq!(counted as usize, stream().len());
    }

    #[test]
    fn too_many_channels_says_so() {
        let data = EventFrameData::F32(vec![0.0; 5 * 4]);
        let message = encode_frame(&data, 5, 2, 2).unwrap_err().to_string();
        assert!(message.contains("4 channels"), "{message}");
        assert!(message.contains('5'), "{message}");
    }

    #[test]
    fn an_empty_window_is_an_empty_packet() {
        let empty = EventStreamBuilder::new(640, 480, 0.001).build();
        let packet = EventPacket::encode(&empty, 1, EvtVersion::Evt3).unwrap();
        assert_eq!(packet.decode().unwrap().len(), 0);
    }
}
