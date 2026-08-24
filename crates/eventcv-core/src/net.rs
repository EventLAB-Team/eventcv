//! UDP event streaming, wire-compatible with aestream and SPIF.
//!
//! Events are packed into 32-bit words and sent as UDP datagrams — the transport SpiNNaker boards
//! and the Open Neuromorphic tooling speak. There is no handshake, no acknowledgement and no
//! retransmission: UDP suits event data precisely because dropping a late packet is better than
//! delaying every packet behind it.
//!
//! ```no_run
//! # use eventcv_core::net::{UdpSender, UdpReceiver, WireFormat};
//! # fn demo(stream: &eventcv_core::EventStream) -> std::io::Result<()> {
//! let sender = UdpSender::connect("127.0.0.1:3333", WireFormat::default())?;
//! sender.send(stream)?;
//!
//! let receiver = UdpReceiver::bind("127.0.0.1:3333", 640, 480, WireFormat::default())?;
//! let events = receiver.recv_window(std::time::Duration::from_millis(30))?;
//! # Ok(())
//! # }
//! ```
//!
//! # The wire format
//!
//! Taken from aestream's `dvs_to_udp.cpp`, which is the de-facto reference:
//!
//! - **Untimestamped** — one word per event:
//!   `((x | 0x8000) << 16) | (polarity ? y | 0x8000 : y & 0x7FFF)`
//! - **Timestamped** — two words: `((x & 0x7FFF) << 16) | (y with the polarity bit)`, then the
//!   timestamp.
//!
//! Bit 31 distinguishes the two modes, bit 15 of the low half carries polarity, and coordinates are
//! 15-bit. A receiver can therefore tell which mode a packet is in from the first word, which is
//! why [`UdpReceiver`] does not need to be told.
//!
//! # Byte order
//!
//! aestream's source contains comments conceding that the packing *should* use `htons`/`htonl` and
//! does not — it writes host-endian words. Matching a documented bug is the only way to actually
//! interoperate, so [`WireFormat::host_endian`] is the default. [`WireFormat::network_endian`]
//! sends correct network byte order for anything that expects it, at the cost of not talking to
//! aestream on a little-endian machine.

use std::io;
use std::net::{ToSocketAddrs, UdpSocket};
use std::time::{Duration, Instant};

use crate::{EventStream, EventStreamBuilder};

/// Largest UDP payload sent, in bytes. Comfortably inside the 1500-byte Ethernet MTU once IP and
/// UDP headers are accounted for, so datagrams are not fragmented — a fragmented datagram is lost
/// entirely if any fragment is, which multiplies the loss rate.
const MAX_PAYLOAD_BYTES: usize = 1400;

/// Words per datagram, from [`MAX_PAYLOAD_BYTES`].
const MAX_WORDS: usize = MAX_PAYLOAD_BYTES / 4;

/// Bit marking the untimestamped mode, in the high half of the first word.
const NO_TIMESTAMP_FLAG: u32 = 0x8000_0000;

/// Bit carrying polarity, in the low half.
const POLARITY_FLAG: u32 = 0x0000_8000;

/// Coordinate mask — 15 bits per axis.
const COORD_MASK: u32 = 0x7FFF;

/// How events are laid out on the wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WireFormat {
    /// Include a timestamp word per event. Doubles the bandwidth; without it the receiver stamps
    /// events on arrival, which is fine for live display and wrong for anything measuring latency.
    pub timestamps: bool,
    /// Write words in host byte order. Default `true`, to match aestream.
    pub host_endian: bool,
}

impl Default for WireFormat {
    fn default() -> Self {
        Self {
            timestamps: false,
            host_endian: true,
        }
    }
}

impl WireFormat {
    /// aestream-compatible: host byte order, no timestamps.
    pub fn host_endian() -> Self {
        Self::default()
    }

    /// Correct network byte order. Will not interoperate with aestream on a little-endian host.
    pub fn network_endian() -> Self {
        Self {
            timestamps: false,
            host_endian: false,
        }
    }

    /// The same format with timestamps included.
    pub fn with_timestamps(mut self) -> Self {
        self.timestamps = true;
        self
    }

    fn encode_word(self, word: u32) -> [u8; 4] {
        if self.host_endian {
            word.to_ne_bytes()
        } else {
            word.to_be_bytes()
        }
    }

    fn decode_word(self, bytes: [u8; 4]) -> u32 {
        if self.host_endian {
            u32::from_ne_bytes(bytes)
        } else {
            u32::from_be_bytes(bytes)
        }
    }
}

/// Packs an event into its data word.
fn encode_event(x: u16, y: u16, polarity: bool, timestamped: bool) -> u32 {
    let x = u32::from(x) & COORD_MASK;
    let y = u32::from(y) & COORD_MASK;
    // Bit 31 is set only in the untimestamped mode, which is how the receiver tells them apart.
    let high = if timestamped { x } else { x | 0x8000 };
    let low = if polarity { y | POLARITY_FLAG } else { y };
    (high << 16) | low
}

/// Unpacks a data word into `(x, y, polarity)`.
fn decode_event(word: u32) -> (u16, u16, bool) {
    let x = ((word >> 16) & COORD_MASK) as u16;
    let y = (word & COORD_MASK) as u16;
    let polarity = word & POLARITY_FLAG != 0;
    (x, y, polarity)
}

/// Sends events over UDP.
pub struct UdpSender {
    socket: UdpSocket,
    format: WireFormat,
}

impl UdpSender {
    /// Binds an ephemeral local port and connects to `target`.
    pub fn connect(target: impl ToSocketAddrs, format: WireFormat) -> io::Result<Self> {
        // 0.0.0.0:0 lets the OS pick the local address and port; `connect` on a UDP socket only
        // fixes the default destination, it does not exchange anything.
        let socket = UdpSocket::bind("0.0.0.0:0")?;
        socket.connect(target)?;
        Ok(Self { socket, format })
    }

    /// The local address, useful when the port was chosen by the OS.
    pub fn local_addr(&self) -> io::Result<std::net::SocketAddr> {
        self.socket.local_addr()
    }

    /// Sends every event, split across as many datagrams as it takes. Returns the number sent.
    ///
    /// Events are never split across datagrams: a timestamped event is two words and both go in the
    /// same packet, so a lost packet costs whole events rather than corrupting the next one.
    ///
    /// **The count returned is what was handed to the socket, not what arrived.** UDP does not
    /// retransmit, and a large stream sent in one burst will overrun the receiver's kernel buffer
    /// unless something is draining it concurrently — even over loopback. The intended pattern is a
    /// receiver looping on [`UdpReceiver::recv_window`] while the sender runs, not a send followed
    /// by a receive.
    pub fn send(&self, stream: &EventStream) -> io::Result<usize> {
        let words_per_event = if self.format.timestamps { 2 } else { 1 };
        let events_per_packet = MAX_WORDS / words_per_event;
        if events_per_packet == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "wire format does not fit in a datagram",
            ));
        }

        let (xs, ys, ts, ps) = (stream.xs(), stream.ys(), stream.ts(), stream.ps());
        let mut buffer = Vec::with_capacity(MAX_PAYLOAD_BYTES);
        let mut sent = 0;

        for chunk in (0..stream.len())
            .collect::<Vec<_>>()
            .chunks(events_per_packet)
        {
            buffer.clear();
            for &index in chunk {
                let word = encode_event(xs[index], ys[index], ps[index], self.format.timestamps);
                buffer.extend_from_slice(&self.format.encode_word(word));
                if self.format.timestamps {
                    // Timestamps are truncated to 32 bits, which wraps after ~71 minutes at
                    // microsecond resolution. The wire format has no more room; a receiver needing
                    // absolute time across a longer session has to track the wrap itself.
                    buffer.extend_from_slice(&self.format.encode_word(ts[index] as u32));
                }
            }
            if !buffer.is_empty() {
                self.socket.send(&buffer)?;
                sent += chunk.len();
            }
        }
        Ok(sent)
    }
}

/// Receives events over UDP.
pub struct UdpReceiver {
    socket: UdpSocket,
    width: usize,
    height: usize,
    format: WireFormat,
}

impl UdpReceiver {
    /// Binds `address` and prepares to decode onto a `width` × `height` sensor.
    ///
    /// The sensor size is needed because the wire format carries coordinates but not dimensions;
    /// events outside the grid are dropped by the stream builder.
    pub fn bind(
        address: impl ToSocketAddrs,
        width: usize,
        height: usize,
        format: WireFormat,
    ) -> io::Result<Self> {
        let socket = UdpSocket::bind(address)?;
        Ok(Self {
            socket,
            width,
            height,
            format,
        })
    }

    pub fn local_addr(&self) -> io::Result<std::net::SocketAddr> {
        self.socket.local_addr()
    }

    /// Collects events for `window`, returning whatever arrived.
    ///
    /// Returns an empty stream rather than erroring when nothing arrives — silence is the normal
    /// state of an event stream, not a failure.
    pub fn recv_window(&self, window: Duration) -> io::Result<EventStream> {
        let deadline = Instant::now() + window;
        let mut builder = EventStreamBuilder::new(self.width, self.height, 0.001);
        let mut packet = vec![0_u8; MAX_PAYLOAD_BYTES * 2];
        let mut received = 0_i64;

        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            // A read timeout is what bounds the window: without it a quiet link blocks forever.
            self.socket.set_read_timeout(Some(remaining))?;
            match self.socket.recv(&mut packet) {
                Ok(size) => {
                    received += self.decode_into(&packet[..size], &mut builder, received);
                }
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                    ) =>
                {
                    break
                }
                Err(error) => return Err(error),
            }
        }
        Ok(builder.build())
    }

    /// Decodes one datagram, returning how many events it held.
    fn decode_into(
        &self,
        payload: &[u8],
        builder: &mut EventStreamBuilder,
        arrival_index: i64,
    ) -> i64 {
        let mut words = payload
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| self.format.decode_word(*bytes));
        let mut count = 0;
        while let Some(word) = words.next() {
            // Bit 31 tells us whether a timestamp word follows, so a receiver reads either format
            // without being configured for it.
            let timestamped = word & NO_TIMESTAMP_FLAG == 0;
            let (x, y, polarity) = decode_event(word);
            let timestamp = if timestamped {
                match words.next() {
                    Some(t) => i64::from(t),
                    // A truncated datagram: the timestamp word never arrived, so there is no event.
                    None => break,
                }
            } else {
                // Without timestamps on the wire, order of arrival is all the ordering there is.
                arrival_index + count
            };
            builder.push(x, y, timestamp, polarity);
            count += 1;
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(width: usize, height: usize, count: usize) -> EventStream {
        let mut builder = EventStreamBuilder::new(width, height, 0.001);
        for index in 0..count {
            builder.push(
                (index % width) as u16,
                (index % height) as u16,
                index as i64 * 100,
                index % 2 == 0,
            );
        }
        builder.build()
    }

    #[test]
    fn encoding_round_trips_through_the_word() {
        for (x, y, polarity) in [
            (0, 0, false),
            (1, 2, true),
            (639, 479, true),
            (0x7FFF, 0x7FFF, false),
        ] {
            for timestamped in [false, true] {
                let word = encode_event(x, y, polarity, timestamped);
                assert_eq!(decode_event(word), (x, y, polarity), "{x},{y},{polarity}");
                // Bit 31 must identify the mode, which is what lets a receiver auto-detect it.
                assert_eq!(
                    word & NO_TIMESTAMP_FLAG == 0,
                    timestamped,
                    "mode flag wrong for timestamped={timestamped}"
                );
            }
        }
    }

    #[test]
    fn the_layout_matches_aestreams() {
        // Pinned against `aestream/src/cpp/output/dvs_to_udp.cpp`: untimestamped sets 0x8000 in the
        // high half, and polarity sets 0x8000 in the low half. A change here breaks interop
        // silently, so it is asserted literally rather than by round trip.
        assert_eq!(
            encode_event(3, 5, true, false),
            ((3 | 0x8000) << 16) | (5 | 0x8000)
        );
        assert_eq!(encode_event(3, 5, false, false), ((3 | 0x8000) << 16) | 5);
        assert_eq!(encode_event(3, 5, true, true), (3 << 16) | (5 | 0x8000));
        assert_eq!(encode_event(3, 5, false, true), (3 << 16) | 5);
    }

    #[test]
    fn byte_order_is_selectable() {
        let host = WireFormat::host_endian();
        let network = WireFormat::network_endian();
        let word = 0x1234_5678_u32;
        assert_eq!(network.encode_word(word), word.to_be_bytes());
        assert_eq!(host.encode_word(word), word.to_ne_bytes());
        // Each decodes what it encoded, whatever the platform.
        assert_eq!(host.decode_word(host.encode_word(word)), word);
        assert_eq!(network.decode_word(network.encode_word(word)), word);
    }

    #[test]
    fn a_stream_round_trips_over_loopback() {
        let format = WireFormat::default();
        let receiver = UdpReceiver::bind("127.0.0.1:0", 64, 64, format).unwrap();
        let address = receiver.local_addr().unwrap();
        let sender = UdpSender::connect(address, format).unwrap();

        let original = sample(64, 64, 500);
        let sent = sender.send(&original).unwrap();
        assert_eq!(sent, original.len());

        let received = receiver.recv_window(Duration::from_millis(300)).unwrap();
        assert_eq!(
            received.len(),
            original.len(),
            "every event should arrive on loopback"
        );
        assert_eq!(received.xs(), original.xs());
        assert_eq!(received.ys(), original.ys());
        assert_eq!(received.ps(), original.ps());
    }

    #[test]
    fn timestamps_survive_when_the_format_carries_them() {
        let format = WireFormat::default().with_timestamps();
        let receiver = UdpReceiver::bind("127.0.0.1:0", 64, 64, format).unwrap();
        let sender = UdpSender::connect(receiver.local_addr().unwrap(), format).unwrap();

        let original = sample(64, 64, 200);
        sender.send(&original).unwrap();
        let received = receiver.recv_window(Duration::from_millis(300)).unwrap();
        assert_eq!(received.len(), original.len());
        assert_eq!(
            received.ts(),
            original.ts(),
            "timestamps must survive the wire"
        );
    }

    #[test]
    fn a_large_stream_is_split_across_datagrams() {
        // Enough events to need many datagrams, with the receiver draining *concurrently* — the
        // only way this works in practice. A sender blasting into a socket nobody is reading
        // overruns the kernel's receive buffer and most of it is dropped, which is UDP behaving as
        // designed rather than a defect. An earlier version of this test used a stream small enough
        // to fit that buffer and so passed without ever exercising the case.
        let format = WireFormat::default();
        let receiver = UdpReceiver::bind("127.0.0.1:0", 128, 128, format).unwrap();
        let address = receiver.local_addr().unwrap();

        let count = MAX_WORDS * 8;
        let handle = std::thread::spawn(move || {
            let mut total = 0;
            let deadline = Instant::now() + Duration::from_secs(2);
            while Instant::now() < deadline {
                let received = receiver
                    .recv_window(Duration::from_millis(100))
                    .expect("receive should not error");
                if received.is_empty() && total > 0 {
                    break;
                }
                total += received.len();
                if total >= count {
                    break;
                }
            }
            total
        });

        // Give the receiver a moment to reach its first recv before sending.
        std::thread::sleep(Duration::from_millis(50));
        let sender = UdpSender::connect(address, format).unwrap();
        let original = sample(128, 128, count);
        assert_eq!(sender.send(&original).unwrap(), count);

        let received = handle.join().expect("receiver thread should not panic");
        // Still not asserting every event: loopback UDP can drop, and a test that demands
        // reliability from a protocol that does not offer it would be flaky by construction.
        assert!(
            received > count / 2,
            "received {received} of {count} with a concurrent receiver"
        );
    }

    #[test]
    fn a_quiet_link_returns_an_empty_stream() {
        let receiver = UdpReceiver::bind("127.0.0.1:0", 32, 32, WireFormat::default()).unwrap();
        let events = receiver.recv_window(Duration::from_millis(20)).unwrap();
        assert!(events.is_empty(), "silence is not an error");
    }

    #[test]
    fn a_truncated_datagram_does_not_produce_a_bogus_event() {
        // A timestamped event whose timestamp word was cut off must be dropped, not completed with
        // whatever follows.
        let format = WireFormat::default().with_timestamps();
        let receiver = UdpReceiver::bind("127.0.0.1:0", 32, 32, format).unwrap();
        let mut builder = EventStreamBuilder::new(32, 32, 0.001);
        let word = encode_event(4, 4, true, true);
        let truncated = format.encode_word(word); // the data word only, no timestamp
        assert_eq!(receiver.decode_into(&truncated, &mut builder, 0), 0);
        assert!(builder.build().is_empty());
    }

    #[test]
    fn coordinates_outside_the_sensor_are_dropped() {
        let receiver = UdpReceiver::bind("127.0.0.1:0", 16, 16, WireFormat::default()).unwrap();
        let mut builder = EventStreamBuilder::new(16, 16, 0.001);
        let mut payload = Vec::new();
        for word in [
            encode_event(4, 4, true, false),     // inside
            encode_event(900, 900, true, false), // outside
        ] {
            payload.extend_from_slice(&receiver.format.encode_word(word));
        }
        receiver.decode_into(&payload, &mut builder, 0);
        assert_eq!(
            builder.build().len(),
            1,
            "only the in-bounds event survives"
        );
    }
}
