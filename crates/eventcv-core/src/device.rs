//! Live USB event-camera capture — the streaming counterpart to [`io`](crate::io).
//!
//! Where the readers in [`io`](crate::io) turn a *file* into [`EventStream`]s, [`Capture`] turns a
//! *camera* into them. Events are pulled from the device over USB by the `neuromorphic-drivers`
//! crate (Prophesee EVK4 / EVK3 HD, iniVation DVXplorer / DAVIS346, CenturyArks VGA), decoded, and
//! accumulated into fixed windows — either a fixed duration ([`Window::Duration`], the `dt_ms`
//! twin) or a fixed event count ([`Window::Count`], the `max_events` twin) — mirroring
//! [`io::open`](crate::io::open) so everything downstream (representations, features, `.view()`,
//! saving) composes on a live stream exactly as it does on a file.
//!
//! Only non-empty windows are emitted: purely idle time advances the window grid but never yields
//! an empty [`EventStream`], so a `for window in camera` loop never spins on nothing. Timestamps are
//! device microseconds, so streams are built with a `0.001` ms scale (matching `stream.numpy()`).
//!
//! This module is compiled only with the `camera` feature, which pulls in the driver crate and a
//! statically-linked vendored libusb (no system libusb at runtime). On Linux the camera needs udev
//! rules for non-root USB access — see the crate README.

use std::collections::VecDeque;
use std::time::Duration;

use neuromorphic_drivers as nd;
use neuromorphic_drivers::types::Polarity;
use neuromorphic_drivers::Adapter;

use crate::{EventStream, EventStreamBuilder};

/// Device timestamps are microseconds, so one timestamp unit is `0.001` ms. Live streams carry
/// this scale, so `stream.numpy()[:, 2]` reads in microseconds just like a file-backed stream.
const TIMESTAMP_SCALE_MS: f64 = 0.001;

/// How long each [`Capture::poll`] blocks waiting for USB data before returning to the caller so it
/// can observe a stop request. Short enough to feel responsive; the driver's own ring absorbs the
/// events that arrive between polls.
const POLL_TIMEOUT: Duration = Duration::from_millis(5);

/// A camera discovered by [`list_cameras`].
#[derive(Clone, Debug)]
pub struct CameraInfo {
    /// Machine-readable type, e.g. `"prophesee_evk4"` (the driver's module name).
    pub kind: String,
    /// Human-readable model name, e.g. `"Prophesee EVK4"`.
    pub name: String,
    /// Device serial, when it could be read.
    pub serial: Option<String>,
    /// USB bus the device is on.
    pub bus_number: u8,
    /// USB address on the bus.
    pub address: u8,
    /// Negotiated USB speed (`"super+"`, `"super"`, `"high"`, …).
    pub speed: String,
}

/// Enumerates every connected, supported event camera. The returned [`CameraInfo::serial`] can be
/// passed to [`Capture::open`] to select a specific device when several are plugged in.
pub fn list_cameras() -> Result<Vec<CameraInfo>, String> {
    let listed = nd::list_devices().map_err(|error| error.to_string())?;
    Ok(listed
        .into_iter()
        .map(|device| CameraInfo {
            kind: device.device_type.to_string(),
            name: device.device_type.name().to_owned(),
            serial: device.serial.ok(),
            bus_number: device.bus_number,
            address: device.address,
            speed: speed_name(device.speed).to_owned(),
        })
        .collect())
}

/// Turns a raw `nd::open` failure into an actionable message. The driver reports a device that is
/// present but can't be claimed (already open elsewhere, or missing USB permissions) the same way as
/// a truly absent one — "no device found". So when enumeration *can* still see a supported camera,
/// we say the device is present-but-unavailable rather than repeat the misleading "not found".
fn open_error(serial: Option<&str>, error: impl std::fmt::Display) -> String {
    let visible = list_cameras().unwrap_or_default();
    let matched = match serial {
        Some(serial) => visible.iter().find(|camera| camera.serial.as_deref() == Some(serial)),
        None => visible.first(),
    };
    match matched {
        Some(camera) => format!(
            "found {} but couldn't open it ({error}). It may already be open in this or another \
             process — an earlier eventcv.stream()/EventCamera that is still alive holds the device \
             until it is closed (use `with eventcv.stream() as cam:` or call `cam.close()`); or, on \
             Linux, the USB udev rules may be missing. Only one handle can open a camera at a time.",
            camera.name,
        ),
        None => match serial {
            Some(serial) => format!("no camera with serial {serial} found ({error})"),
            None => format!("no event camera found ({error})"),
        },
    }
}

fn speed_name(speed: nd::usb::Speed) -> &'static str {
    match speed {
        nd::usb::Speed::Unknown => "unknown",
        nd::usb::Speed::Low => "low",
        nd::usb::Speed::Full => "full",
        nd::usb::Speed::High => "high",
        nd::usb::Speed::Super => "super",
        nd::usb::Speed::SuperPlus => "super+",
    }
}

/// How the incoming event stream is cut into [`EventStream`] windows — the live twin of
/// [`io::open`](crate::io::open)'s `dt_ms` / `max_events`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Window {
    /// Emit a window every `dt` **milliseconds of event time** (fixed-duration frames).
    Duration(f64),
    /// Emit a window every `n` events (fixed event count per frame).
    Count(usize),
}

/// One completed window handed back by [`Capture::poll`].
#[derive(Clone, Debug)]
pub struct CaptureWindow {
    /// The events accumulated over this window.
    pub stream: EventStream,
    /// `true` when the device's ring buffer overflowed just before this window — some events were
    /// dropped by the driver. Surfaced so callers can warn rather than lose data silently.
    pub first_after_overflow: bool,
}

/// Accumulates decoded events into [`CaptureWindow`]s. Split out from the USB plumbing so its
/// windowing logic is unit-testable without a camera.
struct Windower {
    width: usize,
    height: usize,
    window: Window,
    /// Duration-mode window length in microseconds (`0` in count mode).
    dt_us: i64,
    builder: EventStreamBuilder,
    /// Duration mode: the end timestamp (µs) of the window currently filling; `None` until the
    /// first event anchors the grid.
    next_boundary_t: Option<i64>,
    pending: VecDeque<CaptureWindow>,
    /// Carries a pending overflow flag onto the next sealed window.
    overflow_pending: bool,
}

impl Windower {
    fn new(width: usize, height: usize, window: Window) -> Self {
        let dt_us = match window {
            // At least 1 µs so the grid always advances, even for an absurdly small dt.
            Window::Duration(dt_ms) => ((dt_ms * 1000.0).max(1.0)) as i64,
            Window::Count(_) => 0,
        };
        Self {
            width,
            height,
            window,
            dt_us,
            builder: EventStreamBuilder::new(width, height, TIMESTAMP_SCALE_MS),
            next_boundary_t: None,
            pending: VecDeque::new(),
            overflow_pending: false,
        }
    }

    /// Records that the device ring overflowed; the flag rides along on the next sealed window.
    fn mark_overflow(&mut self) {
        self.overflow_pending = true;
    }

    /// Appends one decoded event, sealing windows as boundaries are crossed.
    fn push(&mut self, x: u16, y: u16, t: i64, polarity: bool) {
        match self.window {
            Window::Duration(_) => {
                match self.next_boundary_t {
                    None => self.next_boundary_t = Some(t + self.dt_us),
                    Some(_) => self.advance_to(t),
                }
                self.builder.push(x, y, t, polarity);
            }
            Window::Count(n) => {
                self.builder.push(x, y, t, polarity);
                if self.builder.len() >= n {
                    self.seal();
                }
            }
        }
    }

    /// Advances the duration-mode grid so the window currently filling contains time `t`, sealing
    /// the current window if it holds events. Idle bins are skipped (no empty windows emitted): an
    /// empty current window jumps straight to the bin containing `t` in one step.
    fn advance_to(&mut self, t: i64) {
        let Some(mut boundary) = self.next_boundary_t else {
            return;
        };
        while t >= boundary {
            if self.builder.is_empty() {
                let steps = (t - boundary) / self.dt_us + 1;
                boundary += steps * self.dt_us;
            } else {
                self.seal();
                boundary += self.dt_us;
            }
        }
        self.next_boundary_t = Some(boundary);
    }

    /// Flushes the current window if its boundary has passed by `t_now` — keeps latency bounded when
    /// events pause mid-window. Idle time only advances the grid; it never emits empty windows.
    fn seal_until(&mut self, t_now: i64) {
        self.advance_to(t_now);
    }

    /// Seals the current window into `pending` and starts a fresh one.
    ///
    /// The replacement is sized from the window just sealed: consecutive windows hold similar event
    /// counts, so the next one fills without reallocating. Starting from zero meant a busy window
    /// (millions of events on a high-resolution sensor) grew its four column vectors through ~20
    /// doublings, copying everything accumulated so far each time.
    fn seal(&mut self) {
        let fresh = EventStreamBuilder::with_capacity(
            self.width,
            self.height,
            TIMESTAMP_SCALE_MS,
            self.builder.len(),
        );
        let builder = std::mem::replace(&mut self.builder, fresh);
        self.pending.push_back(CaptureWindow {
            stream: builder.build(),
            first_after_overflow: self.overflow_pending,
        });
        self.overflow_pending = false;
    }

    /// Seals a final partial window (called when the stream stops).
    fn flush(&mut self) {
        if !self.builder.is_empty() {
            self.seal();
        }
    }
}

/// A live capture session bound to one camera. Construct with [`Capture::open`], then drive it by
/// calling [`Capture::poll`] in a loop (typically on a dedicated thread that owns the `Capture`).
pub struct Capture {
    device: nd::Device,
    adapter: Adapter,
    // The USB event loop must stay alive for the device to keep reading; held, not used directly.
    _event_loop: std::sync::Arc<nd::UsbEventLoop>,
    flag: nd::Flag<nd::Error, nd::UsbOverflow>,
    windower: Windower,
    width: usize,
    height: usize,
    name: String,
    serial: String,
}

impl Capture {
    /// Opens a camera and begins streaming. `serial` selects a specific device (from
    /// [`CameraInfo::serial`]); `None` opens the first supported camera found. `window` sets how the
    /// stream is cut into [`CaptureWindow`]s.
    ///
    /// The device is opened with its default configuration (biases, ROI, …). Errors — no camera
    /// found, permission denied (missing udev rules on Linux), or an unreadable serial — are
    /// returned as a human-readable string.
    pub fn open(serial: Option<&str>, window: Window) -> Result<Self, String> {
        let (flag, event_loop) = nd::flag_and_event_loop().map_err(|error| error.to_string())?;
        let selector = match serial {
            Some(serial) => nd::SerialOrBusNumberAndAddress::Serial(serial),
            None => nd::SerialOrBusNumberAndAddress::None,
        };
        let device = nd::open(selector, None, None, event_loop.clone(), flag.clone())
            .map_err(|error| open_error(serial, error))?;
        let (width, height) = device_dimensions(&device);
        let name = device.name().to_owned();
        let serial = device.serial();
        let adapter = device.create_adapter();
        Ok(Self {
            device,
            adapter,
            _event_loop: event_loop,
            flag,
            windower: Windower::new(width, height, window),
            width,
            height,
            name,
            serial,
        })
    }

    /// Sensor width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Sensor height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Human-readable model name, e.g. `"Prophesee EVK4"`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Device serial number.
    pub fn serial(&self) -> &str {
        &self.serial
    }

    /// Number of buffers waiting in the driver's ring — a rising value means the consumer is falling
    /// behind the camera.
    pub fn backlog(&self) -> usize {
        self.device.backlog()
    }

    /// Polls the camera for up to a few milliseconds, decodes whatever arrived, and returns the next
    /// completed window if one is ready (or was already buffered).
    ///
    /// Returns `Ok(None)` when no window completed this poll — the caller should check its own stop
    /// condition and call again. Returns `Err` on a device error (e.g. the camera was unplugged).
    /// Call [`Capture::finish`] once, after the loop, to recover a final partial window.
    pub fn poll(&mut self) -> Result<Option<CaptureWindow>, String> {
        if let Some(window) = self.windower.pending.pop_front() {
            return Ok(Some(window));
        }
        // Surface any error raised by the driver's background USB thread (e.g. a disconnect).
        self.flag.load_error().map_err(|error| error.to_string())?;

        if let Some(view) = self.device.next_with_timeout(&POLL_TIMEOUT) {
            if view.first_after_overflow {
                self.windower.mark_overflow();
            }
            let slice = view.slice;
            // Disjoint field borrows: `slice` borrows `self.device`, the closure borrows
            // `self.windower`, and `convert_buffer` borrows `self.adapter`.
            let windower = &mut self.windower;
            convert_buffer(&mut self.adapter, slice, |x, y, t, polarity| {
                windower.push(x, y, t, polarity)
            });
            // Flush the current window if its boundary has passed, so a mid-window pause doesn't
            // stall delivery. `current_t` advances with the device clock even during quiet periods.
            let current_t = self.adapter.current_t() as i64;
            self.windower.seal_until(current_t);
        }
        Ok(self.windower.pending.pop_front())
    }

    /// Decodes **at most one** queued buffer into the window grid, waiting up to `timeout` for one
    /// to arrive. Returns whether a buffer was decoded; completed windows are collected with
    /// [`take_pending`].
    ///
    /// This is the windowed twin of [`drain_events`](Self::drain_events), deliberately stopping
    /// after a single buffer: a caller that decodes ahead of its consumer must be able to pause. A
    /// full-ring drain would let the driver's ring — half a gigabyte of encoded events on a
    /// Prophesee sensor — expand into many times that as decoded columns before the consumer got a
    /// look in. Decoding one buffer at a time keeps that bounded: when the consumer stops taking
    /// windows the caller simply stops calling this, and the ring (not memory) absorbs the excess.
    pub fn decode_next(&mut self, timeout: Duration) -> Result<bool, String> {
        self.flag.load_error().map_err(|error| error.to_string())?;
        let Some(view) = self.device.next_with_timeout(&timeout) else {
            // Nothing arrived, but the device clock still moved: flush a window whose boundary has
            // passed so a mid-window pause doesn't stall delivery.
            let current_t = self.adapter.current_t() as i64;
            self.windower.seal_until(current_t);
            return Ok(false);
        };
        if view.first_after_overflow {
            self.windower.mark_overflow();
        }
        let slice = view.slice;
        let windower = &mut self.windower;
        convert_buffer(&mut self.adapter, slice, |x, y, t, polarity| {
            windower.push(x, y, t, polarity)
        });
        let current_t = self.adapter.current_t() as i64;
        self.windower.seal_until(current_t);
        Ok(true)
    }

    /// Takes the next window that has **already** completed, without touching the device. Returns
    /// `None` once the queue sealed by [`poll`] / [`catch_up`] is empty.
    pub fn take_pending(&mut self) -> Option<CaptureWindow> {
        self.windower.pending.pop_front()
    }

    /// Seals and returns a final partial window after the stream is stopped, if any events remain.
    pub fn finish(&mut self) -> Option<CaptureWindow> {
        self.windower.flush();
        self.windower.pending.pop_front()
    }

    /// Non-blocking: decodes **every** USB buffer currently queued in the driver's ring, calling
    /// `on_event(x, y, t_us, polarity)` for each polarity event, and returns whether the ring
    /// overflowed. `t` is in microseconds.
    ///
    /// Unlike [`poll`](Self::poll), this bypasses windowing and fully drains the ring in one call,
    /// so a display-paced consumer (the live viewer, which renders on its own clock) never lets the
    /// driver back up. It reads with a zero timeout, so it stops as soon as the ring is momentarily
    /// empty rather than waiting for a window to complete.
    pub fn drain_events<F>(&mut self, mut on_event: F) -> Result<bool, String>
    where
        F: FnMut(u16, u16, i64, bool),
    {
        self.flag.load_error().map_err(|error| error.to_string())?;
        let mut overflow = false;
        while let Some(view) = self.device.next_with_timeout(&Duration::ZERO) {
            if view.first_after_overflow {
                overflow = true;
            }
            convert_buffer(&mut self.adapter, view.slice, &mut on_event);
        }
        Ok(overflow)
    }

    /// Like [`drain_events`](Self::drain_events), but caps how long is spent **decoding** per call.
    ///
    /// Every queued buffer is still read from the ring (so the driver never backs up), but once
    /// `budget` of wall-clock time has been spent decoding, any remaining buffers are dropped
    /// without being decoded. This bounds per-call latency for a display-paced consumer at very high
    /// event rates — e.g. a 1280×720 Prophesee sensor emitting hundreds of millions of events per
    /// second, where decoding a whole frame's worth of events takes far longer than a display frame.
    /// The viewer visualises the freshest `budget` of decode work each frame and skips the surplus
    /// (which is invisible at that rate anyway), keeping the display at ~60 FPS. `t` is in
    /// microseconds. Returns whether the ring overflowed.
    pub fn drain_events_budgeted<F>(
        &mut self,
        budget: Duration,
        mut on_event: F,
    ) -> Result<bool, String>
    where
        F: FnMut(u16, u16, i64, bool),
    {
        self.flag.load_error().map_err(|error| error.to_string())?;
        let start = std::time::Instant::now();
        let mut overflow = false;
        let mut decoding = true;
        while let Some(view) = self.device.next_with_timeout(&Duration::ZERO) {
            if view.first_after_overflow {
                overflow = true;
            }
            if !decoding {
                continue; // over budget: drop the buffer, freeing its ring slot without decoding.
            }
            convert_buffer(&mut self.adapter, view.slice, &mut on_event);
            if start.elapsed() >= budget {
                decoding = false;
            }
        }
        Ok(overflow)
    }
}

/// Decodes one USB buffer with whichever adapter the open device uses, forwarding each polarity
/// event as `(x, y, t_us, positive)` and discarding the IMU/trigger/APS streams. The single place
/// the per-device adapter variants are matched, shared by every decode path above.
fn convert_buffer(
    adapter: &mut Adapter,
    slice: &[u8],
    mut on_event: impl FnMut(u16, u16, i64, bool),
) {
    let handle = |event: nd::types::PolarityEvent<u64, u16, u16>| {
        on_event(
            event.x,
            event.y,
            event.t as i64,
            matches!(event.polarity, Polarity::On),
        );
    };
    match adapter {
        Adapter::Evt3(adapter) => adapter.convert(slice, handle, |_trigger| {}),
        Adapter::Dvxplorer(adapter) => adapter.convert(slice, handle, |_imu| {}, |_trigger| {}),
        Adapter::Davis346(adapter) => {
            adapter.convert(slice, handle, |_imu| {}, |_trigger| {}, |_frame| {})
        }
    }
}

/// Reads the sensor resolution from the device properties. Every variant wraps the same
/// `Camera { width, height, .. }` record.
fn device_dimensions(device: &nd::Device) -> (usize, usize) {
    use nd::Properties;
    let (width, height) = match device.properties() {
        Properties::InivationDavis346(properties) => (properties.width, properties.height),
        Properties::InivationDvxplorer(properties) => (properties.width, properties.height),
        Properties::PropheseeEvk3Hd(properties) => (properties.width, properties.height),
        Properties::PropheseeEvk4(properties) => (properties.width, properties.height),
        Properties::CenturyarksVga(properties) => (properties.width, properties.height),
    };
    (width as usize, height as usize)
}

#[cfg(test)]
mod tests {
    use super::{Window, Windower};

    #[test]
    fn count_windows_split_by_event_count() {
        let mut windower = Windower::new(10, 10, Window::Count(2));
        for t in 0..5 {
            windower.push(1, 1, t, true);
        }
        // Sealed at 2 and 4 events; one event still buffered.
        assert_eq!(windower.pending.len(), 2);
        assert_eq!(windower.pending[0].stream.len(), 2);
        assert_eq!(windower.pending[1].stream.len(), 2);

        windower.flush();
        assert_eq!(windower.pending.len(), 3);
        assert_eq!(windower.pending[2].stream.len(), 1);
    }

    #[test]
    fn duration_windows_split_by_event_time() {
        // dt = 1 ms = 1000 µs; events at t = 0, 500, 1500, 2500 µs.
        let mut windower = Windower::new(10, 10, Window::Duration(1.0));
        windower.push(1, 1, 0, true); // anchors the grid: boundary -> 1000
        windower.push(1, 1, 500, true); // < 1000, same window
        windower.push(1, 1, 1500, true); // >= 1000: seals [.,1000) (2 events), boundary -> 2000
        windower.push(1, 1, 2500, true); // >= 2000: seals [1000,2000) (1 event), boundary -> 3000

        assert_eq!(windower.pending.len(), 2);
        assert_eq!(windower.pending[0].stream.len(), 2);
        assert_eq!(windower.pending[1].stream.len(), 1);

        windower.flush(); // final partial window holds t = 2500
        assert_eq!(windower.pending.len(), 3);
        assert_eq!(windower.pending[2].stream.len(), 1);
    }

    #[test]
    fn idle_gap_emits_one_window_not_many_empties() {
        // A large gap between two events must seal exactly one (non-empty) window, not one empty
        // window per skipped bin.
        let mut windower = Windower::new(10, 10, Window::Duration(1.0));
        windower.push(1, 1, 0, true); // boundary -> 1000
        windower.push(1, 1, 10_000, true); // 10 ms later

        assert_eq!(windower.pending.len(), 1);
        assert_eq!(windower.pending[0].stream.len(), 1);
    }

    #[test]
    fn seal_until_flushes_a_stalled_window() {
        // Events then a pause: seal_until (driven by the device clock) flushes the partial window.
        let mut windower = Windower::new(10, 10, Window::Duration(1.0));
        windower.push(1, 1, 100, true); // boundary -> 1100
        assert_eq!(windower.pending.len(), 0);
        windower.seal_until(2000); // clock passed the boundary
        assert_eq!(windower.pending.len(), 1);
        assert_eq!(windower.pending[0].stream.len(), 1);
    }

    #[test]
    fn overflow_flag_rides_on_the_next_window() {
        let mut windower = Windower::new(10, 10, Window::Count(1));
        windower.mark_overflow();
        windower.push(1, 1, 0, true); // seals immediately
        windower.push(2, 2, 1, true); // seals again, no overflow now

        assert!(windower.pending[0].first_after_overflow);
        assert!(!windower.pending[1].first_after_overflow);
    }
}
