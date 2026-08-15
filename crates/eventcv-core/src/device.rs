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
use std::time::{Duration, Instant};

use neuromorphic_drivers as nd;
use neuromorphic_drivers::types::Polarity;
use neuromorphic_drivers::Adapter;

use crate::bias::{BiasConfig, BiasController, BiasOverrides, BiasState, BiasValues};
use crate::{EventStream, EventStreamBuilder};

/// Device timestamps are microseconds, so one timestamp unit is `0.001` ms. Live streams carry
/// this scale, so `stream.numpy()[:, 2]` reads in microseconds just like a file-backed stream.
const TIMESTAMP_SCALE_MS: f64 = 0.001;

/// How long each [`Capture::poll`] blocks waiting for USB data before returning to the caller so it
/// can observe a stop request. Short enough to feel responsive; the driver's own ring absorbs the
/// events that arrive between polls.
const POLL_TIMEOUT: Duration = Duration::from_millis(5);

/// A camera discovered by [`list_cameras`].
#[derive(Clone, Debug, PartialEq, Eq)]
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

/// What enumeration found, and anything worth telling the user about how it went.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CameraListing {
    /// Connected, supported cameras. Empty is a normal answer, not a failure.
    pub cameras: Vec<CameraInfo>,
    /// Set only when enumeration was blocked by *permissions* rather than finding nothing — the
    /// case where a camera is probably attached and udev rules are the fix. Callers should surface
    /// this (as a warning, not an error): the list is still empty either way, but only here does
    /// the emptiness mean something is wrong.
    pub warning: Option<String>,
}

/// Enumerates every connected, supported event camera. The returned [`CameraInfo::serial`] can be
/// passed to [`Capture::open`] to select a specific device when several are plugged in.
///
/// **Never fails.** Asking what is attached is a question, and "nothing" is a valid answer — on a
/// machine with no camera, in a container without USB passthrough, or on a CI runner with no
/// `/dev/bus/usb` at all. Returning an error for those made a reasonable question fatal, and made
/// the behaviour differ by platform: the same call already returned an empty list on macOS.
///
/// A permissions failure is different in kind — something is there and we are not allowed to look —
/// so that comes back in [`CameraListing::warning`] rather than being silently indistinguishable.
pub fn list_cameras() -> CameraListing {
    let listed = match nd::list_devices() {
        Ok(listed) => listed,
        Err(error) => {
            return CameraListing {
                cameras: Vec::new(),
                warning: enumeration_warning(error),
            }
        }
    };
    CameraListing {
        cameras: listed
            .into_iter()
            .map(|device| CameraInfo {
                kind: device.device_type.to_string(),
                name: device.device_type.name().to_owned(),
                serial: device.serial.ok(),
                bus_number: device.bus_number,
                address: device.address,
                speed: speed_name(device.speed).to_owned(),
            })
            .collect(),
        warning: None,
    }
}

/// Decides whether an enumeration failure is worth reporting.
///
/// `Access` and `Busy` mean libusb saw a device and could not use it — on Linux that is almost
/// always missing udev rules, and the raw message ("Access denied") does not say so. Everything
/// else means there was nothing to enumerate in the first place: `Other` is what `libusb_init`
/// returns when the usbfs backend cannot start, which is the ordinary state of a VM or container
/// with no USB subsystem, and reporting it would cry wolf on every headless machine.
///
/// Split out from [`list_cameras`] so the decision is testable without a USB stack — it is the only
/// part of enumeration with a judgement in it.
fn enumeration_warning(error: nd::rusb::Error) -> Option<String> {
    match error {
        nd::rusb::Error::Access | nd::rusb::Error::Busy => Some(format!(
            "a USB event camera appears to be attached but could not be read ({error}). On Linux \
             this is usually missing udev rules for the device (see the neuromorphic-drivers \
             README, then re-plug it); on Windows, a missing WinUSB driver for it"
        )),
        _ => None,
    }
}

/// Turns a raw `nd::open` failure into an actionable message. The driver reports a device that is
/// present but can't be claimed (already open elsewhere, or missing USB permissions) the same way as
/// a truly absent one — "no device found". So when enumeration *can* still see a supported camera,
/// we say the device is present-but-unavailable rather than repeat the misleading "not found".
fn open_error(serial: Option<&str>, error: impl std::fmt::Display) -> String {
    let visible = list_cameras().cameras;
    let matched = match serial {
        Some(serial) => visible
            .iter()
            .find(|camera| camera.serial.as_deref() == Some(serial)),
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

/// The reference period the hardware event-rate controller averages over. The register field is 10
/// bits, so this is the largest round period that fits — long enough to smooth bursts, short enough
/// that the cap still tracks a changing scene.
const RATE_LIMIT_PERIOD_US: u16 = 1000;

/// Widest cap the rate controller's 22-bit budget register can express, per reference period.
const RATE_LIMIT_MAX_PER_PERIOD: u64 = (1 << 22) - 1;

/// Limits applied to the **sensor itself**, so it never emits more than the pipeline can consume.
///
/// Everything downstream of the USB cable — decoding, windowing, representations — costs time per
/// event, so the cheapest event is one the camera never sends. These are hardware features: the
/// Prophesee event-rate controller drops events on-chip, and the region masks stop pixels producing
/// them at all, neither of which costs the host anything.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Limits {
    /// Ceiling on events per second, enforced on-chip. `None` leaves the sensor unlimited.
    pub max_event_rate: Option<u64>,
    /// Only pixels inside `(x0, y0, width, height)` produce events. `None` uses the whole sensor.
    pub roi: Option<(usize, usize, usize, usize)>,
}

impl Limits {
    /// Whether anything is actually limited (an all-`None` set is left unapplied).
    fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// Where a [`Limits::roi`] rectangle ended up being enforced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoiPlacement {
    /// Blocked on-chip by the sensor's region masks — the pixels outside never produce an event, so
    /// they cost no USB bandwidth, no decoding, and nothing downstream.
    Hardware,
    /// Filtered on the host as events are decoded, because this sensor has no region masks. The
    /// events still cross the cable and are still decoded; only the saving downstream is real.
    Host,
}

/// How a [`Limits`] request is carried out on one particular sensor: what to configure it with at
/// open, and what has to be enforced on the host instead because the sensor cannot do it.
#[derive(Default)]
struct LimitsPlan {
    configuration: Option<nd::Configuration>,
    host_roi: Option<(usize, usize, usize, usize)>,
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

/// The adaptive-bias controller bound to an open device, with the clock that paces it.
///
/// The controller itself is device-agnostic and knows nothing about USB; this pairs it with the
/// wall clock so each decode pass can report how much time it covered.
struct AdaptiveBias {
    controller: BiasController,
    /// When the previous decode pass was reported.
    last_pass: Instant,
}

impl AdaptiveBias {
    /// Starts a controller from the biases `device` is already running — its stock configuration,
    /// unless something else has changed them since it was opened.
    ///
    /// Errors when the camera isn't one the controller can drive, rather than quietly doing
    /// nothing: a caller that asked for a steady event rate should not be left believing it got
    /// one. (Only the DAVIS346 is wired up so far.)
    fn new(overrides: BiasOverrides, device: &nd::Device) -> Result<Self, String> {
        // The defaults the request lands on are the sensor's own: rates scale with pixel count, and
        // the registers are a 2041-step current ladder on one sensor and plain bytes on the other.
        let (base, start) = match device.current_configuration() {
            nd::Configuration::InivationDavis346(configuration) => (
                BiasConfig::davis346(),
                davis346_biases(&configuration)?,
            ),
            nd::Configuration::PropheseeEvk4(configuration) => (
                BiasConfig::prophesee_evk4(),
                evk4_biases(&configuration),
            ),
            _ => {
                return Err(format!(
                    "adaptive_bias is not supported on the {} — it is implemented for the \
                     iniVation DAVIS346 and the Prophesee EVK4 so far",
                    device.name(),
                ))
            }
        };
        let config = overrides.apply(base);
        config.validate()?;
        Ok(Self {
            controller: BiasController::new(config, start),
            last_pass: Instant::now(),
        })
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
    /// Closed-loop bias control, when `stream(adaptive_bias=…)` asked for it.
    bias: Option<AdaptiveBias>,
    /// Region-of-interest mask (row-major `width·height`, `true` = keep), applied as events are
    /// decoded. `None` keeps everything. See [`set_mask`](Self::set_mask).
    mask: Option<Vec<bool>>,
    /// The [`Limits::roi`] rectangle this camera was opened with, and where it is enforced.
    roi: Option<((usize, usize, usize, usize), RoiPlacement)>,
    width: usize,
    height: usize,
    name: String,
    serial: String,
}

impl Capture {
    /// Opens a camera and begins streaming. `serial` selects a specific device (from
    /// [`CameraInfo::serial`]); `None` opens the first supported camera found. `window` sets how the
    /// stream is cut into [`CaptureWindow`]s, and `limits` caps what the sensor emits. `bias` turns
    /// on closed-loop [adaptive biasing](crate::bias), which holds the event rate steady across
    /// changing light by retuning the sensor's biases as it runs.
    ///
    /// Apart from `limits` the device is opened with its default configuration (biases, clock, …),
    /// and adaptive biasing starts from those stock biases. Errors — no camera found, permission
    /// denied (missing udev rules on Linux), an unreadable serial, or limits the device can't
    /// honour — are returned as a human-readable string.
    pub fn open(
        serial: Option<&str>,
        window: Window,
        limits: Limits,
        bias: Option<BiasOverrides>,
    ) -> Result<Self, String> {
        let (flag, event_loop) = nd::flag_and_event_loop().map_err(|error| error.to_string())?;
        // A limits request is device-specific, so the sensor has to be identified before it is
        // opened. Having enumerated it, open *that* device by bus address rather than "the first
        // one": with two cameras attached, the driver's own choice of first need not be ours, and a
        // configuration written for one sensor must never land on another.
        let (selector, plan) = if limits.is_empty() {
            let selector = match serial {
                Some(serial) => nd::SerialOrBusNumberAndAddress::Serial(serial),
                None => nd::SerialOrBusNumberAndAddress::None,
            };
            (selector, LimitsPlan::default())
        } else {
            let listed = select_device(serial)?;
            let plan = plan_limits(&listed, limits)?;
            let address = (listed.bus_number, listed.address);
            (
                nd::SerialOrBusNumberAndAddress::BusNumberAndAddress(address),
                plan,
            )
        };
        // Applied as part of the opening handshake: the driver writes the rate controller and the
        // region masks while configuring the sensor, and only re-writes the masks afterwards, so a
        // rate cap set after open would be silently ignored.
        let device = nd::open(
            selector,
            plan.configuration,
            None,
            event_loop.clone(),
            flag.clone(),
        )
        .map_err(|error| open_error(serial, error))?;
        let (width, height) = device_dimensions(&device);
        let name = device.name().to_owned();
        let serial = device.serial();
        let adapter = device.create_adapter();
        let bias = bias
            .map(|overrides| AdaptiveBias::new(overrides, &device))
            .transpose()?;
        let mut capture = Self {
            device,
            adapter,
            _event_loop: event_loop,
            flag,
            windower: Windower::new(width, height, window),
            bias,
            mask: None,
            roi: limits.roi.map(|rect| {
                let placement = match plan.host_roi {
                    Some(_) => RoiPlacement::Host,
                    None => RoiPlacement::Hardware,
                };
                (rect, placement)
            }),
            width,
            height,
            name,
            serial,
        };
        // A sensor without region masks gets the same rectangle as a host-side mask, so `roi=`
        // means the same thing everywhere — it just stops costing less than the whole sensor.
        if let Some((x0, y0, w, h)) = plan.host_roi {
            let rect = crate::mask::rect(width, height, x0 as f64, y0 as f64, w as f64, h as f64);
            capture.set_mask(Some(rect))?;
        }
        Ok(capture)
    }

    /// The `(x0, y0, width, height)` region the camera was opened with, and whether the sensor is
    /// enforcing it on-chip or eventcv is filtering it on the host. `None` when the whole sensor is
    /// in use.
    pub fn roi(&self) -> Option<((usize, usize, usize, usize), RoiPlacement)> {
        self.roi
    }

    /// A snapshot of the adaptive-bias controller, or `None` when the camera was opened without it.
    pub fn bias_state(&self) -> Option<BiasState> {
        self.bias.as_ref().map(|bias| bias.controller.state())
    }

    /// Reports one decode pass to the adaptive-bias controller and writes back whatever bias change
    /// it asks for. `undercounted` marks a pass that dropped buffers without decoding them, so
    /// `events` understates the rate.
    ///
    /// Called on **every** pass, including ones that decoded nothing — a scene too dark to trigger
    /// an event is exactly when the controller has to open the sensor up, and it can only do that
    /// if the empty passes keep its clock running.
    fn tick_bias(&mut self, events: u64, undercounted: bool) {
        let update = {
            let Some(bias) = self.bias.as_mut() else {
                return;
            };
            let now = Instant::now();
            let elapsed = now.duration_since(bias.last_pass);
            bias.last_pass = now;
            if undercounted {
                bias.controller.invalidate();
            }
            bias.controller.observe(events, elapsed)
        };
        let Some(values) = update else {
            return;
        };
        // Only unreachable failures are possible here: the device type and the bias field names
        // were both proven when the controller was built. Dropping an update would at worst leave
        // the biases as they are, which is far better than ending a live capture.
        let _ = self.apply_biases(values);
    }

    /// Hands a new bias set to the driver. This does not touch the wire: the driver diffs it
    /// against the current configuration and writes only the changed registers, on its own thread.
    fn apply_biases(&mut self, values: BiasValues) -> Result<(), String> {
        let updated = match self.device.current_configuration() {
            nd::Configuration::InivationDavis346(configuration) => {
                nd::Configuration::InivationDavis346(davis346_with_biases(&configuration, values)?)
            }
            nd::Configuration::PropheseeEvk4(configuration) => {
                nd::Configuration::PropheseeEvk4(evk4_with_biases(&configuration, values))
            }
            // Unreachable: the device type was checked when the controller was built.
            _ => return Err("adaptive biasing is not implemented for this camera".to_owned()),
        };
        self.device
            .update_configuration(updated)
            .map_err(|error| error.to_string())
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

    /// The region-of-interest mask events are filtered through, if any.
    pub fn mask(&self) -> Option<&[bool]> {
        self.mask.as_deref()
    }

    /// Sets (or clears, with `None`) the region-of-interest mask: a row-major `width·height` grid
    /// where `true` keeps the pixel. Events outside it are dropped **as they are decoded**, before
    /// windowing, so one mask covers everything the camera feeds — the windows a loop reads, the
    /// file a recording writes, and the live view — and the masked pixels cost nothing downstream.
    ///
    /// Unlike [`Limits::roi`] this is enforced on the host, so it works on any camera and takes any
    /// shape. A mask that doesn't match the sensor grid is rejected.
    pub fn set_mask(&mut self, mask: Option<Vec<bool>>) -> Result<(), String> {
        if let Some(mask) = &mask {
            if mask.len() != self.width * self.height {
                return Err(format!(
                    "mask has {} pixels, expected {} for this {}x{} sensor",
                    mask.len(),
                    self.width * self.height,
                    self.width,
                    self.height
                ));
            }
        }
        self.mask = mask;
        Ok(())
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

        let mut events = 0;
        if let Some(view) = self.device.next_with_timeout(&POLL_TIMEOUT) {
            if view.first_after_overflow {
                self.windower.mark_overflow();
            }
            let slice = view.slice;
            // Disjoint field borrows: `slice` borrows `self.device`, the closure borrows
            // `self.windower`, and `convert_buffer` borrows `self.adapter`.
            let windower = &mut self.windower;
            convert_buffer(
                &mut self.adapter,
                slice,
                self.mask.as_deref(),
                self.width,
                |x, y, t, polarity| {
                    events += 1;
                    windower.push(x, y, t, polarity)
                },
            );
            // Flush the current window if its boundary has passed, so a mid-window pause doesn't
            // stall delivery. `current_t` advances with the device clock even during quiet periods.
            let current_t = self.adapter.current_t() as i64;
            self.windower.seal_until(current_t);
        }
        self.tick_bias(events, false);
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
        let mut events = 0;
        // Scoped so the buffer view — which borrows the device and frees its ring slot when
        // dropped — is released before the bias controller is handed the device below.
        let decoded = if let Some(view) = self.device.next_with_timeout(&timeout) {
            if view.first_after_overflow {
                self.windower.mark_overflow();
            }
            let slice = view.slice;
            let windower = &mut self.windower;
            convert_buffer(
                &mut self.adapter,
                slice,
                self.mask.as_deref(),
                self.width,
                |x, y, t, polarity| {
                    events += 1;
                    windower.push(x, y, t, polarity)
                },
            );
            true
        } else {
            false
        };
        // Whether or not anything arrived, the device clock moved: flush a window whose boundary
        // has passed so a mid-window pause doesn't stall delivery.
        let current_t = self.adapter.current_t() as i64;
        self.windower.seal_until(current_t);
        self.tick_bias(events, false);
        Ok(decoded)
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
        let mut events = 0;
        while let Some(view) = self.device.next_with_timeout(&Duration::ZERO) {
            if view.first_after_overflow {
                overflow = true;
            }
            convert_buffer(
                &mut self.adapter,
                view.slice,
                self.mask.as_deref(),
                self.width,
                |x, y, t, polarity| {
                    events += 1;
                    on_event(x, y, t, polarity)
                },
            );
        }
        self.tick_bias(events, false);
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
        let start = Instant::now();
        let mut overflow = false;
        let mut decoding = true;
        let mut events = 0;
        let mut dropped = false;
        while let Some(view) = self.device.next_with_timeout(&Duration::ZERO) {
            if view.first_after_overflow {
                overflow = true;
            }
            if !decoding {
                // Over budget: drop the buffer, freeing its ring slot without decoding.
                dropped = true;
                continue;
            }
            convert_buffer(
                &mut self.adapter,
                view.slice,
                self.mask.as_deref(),
                self.width,
                |x, y, t, polarity| {
                    events += 1;
                    on_event(x, y, t, polarity)
                },
            );
            if start.elapsed() >= budget {
                decoding = false;
            }
        }
        // Dropped buffers make `events` a lower bound, and a bias controller that mistook the
        // undercount for a quiet scene would open the sensor up and deepen the overload.
        self.tick_bias(events, dropped);
        Ok(overflow)
    }
}

/// Picks the device a [`Limits`] request is being planned for: the one matching `serial`, or the
/// first camera found. Enumeration is the only way to learn a device's type before opening it, and
/// the configuration variant has to match that type.
fn select_device(serial: Option<&str>) -> Result<nd::devices::ListedDevice, String> {
    let listed = nd::list_devices().map_err(|error| error.to_string())?;
    match serial {
        Some(serial) => listed
            .into_iter()
            .find(|device| device.serial.as_deref().ok() == Some(serial))
            .ok_or_else(|| format!("no camera with serial {serial} found")),
        None => listed
            .into_iter()
            .next()
            .ok_or_else(|| "no event camera found".to_owned()),
    }
}

/// Works out how `limits` can be met on `device`: what to open the sensor with, and what eventcv
/// has to do itself.
///
/// Gated on what the sensor *can do*, not on which model it is. The rate controller and the region
/// masks are separate capabilities — the three sensors built on Prophesee's pipeline (EVK4, EVK3
/// HD, CenturyArks VGA) have both; the iniVation cameras have neither. A region of interest they
/// can't block on-chip is still worth honouring on the host, so it comes back as `host_roi` rather
/// than an error. A rate cap has no host equivalent — dropping events after they were decoded saves
/// nothing, and pretending otherwise would misrepresent the source — so that stays an error.
fn plan_limits(device: &nd::devices::ListedDevice, limits: Limits) -> Result<LimitsPlan, String> {
    macro_rules! regions {
        ($module:ident, $variant:ident) => {{
            use nd::devices::$module::{RateLimiter, DEFAULT_CONFIGURATION, PROPERTIES};
            let mut configuration = DEFAULT_CONFIGURATION;
            configuration.rate_limiter = rate_limiter(limits.max_event_rate).map(
                |(reference_period_us, maximum_events_per_period)| RateLimiter {
                    reference_period_us,
                    maximum_events_per_period,
                },
            );
            if let Some(roi) = limits.roi {
                let (x_mask, y_mask) =
                    region_masks(roi, PROPERTIES.width as usize, PROPERTIES.height as usize)?;
                configuration.x_mask = x_mask;
                configuration.y_mask = y_mask;
                configuration.mask_intersection_only = false;
            }
            LimitsPlan {
                configuration: Some(nd::Configuration::$variant(configuration)),
                host_roi: None,
            }
        }};
    }
    Ok(match device.device_type {
        nd::Type::PropheseeEvk4 => regions!(prophesee_evk4, PropheseeEvk4),
        nd::Type::PropheseeEvk3Hd => regions!(prophesee_evk3_hd, PropheseeEvk3Hd),
        nd::Type::CenturyarksVga => regions!(centuryarks_vga, CenturyarksVga),
        other => {
            if limits.max_event_rate.is_some() {
                return Err(format!(
                    "{} has no on-chip event-rate controller — max_event_rate is a sensor feature, \
                     and capping the rate on the host would not save any of the work it exists to \
                     avoid. It is supported on the Prophesee EVK4 and EVK3 HD and the CenturyArks \
                     VGA",
                    other.name(),
                ));
            }
            LimitsPlan {
                configuration: None,
                host_roi: limits.roi,
            }
        }
    })
}

/// Converts an events-per-second ceiling into the rate controller's `(reference period, events
/// allowed per period)` pair. `None` (no ceiling) turns the controller off. A cap below one event
/// per period still admits one, since the register cannot express "none".
fn rate_limiter(max_event_rate: Option<u64>) -> Option<(u16, u32)> {
    let rate = max_event_rate?;
    let per_period = (rate * RATE_LIMIT_PERIOD_US as u64 / 1_000_000)
        .clamp(1, RATE_LIMIT_MAX_PER_PERIOD) as u32;
    Some((RATE_LIMIT_PERIOD_US, per_period))
}

/// Builds the column and row masks that leave only `(x0, y0, width, height)` producing events.
///
/// The masks are bitmaps over sensor columns and rows, and are applied as regions of *non*-interest
/// (the driver's `mask_intersection_only: false`), so a set bit blocks that column or row: we set
/// every bit outside the rectangle.
///
/// Columns map straight across — bit `i` is column `i` — but **rows are stored bottom-up**: bit `i`
/// is row `height - 1 - i`. Verified on an EVK4, where a top-left `(0, 0, 200, 200)` region built
/// row-major came back as rows 520..719, the exact mirror. A vertically symmetric rectangle looks
/// correct either way, so this is easy to miss.
///
/// The bitmap lengths are the driver's, and differ per sensor (20×12 words for the 1280×720
/// Prophesee sensors, 10×8 for the 640×480 CenturyArks), so they come in as const parameters.
fn region_masks<const NX: usize, const NY: usize>(
    roi: (usize, usize, usize, usize),
    sensor_width: usize,
    sensor_height: usize,
) -> Result<([u64; NX], [u64; NY]), String> {
    let (x0, y0, width, height) = roi;
    if width == 0 || height == 0 {
        return Err("roi width and height must be at least 1".to_owned());
    }
    let (x1, y1) = (x0 + width, y0 + height);
    if x1 > sensor_width || y1 > sensor_height {
        return Err(format!(
            "roi ({x0}, {y0}, {width}, {height}) reaches ({x1}, {y1}), outside the \
             {sensor_width}x{sensor_height} sensor"
        ));
    }
    let mut x_mask = [0_u64; NX];
    let mut y_mask = [0_u64; NY];
    for column in (0..sensor_width).filter(|column| !(x0..x1).contains(column)) {
        x_mask[column / 64] |= 1 << (column % 64);
    }
    for row in (0..sensor_height).filter(|row| !(y0..y1).contains(row)) {
        let bit = sensor_height - 1 - row;
        y_mask[bit / 64] |= 1 << (bit % 64);
    }
    Ok((x_mask, y_mask))
}

/// The DAVIS346 bias registers the controller drives, named as the driver's `Biases` struct spells
/// them. Their values are indices into the sensor's monotonic coarse/fine current ladder, which is
/// the space [`BiasController`] already works in, so no conversion is needed either way.
///
/// The driver keeps these fields private and offers no accessors, so the `Serialize`/`Deserialize`
/// it derives on `Biases` is the only way to read and rewrite them. Both helpers below go through
/// a `serde_json::Value`; they run at the control period (a few times a second at most), never per
/// event. Once the fields are public upstream, both collapse to field access.
/// Listed in [`BiasValues`] field order: refractory, photoreceptor, follower, ON, OFF.
const DAVIS346_BIASES: [&str; 5] = ["refrbp", "prbp", "prsfbp", "onbn", "offbn"];

/// Reads the controlled biases out of a DAVIS346 configuration.
fn davis346_biases(
    configuration: &nd::devices::inivation_davis346::Configuration,
) -> Result<BiasValues, String> {
    let value = serde_json::to_value(&configuration.biases)
        .map_err(|error| format!("could not read the camera's biases ({error})"))?;
    let read = |field: &str| {
        value
            .get(field)
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| u16::try_from(value).ok())
            .ok_or_else(|| format!("the camera's biases have no readable {field}"))
    };
    Ok(BiasValues {
        refractory: read(DAVIS346_BIASES[0])?,
        photoreceptor: read(DAVIS346_BIASES[1])?,
        follower: read(DAVIS346_BIASES[2])?,
        on_threshold: read(DAVIS346_BIASES[3])?,
        off_threshold: read(DAVIS346_BIASES[4])?,
    })
}

/// Returns `configuration` with the controlled biases replaced by `values` and everything else —
/// the other seventeen biases, the ROI, the filters, the APS timings — left exactly as it was.
fn davis346_with_biases(
    configuration: &nd::devices::inivation_davis346::Configuration,
    values: BiasValues,
) -> Result<nd::devices::inivation_davis346::Configuration, String> {
    let mut biases = serde_json::to_value(&configuration.biases)
        .map_err(|error| format!("could not read the camera's biases ({error})"))?;
    let written = [
        values.refractory,
        values.photoreceptor,
        values.follower,
        values.on_threshold,
        values.off_threshold,
    ];
    for (field, value) in DAVIS346_BIASES.iter().zip(written) {
        biases[field] = serde_json::Value::from(value);
    }
    let mut configuration = configuration.clone();
    configuration.biases = serde_json::from_value(biases)
        .map_err(|error| format!("could not apply the new biases ({error})"))?;
    Ok(configuration)
}

/// Reads the controlled biases out of a Prophesee EVK4 configuration.
///
/// The five map one-to-one onto the DAVIS346's, with the same directions, so the control law needs
/// no per-sensor special casing: `pr`/`fo` are the photoreceptor and its follower, `diff_on` and
/// `diff_off` sit either side of `diff` as the ON and OFF contrast thresholds, and `refr` sets the
/// refractory period. `hpf` is left alone — its high-pass cutoff removes slow signal rather than
/// scaling sensitivity, so it is not a lever the rate controller should be pulling.
///
/// Unlike the DAVIS346's these fields are public, so this is plain field access. Values are `u8`
/// registers widened into the controller's `u16` ladder space.
fn evk4_biases(configuration: &nd::devices::prophesee_evk4::Configuration) -> BiasValues {
    let biases = &configuration.biases;
    BiasValues {
        refractory: biases.refr.into(),
        photoreceptor: biases.pr.into(),
        follower: biases.fo.into(),
        on_threshold: biases.diff_on.into(),
        off_threshold: biases.diff_off.into(),
    }
}

/// Returns `configuration` with the controlled biases replaced by `values`, everything else — the
/// other biases, the masks, the rate limiter, the clock — left exactly as it was.
fn evk4_with_biases(
    configuration: &nd::devices::prophesee_evk4::Configuration,
    values: BiasValues,
) -> nd::devices::prophesee_evk4::Configuration {
    let mut configuration = configuration.clone();
    // Saturating: the controller's ladder is `u16`, these registers are `u8`. `BiasConfig::limits`
    // already holds the values in range, so this only guards a caller that widened them.
    let byte = |value: u16| value.min(u16::from(u8::MAX)) as u8;
    configuration.biases.refr = byte(values.refractory);
    configuration.biases.pr = byte(values.photoreceptor);
    configuration.biases.fo = byte(values.follower);
    configuration.biases.diff_on = byte(values.on_threshold);
    configuration.biases.diff_off = byte(values.off_threshold);
    configuration
}

/// Decodes one USB buffer with whichever adapter the open device uses, forwarding each polarity
/// event as `(x, y, t_us, positive)` and discarding the IMU/trigger/APS streams. The single place
/// the per-device adapter variants are matched, shared by every decode path above.
///
/// Events outside `mask` (row-major `width`-wide, `true` = keep) are dropped here rather than
/// downstream, so the one filter covers windowing, recording, and the live view — and the callers'
/// event counters, which pace the bias controller, track the events actually kept.
fn convert_buffer(
    adapter: &mut Adapter,
    slice: &[u8],
    mask: Option<&[bool]>,
    width: usize,
    mut on_event: impl FnMut(u16, u16, i64, bool),
) {
    let handle = |event: nd::types::PolarityEvent<u64, u16, u16>| {
        if let Some(mask) = mask {
            if mask.get(event.y as usize * width + event.x as usize) != Some(&true) {
                return;
            }
        }
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
    use super::{rate_limiter, region_masks, Window, Windower};

    #[test]
    fn rate_limiter_converts_events_per_second_to_a_period_budget() {
        // 50 Mev/s over a 1 ms reference period is 50k events per period.
        assert_eq!(rate_limiter(Some(50_000_000)), Some((1000, 50_000)));
        assert_eq!(rate_limiter(None), None);
        // Below one event per period the register still has to admit one.
        assert_eq!(rate_limiter(Some(100)), Some((1000, 1)));
        // Beyond the 22-bit budget the cap saturates rather than wrapping.
        assert_eq!(
            rate_limiter(Some(u64::MAX / 1000)),
            Some((1000, (1 << 22) - 1))
        );
    }

    #[test]
    fn region_masks_block_everything_outside_the_rectangle() {
        let (x_mask, y_mask) = region_masks::<20, 12>((100, 50, 200, 100), 1280, 720).unwrap();
        let bit = |mask: &[u64], index: usize| mask[index / 64] & (1 << (index % 64)) != 0;
        // Columns map straight across; rows are stored bottom-up.
        let column_blocked = |column: usize| bit(&x_mask, column);
        let row_blocked = |row: usize| bit(&y_mask, 720 - 1 - row);

        assert!(column_blocked(99) && column_blocked(300)); // just outside
        assert!(!column_blocked(100) && !column_blocked(299)); // the edges are kept
        assert!(row_blocked(49) && row_blocked(150));
        assert!(!row_blocked(50) && !row_blocked(149));
        // Bits past the sensor height belong to no row, so they stay clear.
        assert!(!bit(&y_mask, 720));
    }

    #[test]
    fn region_masks_store_rows_bottom_up() {
        // A top-left region: rows 0..199 kept. Stored bottom-up, the *kept* bits are the top ones.
        let (_, y_mask) = region_masks::<20, 12>((0, 0, 200, 200), 1280, 720).unwrap();
        let bit = |index: usize| y_mask[index / 64] & (1 << (index % 64)) != 0;
        assert!(
            bit(0) && bit(519),
            "rows 200..719 must be blocked, low bits set"
        );
        assert!(
            !bit(520) && !bit(719),
            "rows 0..199 must be kept, high bits clear"
        );
    }

    #[test]
    fn region_masks_fit_the_centuryarks_bitmaps() {
        // The CenturyArks VGA has the same on-chip region masks as the Prophesee sensors, but its
        // bitmaps are 10x8 words for a 640x480 grid rather than 20x12 for 1280x720.
        let (x_mask, y_mask) = region_masks::<10, 8>((320, 240, 320, 240), 640, 480).unwrap();
        assert_eq!((x_mask.len(), y_mask.len()), (10, 8));
        let bit = |mask: &[u64], index: usize| mask[index / 64] & (1 << (index % 64)) != 0;
        assert!(bit(&x_mask, 319) && !bit(&x_mask, 320)); // columns left of the rectangle blocked
        assert!(bit(&y_mask, 480 - 1 - 239) && !bit(&y_mask, 480 - 1 - 240)); // rows, bottom-up
                                                                              // The row bitmap has spare bits past the sensor (8 words = 512 bits for 480 rows); they
                                                                              // belong to no row, so they stay clear. The column bitmap is exactly 640 bits, with none.
        assert!(!bit(&y_mask, 480));
    }

    #[test]
    fn region_masks_reject_a_rectangle_off_the_sensor() {
        assert!(region_masks::<20, 12>((0, 0, 1280, 720), 1280, 720).is_ok());
        assert!(region_masks::<20, 12>((1, 0, 1280, 720), 1280, 720).is_err());
        assert!(region_masks::<20, 12>((0, 0, 0, 100), 1280, 720).is_err());
        // The same rectangle is off a 640x480 sensor, so the check is per sensor, not per model.
        assert!(region_masks::<10, 8>((0, 0, 1280, 720), 640, 480).is_err());
    }

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

    #[test]
    fn only_permission_failures_warn() {
        use neuromorphic_drivers::rusb::Error;

        // A device is there and we are not allowed to touch it — the genuine udev case, and the
        // only one where an empty list means something is wrong.
        for error in [Error::Access, Error::Busy] {
            let warning = super::enumeration_warning(error).expect("should warn");
            assert!(warning.contains("udev"), "{warning}");
            assert!(warning.contains("attached"), "{warning}");
        }
    }

    #[test]
    fn an_absent_usb_stack_is_silent() {
        use neuromorphic_drivers::rusb::Error;

        // `Other` is what `libusb_init` returns when the usbfs backend cannot start — the ordinary
        // state of a VM, a container without USB passthrough, or a CI runner. Warning about it
        // would cry wolf on every headless machine, which is exactly what broke CI.
        for error in [
            Error::Other,
            Error::NoDevice,
            Error::NotFound,
            Error::NotSupported,
            Error::Io,
        ] {
            assert_eq!(
                super::enumeration_warning(error),
                None,
                "{error:?} should be silent"
            );
        }
    }

    #[test]
    fn enumeration_never_fails() {
        // The contract the docstring always promised and the implementation did not keep. This runs
        // on any machine — with a camera, without one, or with no USB subsystem at all — and must
        // return a list every time.
        let listing = super::list_cameras();
        // A warning is only legitimate alongside an empty list: if we could see cameras, nothing
        // was blocked.
        if listing.warning.is_some() {
            assert!(listing.cameras.is_empty());
        }
    }
}
