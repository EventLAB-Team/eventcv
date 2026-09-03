//! Background capture pump — the thread that keeps a live camera decoded.
//!
//! `Capture::poll` decodes on the *calling* thread, so nothing drained the driver's ring while
//! Python worked on the previous window. On a Prophesee EVK4 that put a hard ceiling on a `read()`
//! loop: a 50 ms-window loop doing one NumPy conversion per frame ran at 7 fps with the ring pegged
//! full, and the driver dropped events for the whole session.
//!
//! The pump moves decoding — and, when `stream(record=…)` is set, writing to disk — onto a thread
//! that owns the [`Capture`]. The ring then drains continuously whatever the Python loop is doing,
//! and the loop only collects windows that are already decoded. What happens to windows the
//! consumer hasn't collected is the caller's choice: [`Backpressure::Buffer`] keeps them in order
//! and pauses the pump when the queue fills, [`Backpressure::Latest`] keeps only the newest and
//! counts the rest as skipped.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use eventcv_core::bias::BiasState;
use eventcv_core::device::{Capture, CaptureWindow};

use crate::Recorder;

/// How the pump treats finished windows the consumer has not collected yet.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Backpressure {
    /// Keep every window, in order, pausing the pump while the queue is full (`stream()` default).
    Buffer,
    /// Keep only the newest window, counting the ones it overtook (`stream(latest=True)`).
    Latest,
}

/// Decoded windows the queue holds before [`Backpressure::Buffer`] pauses the pump. Deep enough to
/// ride out a burst (a quarter-second at a typical `dt_ms`), shallow enough to bound memory at the
/// millions-of-events-per-window rates a 1280×720 sensor reaches.
const QUEUE_DEPTH: usize = 8;

/// A second cap on the same queue, by events rather than windows: eight windows is nothing at rest
/// but hundreds of megabytes mid-burst, when a single window can hold millions of events. Whichever
/// limit is reached first pauses the pump. Roughly 100 MB of decoded columns.
const QUEUE_EVENT_BUDGET: usize = 8_000_000;

/// How long the pump waits on the driver for a buffer before looping to re-check the stop flag.
const DECODE_TIMEOUT: Duration = Duration::from_millis(5);

/// How long the pump naps when the driver ring is momentarily empty, so an idle scene doesn't peg a
/// core. Also the slice a full-queue wait blocks for before re-checking the stop flag.
const IDLE_NAP: Duration = Duration::from_millis(1);

/// State shared between the pump thread and the Python thread.
struct Shared {
    queue: Mutex<Queue>,
    /// Signals both directions: a window arrived, or the consumer freed space / asked to stop.
    signal: Condvar,
    stop: AtomicBool,
    /// Driver ring buffers waiting, republished by the pump each pass.
    backlog: AtomicUsize,
    /// The adaptive-bias controller's state, republished by the pump each pass so Python can read
    /// it without stopping the thread that owns the camera. `None` when biasing is off.
    bias: Mutex<Option<BiasState>>,
    /// Events written to the `record=` file so far.
    recorded: AtomicUsize,
    /// Windows that arrived first-after-overflow — i.e. the driver dropped events before them.
    overflows: AtomicUsize,
    /// Events discarded on purpose in [`Backpressure::Latest`] mode, decoded but never built into a
    /// window. Counted exactly, because a drop policy nobody can quantify is a bias on everything
    /// downstream of it.
    skimmed: AtomicUsize,
}

#[derive(Default)]
struct Queue {
    windows: VecDeque<CaptureWindow>,
    /// Events held across `windows`, tracked so the queue can be capped by memory as well as depth.
    queued_events: usize,
    /// Windows dropped un-collected in [`Backpressure::Latest`] mode.
    skipped: usize,
    /// A device or write error that ended the pump; taken by the next read.
    error: Option<String>,
}

/// A running capture thread. Dropping or [`stop`](Self::stop)ping it returns the camera.
pub(crate) struct Pump {
    shared: Arc<Shared>,
    handle: Option<JoinHandle<(Capture, Option<Recorder>)>>,
}

impl Pump {
    /// Spawns the pump thread and starts decoding immediately.
    pub(crate) fn start(capture: Capture, recorder: Option<Recorder>, mode: Backpressure) -> Self {
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue::default()),
            signal: Condvar::new(),
            stop: AtomicBool::new(false),
            backlog: AtomicUsize::new(0),
            bias: Mutex::new(capture.bias_state()),
            recorded: AtomicUsize::new(recorder.as_ref().map_or(0, Recorder::n_events)),
            overflows: AtomicUsize::new(0),
            skimmed: AtomicUsize::new(0),
        });
        let worker = Arc::clone(&shared);
        // The camera and the open recording are owned *outside* the unwind boundary so a panic in
        // the decode loop still hands them back: the file gets flushed and closed, and the USB
        // device released, instead of both being lost with the thread.
        let handle = std::thread::spawn(move || {
            let mut capture = capture;
            let mut recorder = recorder;
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run(&mut capture, &mut recorder, &worker, mode)
            }));
            if let Err(panic) = outcome {
                fail(
                    &worker,
                    format!("capture thread panicked: {}", panic_message(&panic)),
                );
            }
            (capture, recorder)
        });
        Self {
            shared,
            handle: Some(handle),
        }
    }

    /// Takes the next decoded window, waiting up to `wait` for one to arrive. `Ok(None)` means the
    /// wait elapsed — the caller can check its own deadline and Ctrl+C, then ask again.
    pub(crate) fn next_window(&self, wait: Duration) -> Result<Option<CaptureWindow>, String> {
        let mut queue = self.shared.queue.lock().unwrap();
        if queue.windows.is_empty() && queue.error.is_none() {
            queue = self.shared.signal.wait_timeout(queue, wait).unwrap().0;
        }
        if let Some(window) = queue.windows.pop_front() {
            queue.queued_events = queue.queued_events.saturating_sub(window.stream.len());
            // Wake a Buffer-mode pump that paused on a full queue.
            self.shared.signal.notify_all();
            return Ok(Some(window));
        }
        match queue.error.take() {
            Some(message) => Err(message),
            None => Ok(None),
        }
    }

    pub(crate) fn backlog(&self) -> usize {
        self.shared.backlog.load(Ordering::Relaxed)
    }

    pub(crate) fn bias_state(&self) -> Option<BiasState> {
        *self.shared.bias.lock().unwrap()
    }

    pub(crate) fn n_recorded(&self) -> usize {
        self.shared.recorded.load(Ordering::Relaxed)
    }

    pub(crate) fn n_overflows(&self) -> usize {
        self.shared.overflows.load(Ordering::Relaxed)
    }

    pub(crate) fn n_skipped(&self) -> usize {
        self.shared.queue.lock().unwrap().skipped
    }

    pub(crate) fn n_skimmed_events(&self) -> usize {
        self.shared.skimmed.load(Ordering::Relaxed)
    }

    /// Stops the thread and hands back the camera (and the open recording, if any). A panic inside
    /// the decode loop is caught and republished as an error, so both still come back; `None` means
    /// the thread died in a way even that could not recover from.
    pub(crate) fn stop(&mut self) -> (Option<Capture>, Option<Recorder>) {
        self.shared.stop.store(true, Ordering::Release);
        self.shared.signal.notify_all();
        match self.handle.take() {
            Some(handle) => match handle.join() {
                Ok((capture, recorder)) => (Some(capture), recorder),
                Err(_) => (None, None),
            },
            None => (None, None),
        }
    }
}

/// Stops the thread if the camera was dropped without `close()` — otherwise the pump would keep
/// decoding forever and hold the USB claim, so the next `stream()` could not open the device.
impl Drop for Pump {
    fn drop(&mut self) {
        self.stop();
    }
}

/// The panic payload's message, for the two shapes `panic!` produces.
fn panic_message(panic: &Box<dyn std::any::Any + Send>) -> &str {
    match panic.downcast_ref::<&'static str>() {
        Some(message) => message,
        None => match panic.downcast_ref::<String>() {
            Some(message) => message.as_str(),
            None => "unknown panic",
        },
    }
}

/// The pump thread: drain the ring into windows, archive them, hand them on — until asked to stop.
fn run(
    capture: &mut Capture,
    recorder: &mut Option<Recorder>,
    shared: &Shared,
    mode: Backpressure,
) {
    'pump: while !shared.stop.load(Ordering::Acquire) {
        // In `Latest` mode the consumer has said it only wants the newest window, so anything
        // queued behind the freshest buffer is going to be discarded by `offer` anyway. Skimming
        // those buffers — decoding their words for the stream state, discarding their events —
        // reaches the same answer without building the columns first, and it keeps the driver's
        // ring drained. That second part is the one that matters: a full ring back-pressures into
        // the sensor, where events are lost with nothing to count them, so the cheapest drop is
        // also the only one that stays visible.
        // Not while a recording is open. `record=` promises the file gets every window even in
        // `Latest` mode --- that is the whole point of archiving on this thread rather than the
        // consumer's --- and a skimmed buffer's events are gone before the recorder could see
        // them. Someone who asked for a complete recording asked for complete decoding with it.
        if matches!(mode, Backpressure::Latest) && recorder.is_none() {
            while capture.backlog() > 1 {
                match capture.skim_next(Duration::ZERO) {
                    Ok(Some(events)) => {
                        shared.skimmed.fetch_add(events, Ordering::Relaxed);
                    }
                    Ok(None) => break,
                    Err(message) => {
                        fail(shared, message);
                        break 'pump;
                    }
                }
            }
        }
        // One buffer per pass, its windows handed on before the next: `offer` is what throttles
        // decoding when the consumer stalls, so in `Buffer` mode the backlog waits in the driver's
        // ring rather than ballooning into decoded columns here.
        let decoded = match capture.decode_next(DECODE_TIMEOUT) {
            Ok(decoded) => decoded,
            Err(message) => {
                fail(shared, message);
                break;
            }
        };
        shared.backlog.store(capture.backlog(), Ordering::Relaxed);
        if let Some(state) = capture.bias_state() {
            *shared.bias.lock().unwrap() = Some(state);
        }
        while let Some(window) = capture.take_pending() {
            if window.first_after_overflow {
                shared.overflows.fetch_add(1, Ordering::Relaxed);
            }
            // Archived here rather than on the consumer's thread, so the recording stays complete
            // even in `Latest` mode where most windows never reach Python.
            if let Some(recorder) = recorder.as_mut() {
                if let Err(error) = recorder.append(&window.stream) {
                    fail(shared, error.to_string());
                    break 'pump;
                }
                shared
                    .recorded
                    .store(recorder.n_events(), Ordering::Relaxed);
            }
            if !offer(shared, window, mode) {
                break 'pump;
            }
        }
        if !decoded {
            std::thread::park_timeout(IDLE_NAP);
        }
    }
}

/// Queues one finished window under the chosen policy. Returns `false` when the pump should stop.
fn offer(shared: &Shared, window: CaptureWindow, mode: Backpressure) -> bool {
    let mut queue = shared.queue.lock().unwrap();
    match mode {
        Backpressure::Latest => {
            // Anything the consumer hasn't taken is already stale, so the newest window replaces it.
            queue.skipped += queue.windows.len();
            queue.windows.clear();
            queue.queued_events = 0;
        }
        Backpressure::Buffer => {
            while queue.windows.len() >= QUEUE_DEPTH || queue.queued_events >= QUEUE_EVENT_BUDGET {
                if shared.stop.load(Ordering::Acquire) {
                    return false;
                }
                queue = shared.signal.wait_timeout(queue, IDLE_NAP).unwrap().0;
            }
        }
    }
    queue.queued_events += window.stream.len();
    queue.windows.push_back(window);
    shared.signal.notify_all();
    true
}

/// Publishes the error that ended the pump; the next read raises it.
fn fail(shared: &Shared, message: String) {
    let mut queue = shared.queue.lock().unwrap();
    queue.error = Some(message);
    shared.signal.notify_all();
}

#[cfg(test)]
mod tests {
    use super::panic_message;

    #[test]
    fn panic_message_reads_both_payload_shapes() {
        let literal = std::panic::catch_unwind(|| panic!("static message")).unwrap_err();
        assert_eq!(panic_message(&literal), "static message");
        let formatted = std::panic::catch_unwind(|| panic!("formatted {}", 1 + 1)).unwrap_err();
        assert_eq!(panic_message(&formatted), "formatted 2");
        let payload: Box<dyn std::any::Any + Send> = Box::new(7_u8);
        assert_eq!(panic_message(&payload), "unknown panic");
    }
}
