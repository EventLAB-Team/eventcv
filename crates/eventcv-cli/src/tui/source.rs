//! Where the TUI's events come from: a recording played back against the wall clock, or a live
//! camera drained once per screen refresh.

use std::time::Duration;

use eventcv_core::io::Reader;
use eventcv_core::EventStream;

use crate::Failure;

/// A playhead over a recording: `cursor_us` is the end of the window on screen, measured from the
/// first event, and advances at `speed` times the wall clock while playing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Playback {
    pub cursor_us: i64,
    pub duration_us: i64,
    pub speed: f64,
    pub playing: bool,
}

impl Playback {
    pub fn advance(&mut self, elapsed: Duration) {
        if !self.playing {
            return;
        }
        let step = (elapsed.as_secs_f64() * 1e6 * self.speed).round() as i64;
        self.cursor_us = self.cursor_us.saturating_add(step).min(self.duration_us);
        // Stop at the end rather than loop, so the last frame stays up to be looked at.
        self.playing = self.cursor_us < self.duration_us;
    }

    pub fn seek(&mut self, delta_us: i64) {
        self.cursor_us = self
            .cursor_us
            .saturating_add(delta_us)
            .clamp(0, self.duration_us);
    }
}

/// What one step read: the events new since the last step, and the whole window ending now.
pub(super) struct Step {
    pub fresh: EventStream,
    pub window: EventStream,
}

pub(super) struct FileFeed {
    pub reader: Reader,
    pub start_us: i64,
    pub playback: Playback,
    /// The window on screen and where it ended, so continuous play reads only the new events.
    last: Option<(i64, EventStream)>,
}

impl FileFeed {
    pub fn new(reader: Reader, speed: f64, dt_us: i64) -> Self {
        let (lo, hi) = reader.time_span();
        // +1 so the playhead can reach past the last event and show it.
        let duration_us = if reader.n_events() == 0 {
            0
        } else {
            hi - lo + 1
        };
        Self {
            reader,
            start_us: lo,
            playback: Playback {
                cursor_us: dt_us.min(duration_us),
                duration_us,
                speed,
                playing: true,
            },
            last: None,
        }
    }

    /// Forgets the window on screen, so the next [`step`](Self::step) reads afresh — after a seek
    /// or a change of window or view.
    pub fn reset(&mut self) {
        self.last = None;
    }

    /// The window `[cursor - dt, cursor)`, or `None` when the playhead has not moved. `lookback_us`
    /// is how much history a fresh read hands back as `fresh`, so a raw view's fading trails
    /// survive a seek instead of starting from black.
    pub fn step(&mut self, dt_us: i64, lookback_us: i64) -> Result<Option<Step>, Failure> {
        let t1 = self.start_us + self.playback.cursor_us;
        let t0 = t1 - dt_us;
        let (fresh, window) = match self.last.take() {
            Some((last_t1, window)) if last_t1 == t1 => {
                self.last = Some((last_t1, window));
                return Ok(None);
            }
            // Playing on: append what is new and trim the front, as the GUI player does.
            Some((last_t1, window)) if last_t1 < t1 && t1 - last_t1 < dt_us => {
                let fresh = self.reader.slice_time(last_t1, t1)?;
                let window = window.concat(&[&fresh]).time_window(t0, t1);
                (fresh, window)
            }
            _ => {
                let fresh = self.reader.slice_time(t1 - dt_us.max(lookback_us), t1)?;
                let window = fresh.time_window(t0, t1);
                (fresh, window)
            }
        };
        self.last = Some((t1, window.clone()));
        Ok(Some(Step { fresh, window }))
    }
}

#[cfg(feature = "camera")]
pub(super) struct LiveFeed {
    pub capture: eventcv_core::device::Capture,
    pub overflows: usize,
    window: Option<EventStream>,
}

#[cfg(feature = "camera")]
impl LiveFeed {
    pub fn new(capture: eventcv_core::device::Capture) -> Self {
        Self {
            capture,
            overflows: 0,
            window: None,
        }
    }

    pub fn reset(&mut self) {
        self.window = None;
    }

    /// Drains everything the driver has queued. The ring is emptied on every refresh — paused or
    /// not — so it never backs up; past the decode budget the surplus is dropped undecoded, which
    /// keeps the screen responsive at rates no terminal could show anyway. `None` while paused.
    pub fn step(&mut self, dt_us: i64, paused: bool) -> Result<Option<Step>, Failure> {
        use eventcv_core::device::LIVE_DRAIN_BUDGET_MS;
        use eventcv_core::EventStreamBuilder;

        let (width, height) = (self.capture.width(), self.capture.height());
        // Device timestamps are microseconds.
        let mut builder = EventStreamBuilder::new(width, height, 0.001);
        let overflowed = self.capture.drain_events_budgeted(
            Duration::from_millis(LIVE_DRAIN_BUDGET_MS),
            |x, y, t, polarity| {
                builder.push(x, y, t, polarity);
            },
        )?;
        self.overflows += usize::from(overflowed);
        if paused {
            return Ok(None);
        }
        let fresh = builder.build();
        let window = match (self.window.take(), fresh.ts().last()) {
            (Some(window), Some(&last)) => window
                .concat(&[&fresh])
                .time_window(last + 1 - dt_us, last + 1),
            (Some(window), None) => window,
            (None, _) => fresh.clone(),
        };
        self.window = Some(window.clone());
        Ok(Some(Step { fresh, window }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn playback() -> Playback {
        Playback {
            cursor_us: 0,
            duration_us: 1_000_000,
            speed: 1.0,
            playing: true,
        }
    }

    #[test]
    fn playback_follows_the_clock_at_speed() {
        let mut playback = Playback {
            speed: 0.5,
            ..playback()
        };
        playback.advance(Duration::from_millis(100));
        assert_eq!(playback.cursor_us, 50_000);
    }

    #[test]
    fn playback_stops_at_the_end() {
        let mut playback = playback();
        playback.advance(Duration::from_secs(5));
        assert_eq!(playback.cursor_us, 1_000_000);
        assert!(!playback.playing);
        playback.advance(Duration::from_secs(1));
        assert_eq!(playback.cursor_us, 1_000_000);
    }

    #[test]
    fn seeking_stays_inside_the_recording() {
        let mut playback = playback();
        playback.seek(-5_000_000);
        assert_eq!(playback.cursor_us, 0);
        playback.seek(5_000_000);
        assert_eq!(playback.cursor_us, 1_000_000);
    }

    #[test]
    fn paused_playback_holds_still() {
        let mut playback = Playback {
            playing: false,
            ..playback()
        };
        playback.advance(Duration::from_secs(1));
        assert_eq!(playback.cursor_us, 0);
    }
}
