//! Adaptive biasing — closed-loop control of a DVS sensor's bias currents so the event rate stays
//! comparable as scene brightness changes.
//!
//! A port of the fast-and-slow controller from Nair, Milford & Fischer, *Enhancing Visual Place
//! Recognition via Fast and Slow Adaptive Biasing in Event Cameras* (IROS 2024). Two nested loops
//! run off one measurement, the event rate over a fixed control period:
//!
//! - the **fast** loop maps the rate linearly onto the refractory bias, throttling the sensor's
//!   maximum per-pixel firing rate several times a second;
//! - the **slow** loop waits for the fast loop to run out of [`Authority`] — to sit on a rail, or
//!   to oscillate without ever reaching the band — then shifts the photoreceptor and threshold
//!   biases a step in the helpful direction, giving the fast loop room again.
//!
//! This module is deliberately free of any device: [`BiasController`] is a pure state machine over
//! bias *values*, so the control law is unit-testable without a camera and is shared by every
//! sensor. Translating a [`BiasValues`] into register writes is the caller's job.
//!
//! **Bias values are ladder indices, not currents.** Every field of [`BiasValues`] is an index into
//! whatever monotonic bias ladder the device exposes, where a larger index means more current. On a
//! DAVIS346 that is the driver's `0..=2040` coarse/fine map; the controller only relies on the
//! ordering, never on the units.

use std::time::Duration;

/// Periods spent measuring the scene before the loops engage — about a second at the default
/// period, long enough to average over a few bursts without a visible delay before control starts.
const CALIBRATION_PERIODS: u32 = 4;

/// How far each end of a calibrated band sits from the measured rate, as a ratio. The square root
/// of five, so the band spans the same 5:1 the reference paper's does — wide enough that ordinary
/// scene variation does not constantly saturate the fast loop, narrow enough to still mean
/// something.
const CALIBRATION_SPREAD: f64 = 2.236;

/// A calibrated band is only trustworthy if there was something to measure. Below this the scene is
/// dark or the lens is capped, and the sensor's own default band is a better guess than one built
/// out of noise.
const CALIBRATION_FLOOR: f64 = 1000.0;

/// The five bias currents the controller drives, as ladder indices (larger = more current).
///
/// `refractory` is the fast loop's actuator; the other four are the slow loop's. The names are the
/// DAVIS346's, since that is the sensor the reference controller was built and tuned on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BiasValues {
    /// Refractory period. More current = shorter refractory = higher maximum firing rate.
    pub refractory: u16,
    /// Photoreceptor bandwidth. More current = faster photoreceptor = more events.
    pub photoreceptor: u16,
    /// Photoreceptor source-follower bandwidth. More current = more events.
    pub follower: u16,
    /// ON threshold. More current = **higher** threshold = fewer ON events.
    pub on_threshold: u16,
    /// OFF threshold. More current = **lower** threshold magnitude = more OFF events.
    pub off_threshold: u16,
}

/// How much authority the fast loop still has — the signal the slow loop counts.
///
/// This describes the *actuator*, not the rate: a rate far outside the band still reads as
/// [`Tracking`](Self::Tracking) while the fast loop is slewing toward it, because a loop that is
/// still moving has not run out of anything yet. The three non-tracking states are the ways it can
/// run out, and each of them is what asks the slow loop to shift the operating point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Authority {
    /// The fast loop is coping: the rate is inside the band, or the loop is still on its way there.
    Tracking,
    /// As permissive as it can be, and the rate is *still* below the band.
    Starved,
    /// Throttling as hard as it can, and the rate is *still* above the band.
    Flooded,
    /// Oscillating: the loop keeps reversing without ever landing inside the band, so the band is
    /// not reachable from this operating point at all.
    ///
    /// This is what a sensor whose rate jumps by orders of magnitude over a few bias steps does —
    /// the band falls in the jump, so every setting is either far below it or far above, and no
    /// amount of refractory tuning finds the middle. Distinguished from the rails because a
    /// hunting loop never reaches one, which would otherwise leave the slow loop — the only thing
    /// that *can* help, by moving the operating point out from under the jump — locked out.
    Hunting,
}

/// Tuning for [`BiasController`]. [`BiasConfig::davis346`] is the reference paper's configuration;
/// [`BiasConfig::prophesee_evk4`] re-scales it for that sensor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiasConfig {
    /// How much time each rate measurement covers.
    pub period: Duration,
    /// The event rates (events/second) at the ends of the target band. The fast loop maps this
    /// range onto `throttle_range`; outside it the loop is saturated.
    ///
    /// Only used as given when `calibrate` is zero — otherwise it is the fallback for a scene too
    /// quiet to measure.
    pub target_rate: (f64, f64),
    /// Periods to spend measuring the scene, at the camera's stock biases, before either loop
    /// engages. The band is then re-centred on what was measured and `target_rate` is ignored.
    ///
    /// This is what makes an un-tuned `adaptive_bias=True` useful. A band has to match what the
    /// scene can actually produce — ask for a rate it cannot supply and the sensor gets driven into
    /// the regime where it is amplifying its own noise — but the right number depends on the room,
    /// the lens and the motion, so no fixed default is right for everyone. Measuring first turns
    /// the feature into "hold the rate wherever this scene started", which is the useful promise:
    /// the reference condition is whatever the camera saw at startup, and the loops then work to
    /// keep later conditions looking like it.
    ///
    /// Set to zero to use `target_rate` verbatim, which is what passing an explicit band does.
    pub calibrate: u32,
    /// The refractory bias at each end of its travel, as `(most throttled, least throttled)`.
    /// The lower index is applied at the top of `target_rate` and vice versa.
    pub throttle_range: (u16, u16),
    /// The furthest the fast loop may move the refractory bias in one period.
    ///
    /// The reference jumps straight to whatever its rate→bias map computes, which assumes the
    /// sensor responds gently across that travel. Measured on a DAVIS346 it does not: the event
    /// rate is flat over most of the range and then climbs three orders of magnitude near the top,
    /// so an undamped jump overshoots, the next measurement asks for the opposite rail, and the
    /// controller locks into a rail-to-rail limit cycle instead of converging. Approaching the
    /// target a bounded step at a time removes the limit cycle without changing where the loop is
    /// aiming; on a gentle sensor the bound rarely binds and the behaviour is the reference's.
    ///
    /// Smaller is steadier but slower to chase a lighting change: at the default period, 16 crosses
    /// the DAVIS346's whole refractory travel in about five seconds.
    pub max_slew: u16,
    /// Consecutive periods pinned to a rail before the slow loop takes a step. The paper's
    /// ablation sweeps this (`n2`/`n7`/`n10`).
    ///
    /// [`Authority::Hunting`] is held to twice this, since one overshoot-and-correct also reverses
    /// and only a sustained oscillation means the band is genuinely unreachable.
    pub patience: u32,
    /// How far the slow loop moves each bias per step, in ladder indices.
    pub step: u16,
    /// Inclusive bounds each slow-loop bias is held within.
    pub limits: BiasLimits,
}

/// Per-bias bounds for the slow loop. The refractory bias is not here — the fast loop already
/// bounds it with [`BiasConfig::throttle_range`].
///
/// One shared range would do for a DAVIS346, whose biases all live on the same wide ladder, but not
/// for a sensor whose thresholds are defined *relative to each other*: on an IMX636 the ON
/// threshold must stay above the `diff` reference and the OFF threshold below it, so their safe
/// ranges do not overlap at all and no single pair can describe both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BiasLimits {
    pub photoreceptor: (u16, u16),
    pub follower: (u16, u16),
    pub on_threshold: (u16, u16),
    pub off_threshold: (u16, u16),
}

impl BiasLimits {
    /// The same bounds for every bias — the right model for a sensor with one uniform ladder.
    pub fn uniform(low: u16, high: u16) -> Self {
        Self {
            photoreceptor: (low, high),
            follower: (low, high),
            on_threshold: (low, high),
            off_threshold: (low, high),
        }
    }

    fn each(&self) -> [(u16, u16); 4] {
        [
            self.photoreceptor,
            self.follower,
            self.on_threshold,
            self.off_threshold,
        ]
    }
}

impl BiasConfig {
    /// The reference controller's tuning, translated to a DAVIS346's `0..=2040` bias ladder.
    ///
    /// The rates and the 0.3 s period are the paper's as published. The refractory travel is its
    /// `RefrBp` fine sweep of 4..=57: the reference only ever sends a *fine* value, leaving the
    /// coarse at the stock 4, and coarse 4 fine 4 and fine 57 are ladder indices 801 and 1084.
    /// (The camera's own default, 993, sits about two thirds of the way up that travel.) `step` is
    /// the reference's ±5 fine, which is worth 5–7 indices at the operating points it uses.
    pub fn davis346() -> Self {
        Self {
            period: Duration::from_millis(300),
            target_rate: (5e5, 2.5e6),
            calibrate: CALIBRATION_PERIODS,
            throttle_range: (801, 1084),
            max_slew: 16,
            patience: 2,
            step: 6,
            limits: BiasLimits::uniform(0, 2040),
        }
    }

    /// Tuning for a Prophesee EVK4 (IMX636). The reference paper's controller was never run on one,
    /// so this is the same control law re-scaled to the sensor rather than published numbers.
    ///
    /// `target_rate` is the paper's band scaled by pixel count — 1280×720 against the DAVIS346's
    /// 346×260 is 10.2× the pixels, so 5–25 Mev/s. The biases are plain `u8` registers rather than
    /// a coarse/fine ladder, so `step` and `max_slew` are much smaller numbers for the same
    /// physical effect.
    ///
    /// `throttle_range` is the whole `refr` register, measured open-loop on an EVK4: the rate rises
    /// monotonically with it — 176 kev/s at 0, a plateau around 450 kev/s from 40 to 100, then
    /// 2.1 Mev/s at 160 and 6.9 Mev/s at 235. That is a far better-behaved actuator than the
    /// DAVIS346's, which is flat for most of its travel and then jumps three orders of magnitude,
    /// so this sensor is much less prone to the hunting that motivated `max_slew`. `max_slew` is
    /// still sized to cross the full travel in about twenty periods, as on the DAVIS346.
    ///
    /// The limits are deliberately conservative and are the part most worth revisiting: `diff_on`
    /// and `diff_off` are held clear of the `diff` reference (77 at stock) in both directions,
    /// because thresholds that cross it invert the comparator and flood the sensor. Note the stock
    /// `diff_off` of 73 already sits near its ceiling, so this sensor has little room to grow *more*
    /// sensitive by lowering the OFF threshold — the slow loop leans on the photoreceptor biases
    /// for that direction.
    pub fn prophesee_evk4() -> Self {
        Self {
            period: Duration::from_millis(300),
            target_rate: (5e6, 2.5e7),
            calibrate: CALIBRATION_PERIODS,
            throttle_range: (0, 235),
            max_slew: 12,
            patience: 2,
            step: 2,
            limits: BiasLimits {
                photoreceptor: (60, 160),
                follower: (55, 130),
                on_threshold: (83, 140),
                off_threshold: (35, 76),
            },
        }
    }

    /// Rejects tuning the controller cannot act on, so a caller can validate before opening a
    /// device. Returns a message suitable for surfacing straight to a user.
    pub fn validate(&self) -> Result<(), String> {
        let (rate_min, rate_max) = self.target_rate;
        if !rate_min.is_finite() || !rate_max.is_finite() || rate_min < 0.0 {
            return Err("target_rate must be finite and non-negative".to_owned());
        }
        if rate_max <= rate_min {
            return Err("target_rate must be (low, high) with high greater than low".to_owned());
        }
        if self.period.is_zero() {
            return Err("period must be greater than zero".to_owned());
        }
        if self.throttle_range.0 > self.throttle_range.1 {
            return Err(
                "throttle_range must be (most throttled, least throttled), lowest index first"
                    .to_owned(),
            );
        }
        if self.limits.each().iter().any(|(low, high)| low > high) {
            return Err("every entry in limits must be (low, high)".to_owned());
        }
        if self.max_slew == 0 {
            return Err("max_slew must be at least 1, or the fast loop can never move".to_owned());
        }
        Ok(())
    }
}

/// What the caller asked for, on top of whichever per-sensor default applies.
///
/// The sensible base — rates, register ranges, step sizes — differs per sensor and is only known
/// once the camera is open, so a request is carried as overrides and resolved against
/// [`BiasConfig::davis346`] or [`BiasConfig::prophesee_evk4`] at that point. Every `None` keeps the
/// sensor's own value.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BiasOverrides {
    pub period: Option<Duration>,
    pub target_rate: Option<(f64, f64)>,
    pub throttle_range: Option<(u16, u16)>,
    pub max_slew: Option<u16>,
    pub patience: Option<u32>,
    pub step: Option<u16>,
    pub limits: Option<BiasLimits>,
    pub calibrate: Option<u32>,
}

impl BiasOverrides {
    /// Resolves the request against a sensor's defaults.
    pub fn apply(&self, base: BiasConfig) -> BiasConfig {
        BiasConfig {
            period: self.period.unwrap_or(base.period),
            target_rate: self.target_rate.unwrap_or(base.target_rate),
            // An explicit band is a deliberate choice, so measuring one would only override it.
            calibrate: self
                .calibrate
                .unwrap_or(if self.target_rate.is_some() { 0 } else { base.calibrate }),
            throttle_range: self.throttle_range.unwrap_or(base.throttle_range),
            max_slew: self.max_slew.unwrap_or(base.max_slew),
            patience: self.patience.unwrap_or(base.patience),
            step: self.step.unwrap_or(base.step),
            limits: self.limits.unwrap_or(base.limits),
        }
    }

    /// Checks whatever the caller actually supplied, so a bad option can be rejected before any
    /// camera is opened even though the defaults it will sit on are not known yet.
    pub fn validate(&self) -> Result<(), String> {
        // Resolving against both sensors catches anything wrong with the request itself, while
        // ignoring differences that are only about which sensor it lands on.
        self.apply(BiasConfig::davis346()).validate()?;
        self.apply(BiasConfig::prophesee_evk4()).validate()
    }
}

/// Moves one bias along its ladder, held inside `(low, high)`.
fn shift(value: u16, step: i32, (low, high): (u16, u16)) -> u16 {
    (i32::from(value) + step).clamp(i32::from(low), i32::from(high)) as u16
}

/// A snapshot of the controller, for telemetry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiasState {
    /// The most recently measured event rate (events/second), `0.0` before the first period ends.
    pub event_rate: f64,
    /// The biases as they currently stand.
    pub values: BiasValues,
    /// The band the fast loop is aiming at — the calibrated one once calibration has finished.
    pub target_rate: (f64, f64),
    /// Whether the controller is still measuring the scene and has not yet touched a bias.
    pub calibrating: bool,
    /// How the last completed period ended.
    pub authority: Authority,
    /// How many steps the slow loop has taken since the controller started.
    pub slow_steps: u64,
}

/// The fast-and-slow control law. Feed it decoded event counts with
/// [`observe`](Self::observe); it hands back a new [`BiasValues`] whenever one is due.
#[derive(Clone, Debug)]
pub struct BiasController {
    config: BiasConfig,
    values: BiasValues,
    /// Events counted into the period currently filling.
    events: u64,
    /// Time covered by the period currently filling.
    elapsed: Duration,
    /// Set when events went uncounted this period, making `events` a lower bound.
    undercounted: bool,
    /// Consecutive completed periods spent on each rail.
    too_few: u32,
    too_many: u32,
    /// Consecutive out-of-band periods the fast loop spent reversing without reaching a rail.
    hunting: u32,
    /// Events and time across the current run of out-of-band periods. A hunting loop alternates
    /// between extremes, so it is the *average* rate it delivers — not either extreme, which the
    /// last period alone would report — that says which way the operating point has to move.
    unreachable_events: u64,
    unreachable_elapsed: Duration,
    /// Which way the fast loop wanted to move last period (`-1`, `0`, `1`), for spotting reversals.
    last_want: i32,
    /// Calibration periods still to run before the loops engage.
    calibrating: u32,
    /// Events and time measured across the calibration periods.
    calibration_events: u64,
    calibration_elapsed: Duration,
    event_rate: f64,
    authority: Authority,
    slow_steps: u64,
}

impl BiasController {
    /// Starts the controller from the biases the device is already running (its stock
    /// configuration, unless the caller has a tuned operating point to begin from).
    pub fn new(config: BiasConfig, start: BiasValues) -> Self {
        // The camera's own biases need not sit inside limits the caller chose, and a controller
        // that began outside its bounds would report values it never intends to honour. Clamping
        // here means the first update carries them into range.
        let limits = config.limits;
        let start = BiasValues {
            refractory: start.refractory,
            photoreceptor: shift(start.photoreceptor, 0, limits.photoreceptor),
            follower: shift(start.follower, 0, limits.follower),
            on_threshold: shift(start.on_threshold, 0, limits.on_threshold),
            off_threshold: shift(start.off_threshold, 0, limits.off_threshold),
        };
        Self {
            config,
            values: start,
            events: 0,
            elapsed: Duration::ZERO,
            undercounted: false,
            too_few: 0,
            too_many: 0,
            hunting: 0,
            unreachable_events: 0,
            unreachable_elapsed: Duration::ZERO,
            last_want: 0,
            calibrating: config.calibrate,
            calibration_events: 0,
            calibration_elapsed: Duration::ZERO,
            event_rate: 0.0,
            authority: Authority::Tracking,
            slow_steps: 0,
        }
    }

    pub fn state(&self) -> BiasState {
        BiasState {
            event_rate: self.event_rate,
            values: self.values,
            target_rate: self.config.target_rate,
            calibrating: self.calibrating > 0,
            authority: self.authority,
            slow_steps: self.slow_steps,
        }
    }

    /// Folds one batch of decoded events into the measurement, `elapsed` being the time since the
    /// previous call. Returns the biases to write once a control period completes and the law
    /// actually moved something — `None` the rest of the time, which is most calls.
    ///
    /// The caller keeps the clock, which keeps the controller deterministic for testing. It must
    /// call this on **every** decode pass, including passes that produced nothing — a scene too
    /// dark to trigger any event is exactly when the controller needs to open the sensor up, and a
    /// caller that only reports non-empty buffers would leave the period stuck and never do so.
    pub fn observe(&mut self, events: u64, elapsed: Duration) -> Option<BiasValues> {
        self.events += events;
        self.elapsed += elapsed;
        if self.elapsed < self.config.period {
            return None;
        }
        let (events, elapsed) = (self.events, self.elapsed);
        let rate = events as f64 / elapsed.as_secs_f64();
        let undercounted = self.undercounted;
        self.events = 0;
        self.elapsed = Duration::ZERO;
        self.undercounted = false;
        self.control(rate, events, elapsed, undercounted)
    }

    /// Marks that events were dropped before being counted, so the period in flight measures a
    /// lower bound rather than the true rate.
    ///
    /// The live viewer stops decoding once it is over its per-frame budget and discards the
    /// remaining buffers; without this the controller would read the resulting undercount as a
    /// quiet scene and make the sensor *more* sensitive, driving the overload it was meant to
    /// relieve. A period marked here can only ever throttle.
    pub fn invalidate(&mut self) {
        self.undercounted = true;
    }

    /// Applies both loops to one completed measurement. `events` and `elapsed` are the raw
    /// measurement behind `rate`, kept so a run of periods can be averaged.
    fn control(
        &mut self,
        rate: f64,
        events: u64,
        elapsed: Duration,
        undercounted: bool,
    ) -> Option<BiasValues> {
        self.event_rate = rate;
        if self.calibrating > 0 {
            // Measuring only: the biases stay exactly as the camera had them, so whatever the scene
            // is doing now becomes the reference the loops will hold it to. An undercounted period
            // would bias the measurement low, so it is not counted towards the total.
            if !undercounted {
                self.calibration_events += events;
                self.calibration_elapsed += elapsed;
            }
            self.calibrating -= 1;
            if self.calibrating == 0 {
                self.finish_calibration();
            }
            self.authority = Authority::Tracking;
            return None;
        }
        let (rate_min, rate_max) = self.config.target_rate;
        // An undercounted period only bounds the rate from below, so every conclusion that needs
        // an upper bound — that the scene is quiet, that the sensor should open up — is unsafe.
        if undercounted && rate < rate_max {
            self.authority = Authority::Tracking;
            return None;
        }

        let before = self.values;
        let (throttled, open) = self.config.throttle_range;
        let current = self.values.refractory;
        let target = self.throttle_target(rate);
        // Which way the loop *wants* to go, before the slew limit has its say. Reversals are read
        // from the want rather than the movement, so a loop pinned against a rail — which wants to
        // move but cannot — is not mistaken for one that has changed its mind.
        let want = (target as i32 - current as i32).signum();
        let reversed = want != 0 && self.last_want != 0 && want != self.last_want;
        let out_of_band = rate < rate_min || rate > rate_max;
        self.last_want = want;
        self.values.refractory = target.clamp(
            current.saturating_sub(self.config.max_slew),
            current.saturating_add(self.config.max_slew),
        );
        // Railed is judged on where the loop ended up, not where it started, so the period that
        // *arrives* at a rail already counts as out of travel. Judged against the target rather
        // than the direction of travel, so a loop that has come to rest exactly on the rail —
        // wanting to go nowhere — still reads as railed rather than as tracking.
        let moved = self.values.refractory;
        let railed =
            (target >= open && moved >= open) || (target <= throttled && moved <= throttled);

        // Authority is a property of the *actuator*, not the rate: the slow loop exists to help a
        // fast loop that has run out of room, and one still slewing toward its target has plenty.
        // (The reference tests the same thing — it compares its clamped setting against the rails —
        // but with no slew limit the two are the same test.) The rate is checked too, so a
        // degenerate zero-width travel doesn't read as permanently railed.
        self.authority = match (out_of_band, railed, reversed) {
            (false, _, _) => Authority::Tracking,
            (true, true, _) if target >= open => Authority::Starved,
            (true, true, _) => Authority::Flooded,
            (true, false, true) => Authority::Hunting,
            // Out of band but still slewing productively toward the target.
            (true, false, false) => Authority::Tracking,
        };

        // The average that picks the slow loop's direction is taken over the hunt itself: every
        // out-of-band period from the first reversal onward, including the railed ones a hunt near
        // a rail alternates through and the ones merely slewing between reversals. The approach
        // *to* the hunt is excluded — a long one-way slew is not part of the oscillation, and
        // averaging it in drags the figure toward the rung the loop started on.
        //
        // Only a period that actually lands *in* the band clears the evidence. Clearing it
        // whenever the loop happened to be tracking would erase a real hunt, because crossing back
        // after a reversal takes several slewing periods and every one of them reads as tracking:
        // measured on a DAVIS346, that wiped the count between reversals and the slow loop never
        // fired despite 88 hunting periods in 45 seconds.
        if !out_of_band {
            self.forget_unreachable();
        } else if reversed || self.hunting > 0 {
            self.unreachable_events += events;
            self.unreachable_elapsed += elapsed;
        }

        match self.authority {
            Authority::Starved => {
                self.too_many = 0;
                self.too_few += 1;
                if self.too_few >= self.config.patience {
                    self.too_few = 0;
                    self.step_slow(1);
                }
            }
            Authority::Flooded => {
                self.too_few = 0;
                self.too_many += 1;
                if self.too_many >= self.config.patience {
                    self.too_many = 0;
                    self.step_slow(-1);
                }
            }
            Authority::Hunting => {
                // A hunt gives no single direction — the rate straddles the band — so the run's
                // average delivered rate picks one. Held to a longer count than the rails, because
                // a single overshoot-and-correct also reverses, and only a sustained oscillation
                // means the band is genuinely out of reach.
                self.too_few = 0;
                self.too_many = 0;
                self.hunting += 1;
                if self.hunting >= self.hunting_patience() {
                    let average = self.unreachable_events as f64
                        / self.unreachable_elapsed.as_secs_f64().max(f64::EPSILON);
                    let direction = if average > rate_max {
                        -1
                    } else if average < rate_min {
                        1
                    } else {
                        0 // the hunt averages out inside the band: leave the operating point alone
                    };
                    self.forget_unreachable();
                    if direction != 0 {
                        self.step_slow(direction);
                    }
                }
            }
            // Off the rails, so the rail streaks break — but the unreachable-band evidence above
            // is left alone, since being out of band while slewing is not a success.
            Authority::Tracking => {
                self.too_few = 0;
                self.too_many = 0;
            }
        }
        // Every bias may already sit against a limit, in which case there is nothing to write.
        (self.values != before).then_some(self.values)
    }

    /// Re-centres the target band on what the scene actually produced, leaving the configured band
    /// in place when there was too little to measure.
    fn finish_calibration(&mut self) {
        let seconds = self.calibration_elapsed.as_secs_f64();
        if seconds <= 0.0 {
            return;
        }
        let measured = self.calibration_events as f64 / seconds;
        if measured < CALIBRATION_FLOOR {
            return;
        }
        self.config.target_rate = (measured / CALIBRATION_SPREAD, measured * CALIBRATION_SPREAD);
    }

    /// Consecutive hunting periods before the slow loop acts on them.
    fn hunting_patience(&self) -> u32 {
        self.config.patience.saturating_mul(2).max(2)
    }

    /// Clears the evidence that the target band is unreachable.
    fn forget_unreachable(&mut self) {
        self.hunting = 0;
        self.unreachable_events = 0;
        self.unreachable_elapsed = Duration::ZERO;
    }

    /// The fast loop's aim: the target band mapped linearly onto the refractory bias's travel,
    /// inverted so a faster stream is throttled harder. The slew limit is applied by the caller.
    fn throttle_target(&self, rate: f64) -> u16 {
        let (rate_min, rate_max) = self.config.target_rate;
        let (throttled, open) = self.config.throttle_range;
        let fraction = ((rate - rate_min) / (rate_max - rate_min)).clamp(0.0, 1.0);
        // Saturating, so a range `validate` would have rejected cannot overflow its way to a
        // nonsense bias — the worst it can do is pin the actuator.
        let travel = open.saturating_sub(throttled);
        open.saturating_sub((f64::from(travel) * fraction).round() as u16)
    }

    /// The slow loop: shifts the operating point one step toward more events (`direction` 1) or
    /// fewer (`-1`). Raising the photoreceptor biases and shrinking both thresholds each yield
    /// more events — and the thresholds shrink in opposite directions, since the ON threshold
    /// tracks `on_threshold` upward while the OFF threshold tracks `off_threshold` downward.
    fn step_slow(&mut self, direction: i32) {
        let step = i32::from(self.config.step) * direction;
        let limits = self.config.limits;
        self.values.photoreceptor = shift(self.values.photoreceptor, step, limits.photoreceptor);
        self.values.follower = shift(self.values.follower, step, limits.follower);
        self.values.on_threshold = shift(self.values.on_threshold, -step, limits.on_threshold);
        self.values.off_threshold = shift(self.values.off_threshold, step, limits.off_threshold);
        self.slow_steps += 1;
    }
}


#[cfg(test)]
mod tests {
    use super::{Authority, BiasConfig, BiasController, BiasLimits, BiasValues};
    use std::time::Duration;

    /// The camera's stock biases — where `adaptive_bias=True` actually starts from.
    fn start() -> BiasValues {
        BiasValues {
            refractory: 993,
            photoreceptor: 575,
            follower: 153,
            on_threshold: 1564,
            off_threshold: 583,
        }
    }

    /// The reference tuning with calibration switched off — these tests exercise the control law,
    /// and the measuring phase would just shift every period count by a constant.
    fn config() -> BiasConfig {
        BiasConfig { calibrate: 0, ..BiasConfig::davis346() }
    }

    /// Runs one whole control period at `rate` events/second, as a single batch.
    fn period(controller: &mut BiasController, rate: f64) -> Option<BiasValues> {
        let span = controller.config.period;
        let events = (rate * span.as_secs_f64()).round() as u64;
        controller.observe(events, span)
    }

    fn drive(controller: &mut BiasController, rate: f64, periods: usize) {
        for _ in 0..periods {
            period(controller, rate);
        }
    }

    /// Periods needed to slew the refractory bias from the stock start to `target`.
    fn periods_to(target: u16) -> usize {
        let travel = target.abs_diff(start().refractory);
        (travel as usize).div_ceil(config().max_slew as usize)
    }

    #[test]
    fn every_shipped_configuration_is_valid() {
        assert_eq!(BiasConfig::davis346().validate(), Ok(()));
        assert_eq!(BiasConfig::prophesee_evk4().validate(), Ok(()));
    }

    #[test]
    fn the_evk4_thresholds_stay_the_right_side_of_its_diff_reference() {
        // An IMX636 compares against `diff` (77 at stock): an ON threshold below it or an OFF
        // threshold above it inverts the comparator and floods the sensor. The bounds must keep
        // the slow loop clear of that no matter how long it runs.
        let limits = BiasConfig::prophesee_evk4().limits;
        assert!(limits.on_threshold.0 > 77, "ON threshold must stay above diff");
        assert!(limits.off_threshold.1 < 77, "OFF threshold must stay below diff");
    }

    #[test]
    fn evk4_biases_stay_inside_a_u8() {
        // They are written to byte registers, so bounds beyond 255 would silently saturate and the
        // controller would think it had moved something it had not.
        let config = BiasConfig::prophesee_evk4();
        for (_, high) in config.limits.each() {
            assert!(high <= u16::from(u8::MAX), "{high} does not fit a byte register");
        }
        assert!(config.throttle_range.1 <= u16::from(u8::MAX));
    }

    #[test]
    fn calibration_centres_the_band_on_the_scene_and_touches_nothing_first() {
        let calibrated = BiasConfig::davis346();
        let mut controller = BiasController::new(calibrated, start());
        // While measuring, the biases must stay exactly as the camera had them — the whole point is
        // that the reference condition is the scene as the camera actually saw it.
        assert!(controller.state().calibrating);
        for remaining in (0..calibrated.calibrate).rev() {
            assert_eq!(period(&mut controller, 3e4), None, "no bias may move while measuring");
            // The last measuring period is the one that resolves the band.
            assert_eq!(controller.state().calibrating, remaining > 0);
        }
        assert_eq!(controller.state().values, start());

        let (low, high) = controller.state().target_rate;
        assert!(!controller.state().calibrating);
        assert!(low < 3e4 && 3e4 < high, "the band must bracket the measured rate");
        assert!((high / low - 5.0).abs() < 0.1, "spanning the reference paper's 5:1");
        // And now it controls: a rate well outside the *calibrated* band moves the fast loop, even
        // though it sits inside the configured default the calibration replaced.
        assert!(period(&mut controller, 1e6).is_some());
    }

    #[test]
    fn calibration_keeps_the_default_band_when_there_is_nothing_to_measure() {
        // Lens cap on: a band built from a handful of noise events would be meaningless, and would
        // then drive the sensor to reproduce that noise level forever.
        let calibrated = BiasConfig::davis346();
        let mut controller = BiasController::new(calibrated, start());
        drive(&mut controller, 0.0, calibrated.calibrate as usize);
        assert_eq!(controller.state().target_rate, calibrated.target_rate);
    }

    #[test]
    fn an_explicit_band_skips_calibration() {
        let overrides = super::BiasOverrides {
            target_rate: Some((1e4, 1e5)),
            ..Default::default()
        };
        let config = overrides.apply(BiasConfig::davis346());
        assert_eq!(config.calibrate, 0, "an explicit band must not be measured over");
        assert_eq!(config.target_rate, (1e4, 1e5));
        // Asking for calibration *and* a band is allowed — the measurement wins, as documented.
        let both = super::BiasOverrides {
            calibrate: Some(4),
            ..overrides
        };
        assert_eq!(both.apply(BiasConfig::davis346()).calibrate, 4);
    }

    #[test]
    fn an_undercounted_calibration_period_is_not_measured() {
        // Undercounting biases the measurement low, which would then anchor the band too low.
        let calibrated = BiasConfig::davis346();
        let mut controller = BiasController::new(calibrated, start());
        controller.invalidate();
        period(&mut controller, 0.0); // dropped buffers: ignored, not counted as a quiet scene
        drive(&mut controller, 4e4, (calibrated.calibrate - 1) as usize);
        let (low, high) = controller.state().target_rate;
        assert!(low < 4e4 && 4e4 < high, "the band follows only the periods that counted");
    }

    #[test]
    fn a_start_outside_the_limits_is_pulled_into_range() {
        // The camera's stock biases need not satisfy limits the caller picked.
        let narrow = BiasConfig {
            limits: BiasLimits {
                off_threshold: (35, 60),
                ..BiasConfig::prophesee_evk4().limits
            },
            ..BiasConfig::prophesee_evk4()
        };
        let controller = BiasController::new(narrow, BiasValues { off_threshold: 73, ..start() });
        assert_eq!(controller.state().values.off_threshold, 60);
    }

    #[test]
    fn overrides_leave_unnamed_fields_to_the_sensor() {
        let overrides = super::BiasOverrides {
            target_rate: Some((1e4, 1e5)),
            ..Default::default()
        };
        // The same request resolves differently per sensor, which is the whole point of deferring
        // it until the camera is open.
        let davis = overrides.apply(BiasConfig::davis346());
        let evk4 = overrides.apply(BiasConfig::prophesee_evk4());
        assert_eq!(davis.target_rate, (1e4, 1e5));
        assert_eq!(evk4.target_rate, (1e4, 1e5));
        assert_eq!(davis.throttle_range, BiasConfig::davis346().throttle_range);
        assert_eq!(evk4.throttle_range, BiasConfig::prophesee_evk4().throttle_range);
        assert!(overrides.validate().is_ok());
    }

    #[test]
    fn overrides_are_rejected_for_either_sensor() {
        // Validation runs before the camera is open, so a bad request must be caught whichever
        // sensor it would have landed on.
        let overrides = super::BiasOverrides {
            target_rate: Some((1e6, 1e5)),
            ..Default::default()
        };
        assert!(overrides.validate().is_err());
    }

    #[test]
    fn validate_rejects_unusable_tuning() {
        for broken in [
            BiasConfig { target_rate: (2.5e6, 5e5), ..config() },
            BiasConfig { period: Duration::ZERO, ..config() },
            BiasConfig { throttle_range: (1084, 801), ..config() },
            BiasConfig { limits: BiasLimits::uniform(2040, 0), ..config() },
            BiasConfig { max_slew: 0, ..config() },
        ] {
            assert!(broken.validate().is_err(), "{broken:?} should be rejected");
        }
    }

    #[test]
    fn the_fast_loop_moves_by_at_most_one_slew_per_period() {
        let slew = config().max_slew;
        // Starved: open the sensor up, but only a step at a time.
        let mut controller = BiasController::new(config(), start());
        assert_eq!(period(&mut controller, 1e4).unwrap().refractory, start().refractory + slew);
        assert_eq!(period(&mut controller, 1e4).unwrap().refractory, start().refractory + 2 * slew);

        // Flooded: throttle, one step at a time, in the other direction.
        let mut controller = BiasController::new(config(), start());
        assert_eq!(period(&mut controller, 1e8).unwrap().refractory, start().refractory - slew);
    }

    #[test]
    fn the_fast_loop_reaches_the_rails_and_stops_there() {
        let (throttled, open) = config().throttle_range;

        let mut controller = BiasController::new(config(), start());
        drive(&mut controller, 1e4, periods_to(open) + 4);
        assert_eq!(controller.state().values.refractory, open, "never past the rail");

        let mut controller = BiasController::new(config(), start());
        drive(&mut controller, 1e8, periods_to(throttled) + 4);
        assert_eq!(controller.state().values.refractory, throttled);
    }

    #[test]
    fn a_mid_band_rate_settles_mid_travel() {
        let (throttled, open) = config().throttle_range;
        let middle = (throttled + open) / 2;
        let mut controller = BiasController::new(config(), start());
        drive(&mut controller, 1.5e6, periods_to(middle) + 4);
        // Once there it stays: the target stops moving, so the slew has nothing left to do.
        assert_eq!(controller.state().values.refractory, middle);
        assert_eq!(controller.state().authority, Authority::Tracking);
        assert_eq!(controller.state().slow_steps, 0);
    }

    #[test]
    fn nothing_is_reported_until_the_period_completes() {
        let mut controller = BiasController::new(config(), start());
        let slice = Duration::from_millis(100);
        assert_eq!(controller.observe(50_000, slice), None);
        assert_eq!(controller.observe(50_000, slice), None);
        let update = controller.observe(50_000, slice).expect("period completed");
        // 150k events over 0.3 s is 500k/s — the bottom of the band, so the loop opens up.
        assert_eq!(update.refractory, start().refractory + config().max_slew);
        assert_eq!(controller.state().event_rate, 5e5);
    }

    #[test]
    fn a_settled_rate_leaves_the_slow_biases_alone() {
        let mut controller = BiasController::new(config(), start());
        drive(&mut controller, 1.5e6, 10);
        let state = controller.state();
        assert_eq!(state.authority, Authority::Tracking);
        assert_eq!(state.slow_steps, 0);
        assert_eq!(state.values.photoreceptor, start().photoreceptor);
        assert_eq!(state.values.on_threshold, start().on_threshold);
    }

    #[test]
    fn the_slow_loop_waits_until_the_fast_loop_is_out_of_travel() {
        // The whole point of the slow loop: it only helps once the fast loop *cannot*. While the
        // refractory bias is still slewing there is travel left, so nothing else may move.
        let open = config().throttle_range.1;
        let mut controller = BiasController::new(config(), start());
        drive(&mut controller, 1e4, periods_to(open) - 1);
        assert_eq!(controller.state().authority, Authority::Tracking, "still slewing");
        assert_eq!(controller.state().slow_steps, 0);

        period(&mut controller, 1e4); // arrives at the rail: saturated, count 1 of 2
        assert_eq!(controller.state().authority, Authority::Starved);
        assert_eq!(controller.state().slow_steps, 0);

        period(&mut controller, 1e4); // count 2 of 2
        let state = controller.state();
        assert_eq!(state.slow_steps, 1);
        // Too few events: open the photoreceptor up and shrink both thresholds. The ON threshold
        // shrinks as `on_threshold` falls, the OFF threshold as `off_threshold` rises.
        let step = config().step;
        assert_eq!(state.values.photoreceptor, start().photoreceptor + step);
        assert_eq!(state.values.follower, start().follower + step);
        assert_eq!(state.values.on_threshold, start().on_threshold - step);
        assert_eq!(state.values.off_threshold, start().off_threshold + step);
    }

    #[test]
    fn a_flooded_rail_steps_the_other_way() {
        let throttled = config().throttle_range.0;
        let mut controller = BiasController::new(config(), start());
        drive(&mut controller, 1e8, periods_to(throttled) + config().patience as usize);
        let state = controller.state();
        assert_eq!(state.authority, Authority::Flooded);
        assert_eq!(state.slow_steps, 1);
        let step = config().step;
        assert_eq!(state.values.photoreceptor, start().photoreceptor - step);
        assert_eq!(state.values.follower, start().follower - step);
        assert_eq!(state.values.on_threshold, start().on_threshold + step);
        assert_eq!(state.values.off_threshold, start().off_threshold - step);
    }

    #[test]
    fn a_settled_period_resets_the_saturation_count() {
        let open = config().throttle_range.1;
        let mut controller = BiasController::new(config(), start());
        drive(&mut controller, 1e4, periods_to(open)); // at the rail, count 1 of 2
        assert_eq!(controller.state().slow_steps, 0);
        period(&mut controller, 1.5e6); // back inside the band: count cleared, and it slews away
        assert_eq!(controller.state().slow_steps, 0);
        // Having left the rail it must climb back and then spend `patience` periods there again,
        // rather than resuming the count it had before.
        period(&mut controller, 1e4); // back at the rail: count 1 of 2
        assert_eq!(controller.state().slow_steps, 0, "count restarted, not resumed");
        period(&mut controller, 1e4); // count 2 of 2
        assert_eq!(controller.state().slow_steps, 1);
    }

    #[test]
    fn the_rails_do_not_share_a_count() {
        // One period on each rail must never add up to a step on either.
        let (throttled, open) = config().throttle_range;
        let mut controller = BiasController::new(config(), start());
        drive(&mut controller, 1e4, periods_to(open));
        assert_eq!(controller.state().authority, Authority::Starved);
        drive(&mut controller, 1e8, periods_to(throttled) + 4);
        // Reaching the far rail counts from scratch, and one period there is not enough.
        let mut controller2 = BiasController::new(config(), start());
        drive(&mut controller2, 1e8, periods_to(throttled));
        assert_eq!(controller2.state().slow_steps, 0);
    }

    #[test]
    fn the_slow_loop_keeps_stepping_while_the_rail_holds() {
        let open = config().throttle_range.1;
        let mut controller = BiasController::new(config(), start());
        // Reach the rail, then six more periods at patience 2 is three steps.
        drive(&mut controller, 1e4, periods_to(open) - 1 + 6);
        assert_eq!(controller.state().slow_steps, 3);
        assert_eq!(controller.state().values.photoreceptor, start().photoreceptor + 3 * config().step);
    }

    #[test]
    fn biases_stop_at_the_ladder_limits() {
        let limits = config().limits;
        let (low, high) = limits.photoreceptor;
        let mut controller = BiasController::new(
            config(),
            BiasValues {
                refractory: 993,
                photoreceptor: high - 2, // two indices of headroom, then the rail
                follower: 1000,
                on_threshold: 1000,
                off_threshold: 1000,
            },
        );
        drive(&mut controller, 1e4, 40);
        let values = controller.state().values;
        assert_eq!(values.photoreceptor, high, "clamped, never wrapped past the ladder");
        assert!(values.on_threshold >= low);
    }

    #[test]
    fn no_update_is_reported_when_nothing_moves() {
        let open = config().throttle_range.1;
        let mut controller = BiasController::new(config(), start());
        // Reach the rail and take the slow step that follows, which resets the count. The period
        // that arrives at the rail is itself the first of the `patience` count, hence the -1.
        drive(&mut controller, 1e4, periods_to(open) + config().patience as usize - 1);
        assert_eq!(controller.state().slow_steps, 1);
        // ...now a period that neither moves the fast loop nor completes the next slow count has
        // nothing to write.
        assert_eq!(period(&mut controller, 1e4), None);
    }

    #[test]
    fn a_stream_with_no_events_at_all_still_opens_the_sensor_up() {
        // The dark-scene case: nothing is triggering, so the period only completes if the caller
        // keeps reporting empty decode passes. It must, or the sensor never recovers.
        let mut controller = BiasController::new(config(), start());
        let update = controller
            .observe(0, config().period)
            .expect("an empty period is still a measurement");
        assert_eq!(update.refractory, start().refractory + config().max_slew);
        assert_eq!(controller.state().event_rate, 0.0);

        drive(&mut controller, 0.0, periods_to(config().throttle_range.1) + 1);
        assert_eq!(controller.state().authority, Authority::Starved);
        assert!(controller.state().slow_steps >= 1, "the slow loop takes over");
    }

    #[test]
    fn an_undercounted_period_never_increases_sensitivity() {
        let mut controller = BiasController::new(config(), start());
        controller.invalidate();
        // The count says the scene is starved, but events went missing, so the true rate may be
        // anywhere above it: the controller must not act on it.
        assert_eq!(period(&mut controller, 1e4), None);
        let state = controller.state();
        assert_eq!(state.authority, Authority::Tracking);
        assert_eq!(state.slow_steps, 0);
        assert_eq!(state.values, start());
    }

    #[test]
    fn an_undercounted_period_still_throttles() {
        let mut controller = BiasController::new(config(), start());
        controller.invalidate();
        // Undercounting bounds the rate from below, so a rate already over the band is conclusive.
        let update = period(&mut controller, 1e8).expect("throttling is still safe");
        assert_eq!(update.refractory, start().refractory - config().max_slew);
    }

    #[test]
    fn invalidation_only_applies_to_the_period_in_flight() {
        let mut controller = BiasController::new(config(), start());
        controller.invalidate();
        period(&mut controller, 1e4);
        assert_eq!(controller.state().values, start(), "the marked period was ignored");
        // The next period counted cleanly, so the starved reading is trusted again.
        let update = period(&mut controller, 1e4).expect("a clean period acts");
        assert_eq!(update.refractory, start().refractory + config().max_slew);
    }

    /// The plant that broke the reference law on a real DAVIS346: the event rate is near enough a
    /// step function of the refractory bias — flat across most of the travel, then three orders of
    /// magnitude higher at the top. No bias setting lands inside the default band, which falls in
    /// the jump.
    fn cliff(refractory: u16) -> f64 {
        if refractory >= 1070 {
            8e6
        } else {
            6e3
        }
    }

    /// Runs `periods` of closed-loop control against a plant given as rate-from-refractory.
    fn against(controller: &mut BiasController, plant: fn(u16) -> f64, periods: usize) {
        for _ in 0..periods {
            let rate = plant(controller.state().values.refractory);
            period(controller, rate);
        }
    }

    #[test]
    fn a_hunting_loop_still_reaches_the_slow_loop() {
        // The gap this closes: hunting across a step never touches a rail, so a purely rail-based
        // trigger would leave the slow loop — the only thing that can move the operating point out
        // from under the step — switched off forever, and the controller would hunt indefinitely.
        let mut controller = BiasController::new(config(), start());
        against(&mut controller, cliff, 12);
        let state = controller.state();
        assert_eq!(state.authority, Authority::Hunting);
        assert!(state.slow_steps >= 1, "the slow loop must engage on an unreachable band");
        // The hunt delivers ~4 Mev/s on average, above the band, so the operating point has to
        // become *less* sensitive — even though half the periods measured far too few events.
        assert!(state.values.photoreceptor < start().photoreceptor, "photoreceptor turned down");
        assert!(state.values.on_threshold > start().on_threshold, "ON threshold raised");
        assert!(state.values.off_threshold < start().off_threshold, "OFF threshold raised");
    }

    #[test]
    fn a_hunt_that_averages_inside_the_band_is_left_alone() {
        // Same shape of plant, but the fast rung only just clears the band, so alternating across
        // it delivers an in-band rate on average — unevenly, but in the right quantity. Moving the
        // operating point would not improve the average and risks making it worse, so the slow
        // loop stays out of it. (An alternating hunt can never average *below* the band: half of
        // `rate_max` is inside it by definition, so this and "too fast" are the only two cases.)
        fn gentle_cliff(refractory: u16) -> f64 {
            if refractory >= 1070 {
                2.6e6
            } else {
                1e3
            }
        }
        let mut controller = BiasController::new(config(), start());
        against(&mut controller, gentle_cliff, 24);
        let state = controller.state();
        assert_eq!(state.authority, Authority::Hunting, "it is still hunting");
        assert_eq!(state.slow_steps, 0, "but the average is fine, so nothing is moved");
        assert_eq!(state.values.photoreceptor, start().photoreceptor);
        assert_eq!(state.values.on_threshold, start().on_threshold);
    }

    #[test]
    fn a_single_overshoot_is_not_mistaken_for_hunting() {
        // One reversal is ordinary control, not an unreachable band, so nothing slow may happen.
        let mut controller = BiasController::new(config(), start());
        period(&mut controller, 1e4); // wants up
        period(&mut controller, 1e8); // wants down: one reversal, out of band
        assert_eq!(controller.state().authority, Authority::Hunting);
        assert_eq!(controller.state().slow_steps, 0, "one reversal is not enough");
        period(&mut controller, 1.5e6); // back in band: the evidence is discarded
        assert_eq!(controller.state().authority, Authority::Tracking);
        assert_eq!(controller.state().slow_steps, 0);
    }

    #[test]
    fn slewing_between_reversals_does_not_erase_the_hunt() {
        // Crossing back after a reversal takes several periods, all of them out of band and all of
        // them reading as tracking. They must not clear the evidence, or a hunt whose excursions
        // are wider than one slew step could never accumulate — which is exactly what a real
        // DAVIS346 does.
        fn wide_cliff(refractory: u16) -> f64 {
            if refractory >= 1070 {
                8e6
            } else {
                6e3
            }
        }
        let patient = BiasConfig { max_slew: 4, ..config() }; // 4 slewing periods per excursion
        let mut controller = BiasController::new(patient, start());
        against(&mut controller, wide_cliff, 60);
        assert!(
            controller.state().slow_steps >= 1,
            "a hunt with multi-period excursions must still reach the slow loop"
        );
    }

    #[test]
    fn a_long_slew_across_the_band_is_not_mistaken_for_hunting() {
        // Every period of a full traverse is out of band, but the direction never changes, so this
        // must read as tracking throughout rather than tripping the unreachable-band detector.
        let mut controller = BiasController::new(config(), start());
        for _ in 0..(periods_to(config().throttle_range.1) - 1) {
            period(&mut controller, 1e4);
            assert_eq!(controller.state().authority, Authority::Tracking);
        }
        assert_eq!(controller.state().slow_steps, 0);
    }

    #[test]
    fn a_step_response_sensor_does_not_limit_cycle() {
        // Without a slew limit this plant ran the loop rail to rail every single period. It must
        // instead hunt locally around the step.
        let mut controller = BiasController::new(config(), start());
        let mut visited = Vec::new();
        for _ in 0..80 {
            let rate = cliff(controller.state().values.refractory);
            period(&mut controller, rate);
            visited.push(controller.state().values.refractory);
        }
        let settled = &visited[20..];
        let low = *settled.iter().min().expect("80 periods recorded");
        let high = *settled.iter().max().expect("80 periods recorded");
        let (throttled, open) = config().throttle_range;
        assert!(
            high - low <= 4 * config().max_slew,
            "hunting {low}..{high} must stay local, not sweep the {throttled}..{open} travel"
        );
    }
}
