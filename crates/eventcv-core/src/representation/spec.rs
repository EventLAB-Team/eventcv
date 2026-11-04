use std::fmt;
use std::sync::Arc;

use super::{
    AveragedTimeSurface, Binary, CountMask, EventCount, EventFrame, Mcts, Polarity, RedBlue,
    Representation, RepresentationError, Tencode, TimeSurface, VoxelGrid,
};
use crate::flow::FlowError;
use crate::EventStream;

/// Default time span for the representations that have one (`window_ms`, `tau_ms`,
/// `max_window_ms`) when none of its unit forms is given and no slice/capture window applies
/// (an unset span on a windowed stream follows the window — see `PyEventStream::span_ms`).
pub const DEFAULT_SPAN_MS: f64 = 30.0;

/// A representation the caller compiled at run time, carried by [`ReprSpec::Custom`]. The core
/// knows nothing about how it was built — only that it turns a stream into a frame.
pub trait CustomKernel: fmt::Debug + Send + Sync {
    fn generate(&self, stream: &EventStream) -> Result<EventFrame, RepresentationError>;
}

/// A representation choice + its parameters, attached to a reader so each slice renders to a
/// dense frame (the `DataLoader`-friendly mode). One place that maps a repr name to the core
/// generator, shared by the Python bindings (`open(repr=…)`, `with_repr`, `__getitem__`, `batch`)
/// and the command line.
#[derive(Clone, Debug)]
pub enum ReprSpec {
    Custom(Arc<dyn CustomKernel>),
    Binary,
    Count {
        normalize: bool,
    },
    Polarity {
        normalize: bool,
        downsample: usize,
    },
    Voxel {
        bins: usize,
        window_ms: f64,
    },
    TimeSurface {
        tau_ms: f64,
    },
    AveragedTimeSurface {
        tau_ms: f64,
    },
    Tencode {
        window_ms: f64,
    },
    CountMask {
        pct: f64,
        white_frame: bool,
    },
    RedBlue {
        pct: f64,
    },
    Mcts {
        max_window_ms: f64,
        windows: Option<Vec<f64>>,
    },
    Flow {
        window: usize,
    },
}

impl ReprSpec {
    /// Every name [`ReprSpec::new`] accepts, in the order a viewer cycles through them.
    pub const NAMES: [&'static str; 11] = [
        "binary",
        "count",
        "polarity",
        "voxel",
        "tsurf",
        "atsurf",
        "tencode",
        "countmask",
        "redblue",
        "mcts",
        "flow",
    ];

    /// Builds a spec from a repr name and the optional overrides (unset ones take the
    /// same defaults as the `EventStream` methods: count raw `u64`, polarity normalized).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: &str,
        bins: Option<i64>,
        window_ms: Option<f64>,
        tau_ms: Option<f64>,
        max_window_ms: Option<f64>,
        windows: Option<Vec<f64>>,
        window: Option<i64>,
        normalize: Option<bool>,
        pct: Option<f64>,
        white_frame: Option<bool>,
        downsample: Option<i64>,
    ) -> Result<Self, String> {
        let downsample = match downsample {
            Some(factor) if factor >= 1 => factor as usize,
            Some(_) => return Err("downsample must be at least 1".to_owned()),
            None => 1,
        };
        Ok(match name {
            "binary" => Self::Binary,
            "count" => Self::Count {
                normalize: normalize.unwrap_or(false),
            },
            "polarity" => Self::Polarity {
                normalize: normalize.unwrap_or(true),
                downsample,
            },
            "voxel" => Self::Voxel {
                bins: bins
                    .map(|value| {
                        usize::try_from(value).map_err(|_| "bins must be at least 1".to_owned())
                    })
                    .transpose()?
                    .unwrap_or(9),
                window_ms: window_ms.unwrap_or(DEFAULT_SPAN_MS),
            },
            "tsurf" => Self::TimeSurface {
                tau_ms: tau_ms.unwrap_or(DEFAULT_SPAN_MS),
            },
            "atsurf" => Self::AveragedTimeSurface {
                tau_ms: tau_ms.unwrap_or(DEFAULT_SPAN_MS),
            },
            "tencode" => Self::Tencode {
                window_ms: window_ms.unwrap_or(DEFAULT_SPAN_MS),
            },
            "redblue" => Self::RedBlue {
                pct: pct.unwrap_or(99.0),
            },
            "countmask" => Self::CountMask {
                pct: pct.unwrap_or(99.0),
                white_frame: white_frame.unwrap_or(false),
            },
            "mcts" => Self::Mcts {
                max_window_ms: max_window_ms.unwrap_or(DEFAULT_SPAN_MS),
                windows,
            },
            "flow" => Self::Flow {
                window: window
                    .map(|value| {
                        usize::try_from(value)
                            .ok()
                            .filter(|&w| w >= 1)
                            .ok_or_else(|| "window must be at least 1".to_owned())
                    })
                    .transpose()?
                    .unwrap_or(3),
            },
            other => return Err(format!("unknown representation: {other}")),
        })
    }

    /// A spec from a name alone, for the callers that don't accept per-representation options —
    /// `open(repr=…)`, `flatten`/`view`, `camera.show` — which take at most `normalize`.
    pub fn named(name: &str, normalize: Option<bool>) -> Result<Self, String> {
        Self::named_with_span(name, normalize, None)
    }

    /// [`ReprSpec::named_with_span`] plus the polarity `downsample` factor (ignored by the others).
    pub fn named_with_span_downsample(
        name: &str,
        normalize: Option<bool>,
        span_ms: Option<f64>,
        downsample: Option<i64>,
    ) -> Result<Self, String> {
        Self::new(
            name, None, span_ms, span_ms, span_ms, None, None, normalize, None, None, downsample,
        )
    }

    /// [`ReprSpec::named`], with unset time spans defaulting to `span_ms` — the slice window a
    /// reader or stream was fetched as — instead of the fixed 30 ms, mirroring `stream()`'s
    /// live-camera rule so the representation covers exactly the events it is handed.
    pub fn named_with_span(
        name: &str,
        normalize: Option<bool>,
        span_ms: Option<f64>,
    ) -> Result<Self, String> {
        Self::new(
            name, None, span_ms, span_ms, span_ms, None, None, normalize, None, None, None,
        )
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Custom(_) => "custom",
            Self::Binary => "binary",
            Self::Count { .. } => "count",
            Self::Polarity { .. } => "polarity",
            Self::Voxel { .. } => "voxel",
            Self::TimeSurface { .. } => "tsurf",
            Self::AveragedTimeSurface { .. } => "atsurf",
            Self::Tencode { .. } => "tencode",
            Self::CountMask { .. } => "countmask",
            Self::RedBlue { .. } => "redblue",
            Self::Mcts { .. } => "mcts",
            Self::Flow { .. } => "flow",
        }
    }

    /// Renders on the session's device (see `eventcv.set_device`).
    ///
    /// Unlike [`Representation::generate`], which stays the CPU reference implementation, a
    /// `reader = ecv.open(..., repr="voxel")` that never names a device should follow whatever the
    /// session was set to. Representations with no kernel fall back to the CPU on their own.
    pub fn generate(&self, stream: &EventStream) -> Result<EventFrame, RepresentationError> {
        self.generate_on(stream, crate::accel::default_device())
    }

    pub fn generate_on(
        &self,
        stream: &EventStream,
        device: crate::accel::Device,
    ) -> Result<EventFrame, RepresentationError> {
        match self {
            Self::Custom(kernel) => kernel.generate(stream),
            Self::Binary => Binary.generate_on(stream, device),
            Self::Count { normalize } => EventCount::new(*normalize).generate_on(stream, device),
            Self::Polarity {
                normalize,
                downsample,
            } => Polarity::new(*normalize)
                .with_downsample(*downsample)
                .generate_on(stream, device),
            Self::Voxel { bins, window_ms } => {
                VoxelGrid::new(*bins, *window_ms).generate_on(stream, device)
            }
            Self::TimeSurface { tau_ms } => TimeSurface::new(*tau_ms).generate_on(stream, device),
            Self::AveragedTimeSurface { tau_ms } => {
                AveragedTimeSurface::new(*tau_ms).generate_on(stream, device)
            }
            Self::Tencode { window_ms } => Tencode::new(*window_ms).generate_on(stream, device),
            Self::RedBlue { pct } => RedBlue::new(*pct).generate_on(stream, device),
            Self::CountMask { pct, white_frame } => {
                CountMask::new(*pct, *white_frame).generate_on(stream, device)
            }
            Self::Mcts {
                max_window_ms,
                windows,
            } => match windows {
                Some(list) => Mcts::with_windows(list.clone()).generate_on(stream, device),
                None => Mcts::new(*max_window_ms).generate_on(stream, device),
            },
            Self::Flow { window } => stream.optical_flow(*window).map_err(|error| match error {
                FlowError::SizeOverflow => RepresentationError::SizeOverflow,
                FlowError::InvalidParameter(name) => RepresentationError::InvalidParameter(name),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_name_round_trips() {
        for name in ReprSpec::NAMES {
            assert_eq!(ReprSpec::named(name, None).unwrap().name(), name);
        }
    }

    #[test]
    fn unknown_name_is_an_error() {
        assert_eq!(
            ReprSpec::named("nope", None).unwrap_err(),
            "unknown representation: nope"
        );
    }

    #[test]
    fn span_fills_unset_windows() {
        match ReprSpec::named_with_span("tencode", None, Some(50.0)).unwrap() {
            ReprSpec::Tencode { window_ms } => assert_eq!(window_ms, 50.0),
            other => panic!("unexpected spec {other:?}"),
        }
    }
}
