//! Lean playback GUI: egui controls rendered through the viewer's process-global winit/wgpu loop.
//! File IO, event processing and frame generation stay on one worker thread; the main thread only
//! owns widgets and the GPU surface.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use egui::{Color32, Pos2, Stroke, TextureHandle, TextureOptions, Vec2};
use egui_wgpu::{Renderer as EguiRenderer, ScreenDescriptor};
use eventcv_core::io::{self, LoadOptions, MemorySliceSource, Reader};
use eventcv_core::viz::{Colormap, RawSurface, Rgb8Image, Scale};
use eventcv_core::EventStream;
#[cfg(feature = "camera")]
use eventcv_core::EventStreamBuilder;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoopProxy};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use super::gpu::{self, Failable, GpuContext, UserEvent};
use crate::{apply_slice_ops, RenderView, ReprSpec, SliceOps};

const DEFAULT_DT_MS: f64 = 30.0;
const DEFAULT_REFRESH_HZ: f64 = 60.0;
const MAX_REFRESH_HZ: f64 = 240.0;
const HISTORY_LIMIT: usize = 2_048;
const ACTIVE_POLL: Duration = Duration::from_millis(4);
const IDLE_POLL: Duration = Duration::from_millis(100);

#[derive(Clone, Copy)]
pub(crate) struct Options {
    pub(crate) dt_ms: f64,
    pub(crate) refresh_hz: f64,
    pub(crate) speed: f64,
    pub(crate) loop_: bool,
    pub(crate) max_frames: Option<usize>,
    pub(crate) repr: Option<ReprSpec>,
    pub(crate) colormap: Colormap,
    pub(crate) clim: Option<f64>,
    pub(crate) decay_ms: Option<f64>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            dt_ms: DEFAULT_DT_MS,
            refresh_hz: DEFAULT_REFRESH_HZ,
            speed: 1.0,
            loop_: false,
            max_frames: None,
            repr: None,
            colormap: Colormap::Viridis,
            clim: None,
            decay_ms: None,
        }
    }
}

struct PlaybackSource {
    reader: Arc<Mutex<Reader>>,
    ops: SliceOps,
    hot_pixel_mask: Option<Arc<[bool]>>,
    name: String,
}

impl PlaybackSource {
    fn opened(path: &Path) -> Result<Self, String> {
        let reader = io::open(path, LoadOptions::default()).map_err(|error| error.to_string())?;
        Ok(Self {
            reader: Arc::new(Mutex::new(reader)),
            ops: crate::no_slice_ops(),
            hot_pixel_mask: None,
            name: path.display().to_string(),
        })
    }

    fn metadata(&self) -> Metadata {
        let guard = self.reader.lock().unwrap();
        let (width, height) = guard.sensor_size();
        let (start_us, end_us) = guard.time_span();
        Metadata {
            name: self.name.clone(),
            width,
            height,
            n_events: guard.n_events(),
            start_us,
            end_us,
            live: false,
            skipped_windows: 0,
        }
    }

    fn fetch(&self, t0: i64, t1: i64, index: usize) -> Result<EventStream, String> {
        let stream = self
            .reader
            .lock()
            .map_err(|_| "event reader lock was poisoned".to_owned())?
            .slice_time(t0, t1)
            .map_err(|error| error.to_string())?;
        let stream = match &self.hot_pixel_mask {
            Some(mask) => stream.drop_masked_pixels(mask),
            None => stream,
        };
        Ok(apply_slice_ops(&self.ops, index, stream))
    }
}

#[derive(Clone)]
struct Metadata {
    name: String,
    width: usize,
    height: usize,
    n_events: usize,
    start_us: i64,
    end_us: i64,
    live: bool,
    skipped_windows: usize,
}

impl Metadata {
    fn duration_us(&self) -> i64 {
        if self.n_events == 0 {
            0
        } else {
            self.end_us.saturating_sub(self.start_us).max(0)
        }
    }

    fn mean_rate(&self) -> f64 {
        let seconds = self.duration_us() as f64 / 1_000_000.0;
        if seconds > 0.0 {
            self.n_events as f64 / seconds
        } else {
            0.0
        }
    }
}

#[derive(Clone, Copy, Default)]
struct WindowStats {
    total: u64,
    positive: u64,
    negative: u64,
    rate: f64,
}

fn window_stats(stream: &EventStream, bin_us: i64) -> WindowStats {
    let rate = stream.event_rate(bin_us);
    let total = rate.counts.iter().sum();
    let positive = rate.positive.iter().sum();
    let negative = rate.negative.iter().sum();
    WindowStats {
        total,
        positive,
        negative,
        rate: total as f64 / (bin_us.max(1) as f64 / 1_000_000.0),
    }
}

struct ProcessorPlugin {
    name: &'static str,
    default_dt_us: i64,
    build: fn(i64) -> Box<dyn Processor>,
}

static PROCESSOR_PLUGINS: [ProcessorPlugin; 2] = [
    ProcessorPlugin {
        name: "Background activity",
        default_dt_us: 25_000,
        build: |dt_us| Box::new(BackgroundProcessor::new(dt_us)),
    },
    ProcessorPlugin {
        name: "Refractory",
        default_dt_us: 1_000,
        build: |dt_us| Box::new(RefractoryProcessor::new(dt_us)),
    },
];

#[derive(Clone, Copy)]
struct ProcessorConfig {
    plugin: &'static ProcessorPlugin,
    enabled: bool,
    dt_us: i64,
}

impl PartialEq for ProcessorConfig {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.plugin, other.plugin)
            && self.enabled == other.enabled
            && self.dt_us == other.dt_us
    }
}

impl ProcessorConfig {
    fn new(plugin: &'static ProcessorPlugin) -> Self {
        Self {
            plugin,
            enabled: true,
            dt_us: plugin.default_dt_us,
        }
    }

    fn name(&self) -> &'static str {
        self.plugin.name
    }

    fn enabled_mut(&mut self) -> &mut bool {
        &mut self.enabled
    }

    fn dt_mut(&mut self) -> &mut i64 {
        &mut self.dt_us
    }

    fn build(&self) -> Option<Box<dyn Processor>> {
        self.enabled.then(|| (self.plugin.build)(self.dt_us))
    }
}

/// Internal compiled-in processor boundary. The retained input tail makes the existing stateless
/// EventStream filters behave continuously across playback windows and later live-camera packets.
trait Processor: Send {
    fn lookback_us(&self) -> i64;
    fn process(&mut self, window: (i64, i64), stream: EventStream) -> EventStream;
    fn reset(&mut self);
}

fn process_with_history(
    history: &mut Option<EventStream>,
    dt_us: i64,
    window: (i64, i64),
    stream: EventStream,
    filter: impl FnOnce(&EventStream) -> EventStream,
) -> EventStream {
    let combined = match history.take() {
        Some(previous) if !previous.is_empty() => previous.concat(&[&stream]),
        _ => stream,
    };
    let filtered = filter(&combined).time_window(window.0, window.1);
    *history = Some(combined.time_window(window.1.saturating_sub(dt_us), window.1));
    filtered
}

struct BackgroundProcessor {
    dt_us: i64,
    history: Option<EventStream>,
}

impl BackgroundProcessor {
    fn new(dt_us: i64) -> Self {
        Self {
            dt_us: dt_us.max(1),
            history: None,
        }
    }
}

impl Processor for BackgroundProcessor {
    fn lookback_us(&self) -> i64 {
        self.dt_us
    }

    fn process(&mut self, window: (i64, i64), stream: EventStream) -> EventStream {
        process_with_history(&mut self.history, self.dt_us, window, stream, |events| {
            events.background_activity_filter(self.dt_us)
        })
    }

    fn reset(&mut self) {
        self.history = None;
    }
}

struct RefractoryProcessor {
    dt_us: i64,
    history: Option<EventStream>,
}

impl RefractoryProcessor {
    fn new(dt_us: i64) -> Self {
        Self {
            dt_us: dt_us.max(1),
            history: None,
        }
    }
}

impl Processor for RefractoryProcessor {
    fn lookback_us(&self) -> i64 {
        self.dt_us
    }

    fn process(&mut self, window: (i64, i64), stream: EventStream) -> EventStream {
        process_with_history(&mut self.history, self.dt_us, window, stream, |events| {
            events.refractory_filter(self.dt_us)
        })
    }

    fn reset(&mut self) {
        self.history = None;
    }
}

struct Pipeline {
    processors: Vec<Box<dyn Processor>>,
}

impl Pipeline {
    fn new(config: &[ProcessorConfig]) -> Self {
        Self {
            processors: config.iter().filter_map(ProcessorConfig::build).collect(),
        }
    }

    fn lookback_us(&self) -> i64 {
        // Chained filters can extend one another's required input history, so sum is the safe bound.
        self.processors.iter().fold(0_i64, |total, processor| {
            total.saturating_add(processor.lookback_us())
        })
    }

    fn reset(&mut self) {
        for processor in &mut self.processors {
            processor.reset();
        }
    }

    fn process(&mut self, window: (i64, i64), mut stream: EventStream) -> EventStream {
        for processor in &mut self.processors {
            stream = processor.process(window, stream);
        }
        stream
    }
}

#[derive(Clone, Copy)]
struct ViewConfig {
    repr: Option<ReprSpec>,
    colormap: Colormap,
    clim: Option<f64>,
    decay_ms: Option<f64>,
}

impl From<Options> for ViewConfig {
    fn from(options: Options) -> Self {
        Self {
            repr: options.repr,
            colormap: options.colormap,
            clim: options.clim,
            decay_ms: options.decay_ms,
        }
    }
}

/// Built-in source boundary. File and memory readers share the seekable arm; the camera arm feeds
/// the same rolling window and processor pipeline without inventing a native plugin ABI.
enum InputSource {
    Recording(PlaybackSource),
    #[cfg(feature = "camera")]
    Live {
        pump: Arc<Mutex<crate::capture::Pump>>,
    },
}

struct WorkerSource {
    source: InputSource,
    metadata: Metadata,
    view_config: ViewConfig,
    view: Option<RenderView>,
    pipeline: Pipeline,
    pipeline_config: Vec<ProcessorConfig>,
    raw_window: Option<EventStream>,
    processed_window: Option<EventStream>,
    last_start_us: Option<i64>,
    last_end_us: Option<i64>,
    last_dt_us: i64,
}

impl WorkerSource {
    fn new(source: PlaybackSource, view_config: ViewConfig) -> Self {
        let metadata = source.metadata();
        Self {
            source: InputSource::Recording(source),
            metadata,
            view_config,
            view: None,
            pipeline: Pipeline::new(&[]),
            pipeline_config: Vec::new(),
            raw_window: None,
            processed_window: None,
            last_start_us: None,
            last_end_us: None,
            last_dt_us: 0,
        }
    }

    #[cfg(feature = "camera")]
    fn live(
        pump: Arc<Mutex<crate::capture::Pump>>,
        name: String,
        width: usize,
        height: usize,
        view_config: ViewConfig,
    ) -> Self {
        Self {
            source: InputSource::Live { pump },
            metadata: Metadata {
                name,
                width,
                height,
                n_events: 0,
                start_us: 0,
                end_us: 0,
                live: true,
                skipped_windows: 0,
            },
            view_config,
            view: None,
            pipeline: Pipeline::new(&[]),
            pipeline_config: Vec::new(),
            raw_window: None,
            processed_window: None,
            last_start_us: None,
            last_end_us: None,
            last_dt_us: 0,
        }
    }

    fn recording(&self) -> &PlaybackSource {
        match &self.source {
            InputSource::Recording(source) => source,
            #[cfg(feature = "camera")]
            InputSource::Live { .. } => unreachable!("live input has its own render path"),
        }
    }

    fn reset(&mut self, dt_us: i64, config: &[ProcessorConfig]) {
        self.pipeline = Pipeline::new(config);
        self.pipeline.reset();
        self.pipeline_config = config.to_vec();
        self.raw_window = None;
        self.processed_window = None;
        self.last_start_us = None;
        self.last_end_us = None;
        self.last_dt_us = dt_us;
        self.view = Some(match self.view_config.repr {
            None => RenderView::Raw(RawSurface::new(
                self.metadata.width,
                self.metadata.height,
                self.view_config.decay_ms.unwrap_or(dt_us as f64 / 1_000.0),
            )),
            Some(spec) => RenderView::Repr {
                spec,
                colormap: self.view_config.colormap,
                scale: match self.view_config.clim {
                    Some(value) if value > 0.0 => Scale::Fixed(value),
                    _ => Scale::Auto,
                },
            },
        });
    }

    fn render(&mut self, request: &RenderRequest) -> Result<FrameResult, String> {
        #[cfg(feature = "camera")]
        if matches!(self.source, InputSource::Live { .. }) {
            return self.render_live(request);
        }
        let t0 = self.metadata.start_us.saturating_add(request.cursor_us);
        let t1 = t0.saturating_add(request.dt_us);
        let incremental = !request.discontinuous
            && self.recording().ops.is_empty()
            && self.last_start_us.is_some_and(|last| t0 > last)
            && self.last_end_us.is_some_and(|last| t0 <= last && t1 > last)
            && self.last_dt_us == request.dt_us
            && self.pipeline_config == request.processors;
        if !incremental || self.view.is_none() {
            self.reset(request.dt_us, &request.processors);
            let render_lookback = if self.view_config.repr.is_none() {
                (self
                    .view_config
                    .decay_ms
                    .unwrap_or(request.dt_us as f64 / 1_000.0)
                    * 6_500.0)
                    .round() as i64
            } else {
                0
            };
            let mut remaining = self.pipeline.lookback_us().saturating_add(render_lookback);
            let mut end = t0;
            let mut index = request.index;
            let mut preroll = Vec::new();
            while remaining > 0 && end > self.metadata.start_us {
                let start = end
                    .saturating_sub(request.dt_us)
                    .max(self.metadata.start_us);
                index = index.saturating_sub(1);
                preroll.push((start, end, index));
                remaining = remaining.saturating_sub(end - start);
                end = start;
            }
            // Apply source-side lazy operations with the same per-window indices they get during
            // uninterrupted playback, then warm the GUI processors from oldest to newest.
            for (start, end, index) in preroll.into_iter().rev() {
                let pre = self.recording().fetch(start, end, index)?;
                let processed = self.pipeline.process((start, end), pre);
                if matches!(&self.view, Some(RenderView::Raw(_))) {
                    let _ = self
                        .view
                        .as_mut()
                        .expect("reset creates a view")
                        .render(&processed)
                        .map_err(|error| error.to_string())?;
                }
            }
        }

        let (raw, processed) = if incremental {
            let tail_start = self
                .last_end_us
                .expect("incremental render has a previous end");
            let tail = self.recording().fetch(tail_start, t1, request.index)?;
            let raw = self
                .raw_window
                .take()
                .expect("incremental render has a raw window")
                .concat(&[&tail])
                .time_window(t0, t1);
            let processed_tail = self.pipeline.process((tail_start, t1), tail);
            let processed = self
                .processed_window
                .take()
                .expect("incremental render has a processed window")
                .concat(&[&processed_tail])
                .time_window(t0, t1);
            (raw, processed)
        } else {
            let raw = self.recording().fetch(t0, t1, request.index)?;
            let processed = self.pipeline.process((t0, t1), raw.clone());
            (raw, processed)
        };
        let raw_stats = window_stats(&raw, request.dt_us);
        let processed_stats = window_stats(&processed, request.dt_us);
        let image = self
            .view
            .as_mut()
            .expect("reset creates a view")
            .render(&processed)
            .map_err(|error| error.to_string())?;
        self.raw_window = Some(raw);
        self.processed_window = Some(processed);
        self.last_start_us = Some(t0);
        self.last_end_us = Some(t1);
        Ok(FrameResult {
            generation: request.generation,
            cursor_us: request.cursor_us,
            image: Some(image),
            raw: raw_stats,
            processed: processed_stats,
            metadata: None,
        })
    }

    #[cfg(feature = "camera")]
    fn render_live(&mut self, request: &RenderRequest) -> Result<FrameResult, String> {
        let mut reconfigure = request.discontinuous
            || self.last_dt_us != request.dt_us
            || self.pipeline_config != request.processors
            || self.view.is_none();
        let previous_raw = self.raw_window.take();
        let previous_processed = self.processed_window.take();
        let previous_end = self.last_end_us;
        if reconfigure {
            self.reset(request.dt_us, &request.processors);
        }

        let (packet, skipped) = match &self.source {
            InputSource::Live { pump } => {
                let pump = pump
                    .lock()
                    .map_err(|_| "camera pump lock was poisoned".to_owned())?;
                (
                    pump.next_window(Duration::ZERO)?
                        .map(|window| window.stream),
                    pump.n_skipped(),
                )
            }
            InputSource::Recording(_) => unreachable!("checked before entering live render"),
        };
        let has_packet = packet.is_some();
        if skipped > self.metadata.skipped_windows && !reconfigure {
            reconfigure = true;
            self.reset(request.dt_us, &request.processors);
        }
        self.metadata.skipped_windows = skipped;

        if let Some(packet) = packet {
            self.metadata.n_events = self.metadata.n_events.saturating_add(packet.len());
            if let (Some(&first), Some(&last)) = (packet.ts().first(), packet.ts().last()) {
                if self.metadata.n_events == packet.len() {
                    self.metadata.start_us = first;
                }
                self.metadata.end_us = last;
            }
            let packet_window = (
                packet.ts().first().copied().unwrap_or(self.metadata.end_us),
                packet
                    .ts()
                    .last()
                    .copied()
                    .unwrap_or(self.metadata.end_us)
                    .saturating_add(1),
            );
            let raw = previous_raw
                .map(|events| events.concat(&[&packet]))
                .unwrap_or_else(|| packet.clone());
            let end = packet_window.1;
            let start = end.saturating_sub(request.dt_us);
            let raw = raw.time_window(start, end);
            let processed = if reconfigure {
                self.pipeline.process((start, end), raw.clone())
            } else {
                let tail = self.pipeline.process(packet_window, packet);
                previous_processed
                    .map(|events| events.concat(&[&tail]))
                    .unwrap_or(tail)
                    .time_window(start, end)
            };
            self.raw_window = Some(raw);
            self.processed_window = Some(processed);
            self.last_start_us = Some(start);
            self.last_end_us = Some(end);
        } else if reconfigure {
            let end = previous_end.unwrap_or(0);
            let start = end.saturating_sub(request.dt_us);
            let raw = previous_raw
                .unwrap_or_else(|| {
                    EventStreamBuilder::new(self.metadata.width, self.metadata.height, 0.001)
                        .build()
                })
                .time_window(start, end);
            let processed = self.pipeline.process((start, end), raw.clone());
            self.raw_window = Some(raw);
            self.processed_window = Some(processed);
            self.last_start_us = Some(start);
            self.last_end_us = previous_end;
        } else {
            self.raw_window = previous_raw;
            self.processed_window = previous_processed;
            self.last_end_us = previous_end;
        }

        let raw = self.raw_window.clone().unwrap_or_else(|| {
            EventStreamBuilder::new(self.metadata.width, self.metadata.height, 0.001).build()
        });
        let processed = self.processed_window.clone().unwrap_or_else(|| raw.clone());
        let raw_stats = window_stats(&raw, request.dt_us);
        let processed_stats = window_stats(&processed, request.dt_us);
        let image = if has_packet || reconfigure {
            Some(
                self.view
                    .as_mut()
                    .expect("live reset creates a view")
                    .render(&processed)
                    .map_err(|error| error.to_string())?,
            )
        } else {
            None
        };
        Ok(FrameResult {
            generation: request.generation,
            cursor_us: 0,
            image,
            raw: raw_stats,
            processed: processed_stats,
            metadata: Some(self.metadata.clone()),
        })
    }
}

#[derive(Clone)]
struct RenderRequest {
    generation: u64,
    cursor_us: i64,
    dt_us: i64,
    index: usize,
    processors: Vec<ProcessorConfig>,
    discontinuous: bool,
}

enum Command {
    SetSource {
        generation: u64,
        source: PlaybackSource,
        view: ViewConfig,
    },
    #[cfg(feature = "camera")]
    SetLive {
        generation: u64,
        pump: Arc<Mutex<crate::capture::Pump>>,
        name: String,
        width: usize,
        height: usize,
        view: ViewConfig,
    },
    Open {
        generation: u64,
        path: PathBuf,
        view: ViewConfig,
    },
    Render(RenderRequest),
    Shutdown,
}

struct FrameResult {
    generation: u64,
    cursor_us: i64,
    image: Option<Rgb8Image>,
    raw: WindowStats,
    processed: WindowStats,
    metadata: Option<Metadata>,
}

enum ResultMessage {
    Loaded { generation: u64, metadata: Metadata },
    Frame(FrameResult),
    Error { generation: u64, message: String },
}

fn worker_loop(commands: Receiver<Command>, results: Sender<ResultMessage>) {
    let mut source: Option<WorkerSource> = None;
    let mut queued = None;
    loop {
        let command = match queued.take() {
            Some(command) => command,
            None => match commands.recv() {
                Ok(command) => command,
                Err(_) => return,
            },
        };
        match command {
            Command::Shutdown => return,
            Command::SetSource {
                generation,
                source: new_source,
                view,
            } => {
                let new_source = WorkerSource::new(new_source, view);
                let metadata = new_source.metadata.clone();
                source = Some(new_source);
                let _ = results.send(ResultMessage::Loaded {
                    generation,
                    metadata,
                });
            }
            #[cfg(feature = "camera")]
            Command::SetLive {
                generation,
                pump,
                name,
                width,
                height,
                view,
            } => {
                let new_source = WorkerSource::live(pump, name, width, height, view);
                let metadata = new_source.metadata.clone();
                source = Some(new_source);
                let _ = results.send(ResultMessage::Loaded {
                    generation,
                    metadata,
                });
            }
            Command::Open {
                generation,
                path,
                view,
            } => match PlaybackSource::opened(&path) {
                Ok(new_source) => {
                    let new_source = WorkerSource::new(new_source, view);
                    let metadata = new_source.metadata.clone();
                    source = Some(new_source);
                    let _ = results.send(ResultMessage::Loaded {
                        generation,
                        metadata,
                    });
                }
                Err(message) => {
                    let _ = results.send(ResultMessage::Error {
                        generation,
                        message,
                    });
                }
            },
            Command::Render(mut request) => {
                // Keep the newest interactive seek/configuration request instead of rendering a
                // backlog the user can no longer see. Source/open/shutdown commands keep order.
                while let Ok(next) = commands.try_recv() {
                    match next {
                        Command::Render(newer) => request = newer,
                        other => {
                            queued = Some(other);
                            break;
                        }
                    }
                }
                let result = match source.as_mut() {
                    Some(source) => source.render(&request).map(ResultMessage::Frame),
                    None => Err("open a recording before playing".to_owned()),
                };
                let message = match result {
                    Ok(message) => message,
                    Err(message) => ResultMessage::Error {
                        generation: request.generation,
                        message,
                    },
                };
                let _ = results.send(message);
            }
        }
    }
}

struct RateSample {
    raw: f64,
    processed: f64,
}

struct Model {
    commands: Sender<Command>,
    results: Receiver<ResultMessage>,
    view: ViewConfig,
    metadata: Option<Metadata>,
    processors: Vec<ProcessorConfig>,
    history: VecDeque<RateSample>,
    cursor_us: i64,
    dt_us: i64,
    refresh_hz: f64,
    speed: f64,
    loop_: bool,
    max_frames: Option<usize>,
    playing: bool,
    pending: bool,
    loading: bool,
    play_after_load: bool,
    generation: u64,
    next_due: Instant,
    raw_stats: WindowStats,
    processed_stats: WindowStats,
    error: Option<String>,
    image: Option<Rgb8Image>,
    texture: Option<TextureHandle>,
    image_size: [usize; 2],
}

impl Model {
    fn new(commands: Sender<Command>, results: Receiver<ResultMessage>, options: Options) -> Self {
        Self {
            commands,
            results,
            view: options.into(),
            metadata: None,
            processors: Vec::new(),
            history: VecDeque::new(),
            cursor_us: 0,
            dt_us: ms_to_us(options.dt_ms),
            refresh_hz: options.refresh_hz.clamp(1.0, MAX_REFRESH_HZ),
            speed: options.speed,
            loop_: options.loop_,
            max_frames: options.max_frames,
            playing: false,
            pending: false,
            loading: false,
            play_after_load: false,
            generation: 0,
            next_due: Instant::now(),
            raw_stats: WindowStats::default(),
            processed_stats: WindowStats::default(),
            error: None,
            image: None,
            texture: None,
            image_size: [0, 0],
        }
    }

    fn next_generation(&mut self) -> u64 {
        self.generation = self.generation.wrapping_add(1);
        self.generation
    }

    fn set_initial_source(&mut self, source: PlaybackSource) {
        let generation = self.next_generation();
        self.loading = true;
        self.pending = true;
        self.play_after_load = true;
        let _ = self.commands.send(Command::SetSource {
            generation,
            source,
            view: self.view,
        });
    }

    #[cfg(feature = "camera")]
    fn set_live_source(
        &mut self,
        pump: Arc<Mutex<crate::capture::Pump>>,
        name: String,
        width: usize,
        height: usize,
    ) {
        let generation = self.next_generation();
        self.loading = true;
        self.pending = true;
        self.play_after_load = true;
        let _ = self.commands.send(Command::SetLive {
            generation,
            pump,
            name,
            width,
            height,
            view: self.view,
        });
    }

    fn open(&mut self, path: PathBuf) {
        let generation = self.next_generation();
        self.loading = true;
        self.pending = true;
        self.playing = false;
        self.play_after_load = false;
        self.error = None;
        let _ = self.commands.send(Command::Open {
            generation,
            path,
            view: self.view,
        });
    }

    fn choose_file(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter(
                "Event recordings",
                &[
                    "npz", "txt", "csv", "h5", "hdf5", "bag", "aedat", "aedat4", "dat", "raw",
                ],
            )
            .pick_file()
        {
            self.open(path);
        }
    }

    fn poll(&mut self) -> bool {
        let mut changed = false;
        while let Ok(message) = self.results.try_recv() {
            match message {
                ResultMessage::Loaded {
                    generation,
                    metadata,
                } if generation == self.generation => {
                    self.metadata = Some(metadata);
                    self.cursor_us = 0;
                    self.history.clear();
                    self.image = None;
                    self.texture = None;
                    self.loading = false;
                    self.pending = false;
                    self.playing = self.play_after_load
                        && self
                            .metadata
                            .as_ref()
                            .is_some_and(|meta| meta.live || meta.n_events > 0);
                    self.request_render(true);
                    changed = true;
                }
                ResultMessage::Frame(frame) if frame.generation == self.generation => {
                    self.pending = false;
                    self.cursor_us = frame.cursor_us;
                    if let Some(metadata) = frame.metadata {
                        self.metadata = Some(metadata);
                    }
                    if let Some(image) = frame.image {
                        self.raw_stats = frame.raw;
                        self.processed_stats = frame.processed;
                        self.image_size = [image.width, image.height];
                        self.image = Some(image);
                        self.push_history(frame.raw.rate, frame.processed.rate);
                        changed = true;
                    }
                }
                ResultMessage::Error {
                    generation,
                    message,
                } if generation == self.generation => {
                    self.error = Some(message);
                    self.pending = false;
                    self.loading = false;
                    self.playing = false;
                    changed = true;
                }
                _ => {}
            }
        }
        changed
    }

    fn push_history(&mut self, raw: f64, processed: f64) {
        self.history.push_back(RateSample { raw, processed });
        if self.history.len() > HISTORY_LIMIT {
            self.history.pop_front();
        }
    }

    fn max_cursor_us(&self) -> i64 {
        let Some(metadata) = &self.metadata else {
            return 0;
        };
        max_cursor_us(
            metadata.duration_us(),
            self.dt_us,
            self.step_us(),
            self.max_frames,
        )
    }

    fn step_us(&self) -> i64 {
        (1_000_000.0 * self.speed.max(0.01) / self.refresh_hz.max(1.0))
            .round()
            .max(1.0) as i64
    }

    fn frame_interval(&self) -> Duration {
        Duration::from_secs_f64(1.0 / self.refresh_hz.max(1.0))
    }

    fn request_render(&mut self, discontinuous: bool) {
        let Some(metadata) = &self.metadata else {
            return;
        };
        if metadata.n_events == 0 && !metadata.live {
            self.pending = false;
            self.playing = false;
            return;
        }
        self.cursor_us = self.cursor_us.clamp(0, self.max_cursor_us());
        let generation = self.next_generation();
        self.pending = true;
        let index = usize::try_from(self.cursor_us / self.step_us()).unwrap_or(0);
        let _ = self.commands.send(Command::Render(RenderRequest {
            generation,
            cursor_us: self.cursor_us,
            dt_us: self.dt_us,
            index,
            processors: self.processors.clone(),
            discontinuous,
        }));
        self.next_due = Instant::now() + self.frame_interval();
    }

    fn invalidate(&mut self) {
        self.history.clear();
        self.request_render(true);
    }

    fn tick(&mut self, now: Instant) -> bool {
        if !self.playing || self.pending || now < self.next_due {
            return false;
        }
        if self.metadata.as_ref().is_some_and(|metadata| metadata.live) {
            self.request_render(false);
            return true;
        }
        let max = self.max_cursor_us();
        if self.cursor_us >= max {
            if self.loop_ && max > 0 {
                self.cursor_us = 0;
                self.history.clear();
                self.request_render(true);
            } else {
                self.playing = false;
            }
            return true;
        }
        self.cursor_us = self.cursor_us.saturating_add(self.step_us()).min(max);
        self.request_render(false);
        true
    }

    fn step(&mut self, direction: i64) {
        if self.metadata.as_ref().is_none_or(|metadata| metadata.live) {
            return;
        }
        self.playing = false;
        self.cursor_us = self
            .cursor_us
            .saturating_add(self.step_us().saturating_mul(direction))
            .clamp(0, self.max_cursor_us());
        self.history.clear();
        self.request_render(true);
    }

    fn upload_image(&mut self, context: &egui::Context) {
        let Some(image) = self.image.take() else {
            return;
        };
        let color = egui::ColorImage::from_rgb([image.width, image.height], &image.pixels);
        match &mut self.texture {
            Some(texture) => texture.set(color, TextureOptions::NEAREST),
            None => {
                self.texture =
                    Some(context.load_texture("eventcv-frame", color, TextureOptions::NEAREST));
            }
        }
    }

    fn ui(&mut self, context: &egui::Context) {
        self.upload_image(context);
        self.shortcuts(context);

        egui::TopBottomPanel::top("menu").show(context, |ui| {
            ui.horizontal(|ui| {
                let live = self.metadata.as_ref().is_some_and(|metadata| metadata.live);
                if ui
                    .add_enabled(!live, egui::Button::new("Open…"))
                    .on_hover_text("Open a recording (Ctrl/Cmd+O)")
                    .clicked()
                {
                    self.choose_file();
                }
                if let Some(metadata) = &self.metadata {
                    ui.label(&metadata.name);
                } else {
                    ui.label("Drop a recording here, or choose Open…");
                }
                if self.loading {
                    ui.spinner();
                    ui.label("Opening…");
                }
            });
            if let Some(error) = &self.error {
                ui.colored_label(Color32::LIGHT_RED, error);
            }
        });

        egui::SidePanel::right("inspector")
            .default_width(300.0)
            .show(context, |ui| self.inspector(ui));
        egui::TopBottomPanel::bottom("transport").show(context, |ui| self.transport(ui));
        egui::CentralPanel::default().show(context, |ui| self.image_panel(ui));

        if let Some(path) = context.input(|input| {
            input
                .raw
                .dropped_files
                .iter()
                .find_map(|file| file.path.clone())
        }) {
            if !self.metadata.as_ref().is_some_and(|metadata| metadata.live) {
                self.open(path);
            }
        }
    }

    fn shortcuts(&mut self, context: &egui::Context) {
        let open = context.input_mut(|input| {
            input.consume_shortcut(&egui::KeyboardShortcut::new(
                egui::Modifiers::COMMAND,
                egui::Key::O,
            ))
        });
        if open && !self.metadata.as_ref().is_some_and(|metadata| metadata.live) {
            self.choose_file();
        }
        if context.wants_keyboard_input() {
            return;
        }
        if context.input(|input| input.key_pressed(egui::Key::Space))
            && self
                .metadata
                .as_ref()
                .is_some_and(|meta| meta.live || meta.n_events > 0)
        {
            self.playing = !self.playing;
            self.next_due = Instant::now();
        }
        if context.input(|input| input.key_pressed(egui::Key::ArrowLeft)) {
            self.step(-1);
        }
        if context.input(|input| input.key_pressed(egui::Key::ArrowRight)) {
            self.step(1);
        }
    }

    fn transport(&mut self, ui: &mut egui::Ui) {
        let live = self.metadata.as_ref().is_some_and(|meta| meta.live);
        let enabled = self
            .metadata
            .as_ref()
            .is_some_and(|meta| meta.live || meta.n_events > 0);
        ui.add_enabled_ui(enabled, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(!live, egui::Button::new("Previous"))
                    .on_hover_text("Previous accumulation window (Left)")
                    .clicked()
                {
                    self.step(-1);
                }
                if ui
                    .button(if self.playing { "Pause" } else { "Play" })
                    .on_hover_text("Play or pause (Space)")
                    .clicked()
                {
                    self.playing = !self.playing;
                    self.next_due = Instant::now();
                }
                if ui
                    .add_enabled(!live, egui::Button::new("Next"))
                    .on_hover_text("Next accumulation window (Right)")
                    .clicked()
                {
                    self.step(1);
                }
                if live {
                    ui.label("LIVE");
                } else {
                    if ui.checkbox(&mut self.loop_, "Loop").changed() {
                        self.invalidate();
                    }
                    egui::ComboBox::from_label("Speed")
                        .selected_text(format!("{}×", format_number(self.speed)))
                        .show_ui(ui, |ui| {
                            for speed in [0.25, 0.5, 1.0, 2.0, 4.0] {
                                ui.selectable_value(
                                    &mut self.speed,
                                    speed,
                                    format!("{}×", format_number(speed)),
                                );
                            }
                        });
                }
            });

            if !live {
                let mut seconds = self.cursor_us as f64 / 1_000_000.0;
                let max_seconds = self.max_cursor_us() as f64 / 1_000_000.0;
                let response = ui.add(
                    egui::Slider::new(&mut seconds, 0.0..=max_seconds.max(0.0))
                        .show_value(false)
                        .text("Recording position"),
                );
                if response.changed() {
                    self.cursor_us = (seconds * 1_000_000.0).round() as i64;
                    self.history.clear();
                    self.request_render(true);
                }
                ui.label(format!(
                    "{} / {}",
                    format_time(self.cursor_us),
                    format_time(self.metadata.as_ref().map_or(0, Metadata::duration_us))
                ));
            }
        });
    }

    fn inspector(&mut self, ui: &mut egui::Ui) {
        ui.heading("Source");
        if let Some(metadata) = &self.metadata {
            egui::Grid::new("file-stats").show(ui, |ui| {
                stat_row(
                    ui,
                    "Sensor",
                    format!("{} × {}", metadata.width, metadata.height),
                );
                stat_row(ui, "Events", format_count(metadata.n_events as u64));
                stat_row(ui, "Duration", format_time(metadata.duration_us()));
                stat_row(ui, "Mean rate", format_rate(metadata.mean_rate()));
                if metadata.live {
                    stat_row(
                        ui,
                        "Skipped windows",
                        format_count(metadata.skipped_windows as u64),
                    );
                }
            });
        } else {
            ui.label("No recording open");
        }

        ui.separator();
        ui.heading("Display");
        let mut dt_ms = self.dt_us as f64 / 1_000.0;
        let slider = ui.add(
            egui::Slider::new(&mut dt_ms, 0.1..=1_000.0)
                .logarithmic(true)
                .text("Accumulation (ms)"),
        );
        let drag = ui
            .horizontal(|ui| {
                ui.label("Exact accumulation");
                ui.add(
                    egui::DragValue::new(&mut dt_ms)
                        .range(0.1..=1_000.0)
                        .speed(0.1)
                        .suffix(" ms"),
                )
            })
            .inner;
        if slider.changed() || drag.changed() {
            self.dt_us = ms_to_us(dt_ms);
            self.cursor_us = self.cursor_us.min(self.max_cursor_us());
            self.invalidate();
        }
        let refresh = ui.add(
            egui::Slider::new(&mut self.refresh_hz, 1.0..=MAX_REFRESH_HZ).text("Refresh (Hz)"),
        );
        refresh.on_hover_text(
            "Sliding-window update rate; one render is kept in flight so slow processing applies backpressure",
        );

        ui.separator();
        ui.heading("Current window");
        egui::Grid::new("window-stats").show(ui, |ui| {
            stat_row(ui, "Raw events", format_count(self.raw_stats.total));
            stat_row(ui, "Raw rate", format_rate(self.raw_stats.rate));
            stat_row(
                ui,
                "Output events",
                format_count(self.processed_stats.total),
            );
            stat_row(ui, "Processed", format_rate(self.processed_stats.rate));
            stat_row(
                ui,
                "Raw ON / OFF",
                format!(
                    "{} / {}",
                    format_count(self.raw_stats.positive),
                    format_count(self.raw_stats.negative)
                ),
            );
            stat_row(
                ui,
                "Output ON / OFF",
                format!(
                    "{} / {}",
                    format_count(self.processed_stats.positive),
                    format_count(self.processed_stats.negative)
                ),
            );
            let retained = if self.raw_stats.total == 0 {
                100.0
            } else {
                self.processed_stats.total as f64 / self.raw_stats.total as f64 * 100.0
            };
            stat_row(ui, "Retained", format!("{retained:.1}%"));
        });
        rate_plot(ui, &self.history);

        ui.separator();
        ui.heading("Processing");
        let mut action = None;
        let mut changed = false;
        for (index, processor) in self.processors.iter_mut().enumerate() {
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    let name = processor.name();
                    changed |= ui.checkbox(processor.enabled_mut(), name).changed();
                    if ui
                        .small_button("Up")
                        .on_hover_text("Move earlier")
                        .clicked()
                    {
                        action = Some(PipelineAction::Up(index));
                    }
                    if ui
                        .small_button("Down")
                        .on_hover_text("Move later")
                        .clicked()
                    {
                        action = Some(PipelineAction::Down(index));
                    }
                    if ui.small_button("Remove").clicked() {
                        action = Some(PipelineAction::Remove(index));
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Temporal window");
                    changed |= ui
                        .add(
                            egui::DragValue::new(processor.dt_mut())
                                .range(1..=1_000_000)
                                .speed(10.0)
                                .suffix(" µs"),
                        )
                        .on_hover_text("Temporal window in microseconds")
                        .changed();
                });
            });
        }
        ui.menu_button("Add processor", |ui| {
            for plugin in &PROCESSOR_PLUGINS {
                if ui.button(plugin.name).clicked() {
                    self.processors.push(ProcessorConfig::new(plugin));
                    changed = true;
                    ui.close_menu();
                }
            }
        });
        if let Some(action) = action {
            changed |= apply_pipeline_action(&mut self.processors, action);
        }
        if changed {
            self.invalidate();
        }
    }

    fn image_panel(&self, ui: &mut egui::Ui) {
        let Some(texture) = &self.texture else {
            ui.centered_and_justified(|ui| {
                ui.label(if self.loading {
                    "Opening source…"
                } else {
                    "Open or drop an event recording"
                });
            });
            return;
        };
        let available = ui.available_size();
        let source = Vec2::new(self.image_size[0] as f32, self.image_size[1] as f32);
        let scale = (available.x / source.x.max(1.0))
            .min(available.y / source.y.max(1.0))
            .max(0.01);
        let size = source * scale;
        ui.centered_and_justified(|ui| {
            ui.add(
                egui::Image::new(texture)
                    .fit_to_exact_size(size)
                    .texture_options(TextureOptions::NEAREST),
            );
        });
    }
}

enum PipelineAction {
    Up(usize),
    Down(usize),
    Remove(usize),
}

fn apply_pipeline_action(processors: &mut Vec<ProcessorConfig>, action: PipelineAction) -> bool {
    match action {
        PipelineAction::Up(index) if index > 0 => processors.swap(index, index - 1),
        PipelineAction::Down(index) if index + 1 < processors.len() => {
            processors.swap(index, index + 1)
        }
        PipelineAction::Remove(index) if index < processors.len() => {
            processors.remove(index);
        }
        _ => return false,
    }
    true
}

fn max_cursor_us(duration_us: i64, dt_us: i64, step_us: i64, max_frames: Option<usize>) -> i64 {
    if duration_us <= 0 || dt_us <= 0 || step_us <= 0 {
        return 0;
    }
    let natural = if duration_us <= dt_us {
        0
    } else {
        duration_us / step_us * step_us
    };
    max_frames.map_or(natural, |frames| {
        natural.min(
            i64::try_from(frames.saturating_sub(1))
                .unwrap_or(i64::MAX)
                .saturating_mul(step_us),
        )
    })
}

fn ms_to_us(milliseconds: f64) -> i64 {
    (milliseconds.clamp(0.1, 1_000.0) * 1_000.0).round() as i64
}

fn stat_row(ui: &mut egui::Ui, name: &str, value: String) {
    ui.label(name);
    ui.label(value);
    ui.end_row();
}

fn format_count(value: u64) -> String {
    let digits = value.to_string();
    digits
        .chars()
        .rev()
        .enumerate()
        .flat_map(|(index, digit)| {
            (index > 0 && index % 3 == 0)
                .then_some(',')
                .into_iter()
                .chain([digit])
        })
        .collect::<String>()
        .chars()
        .rev()
        .collect()
}

fn format_rate(rate: f64) -> String {
    if rate >= 1_000_000.0 {
        format!("{:.2} Mev/s", rate / 1_000_000.0)
    } else if rate >= 1_000.0 {
        format!("{:.1} kev/s", rate / 1_000.0)
    } else {
        format!("{rate:.0} ev/s")
    }
}

fn format_time(microseconds: i64) -> String {
    let seconds = microseconds.max(0) as f64 / 1_000_000.0;
    if seconds >= 60.0 {
        format!("{}:{:05.2}", (seconds / 60.0) as u64, seconds % 60.0)
    } else {
        format!("{seconds:.3} s")
    }
}

fn format_number(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.0}")
    } else {
        value.to_string()
    }
}

fn rate_plot(ui: &mut egui::Ui, history: &VecDeque<RateSample>) {
    let (rect, _) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), 110.0), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, Color32::from_gray(24));
    if history.len() < 2 {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "Rate history",
            egui::FontId::default(),
            Color32::GRAY,
        );
        return;
    }
    let max_rate = history.iter().fold(1.0_f64, |max, sample| {
        max.max(sample.raw).max(sample.processed)
    });
    let line = |value: fn(&RateSample) -> f64| {
        history
            .iter()
            .enumerate()
            .map(|(index, sample)| {
                let x = rect.left() + index as f32 / (history.len() - 1) as f32 * rect.width();
                let y = rect.bottom() - (value(sample) / max_rate) as f32 * rect.height();
                Pos2::new(x, y)
            })
            .collect::<Vec<_>>()
    };
    painter.add(egui::Shape::line(
        line(|sample| sample.raw),
        Stroke::new(1.5_f32, Color32::LIGHT_RED),
    ));
    painter.add(egui::Shape::line(
        line(|sample| sample.processed),
        Stroke::new(1.5_f32, Color32::LIGHT_GREEN),
    ));
    painter.text(
        rect.left_top() + Vec2::new(6.0, 5.0),
        egui::Align2::LEFT_TOP,
        "raw / processed",
        egui::FontId::proportional(11.0),
        Color32::LIGHT_GRAY,
    );
}

struct GuiRender {
    gpu: GpuContext,
    context: egui::Context,
    state: egui_winit::State,
    renderer: EguiRenderer,
}

impl GuiRender {
    fn new(window: Arc<Window>, proxy: EventLoopProxy<UserEvent>) -> Result<Self, String> {
        let gpu = gpu::create_gpu(window.clone())?;
        let context = egui::Context::default();
        let mut state = egui_winit::State::new(
            context.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(gpu.device.limits().max_texture_dimension_2d as usize),
        );
        state.init_accesskit(window.as_ref(), proxy);
        let renderer = EguiRenderer::new(&gpu.device, gpu.config.format, None, 1, false);
        Ok(Self {
            gpu,
            context,
            state,
            renderer,
        })
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.gpu.config.width = size.width;
        self.gpu.config.height = size.height;
        self.gpu
            .surface
            .configure(&self.gpu.device, &self.gpu.config);
    }

    fn draw(&mut self, window: &Window, model: &mut Model) -> Result<(), String> {
        let input = self.state.take_egui_input(window);
        let output = self.context.run(input, |context| model.ui(context));
        self.state
            .handle_platform_output(window, output.platform_output);
        let pixels_per_point = egui_winit::pixels_per_point(&self.context, window);
        let paint_jobs = self.context.tessellate(output.shapes, pixels_per_point);
        for (id, delta) in &output.textures_delta.set {
            self.renderer
                .update_texture(&self.gpu.device, &self.gpu.queue, *id, delta);
        }

        let frame = match self.gpu.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.gpu
                    .surface
                    .configure(&self.gpu.device, &self.gpu.config);
                return Ok(());
            }
            Err(wgpu::SurfaceError::Timeout) => return Ok(()),
            Err(error) => return Err(error.to_string()),
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let descriptor = ScreenDescriptor {
            size_in_pixels: [self.gpu.config.width, self.gpu.config.height],
            pixels_per_point,
        };
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("eventcv-gui"),
            });
        let mut command_buffers = self.renderer.update_buffers(
            &self.gpu.device,
            &self.gpu.queue,
            &mut encoder,
            &paint_jobs,
            &descriptor,
        );
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("eventcv-gui-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.025,
                            b: 0.05,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.renderer
                .render(&mut pass.forget_lifetime(), &paint_jobs, &descriptor);
        }
        command_buffers.push(encoder.finish());
        self.gpu.queue.submit(command_buffers);
        frame.present();
        for id in &output.textures_delta.free {
            self.renderer.free_texture(id);
        }
        Ok(())
    }
}

enum InitialSource {
    Recording(PlaybackSource),
    #[cfg(feature = "camera")]
    Live {
        pump: Arc<Mutex<crate::capture::Pump>>,
        name: String,
        width: usize,
        height: usize,
    },
}

struct GuiApp {
    initial: Option<InitialSource>,
    model: Model,
    worker: Option<JoinHandle<()>>,
    window: Option<Arc<Window>>,
    render: Option<GuiRender>,
    proxy: Option<EventLoopProxy<UserEvent>>,
    error: Option<String>,
}

impl GuiApp {
    fn new(initial: Option<InitialSource>, options: Options) -> Self {
        let (command_tx, command_rx) = mpsc::channel();
        let (result_tx, result_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || worker_loop(command_rx, result_tx));
        Self {
            initial,
            model: Model::new(command_tx, result_rx, options),
            worker: Some(worker),
            window: None,
            render: None,
            proxy: None,
            error: None,
        }
    }

    fn shutdown(&mut self, event_loop: &ActiveEventLoop) {
        let _ = self.model.commands.send(Command::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        self.render = None;
        self.window = None;
        event_loop.set_control_flow(ControlFlow::Poll);
        event_loop.exit();
    }

    fn fail(&mut self, event_loop: &ActiveEventLoop, message: String) {
        self.error = Some(message);
        self.shutdown(event_loop);
    }
}

impl Failable for GuiApp {
    fn take_error(&mut self) -> Option<String> {
        self.error.take()
    }

    fn set_event_loop_proxy(&mut self, proxy: EventLoopProxy<UserEvent>) {
        self.proxy = Some(proxy);
    }
}

impl ApplicationHandler<UserEvent> for GuiApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title("eventcv - playback")
            .with_inner_size(LogicalSize::new(1_100, 720));
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(error) => return self.fail(event_loop, error.to_string()),
        };
        let Some(proxy) = self.proxy.clone() else {
            return self.fail(
                event_loop,
                "event loop proxy was not initialised".to_owned(),
            );
        };
        match GuiRender::new(window.clone(), proxy) {
            Ok(render) => {
                self.render = Some(render);
                self.window = Some(window.clone());
                if let Some(source) = self.initial.take() {
                    match source {
                        InitialSource::Recording(source) => self.model.set_initial_source(source),
                        #[cfg(feature = "camera")]
                        InitialSource::Live {
                            pump,
                            name,
                            width,
                            height,
                        } => self.model.set_live_source(pump, name, width, height),
                    }
                }
                window.request_redraw();
                event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + ACTIVE_POLL));
            }
            Err(error) => self.fail(event_loop, error),
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if matches!(event, WindowEvent::CloseRequested) {
            return self.shutdown(event_loop);
        }
        let consumed = match (&mut self.render, &self.window) {
            (Some(render), Some(window)) => render.state.on_window_event(window, &event).consumed,
            _ => false,
        };
        if !consumed
            && matches!(
                &event,
                WindowEvent::KeyboardInput { event, .. }
                    if event.state == ElementState::Pressed
                        && event.logical_key == Key::Named(NamedKey::Escape)
            )
        {
            return self.shutdown(event_loop);
        }
        match event {
            WindowEvent::Resized(size) => {
                if let Some(render) = &mut self.render {
                    render.resize(size);
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                let result = match (&mut self.render, &self.window) {
                    (Some(render), Some(window)) => render.draw(window, &mut self.model),
                    _ => Ok(()),
                };
                if let Err(error) = result {
                    self.fail(event_loop, error);
                }
            }
            _ if !consumed => {
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        let UserEvent::AccessKit(event) = event;
        if self.window.as_ref().map(|window| window.id()) != Some(event.window_id) {
            return;
        }
        match event.window_event {
            egui_winit::accesskit_winit::WindowEvent::ActionRequested(request) => {
                if let Some(render) = &mut self.render {
                    render.state.on_accesskit_action_request(request);
                }
            }
            egui_winit::accesskit_winit::WindowEvent::InitialTreeRequested => {}
            egui_winit::accesskit_winit::WindowEvent::AccessibilityDeactivated => {}
        }
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let now = Instant::now();
        let changed = self.model.poll() | self.model.tick(now);
        if changed {
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
        let deadline = if self.model.pending || self.model.loading {
            now + ACTIVE_POLL
        } else if self.model.playing {
            self.model.next_due.max(now)
        } else {
            now + IDLE_POLL
        };
        event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
    }
}

impl Drop for GuiApp {
    fn drop(&mut self) {
        let _ = self.model.commands.send(Command::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

pub(crate) fn run_empty(options: Options) -> Result<(), String> {
    gpu::run_app(GuiApp::new(None, options))
}

pub(crate) fn run_reader(
    reader: Arc<Mutex<Reader>>,
    ops: SliceOps,
    hot_pixel_mask: Option<Arc<[bool]>>,
    options: Options,
) -> Result<(), String> {
    let source = PlaybackSource {
        reader,
        ops,
        hot_pixel_mask,
        name: "recording".to_owned(),
    };
    gpu::run_app(GuiApp::new(Some(InitialSource::Recording(source)), options))
}

pub(crate) fn run_stream(stream: EventStream, options: Options) -> Result<(), String> {
    let source = PlaybackSource {
        reader: Arc::new(Mutex::new(Box::new(MemorySliceSource::new(stream)))),
        ops: crate::no_slice_ops(),
        hot_pixel_mask: None,
        name: "in-memory stream".to_owned(),
    };
    gpu::run_app(GuiApp::new(Some(InitialSource::Recording(source)), options))
}

#[cfg(feature = "camera")]
pub(crate) struct CameraResult {
    pub(crate) capture: Option<eventcv_core::device::Capture>,
    pub(crate) recorder: Option<crate::Recorder>,
    pub(crate) skipped: usize,
    pub(crate) overflows: usize,
    pub(crate) result: Result<(), String>,
}

#[cfg(feature = "camera")]
pub(crate) fn run_camera(
    capture: eventcv_core::device::Capture,
    recorder: Option<crate::Recorder>,
    options: Options,
) -> CameraResult {
    let (width, height) = (capture.width(), capture.height());
    let name = capture.name().to_owned();
    let pump = Arc::new(Mutex::new(crate::capture::Pump::start(
        capture,
        recorder,
        crate::capture::Backpressure::Latest,
    )));
    let result = gpu::run_app(GuiApp::new(
        Some(InitialSource::Live {
            pump: Arc::clone(&pump),
            name,
            width,
            height,
        }),
        options,
    ));
    let pump = match Arc::try_unwrap(pump) {
        Ok(pump) => pump,
        Err(_) => panic!("GUI worker released the camera pump"),
    };
    let mut pump = match pump.into_inner() {
        Ok(pump) => pump,
        Err(poisoned) => poisoned.into_inner(),
    };
    let skipped = pump.n_skipped();
    let overflows = pump.n_overflows();
    let (capture, recorder) = pump.stop();
    CameraResult {
        capture,
        recorder,
        skipped,
        overflows,
        result,
    }
}

#[cfg(test)]
mod tests {
    use eventcv_core::EventStreamBuilder;

    use super::*;

    fn sample(rows: &[(u16, u16, i64, bool)]) -> EventStream {
        let mut builder = EventStreamBuilder::new(4, 4, 0.001);
        for &(x, y, t, p) in rows {
            builder.push(x, y, t, p);
        }
        builder.build()
    }

    fn metadata(duration_us: i64) -> Metadata {
        Metadata {
            name: "test".to_owned(),
            width: 4,
            height: 4,
            n_events: 10,
            start_us: 0,
            end_us: duration_us,
            live: false,
            skipped_windows: 0,
        }
    }

    fn model() -> Model {
        let (commands, _command_rx) = mpsc::channel();
        let (_result_tx, results) = mpsc::channel();
        Model::new(commands, results, Options::default())
    }

    fn worker(stream: EventStream) -> WorkerSource {
        WorkerSource::new(
            PlaybackSource {
                reader: Arc::new(Mutex::new(Box::new(MemorySliceSource::new(stream)))),
                ops: crate::no_slice_ops(),
                hot_pixel_mask: None,
                name: "test".to_owned(),
            },
            view_config(),
        )
    }

    fn view_config() -> ViewConfig {
        ViewConfig {
            repr: None,
            colormap: Colormap::Viridis,
            clim: None,
            decay_ms: Some(1.0),
        }
    }

    #[test]
    fn cursor_is_clamped_to_complete_frame_origins() {
        assert_eq!(max_cursor_us(95_000, 30_000, 10_000, None), 90_000);
        assert_eq!(max_cursor_us(95_000, 30_000, 10_000, Some(2)), 10_000);
        assert_eq!(max_cursor_us(0, 30_000, 10_000, None), 0);
        assert_eq!(max_cursor_us(10_000, 30_000, 10_000, None), 0);
        assert_eq!(ms_to_us(0.001), 100);
        assert_eq!(ms_to_us(2_000.0), 1_000_000);
    }

    #[test]
    fn stats_count_both_polarities() {
        let stats = window_stats(
            &sample(&[(0, 0, 0, true), (1, 0, 500, false), (2, 0, 900, true)]),
            1_000,
        );
        assert_eq!((stats.total, stats.positive, stats.negative), (3, 2, 1));
        assert_eq!(stats.rate, 3_000.0);
    }

    #[test]
    fn refractory_history_crosses_window_boundary() {
        let mut processor = RefractoryProcessor::new(1_000);
        let first = processor.process((0, 1_000), sample(&[(0, 0, 900, true)]));
        let second = processor.process((1_000, 2_000), sample(&[(0, 0, 1_100, true)]));
        assert_eq!(first.len(), 1);
        assert!(second.is_empty());
    }

    #[test]
    fn ordered_pipeline_matches_explicit_filter_composition() {
        let input = sample(&[
            (0, 0, 0, true),
            (1, 0, 100, true),
            (1, 0, 150, false),
            (2, 0, 200, true),
        ]);
        let config = [
            ProcessorConfig {
                plugin: &PROCESSOR_PLUGINS[0],
                enabled: true,
                dt_us: 250,
            },
            ProcessorConfig {
                plugin: &PROCESSOR_PLUGINS[1],
                enabled: true,
                dt_us: 100,
            },
        ];
        let expected = input.background_activity_filter(250).refractory_filter(100);
        let actual = Pipeline::new(&config).process((0, 1_000), input.clone());
        assert_eq!(actual.ts(), expected.ts());
        assert_eq!(actual.xs(), expected.xs());
        let (raw, processed) = (window_stats(&input, 1_000), window_stats(&actual, 1_000));
        assert_eq!(raw.total, input.len() as u64);
        assert_eq!(processed.total, actual.len() as u64);
        assert!(processed.total <= raw.total);
    }

    #[test]
    fn sliding_window_keeps_overlap_and_processes_only_the_tail() {
        let recording = sample(&[
            (0, 0, 100, true),
            (1, 0, 600, true),
            (2, 0, 900, false),
            (3, 0, 1_100, true),
            (0, 1, 1_400, false),
        ]);
        let request = |cursor_us, discontinuous| RenderRequest {
            generation: 0,
            cursor_us,
            dt_us: 1_000,
            index: (cursor_us / 500) as usize,
            processors: Vec::new(),
            discontinuous,
        };
        let mut sliding = worker(recording.clone());
        sliding.render(&request(0, true)).unwrap();
        let actual = sliding.render(&request(500, false)).unwrap();
        let expected = worker(recording).render(&request(500, true)).unwrap();

        assert_eq!(actual.raw.total, 4);
        assert_eq!(actual.raw.total, expected.raw.total);
        assert_eq!(
            sliding.raw_window.as_ref().unwrap().ts(),
            &[600, 900, 1_100, 1_400]
        );
        assert_eq!(sliding.last_start_us, Some(600));
    }

    #[test]
    fn refresh_rate_sets_a_smaller_sliding_stride_than_accumulation() {
        let mut player = model();
        assert_eq!(player.dt_us, 30_000);
        assert_eq!(player.step_us(), 16_667);
        player.speed = 2.0;
        assert_eq!(player.step_us(), 33_333);
        player.refresh_hz = 120.0;
        assert_eq!(player.step_us(), 16_667);
    }

    #[test]
    fn seek_preroll_matches_uninterrupted_processing() {
        let recording = sample(&[(0, 0, 900, true), (0, 0, 1_100, true)]);
        let config = [ProcessorConfig {
            plugin: &PROCESSOR_PLUGINS[1],
            enabled: true,
            dt_us: 1_000,
        }];
        let mut uninterrupted = Pipeline::new(&config);
        let _ = uninterrupted.process((0, 1_000), recording.time_window(0, 1_000));
        let expected = uninterrupted.process((1_000, 2_000), recording.time_window(1_000, 2_000));

        let mut after_seek = Pipeline::new(&config);
        let lookback = after_seek.lookback_us();
        let _ = after_seek.process(
            (1_000 - lookback, 1_000),
            recording.time_window(1_000 - lookback, 1_000),
        );
        let actual = after_seek.process((1_000, 2_000), recording.time_window(1_000, 2_000));
        assert_eq!(actual.ts(), expected.ts());
    }

    #[test]
    fn seek_preroll_matches_uninterrupted_raw_rendering() {
        let recording = sample(&[(0, 0, 0, true), (1, 0, 900, false), (2, 0, 1_100, true)]);
        let request = |cursor_us, discontinuous| RenderRequest {
            generation: 0,
            cursor_us,
            dt_us: 1_000,
            index: (cursor_us / 1_000) as usize,
            processors: Vec::new(),
            discontinuous,
        };

        let mut uninterrupted = worker(recording.clone());
        let _ = uninterrupted.render(&request(0, true)).unwrap();
        let expected = uninterrupted.render(&request(1_000, false)).unwrap();
        let actual = worker(recording).render(&request(1_000, true)).unwrap();
        assert_eq!(actual.image, expected.image);
    }

    #[test]
    fn failed_replacement_keeps_the_current_source() {
        let recording = sample(&[(0, 0, 0, true), (1, 0, 900, false)]);
        let source = PlaybackSource {
            reader: Arc::new(Mutex::new(Box::new(MemorySliceSource::new(recording)))),
            ops: crate::no_slice_ops(),
            hot_pixel_mask: None,
            name: "test".to_owned(),
        };
        let (command_tx, command_rx) = mpsc::channel();
        let (result_tx, result_rx) = mpsc::channel();
        let thread = std::thread::spawn(move || worker_loop(command_rx, result_tx));
        command_tx
            .send(Command::SetSource {
                generation: 1,
                source,
                view: view_config(),
            })
            .unwrap();
        assert!(matches!(
            result_rx.recv().unwrap(),
            ResultMessage::Loaded { .. }
        ));
        command_tx
            .send(Command::Open {
                generation: 2,
                path: PathBuf::from("/eventcv/does-not-exist.npz"),
                view: view_config(),
            })
            .unwrap();
        assert!(matches!(
            result_rx.recv().unwrap(),
            ResultMessage::Error { .. }
        ));
        command_tx
            .send(Command::Render(RenderRequest {
                generation: 3,
                cursor_us: 0,
                dt_us: 1_000,
                index: 0,
                processors: Vec::new(),
                discontinuous: true,
            }))
            .unwrap();
        assert!(matches!(result_rx.recv().unwrap(), ResultMessage::Frame(_)));
        command_tx.send(Command::Shutdown).unwrap();
        thread.join().unwrap();
    }

    #[test]
    fn playback_stops_at_end_and_loop_restarts() {
        let mut player = model();
        player.metadata = Some(metadata(95_000));
        player.cursor_us = player.max_cursor_us();
        player.playing = true;
        player.next_due = Instant::now();
        assert!(player.tick(Instant::now()));
        assert!(!player.playing);

        player.loop_ = true;
        player.playing = true;
        player.pending = false;
        assert!(player.tick(Instant::now()));
        assert_eq!(player.cursor_us, 0);
        assert!(player.pending);
    }

    #[test]
    fn stepping_clamps_and_history_resets() {
        let mut player = model();
        player.metadata = Some(metadata(95_000));
        player.push_history(10.0, 5.0);
        player.step(-1);
        assert_eq!(player.cursor_us, 0);
        assert!(player.history.is_empty());
        player.cursor_us = player.max_cursor_us();
        player.step(1);
        assert_eq!(player.cursor_us, player.max_cursor_us());
    }

    #[test]
    fn stale_frames_are_ignored_and_history_is_bounded() {
        let (commands, _command_rx) = mpsc::channel();
        let (result_tx, results) = mpsc::channel();
        let mut player = Model::new(commands, results, Options::default());
        player.generation = 2;
        for generation in [1, 2] {
            result_tx
                .send(ResultMessage::Frame(FrameResult {
                    generation,
                    cursor_us: generation as i64 * 1_000,
                    image: Some(Rgb8Image {
                        width: 1,
                        height: 1,
                        pixels: vec![0, 0, 0],
                    }),
                    raw: WindowStats::default(),
                    processed: WindowStats::default(),
                    metadata: None,
                }))
                .unwrap();
        }
        assert!(player.poll());
        assert_eq!(player.cursor_us, 2_000);
        assert_eq!(player.history.len(), 1);

        for value in 0..=HISTORY_LIMIT {
            player.push_history(value as f64, 0.0);
        }
        assert_eq!(player.history.len(), HISTORY_LIMIT);
        assert_eq!(player.history.front().unwrap().raw, 1.0);
    }
}
