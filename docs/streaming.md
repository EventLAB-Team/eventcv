# Live camera streaming

{func}`eventcv.stream` opens a USB event camera as an {class}`~eventcv.EventCamera` — the streaming
twin of {func}`~eventcv.open`. It hands back the same {class}`~eventcv.EventStream` windows the file
readers do, so every representation, transform, feature detector, and viewer in EventCV composes on
a live feed exactly as it does on a recording.

```python
import eventcv as ecv

ecv.list_cameras()          # [{'kind': 'prophesee_evk4', 'name': 'Prophesee EVK4', 'serial': ...}]
ecv.stream().show()         # live raw view; Ctrl+C or close the window to stop
```

Supported: Prophesee EVK4 / EVK3-HD, iniVation DVXplorer / DAVIS346, CenturyArks VGA. Camera support
is compiled into the published wheels; on Linux the device needs udev rules for non-root USB access.

## Closing the camera

A camera allows exactly one open handle, so the device — and any `record=` file — must be released
before the next `stream()` can use it. Closing does that, and it happens in one of three ways:

```python
with ecv.stream(dt_ms=50) as cam:   # closed on the way out, even on an exception
    ...

cam = ecv.stream(dt_ms=50)
cam.close()                         # or explicitly

ecv.record("session.h5", seconds=10)  # opened, recorded, and closed in one call
```

Dropping the camera closes it too, but only whenever Python happens to collect the object — so a
throwaway `ecv.stream()` in a REPL can hold the device well past the line that created it. Prefer
one of the three forms above; for a script that only wants a file, {func}`~eventcv.record` is the
shortest and leaves nothing open.

## Reading windows in a loop

`read()` blocks until the next window completes and returns it — an {class}`~eventcv.EventFrame`
when the camera was opened with `repr=`, otherwise a raw {class}`~eventcv.EventStream`. Windowing
mirrors {func}`~eventcv.open`: `dt_ms` for fixed durations, `max_events` for a fixed event count.

```python
cam = ecv.stream(dt_ms=50, repr="mcts")     # one MCTS frame per ~50 ms of events
while running:
    frame = cam.read()                      # blocks for the next window
    infer(frame.numpy())

frame = cam.read(timeout_ms=100)            # None if nothing completed in 100 ms
```

`timeout_ms` is a cap on the *wait*, not the window length — it lets an idle scene fall through to
other work instead of blocking. Iterating (`for frame in cam:`) is the same thing in `for` form, and
`Ctrl+C` breaks either. Only non-empty windows are emitted, so a loop never spins on idle time.

### Representation options

`stream` accepts the same per-representation options as {meth}`~eventcv.EventReader.with_repr` —
`bins`, `window_ms`, `tau_ms`, `max_window_ms`, `window` (for `"flow"`), and `normalize`. Unset time
spans **follow the capture window**, so a live representation covers exactly the events it was
handed:

```python
ecv.stream(dt_ms=50, repr="tencode")                # window_ms = 50, matching the window
ecv.stream(dt_ms=50, repr="tencode", window_ms=20)  # explicit: keep only the newest 20 ms
ecv.stream(dt_ms=50, repr="voxel", bins=5)          # 5 bins across the 50 ms window
ecv.stream(max_events=50_000, repr="tsurf")         # no fixed duration -> tau_ms stays 30 ms
```

Without that default a `dt_ms=50` window rendered with tencode's stock 30 ms span would silently
discard its oldest 20 ms — on a real 50 ms window, 54% of the lit pixels.

## Recording while you process

Pass `record=` an `.h5`/`.hdf5` path and every window has its **raw** events appended to that file
before it is rendered. The loop works on representations while the file keeps the full-resolution
recording for later:

```python
with ecv.stream(dt_ms=50, repr="mcts", record="session.h5") as cam:
    while running:
        infer(cam.read().numpy())       # this window's raw events are already on disk

ecv.open("session.h5")                  # reads back like any other recording
```

Writing happens on the capture thread, so there is no per-window Python round trip, and the file is
flushed about once a second — a crash keeps everything up to a second ago. `compression=` takes an
optional gzip level (`0..=9`); omit it for the fastest writes. {attr}`~eventcv.EventCamera.n_recorded`
counts events written, and the `with` block (or `close()`) finishes the file.

Only windows that are actually **read** are recorded, since `show()` doesn't poll them. To record
without a loop, use {func}`~eventcv.record`, which opens the camera, captures, and closes it:

```python
ecv.record("session.h5", seconds=10)     # blocks 10 s (or Ctrl+C); returns the event count
ecv.record("session.npz")                # npz/txt/bag buffer in memory, write at the end
ecv.open("session.h5")                   # safe straight after: nothing is still writing
```

{meth}`~eventcv.EventCamera.record` is the same thing on a camera you already hold open — use it
when the session continues afterwards. A camera opened with `record=` is already archiving every
window, so calling `record()` on it as well is refused rather than splitting the recording in two.

For full control, drive the writer yourself with an {class}`~eventcv.EventSink` — the event-level
twin of {class}`~eventcv.FrameSink`, which appends windows to an extendable HDF5 file:

```python
with ecv.stream(dt_ms=50) as cam, ecv.EventSink("session.h5") as sink:
    for events in cam:
        sink.append(events)     # to disk, window-by-window
        track(events)           # ...and process the same window live
```

Only events are saved either way — a DAVIS346's APS frames and IMU samples are dropped.

## How capture keeps up

Decoding, and any `record=` writing, run on a **background thread** that owns the camera. The
driver's ring drains continuously whatever your loop is doing, and the loop only collects windows
that are already decoded — so the per-window budget is yours to spend. On an EVK4 at `dt_ms=50`,
40 ms of processing per frame still holds 19.9 of an ideal 20 fps with the ring empty.

{meth}`~eventcv.EventCamera.show`, {meth}`~eventcv.EventCamera.record`, and
{meth}`~eventcv.EventCamera.close` pause the thread to take the camera back; it restarts on the next
read.

Four counters tell you what the pipeline is doing:

| Property | Meaning |
|----------|---------|
| {attr}`~eventcv.EventCamera.backlog` | Driver ring buffers waiting. Should sit near zero. |
| {attr}`~eventcv.EventCamera.n_skipped` | Windows a `latest=True` loop passed over. |
| {attr}`~eventcv.EventCamera.n_overflows` | Times the driver dropped events — loss upstream of EventCV. |
| {attr}`~eventcv.EventCamera.n_recorded` | Events written by `record=`. |

### Staying on live data

Windows are delivered in order, so a loop slower than the camera falls behind: the thread buffers a
few windows, then applies backpressure, and under sustained overload the driver's ring is what
finally overflows. `latest=True` trades completeness for freshness — each read returns the **newest**
decoded window and drops what it overtook, holding latency at about one window however slow the loop
is. On an EVK4 with a 200 ms per-window workload, a 5 s session ends 4.3 s behind in order versus
0.2 s behind with `latest=True`.

Skipped windows are still written by `record=` from the capture thread, so the archive stays complete
even though the loop never sees them:

```python
with ecv.stream(dt_ms=30, repr="count", record="session.h5", latest=True) as cam:
    while running:
        slow_inference(cam.read().numpy())
print(cam.n_skipped, "windows skipped,", cam.n_recorded, "events recorded")
```

### Capping what the sensor sends

Every event costs time to decode, window, and render, so the cheapest event is one the camera never
emits. `max_event_rate` (events per second) enables the sensor's on-chip event-rate controller, and
`roi=(x0, y0, width, height)` masks every pixel outside that rectangle. Both are hardware features,
so neither costs the host anything:

```python
ecv.stream(dt_ms=50, max_event_rate=40_000_000)   # 40 Mev/s ceiling, enforced on-chip
ecv.stream(dt_ms=50, roi=(320, 180, 640, 360))    # centre quarter only
```

Both are on-chip on the sensors built around Prophesee's pipeline — the EVK4, the EVK3-HD, and the
CenturyArks VGA. The iniVation cameras (DVXplorer, DAVIS346) have neither, so there `roi=` falls back
to a host-side mask: the same events are dropped, but only after crossing the cable and being
decoded, and a `RuntimeWarning` says so. `max_event_rate` still raises there, because capping the
rate on the host would save none of the work the setting exists to avoid.
{attr}`~eventcv.EventCamera.roi` reports which happened:

```python
cam = ecv.stream(roi=(320, 180, 640, 360))
cam.roi     # {'rect': (320, 180, 640, 360), 'applied': 'hardware'}
```

A saturating scene on a 1280×720 sensor can emit far more than one core can decode, and capping the
source is the only remedy that keeps the events you *do* receive contiguous rather than riddled with
dropout holes. Measured on an EVK4 waved to saturation, `dt_ms=50` with `repr="tencode"`:

| `max_event_rate` | Frame rate | Backlog | Pixels lit per frame |
|------------------|-----------|---------|----------------------|
| uncapped         | 6.4 fps   | 1845    | 99.6% of the sensor  |
| 80 Mev/s         | 16.9 fps  | 188     | 92.2%                |
| **40 Mev/s**     | **19.9 fps** | **0** | **83.9%**           |
| 20 Mev/s         | 19.9 fps  | 0       | 60.9%                |

40 Mev/s is the knee here: the first cap that clears the backlog entirely and holds the full frame
rate, while keeping 84% of the lit pixels. Tightening further buys no extra frames and only costs
scene content. The controller drops events proportionally rather than spatially, so the structure of
the scene survives the cull — which is why a rate cap is usually the better default, and `roi` is for
when you genuinely only care about part of the frame.

Where the knee falls depends on your host: the decode pipeline costs about 16 ns per event
(~6 ns parsing the sensor's wire format, ~10 ns accumulating events into a window), i.e. a ceiling
near 62 Mev/s on one core. Caps below that keep up; caps above it leave a backlog.

### Masking a region of interest

`roi=` is a rectangle, fixed when the camera opens. When the region you care about is some other
shape — a circular aperture, a window frame, a lens vignette — or when you need to change it while
the camera runs, pass `mask=` instead: an `(H, W)` boolean array covering the sensor, always enforced
on the host. Events outside it are dropped as they are decoded, before windowing, so they never reach
a `record=` file, the windows your loop reads, or `show()`:

```python
aperture = ecv.circle_mask((640, 480), cx=320, cy=240, r=230)

with ecv.stream(dt_ms=50, mask=aperture, record="session.h5") as cam:
    for events in cam:          # already masked — and so is the recording
        track(events)
```

Easiest is to draw it over the live view. {meth}`~eventcv.EventCamera.draw_mask` opens the viewer,
lets you drag shapes, and applies what you drew to the camera:

```python
cam = ecv.stream(dt_ms=50)
cam.draw_mask()                 # drag an ellipse over the aperture, press Enter
ecv.save_mask(cam.mask, "aperture.png")     # reuse it next session with mask=ecv.load_mask(...)
```

Drag to keep an area, shift+drag to drop one, `e`/`r`/`f` to switch between ellipse, rectangle, and
freehand, `a`/`c` to select all or clear, `z` to undo. Whatever stays bright is what the mask keeps;
`Enter` accepts and `Esc` cancels. Assign {attr}`~eventcv.EventCamera.mask` to change it mid-session
(`None` clears it). Unlike `roi=`, a host-side mask does not reduce what crosses the USB cable — on
a Prophesee sensor, combine the two: `roi=` for the bounding box, `mask=` for the shape inside it.

See {ref}`ROI masking <roi-masking>` for building masks without a camera and for applying them to
recordings.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Frame rate below `1000 / dt_ms`, `backlog` climbing | The camera outruns decoding | Cap with `max_event_rate`, or narrow the field with `roi` |
| Noise outside the useful part of the sensor | The scene doesn't fill the sensor (a circular aperture, a vignette) | `mask=` an ROI of any shape, or draw one with `cam.draw_mask()` |
| `n_overflows` rising | The driver dropped events before EventCV saw them | Same — cap the source; a busy scene on a large sensor can exceed any host |
| Frames arrive but are stale | In-order delivery behind a slow loop | `latest=True` (and check `n_skipped`) |
| `n_skipped` rising fast | Per-window work slower than the camera | Expected with `latest=True`; `record=` still archives everything |
| "no event camera found" with a camera attached | Another handle holds the device, or udev rules are missing | Close the other handle (`with ecv.stream() as cam:`), or install udev rules |
| `record=` raises on a `.npz` path | `record=` appends window-by-window, which needs HDF5 | Use `.h5`/`.hdf5`, or `cam.record("out.npz")` to buffer and write once |
| A panic from `usb.rs` ("callback called for a transfer marked as complete") aborting the process, usually on Windows and usually right after a camera is released | A race in the USB teardown inside `neuromorphic-drivers`, the driver crate underneath EventCV: a cancelled transfer's completion callback can arrive after its slot was retired. It is an abort, not a Python exception, so it cannot be caught | Open the camera **once** and keep it for the session rather than opening and closing one per recording. `ecv.record(...)` and `with ecv.stream(...) as cam:` both close deterministically, which keeps the teardown off unpredictable GC timing |

## Live viewer

{meth}`~eventcv.EventCamera.show` opens an interactive window. With no argument it renders the raw
event stream (polarity dots with exponential decay, tuned by `decay_ms`); pass a representation name
to render that instead:

```python
ecv.stream().show()                     # raw polarity view (the default)
ecv.stream(dt_ms=30).show("count")      # a representation, colour-mapped
```

It blocks on the main thread until the window closes, then the camera is usable again.

## Reference

Full signatures and every {class}`~eventcv.EventCamera` method and property live in the
{ref}`API reference <live-camera-streaming>`: {func}`~eventcv.stream`,
{func}`~eventcv.list_cameras`, {class}`~eventcv.EventCamera`, and {class}`~eventcv.EventSink`.
