"""Command-line entry point: ``eventcv <command>`` (or ``python -m eventcv <command>``).

EventCV is a library first, and this stays a thin wrapper over it — every command below is a few
lines over the same public API a script would call. It exists for the jobs where opening a REPL is
the annoying part: checking what is actually in a recording, converting a directory of them, and
producing a video to paste into an issue or a slide.

``--version`` is the oldest and most important of these: which version is installed, and whether it
has HDF5 reading, USB streaming and ONNX inference compiled in, are the first questions a bug report
raises.
"""

from __future__ import annotations

import argparse
import platform
import sys

from . import __version__, _ort, _rust


def _onnx_runtime_line() -> str | None:
    """``ONNX Runtime 1.28.0 — /…/libonnxruntime.so.1.28.0 (bundled)``, when this build has ONNX.

    Which runtime got loaded, and which of the five places it came from, is the first question a
    bug report about `Model` raises — and the one thing nothing else in `--version` can answer,
    since it is decided at run time rather than at build time.
    """
    if "onnx" not in getattr(_rust, "__features__", ()):
        return None
    version = _rust.onnx_runtime_version()
    if version is None:
        return "ONNX Runtime: not found (reinstall eventcv, or `pip install eventcv[onnx]`)"
    return f"ONNX Runtime {version} — {_ort.describe()}"


def version_text() -> str:
    """``eventcv 1.0.6 (hdf5, camera)`` plus the runtime and the interpreter it is installed under."""
    features = ", ".join(getattr(_rust, "__features__", ())) or "no optional features"
    lines = [
        f"eventcv {__version__} ({features})",
        _onnx_runtime_line(),
        f"Python {platform.python_version()} on {platform.platform()}",
    ]
    return "\n".join(line for line in lines if line)


def _format_count(value: int) -> str:
    return f"{value:,}"


def _add_source_options(parser: argparse.ArgumentParser) -> None:
    """Options describing how to *read* the input, shared by every command that opens one.

    Text formats carry no unit or column order in the file, and a rosbag has no single obvious
    connection, so these cannot always be inferred — without `--time-unit` a `.txt` recording can
    report a duration off by a factor of a thousand.
    """
    parser.add_argument(
        "--time-unit",
        default=None,
        choices=("s", "ms", "us", "ns"),
        help="timestamp unit of the source, when it cannot be inferred (text formats)",
    )
    parser.add_argument(
        "--order",
        default=None,
        metavar="COLS",
        help="column order of a text source, e.g. txyp or xytp",
    )
    parser.add_argument(
        "--topic", default=None, metavar="TOPIC", help="rosbag connection to read"
    )


def _source_options(args: argparse.Namespace) -> dict:
    """The `--time-unit`/`--order`/`--topic` flags that were actually given.

    `open`/`load` type these as strings and reject an explicit None, so an unused flag must not
    reach them at all.
    """
    return {
        name: value
        for name, value in (
            ("time_unit", args.time_unit),
            ("order", args.order),
            ("topic", args.topic),
        )
        if value is not None
    }


def _cmd_info(args: argparse.Namespace) -> int:
    """Report what is in a recording without decoding all of it."""
    from . import open as open_reader

    # No dt_ms/max_events: this opens the index only, so it stays fast on a multi-GB file.
    reader = open_reader(args.path, **_source_options(args))
    width, height = reader.sensor_size
    n = reader.n_events
    duration_ms = reader.duration_ms
    seconds = duration_ms / 1000.0

    rows = [
        ("file", args.path),
        ("sensor", f"{width} x {height}"),
        ("events", _format_count(n)),
        ("duration", f"{seconds:.3f} s"),
        # Mean rate is the number people actually compare against a camera's spec sheet.
        ("mean rate", f"{_format_count(int(n / seconds)) if seconds > 0 else 'n/a'} ev/s"),
    ]
    if args.rate_bin_ms:
        rate = reader.slice(t0_ms=0, t1_ms=duration_ms).event_rate(bin_ms=args.rate_bin_ms)
        if len(rate["rate"]):
            rows.append(("peak rate", f"{_format_count(int(rate['rate'].max()))} ev/s"))
            rows.append(("bins", f"{len(rate['t'])} x {args.rate_bin_ms} ms"))

    width_key = max(len(key) for key, _ in rows)
    for key, value in rows:
        print(f"{key.rjust(width_key)}  {value}")
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    """Re-encode a recording into another format."""
    from . import load, open as open_reader

    options = _source_options(args)
    reader = open_reader(args.input, **options)
    n_events = reader.n_events

    # Prefer the streaming path — a reader can append straight to the formats that support it
    # (HDF5, E2VID), so a multi-GB recording never has to fit in memory. The rest have to be
    # loaded whole; rather than hardcode which is which and drift from the library, ask it and
    # fall back on the error it raises for exactly this case.
    try:
        reader.save(args.output)
        streamed = True
    except ValueError:
        del reader
        load(args.input, **options).save(args.output)
        streamed = False

    if not args.quiet:
        how = "streamed" if streamed else "loaded"
        print(f"wrote {args.output} ({_format_count(n_events)} events, {how})")
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    """Render a recording to an animation (.gif / .apng / .mp4)."""
    from . import open as open_reader

    reader = open_reader(args.input, dt_ms=args.dt_ms, **_source_options(args))
    # "raw" is not a representation — it draws the events themselves — so it stays out of
    # `with_repr`, which only knows about the dense ones.
    if args.repr != "raw":
        reader = reader.with_repr(args.repr)
    frames = reader.save_video(
        args.output,
        fps=args.fps,
        colormap=args.colormap,
        clim=args.clim,
        max_frames=args.max_frames,
        repr=args.repr,
        decay_ms=args.decay_ms,
    )
    if not args.quiet:
        print(f"wrote {args.output} ({frames} frames at {args.fps:g} fps)")
    return 0


def _cmd_play(args: argparse.Namespace) -> int:
    """Play a recording in an interactive window."""
    from . import open as open_reader

    reader = open_reader(args.input, **_source_options(args))
    reader.play(
        fps=args.fps,
        dt_ms=args.dt_ms,
        decay_ms=args.decay_ms,
        repr=args.repr,
        colormap=args.colormap,
        speed=args.speed,
        loop_=args.loop,
        max_frames=args.max_frames,
    )
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    """Turn a video into the events a DVS would have produced watching it."""
    from . import simulate

    # `out=` streams the events to disk as they are produced. A realistic sensor emits far more
    # than fits in memory at any real resolution, so the CLI never materialises the whole run.
    result = simulate(
        args.input,
        out=args.output,
        pos_thres=args.threshold,
        neg_thres=args.threshold,
        sigma_thres=args.sigma_thres,
        cutoff_hz=args.cutoff_hz,
        leak_rate_hz=args.leak_rate_hz,
        shot_noise_rate_hz=args.shot_noise_rate_hz,
        refractory_us=args.refractory_us,
        seed=args.seed,
        upsample=args.upsample,
        max_upsample=args.max_upsample,
        scale=tuple(args.scale) if args.scale else None,
        max_frames=args.max_frames,
        compression=args.compression,
        # Progress goes to stderr, so it stays out of a redirected stdout; `-q` silences it along
        # with the summary.
        progress=not args.quiet and sys.stderr.isatty(),
    )
    if not args.quiet:
        print(
            f"wrote {args.output} "
            f"({_format_count(result.events)} events from {result.frames} frames)"
        )
    return 0


def _cmd_reconstruct(args: argparse.Namespace) -> int:
    """Run a reconstruction model over a recording to recover an intensity video."""
    from . import Model, open as open_reader, reconstruct

    reader = open_reader(args.input, dt_ms=args.dt_ms, **_source_options(args)).with_repr(
        args.repr, bins=args.bins
    )
    frames = reconstruct(
        reader,
        Model(args.model),
        args.output,
        fps=args.fps,
        max_frames=args.max_frames,
    )
    if not args.quiet:
        print(f"wrote {args.output} ({frames} frames at {args.fps:g} fps)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eventcv",
        description="EventCV — OpenCV for event-based vision. Import `eventcv` to use it as a library.",
    )
    # Printed by hand rather than with argparse's `version` action, which re-wraps its text to the
    # terminal width and would run the two lines together.
    parser.add_argument(
        "-V",
        "--version",
        action="store_true",
        help="print the installed version, the features it was built with, and the interpreter",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    info = subparsers.add_parser(
        "info", help="summarise a recording (sensor size, event count, duration, rate)"
    )
    info.add_argument("path", help="event file to inspect")
    _add_source_options(info)
    info.add_argument(
        "--rate-bin-ms",
        type=float,
        default=None,
        metavar="MS",
        help="also report the peak rate, measured over bins of this width",
    )
    info.set_defaults(func=_cmd_info)

    convert = subparsers.add_parser(
        "convert", help="convert a recording to another format (by output extension)"
    )
    convert.add_argument("input", help="source recording")
    convert.add_argument("output", help="destination; the extension picks the format")
    _add_source_options(convert)
    convert.add_argument("-q", "--quiet", action="store_true", help="suppress the summary line")
    convert.set_defaults(func=_cmd_convert)

    render = subparsers.add_parser(
        "render", help="render a recording to .gif / .apng / .mp4 (.mp4 needs ffmpeg on PATH)"
    )
    render.add_argument("input", help="source recording")
    render.add_argument("output", help="destination; the extension picks the format")
    _add_source_options(render)
    render.add_argument(
        "--dt-ms", type=float, default=30.0, metavar="MS", help="time per frame (default: 30)"
    )
    render.add_argument(
        "--fps", type=float, default=30.0, help="playback frame rate (default: 30)"
    )
    render.add_argument(
        "--repr",
        default="raw",
        help='what to draw: "raw" for the event stream itself (default), or a representation '
        "name (tencode, count, tsurf, voxel, …)",
    )
    render.add_argument(
        "--decay-ms",
        type=float,
        default=None,
        metavar="MS",
        help="raw view only: fade time for an event's trail (default: one frame's worth)",
    )
    render.add_argument(
        "--colormap", default="viridis", help="colormap for scalar representations"
    )
    render.add_argument(
        "--clim",
        type=float,
        default=None,
        metavar="MAX",
        help="fix the value mapped to the top of the colormap; omit to calibrate from the "
        "first frames, or pass 0 for per-frame auto-contrast",
    )
    render.add_argument(
        "--max-frames", type=int, default=None, metavar="N", help="stop after N frames"
    )
    render.add_argument("-q", "--quiet", action="store_true", help="suppress the summary line")
    render.set_defaults(func=_cmd_render)

    play = subparsers.add_parser(
        "play", help="play a recording in an interactive window"
    )
    play.add_argument("input", help="source recording")
    _add_source_options(play)
    play.add_argument(
        "--dt-ms", type=float, default=30.0, metavar="MS", help="time per frame (default: 30)"
    )
    play.add_argument("--fps", type=float, default=30.0, help="display frame rate (default: 30)")
    play.add_argument(
        "--repr",
        default="raw",
        help='what to draw: "raw" for the event stream itself (default), or a representation name',
    )
    play.add_argument(
        "--decay-ms", type=float, default=None, metavar="MS",
        help="raw view only: fade time for an event's trail (default: one frame's worth)",
    )
    play.add_argument("--colormap", default="viridis", help="colormap for scalar representations")
    play.add_argument(
        "--speed", type=float, default=1.0, help="playback rate multiplier (default: 1)"
    )
    play.add_argument("--loop", action="store_true", help="restart at the end instead of closing")
    play.add_argument(
        "--max-frames", type=int, default=None, metavar="N", help="stop after N frames"
    )
    play.set_defaults(func=_cmd_play)

    simulate = subparsers.add_parser(
        "simulate", help="turn a video into synthetic events (needs ffmpeg on PATH)"
    )
    simulate.add_argument("input", help="source video")
    simulate.add_argument("output", help="destination event file; the extension picks the format")
    simulate.add_argument(
        "--threshold", type=float, default=0.2, metavar="C",
        help="log-contrast threshold for both polarities (default: 0.2)",
    )
    simulate.add_argument(
        "--sigma-thres", type=float, default=0.03, metavar="S",
        help="per-pixel threshold spread; 0 for an ideal sensor (default: 0.03)",
    )
    simulate.add_argument(
        "--cutoff-hz", type=float, default=200.0, metavar="HZ",
        help="photoreceptor bandwidth for a white pixel; 0 disables (default: 200)",
    )
    simulate.add_argument(
        "--leak-rate-hz", type=float, default=1.0, metavar="HZ",
        help="spontaneous ON events per pixel per second (default: 1)",
    )
    simulate.add_argument(
        "--shot-noise-rate-hz", type=float, default=10.0, metavar="HZ",
        help="noise floor in dark pixels (default: 10)",
    )
    simulate.add_argument(
        "--refractory-us", type=int, default=100, metavar="US",
        help="dead time after each event (default: 100)",
    )
    simulate.add_argument(
        "--upsample", default=None, metavar="MODE",
        help='frame subdivision: "adaptive" (default), "off", or an integer factor',
    )
    simulate.add_argument(
        "--scale", type=int, nargs=2, default=None, metavar=("W", "H"),
        help="decode the video at this size instead of its native one",
    )
    simulate.add_argument("--max-frames", type=int, default=None, metavar="N",
                          help="stop after N frames")
    simulate.add_argument(
        "--max-upsample", type=int, default=None, metavar="N",
        help="cap the sub-steps any frame pair is split into (default: 64). Each costs a full "
        "pass over every pixel and the busiest pixel sets the factor, so lowering this bounds "
        "the worst case at some cost in timestamp accuracy",
    )
    simulate.add_argument(
        "--compression", type=int, default=None, metavar="LEVEL",
        help="HDF5 gzip level 0..9; 0 stores uncompressed (default: 1)",
    )
    simulate.add_argument("--seed", type=int, default=0, help="seed for mismatch and noise")
    simulate.add_argument("-q", "--quiet", action="store_true", help="suppress the summary line")
    simulate.set_defaults(func=_cmd_simulate)

    reconstruct = subparsers.add_parser(
        "reconstruct", help="recover an intensity video from events using an ONNX model"
    )
    reconstruct.add_argument("input", help="source recording")
    reconstruct.add_argument("output", help="destination video; the extension picks the format")
    reconstruct.add_argument("model", help="ONNX reconstruction model (e.g. an E2VID export)")
    _add_source_options(reconstruct)
    reconstruct.add_argument("--dt-ms", type=float, default=33.0, metavar="MS",
                             help="time per reconstructed frame (default: 33)")
    reconstruct.add_argument("--fps", type=float, default=30.0, help="playback rate (default: 30)")
    reconstruct.add_argument("--repr", default="voxel",
                             help="representation the model expects (default: voxel)")
    reconstruct.add_argument("--bins", type=int, default=5,
                             help="bins for the voxel representation (default: 5)")
    reconstruct.add_argument("--max-frames", type=int, default=None, metavar="N",
                             help="stop after N frames")
    reconstruct.add_argument("-q", "--quiet", action="store_true",
                             help="suppress the summary line")
    reconstruct.set_defaults(func=_cmd_reconstruct)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(version_text())
        return 0
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except KeyboardInterrupt:
        # A long convert or render is expected to be interrupted; 130 is the shell's convention
        # for it, and a traceback here would just be noise.
        print("interrupted", file=sys.stderr)
        return 130
    except FileNotFoundError as error:
        # The library's message is the OS one ("No such file or directory"), which on its own
        # doesn't say *which* file — and for `render` it could equally be the missing ffmpeg.
        message = str(error)
        path = getattr(args, "path", None) or getattr(args, "input", None)
        if path and path not in message:
            message = f"{message}: {path}"
        print(f"error: {message}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        # These carry the actionable message (unreadable format, bad column order, missing
        # ffmpeg); a traceback would bury it.
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
