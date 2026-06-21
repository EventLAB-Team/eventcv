from . import _rust

EventStream = _rust.EventStream


def load(path: str) -> EventStream:
    """Load N-ImageNet events into a Rust-backed event stream."""
    return _rust.load(path)


__all__ = ["EventStream", "load"]
