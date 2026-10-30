"""Experimental NumPy tracing and Rust compilation for dense representations."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import threading
from types import SimpleNamespace

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin


class _Expr(NDArrayOperatorsMixin):
    def __init__(self, code, dtype='f64', mask=None, vector=False):
        self.code, self.dtype, self.mask, self.vector = code, dtype, mask, vector

    def __bool__(self):
        raise TypeError('data-dependent Python branching is unsupported')

    def __array__(self, *args, **kwargs):
        raise TypeError('symbolic events cannot be converted to a NumPy array')

    def __getitem__(self, mask):
        mask = _expr(mask)
        if mask.dtype != 'bool' or not mask.vector or not self.vector or self.mask is not None:
            raise TypeError('only a single boolean event mask is supported')
        return _Expr(self.code, self.dtype, mask.code, True)

    def astype(self, dtype):
        types = {np.dtype('float32'): 'f32', np.dtype('float64'): 'f64',
                 np.dtype('int64'): 'i64', np.dtype('uint64'): 'u64',
                 np.dtype('bool'): 'bool'}
        target = types.get(np.dtype(dtype))
        if target is None or target == 'bool':
            raise TypeError('supported casts: float32, float64, int64, uint64')
        return _Expr(f'({self.code} as {target})', target, self.mask, self.vector)

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if method != '__call__' or kwargs:
            raise TypeError(f'unsupported NumPy operation: {ufunc.__name__}.{method}')
        values = [_expr(a) for a in args]
        masks = {v.mask for v in values if v.vector}
        if len(masks) > 1:
            raise TypeError('event masks must match')
        mask = next(iter(masks), None)
        name = ufunc.__name__
        comparisons = {'equal': '==', 'not_equal': '!=', 'less': '<', 'less_equal': '<=',
                       'greater': '>', 'greater_equal': '>='}
        binary = {'add': '+', 'subtract': '-', 'multiply': '*', 'divide': '/',
                  'true_divide': '/', 'bitwise_and': '&', 'bitwise_or': '|',
                  'logical_and': '&', 'logical_or': '|'}
        dtype = values[0].dtype if all(v.dtype == values[0].dtype for v in values) else 'f64'
        if name in ('logical_and', 'logical_or', 'bitwise_and', 'bitwise_or') and any(v.dtype != 'bool' for v in values):
            raise TypeError('boolean operations require boolean operands')
        if name in ('divide', 'true_divide'):
            dtype = 'f64'
        codes = [v.code if v.dtype == dtype else f'({v.code} as {dtype})' for v in values]
        if name in comparisons or name in binary:
            op = comparisons.get(name, binary.get(name))
            code = f'({codes[0]} {op} {codes[1]})'
            if name in comparisons:
                dtype = 'bool'
        elif name in ('exp', 'floor', 'minimum', 'maximum', 'negative', 'absolute'):
            if name in ('exp', 'floor') and dtype not in ('f32', 'f64'):
                dtype, codes = 'f64', [f'({v.code} as f64)' for v in values]
            if name == 'negative':
                code = f'(-{codes[0]})'
            elif name in ('minimum', 'maximum'):
                # Propagate NaNs as NumPy does, unlike Rust's min/max.
                op = '<' if name == 'minimum' else '>'
                code = f'{{ let a = {codes[0]}; let b = {codes[1]}; if a != a || a {op} b {{ a }} else {{ b }} }}'
            else:
                code = f'({codes[0]}).{ "abs" if name == "absolute" else name }()'
        else:
            raise TypeError(f'unsupported NumPy ufunc: {name}')
        return _Expr(code, dtype, mask, any(v.vector for v in values))

    def __array_function__(self, func, types, args, kwargs):
        if func is np.where and len(args) == 3 and not kwargs:
            condition, left, right = map(_expr, args)
            if condition.dtype != 'bool':
                raise TypeError('where requires a boolean condition')
            masks = {a.mask for a in (condition, left, right) if a.vector}
            if len(masks) > 1:
                raise TypeError('event masks must match')
            return _Expr(f'(if {condition.code} {{ {left.code} as f64 }} else {{ {right.code} as f64 }})',
                         mask=next(iter(masks), None), vector=any(a.vector for a in (condition, left, right)))
        if func is np.clip and len(args) == 3 and not kwargs:
            return np.minimum(np.maximum(args[0], args[1]), args[2])
        raise TypeError(f'unsupported NumPy function: {func.__name__}')


def _expr(value):
    if isinstance(value, _Expr):
        return value
    if isinstance(value, (bool, np.bool_)):
        return _Expr(str(bool(value)).lower(), 'bool')
    if isinstance(value, (int, np.integer)) and -(2**63) <= int(value) < 2**63:
        return _Expr(f'{int(value)}i64', 'i64')
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return _Expr(f'{float(value)!r}f64')
    raise TypeError('expected symbolic events or a finite scalar')


class _Frame:
    def __array_function__(self, func, types, args, kwargs):
        if func is np.stack and kwargs.get('axis', 0) == 0 and set(kwargs) <= {'axis'}:
            frames = args[0]
            if not frames or any(not isinstance(f, _Frame) or len(f.shape) != 2 or f.shape != frames[0].shape for f in frames):
                raise TypeError('stack requires matching two-dimensional custom frames')
            result = _Frame((len(frames), *frames[0].shape))
            for channel, frame in enumerate(frames):
                for op, values, mask in frame.operations:
                    result.operations.append((op, [_expr(channel), *values[1:]], mask))
            return result
        raise TypeError(f'unsupported frame operation: {func.__name__}')

    def __init__(self, shape):
        self.shape, self.operations = shape, []

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if method != 'at' or ufunc not in (np.add, np.maximum) or kwargs or len(args) != 3:
            raise TypeError('frames support only np.add.at and np.maximum.at')
        target, indices, value = args
        if isinstance(indices, tuple) and len(indices) == 2 and len(self.shape) == 2:
            indices = (0, *indices)
        if target is not self or not isinstance(indices, tuple) or len(indices) != 3:
            raise TypeError('scatter indices must be (channel, y, x)')
        values = [_expr(v) for v in (*indices, value)]
        masks = {v.mask for v in values if v.vector}
        if not masks:
            raise TypeError('scatter requires event-array indices or values')
        if len(masks) > 1:
            raise TypeError('scatter event masks must match')
        self.operations.append((ufunc.__name__, values, next(iter(masks), None)))


class _Dimension(int):
    def __new__(cls, name):
        value = super().__new__(cls, 0)
        value.name = name
        return value


    def _unsupported(self, *args):
        raise TypeError('sensor dimensions can only be used in frame allocation shapes')

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _unsupported
    __floordiv__ = __truediv__ = __mod__ = __pow__ = _unsupported


@contextmanager
def _cache_lock(path):
    with open(path, 'a+b') as lock:
        if os.name == 'nt':
            import msvcrt
            lock.write(b'0')
            lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == 'nt':
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class _Events(SimpleNamespace):
    def __array_function__(self, func, types, args, kwargs):
        if func is np.zeros:
            shape = args[0] if args else kwargs['shape']
            if np.dtype(kwargs.get('dtype', float)) != np.dtype('float32'):
                raise TypeError('custom frames require dtype=np.float32')
            if kwargs.get('order', 'C') != 'C':
                raise TypeError('custom frames require C order')
            return _Frame(tuple(shape))
        raise TypeError(f'unsupported allocation: {func.__name__}')


def _source(frame, channels):
    loops = []
    for operation, values, mask in frame.operations:
        c, y, x, value = values
        for index in (c, y, x):
            if index.dtype not in ('i64', 'u64'):
                raise TypeError('scatter indices require integer expressions; use astype(np.int64)')
        update = 'out[j] += value;' if operation == 'add' else 'out[j] = if out[j].is_nan() || out[j] > value { out[j] } else { value };'
        loops.append(f'''for i in 0..n {{
            if !({mask or 'true'}) {{ continue; }}
            let c = {c.code}; let y = {y.code}; let x = {x.code};
            if c < 0 || y < 0 || x < 0 || c as usize >= {channels} || y as usize >= height || x as usize >= width {{ return 2; }}
            let j = (c as usize * height + y as usize) * width + x as usize;
            let value = ({value.code}) as f32; {update}
        }}''')
    return '''// EventCV custom kernel ABI 1. Generated code; CPU only.
#[no_mangle]
pub unsafe extern "C" fn eventcv_kernel_v1(
    xp: *const u16, yp: *const u16, tp: *const i64, pp: *const bool,
    n: usize, width: usize, height: usize, scale: f64, output: *mut f32, len: usize
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let xs = std::slice::from_raw_parts(xp, n);
        let ys = std::slice::from_raw_parts(yp, n);
        let ts = std::slice::from_raw_parts(tp, n);
        let ps = std::slice::from_raw_parts(pp, n);
        let out = std::slice::from_raw_parts_mut(output, len);
        let last = ts.last().copied().unwrap_or(0);
        out.fill(0.0);
''' + '\n'.join(loops) + '\n0\n})).unwrap_or(3)\n}\n'


class CustomRepresentation:
    """A frozen traced representation; compilation is lazy and CPU-only."""
    def __init__(self, source, names):
        self._source, self._names = source, tuple(names)
        self._native = None
        self._lock = threading.Lock()

    @property
    def rust_source(self):
        return self._source

    def compile(self):
        from . import _rust
        with self._lock:
            if self._native is not None:
                return self
            root = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'eventcv' / 'kernels-v1'
            root.mkdir(parents=True, exist_ok=True)
            identity = hashlib.sha256((self._source + platform.platform() + platform.machine() + '-O').encode()).hexdigest()
            index = root / (identity + '.cache')
            try:
                version = subprocess.run(['rustc', '-vV'], check=True, capture_output=True, text=True).stdout
            except (OSError, subprocess.CalledProcessError) as error:
                if index.exists():
                    cached = index.read_text().strip()
                    if Path(cached).name == cached and (root / cached).is_file():
                        self._native = _rust._CustomKernel(str(root / cached), list(self._names))
                        return self
                raise RuntimeError('custom representations require rustc and a platform linker to compile') from error
            key = hashlib.sha256((identity + version).encode()).hexdigest()
            suffix = '.dll' if os.name == 'nt' else '.dylib' if platform.system() == 'Darwin' else '.so'
            binary = root / (key + suffix)
            with _cache_lock(root / (key + '.lock')):
                # Independent temporary builds prevent partial binaries being loaded by other processes.
                if not binary.exists():
                    with tempfile.TemporaryDirectory(dir=root) as directory:
                        source = Path(directory) / 'kernel.rs'
                        result = Path(directory) / ('kernel' + suffix)
                        source.write_text(self._source)
                        process = subprocess.run(['rustc', '--crate-type', 'cdylib', '--edition', '2021', '-O', str(source), '-o', str(result)], capture_output=True, text=True)
                        if process.returncode:
                            raise RuntimeError('custom representation compilation failed:\n' + process.stderr)
                        if not binary.exists():
                            os.replace(result, binary)
                        (root / (key + '.rs')).write_text(self._source)
            with tempfile.NamedTemporaryFile(mode='w', dir=root, delete=False) as manifest:
                manifest.write(binary.name)
            os.replace(manifest.name, index)
            self._native = _rust._CustomKernel(str(binary), list(self._names))
        return self

    def __call__(self, stream):
        return self.compile()._native(stream)


def representation(*, channels, channel_names=None):
    """Trace a function of symbolic events into a compiled float32 representation.

    Allocate with ``np.zeros((channels, e.height, e.width), dtype=np.float32,
    like=e)`` and accumulate using ``np.add.at(out, (channel, e.y, e.x), value)``.
    """
    if isinstance(channels, bool) or not isinstance(channels, int) or channels < 1:
        raise ValueError('channels must be a positive integer')
    names = list(channel_names) if channel_names is not None else [f'channel_{i}' for i in range(channels)]
    if len(names) != channels or any(not isinstance(n, str) for n in names):
        raise ValueError('channel_names must contain one string per channel')
    def decorate(function):
        events = _Events(x=_Expr('(xs[i] as i64)', 'i64', vector=True), y=_Expr('(ys[i] as i64)', 'i64', vector=True),
                         t=_Expr('ts[i]', 'i64', vector=True), p=_Expr('ps[i]', 'bool', vector=True),
                         age_ms=_Expr('((last as i128 - ts[i] as i128) as f64 * scale)', vector=True),
                         timestamp_scale_ms=_Expr('scale'), width=_Dimension('width'), height=_Dimension('height'))
        frame = function(events)
        if not isinstance(frame, _Frame) or (len(frame.shape) != 3 or frame.shape[0] != channels or frame.shape[1] is not events.height or frame.shape[2] is not events.width):
            raise TypeError('representation must return a float32 frame with shape (channels, e.height, e.width)')
        return CustomRepresentation(_source(frame, channels), names)
    return decorate
