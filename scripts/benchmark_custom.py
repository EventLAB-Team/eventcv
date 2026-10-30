"""Run against a release build: python scripts/benchmark_custom.py [event_count].

Reports compilation separately from warm medians. RSS is the whole-process high
water mark (including input, all backends and compiler orchestration), not kernel
allocation size. Use a fresh cache directory to measure cold compilation.
"""
import statistics
import sys
import time

import numpy as np
import eventcv as ecv

count = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
rng = np.random.default_rng(0)
rows = np.column_stack((rng.integers(0, 346, count), rng.integers(0, 260, count),
                        np.arange(count), rng.integers(0, 2, count)))
stream = ecv.from_numpy(rows, sensor_size=(346, 260), time_unit='us')

@ecv.representation(channels=1)
def custom(e):
    out = np.zeros((1, e.height, e.width), dtype=np.float32, like=e)
    np.add.at(out, (0, e.y, e.x), 1)
    return out


def numpy_count():
    out = np.zeros((1, 260, 346), np.float32)
    np.add.at(out, (0, rows[:, 1], rows[:, 0]), 1)
    return out

start = time.perf_counter()
custom.compile()
print(f'Preparation (cold only with empty cache): {time.perf_counter() - start:.3f}s')
np.testing.assert_array_equal(custom(stream).numpy(), numpy_count())
for name, run in [('custom', lambda: custom(stream)), ('numpy', numpy_count),
                  ('builtin', lambda: stream.count())]:
    run()
    timings = []
    for _ in range(7):
        start = time.perf_counter()
        run()
        timings.append(time.perf_counter() - start)
    median = statistics.median(timings)
    print(f'{name}: {median * 1000:.3f} ms, {count / median / 1e6:.2f} M events/s')
try:
    import resource
    print(f'Process peak RSS (platform units): {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}')
except ImportError:
    pass
