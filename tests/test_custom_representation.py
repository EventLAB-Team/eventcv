"""Generated kernels are checked against independent NumPy accumulations."""
import concurrent.futures
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import eventcv as ecv


def weighted():
    @ecv.representation(channels=2, channel_names=['negative', 'positive'])
    def result(e):
        out = np.zeros((2, e.height, e.width), dtype=np.float32, like=e)
        np.add.at(out, (e.p.astype(np.int64), e.y, e.x), np.exp(-e.age_ms / 30))
        return out
    return result


class TestCustomRepresentation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = patch.dict('os.environ', {'XDG_CACHE_HOME': self.temp.name})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.rows = np.array([[0, 0, 2**54, 1], [0, 0, 2**54 + 12, 1],
                              [1, 1, 2**54 + 100, 0]], dtype=np.int64)
        self.stream = ecv.from_numpy(self.rows, sensor_size=(3, 2), time_unit='us')

    def test_weighted_and_concurrent_calls(self):
        custom = weighted()
        expected = np.zeros((2, 2, 3), np.float32)
        age = (self.rows[-1, 2] - self.rows[:, 2]) * .001
        np.add.at(expected, (self.rows[:, 3], self.rows[:, 1], self.rows[:, 0]), np.exp(-age / 30))
        with concurrent.futures.ThreadPoolExecutor(3) as executor:
            frames = list(executor.map(custom, [self.stream] * 3))
        for frame in frames:
            np.testing.assert_allclose(frame.numpy(), expected, rtol=1e-6)
        with patch('subprocess.run', side_effect=AssertionError('recompiled')):
            custom(self.stream)

    def test_mask_stack_and_empty(self):
        @ecv.representation(channels=2)
        def custom(e):
            first = np.zeros((e.height, e.width), dtype=np.float32, like=e)
            second = np.zeros((e.height, e.width), dtype=np.float32, like=e)
            np.add.at(first, (e.y[e.p], e.x[e.p]), 1)
            np.maximum.at(second, (e.y, e.x), e.age_ms)
            return np.stack([first, second])
        output = custom(self.stream).numpy()
        self.assertEqual(output[0, 0, 0], 2)
        self.assertAlmostEqual(output[1, 0, 0], .1, places=6)
        empty = ecv.from_numpy(np.empty((0, 4), np.int64), sensor_size=(4, 3), time_unit='us')
        np.testing.assert_array_equal(custom(empty).numpy(), np.zeros((2, 3, 4)))

    def test_reader_and_frame_roundtrip(self):
        path = str(Path(self.temp.name) / 'events.npz')
        self.stream.save(path)
        reader = ecv.open(path, max_events=2)
        custom = weighted()
        configured = reader.with_repr(custom)
        np.testing.assert_allclose(configured[0], custom(reader.slice(0)).numpy())
        frame = configured.slice(0)
        self.assertEqual(frame.kind, "custom")
        self.assertEqual(frame.channel_names, ("negative", "positive"))
        output = str(Path(self.temp.name) / 'frame.npz')
        frame.save(output)
        restored = ecv.load_frame(output)
        np.testing.assert_array_equal(restored.numpy(), frame.numpy())
        self.assertEqual(restored.kind, frame.kind)
        self.assertEqual(restored.channel_names, frame.channel_names)
        with self.assertRaises(TypeError):
            reader.with_repr(custom, bins=2)

    def test_invalid_index_and_unsupported_operations(self):
        @ecv.representation(channels=1)
        def bad(e):
            out = np.zeros((1, e.height, e.width), dtype=np.float32, like=e)
            np.add.at(out, (2, e.y, e.x), 1)
            return out
        with self.assertRaisesRegex(Exception, 'scatter index'):
            bad(self.stream)
        with self.assertRaisesRegex(TypeError, 'branching'):
            ecv.representation(channels=1)(lambda e: 1 if e.x > 0 else 0)
        with self.assertRaisesRegex(TypeError, 'unsupported'):
            ecv.representation(channels=1)(lambda e: np.sin(e.age_ms))
        with self.assertRaisesRegex(TypeError, 'float32'):
            ecv.representation(channels=1)(lambda e: np.zeros((1, e.height, e.width), like=e))

    def test_temporal_bins_and_configuration(self):
        def make(bins):
            @ecv.representation(channels=bins)
            def custom(e):
                out = np.zeros((bins, e.height, e.width), dtype=np.float32, like=e)
                channel = np.clip(np.floor(e.age_ms / .04), 0, bins - 1).astype(np.int64)
                np.add.at(out, (channel, e.y, e.x), np.where(e.p, 1, -1))
                return out
            return custom
        for bins in (1, 4):
            custom = make(bins)
            expected = np.zeros((bins, 2, 3), np.float32)
            rows = self.stream.numpy().astype(np.int64)
            ages = (rows[-1, 2] - rows[:, 2]) * .001
            channels = np.clip(np.floor(ages / .04), 0, bins - 1).astype(np.int64)
            np.add.at(expected, (channels, rows[:, 1], rows[:, 0]), np.where(rows[:, 3], 1, -1))
            np.testing.assert_array_equal(custom(self.stream).numpy(), expected)

    def test_compile_failure_and_mask_shape(self):
        import subprocess
        run = subprocess.run
        def fail_compile(args, **kwargs):
            if args == ['rustc', '-vV']:
                return run(args, **kwargs)
            return subprocess.CompletedProcess(args, 1, '', 'linker unavailable')
        with patch('subprocess.run', side_effect=fail_compile):
            with self.assertRaisesRegex(RuntimeError, 'linker unavailable'):
                weighted().compile()
        self.assertFalse(list(Path(self.temp.name).rglob('*.cache')))
        with self.assertRaisesRegex(TypeError, 'masks must match'):
            @ecv.representation(channels=1)
            def bad(e):
                out = np.zeros((1, e.height, e.width), dtype=np.float32, like=e)
                np.add.at(out, (0, e.y[e.p], e.x[e.p]), e.age_ms)
                return out

    def test_cache_reuse_and_missing_compiler(self):
        weighted().compile()
        import subprocess
        run = subprocess.run
        def only_version(args, **kwargs):
            self.assertEqual(args, ['rustc', '-vV'])
            return run(args, **kwargs)
        with patch('subprocess.run', side_effect=only_version):
            weighted().compile()
        with patch('subprocess.run', side_effect=FileNotFoundError):
            weighted().compile()
            with patch.dict('os.environ', {'XDG_CACHE_HOME': str(Path(self.temp.name) / 'missing')}):
                with self.assertRaisesRegex(RuntimeError, 'rustc'):
                    weighted().compile()
