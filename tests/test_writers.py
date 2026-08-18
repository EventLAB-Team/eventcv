"""Byte-parity of the event writers against independent implementations.

`save()` reads nine formats and now writes all of them. Round-tripping through eventcv's own
readers (``test_save.py``) proves the pair is self-consistent, not that anyone else can open the
result — so these tests hand each file to the reference implementation of its format and compare
``x``/``y``/``t``/``p`` element by element.

The references are not eventcv dependencies and are deliberately not installed into the project
environment. Point ``EVENTCV_REFERENCE_PYTHON`` at an interpreter that has them and these tests
run; leave it unset and they skip::

    python -m venv /tmp/dvenv
    /tmp/dvenv/bin/pip install numpy dv-processing expelliarmus evlib
    EVENTCV_REFERENCE_PYTHON=/tmp/dvenv/bin/python pytest tests/test_writers.py

Which reference checks which format is not arbitrary; see ``REFERENCES`` below.
"""

import json
import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

import eventcv

REFERENCE_PYTHON = os.environ.get("EVENTCV_REFERENCE_PYTHON")

# A DAVIS346-sized recording: twenty events per timestamp, every other group an ascending run
# along one row (the shape EVT3's vector encoding is for), and 50 ms between groups — comfortably
# more than EVT3's 4096 µs low half, so the encodings' coarse-timestamp handling is exercised.
SENSOR = (346, 260)


def _reference_events():
    index = np.arange(500, dtype=np.int64)
    group = index // 20
    run = group % 2 == 0
    return {
        "x": np.where(run, index % 20, (index * 7) % 300).astype(np.int64),
        "y": np.where(run, group, (index * 13) % 200).astype(np.int64),
        "t": (group * 50_000).astype(np.int64),
        "p": np.where(run, 1, (index % 3 == 0).astype(np.int64)).astype(np.int64),
    }


# `(file name, save format, reference decoder)`.
#
# `expelliarmus` is the reference for the Prophesee formats it decodes correctly, and `evlib` for
# EVT3. That split is measured, not a preference:
#
#   * expelliarmus round-trips `.dat` and EVT2 exactly, and reads eventcv's files for both.
#   * Its EVT3 *encoder* is lossy — it emits `EVT_TIME_HIGH` once for a whole recording and leans
#     on its decoder's low-half wrap rule, which cannot express a gap wider than 4096 µs. It fails
#     to read back its own EVT3 file (a 1.2 s recording comes back as 20 ms), so it cannot serve as
#     the EVT3 oracle.
#   * evlib decodes eventcv's EVT3 exactly. Its EVT2 reader, on the other hand, misreads
#     expelliarmus's own EVT2 file in precisely the way it misreads eventcv's, so it is not the
#     EVT2 oracle either.
REFERENCES = [
    ("out.dat", None, "expelliarmus"),
    ("evt2.raw", "evt2", "expelliarmus"),
    ("evt3.raw", "evt3", "evlib"),
    ("out.aedat4", None, "dv"),
]

# Run inside the reference interpreter: decode one file, print the four columns as JSON.
_DECODE = textwrap.dedent(
    """
    import json, sys
    import numpy as np

    path, reference = sys.argv[1], sys.argv[2]
    if reference == "expelliarmus":
        from expelliarmus import Wizard
        encoding = "dat" if path.endswith(".dat") else Path(path).stem
        array = Wizard(encoding=encoding, fpath=path).read()
        columns = [array["x"], array["y"], array["t"], array["p"]]
    elif reference == "evlib":
        import evlib
        frame = evlib.load_events(path).collect(engine="cpu")
        columns = [
            frame["x"].to_numpy(),
            frame["y"].to_numpy(),
            frame["t"].to_numpy().astype("int64"),
            # evlib reports polarity as -1/+1; eventcv as 0/1.
            (frame["polarity"].to_numpy() > 0).astype("int64"),
        ]
    else:
        import dv_processing as dv
        recording, columns = dv.io.MonoCameraRecording(path), [[], [], [], []]
        while recording.isRunning():
            batch = recording.getNextEventBatch()
            if batch is None:
                continue
            for event in batch:
                for column, value in zip(
                    columns,
                    (event.x(), event.y(), event.timestamp(), int(event.polarity())),
                ):
                    column.append(value)
    print(json.dumps([np.asarray(column).astype("int64").tolist() for column in columns]))
    """
)


@unittest.skipUnless(
    REFERENCE_PYTHON, "set EVENTCV_REFERENCE_PYTHON to an interpreter with the reference decoders"
)
class ByteParityTests(unittest.TestCase):
    """Every format eventcv writes, decoded by somebody else's implementation."""

    @classmethod
    def setUpClass(cls):
        cls.expected = _reference_events()
        cls.directory = Path(tempfile.mkdtemp())
        columns = np.stack(
            [cls.expected[name] for name in ("x", "y", "t", "p")], axis=1
        ).astype(np.int64)
        cls.stream = eventcv.from_numpy(columns, sensor_size=SENSOR, time_unit="us")
        # `-1` keeps the `.stem`-based encoding lookup above honest for `evt2.raw`/`evt3.raw`.
        cls.script = cls.directory / "decode.py"
        cls.script.write_text("from pathlib import Path\n" + _DECODE)

    def _decode(self, path, reference):
        result = subprocess.run(
            [REFERENCE_PYTHON, str(self.script), str(path), reference],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, f"{reference} failed to decode {path}:\n{result.stderr}")
        # evlib prints a PyTorch notice on import, so take the last line rather than all of stdout.
        return json.loads(result.stdout.strip().splitlines()[-1])

    def test_every_written_format_decodes_identically_elsewhere(self):
        for name, fmt, reference in REFERENCES:
            with self.subTest(format=fmt or name, reference=reference):
                path = self.directory / name
                eventcv.save(self.stream, str(path), **({"format": fmt} if fmt else {}))
                decoded = self._decode(path, reference)
                for column, values in zip(("x", "y", "t", "p"), decoded):
                    np.testing.assert_array_equal(
                        np.asarray(values),
                        self.expected[column],
                        err_msg=f"{name} column {column} differs from what {reference} decoded",
                    )


class RealAedatReencodeTests(unittest.TestCase):
    """AEDAT 2.0 has no pip-installable reference decoder, so a real jAER recording stands in.

    A slice of a genuine DAVIS346 file is read, written back out, and the records compared byte for
    byte against the ones jAER itself produced — a stronger check than any decoder written from the
    specification, because it is the specification as an actual recorder implements it.
    """

    SOURCES = [
        Path("data/development/+0+0+0_l_ref.aedat"),
        Path(
            "/media/adam/vprdatasets/data/event-datasets/fast-slow-raw/Fast_Slow/1/aedats/c_h_ref.aedat"
        ),
    ]

    def test_re_encoded_records_match_the_original_bytes(self):
        source = next((path for path in self.SOURCES if path.exists()), None)
        if source is None:
            self.skipTest("no real AEDAT 2.0 recording available on this machine")

        reader = eventcv.open(str(source), max_events=20_000)
        window = reader.slice(0)
        destination = Path(tempfile.mkdtemp()) / "reencoded.aedat"
        eventcv.save(window, str(destination))

        original = _dvs_records(source, len(window))
        written = _dvs_records(destination, len(window))
        self.assertEqual(len(written), len(window))
        np.testing.assert_array_equal(
            written, original, "re-encoded AEDAT 2.0 records differ from jAER's own bytes"
        )


def _dvs_records(path, count):
    """The first `count` DVS records of an AEDAT 2.0 file as raw `(address, timestamp)` pairs.

    APS and IMU records are skipped: a sink is handed events, so those are what a re-encode can be
    expected to reproduce.
    """
    records = []
    with open(path, "rb") as handle:
        while True:
            position = handle.tell()
            line = handle.readline()
            if not line.startswith(b"#"):
                handle.seek(position)
                break
        while len(records) < count:
            chunk = handle.read(8)
            if len(chunk) < 8:
                break
            address = int.from_bytes(chunk[:4], "big")
            # Bit 31 marks an APS/IMU sample and bit 10 a non-DVS type; both are dropped.
            if address & 0x8000_0000 or address & 0x0000_0400:
                continue
            records.append((address, int.from_bytes(chunk[4:], "big")))
    return np.asarray(records, dtype=np.int64)


if __name__ == "__main__":
    unittest.main()
