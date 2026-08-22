import unittest

import numpy as np

try:
    from obspy import Stream, Trace, UTCDateTime
    from waveform_qc import remove_glitch
    RUNTIME_AVAILABLE = True
except ModuleNotFoundError:
    RUNTIME_AVAILABLE = False


def _no_peak(_):
    return 0


@unittest.skipUnless(RUNTIME_AVAILABLE, "waveform QC tests require ObsPy")
class WaveformQCTests(unittest.TestCase):
    def test_two_available_channels_do_not_require_third_index(self):
        start = UTCDateTime("2026-01-01T00:00:00Z")
        stream = Stream()
        for channel in ("HHE", "HHN"):
            trace = Trace(np.sin(np.linspace(0, 20, 4000)).astype(np.float32))
            trace.stats.network = "CI"
            trace.stats.station = "TEST"
            trace.stats.channel = channel
            trace.stats.starttime = start
            trace.stats.sampling_rate = 100.0
            stream.append(trace)

        result = remove_glitch(
            stream, start + 5, start + 10, 1.0, 100, [5, 8, 3],
            _no_peak, _no_peak,
        )

        self.assertIsInstance(result, bool)

    def test_no_complete_qc_channel_keeps_pick(self):
        result = remove_glitch(
            Stream(), UTCDateTime(0), UTCDateTime(5), 1.0, 100,
            [5, 8, 3], _no_peak, _no_peak,
        )

        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
