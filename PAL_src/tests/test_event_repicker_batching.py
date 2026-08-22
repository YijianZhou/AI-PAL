import time
import unittest
from types import SimpleNamespace

import numpy as np
try:
    import torch
    from obspy import Stream, Trace, UTCDateTime
    from event_repicker import EventRepicker, PositivePickerAdapter
    RUNTIME_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    EventRepicker = None
    RUNTIME_AVAILABLE = False


class _FakeDecoder(object):
    def decode_one_window(self, probability, start):
        return [(start + 1.0, float(probability[0]))], [
            (start + 2.0, float(probability[1]))
        ]


class _FakeAdapter(object):
    def __init__(self, name, device="cpu"):
        self.name = name
        self.device = torch.device(device)
        self.picker = _FakeDecoder()
        self.batch_sizes = []

    def predict_preprocessed_batch(self, batch):
        self.batch_sizes.append(len(batch))
        return np.tile(
            np.asarray([[0.9, 0.8]], dtype=np.float32),
            (len(batch), 1),
        )

    def finalize_shared_votes(self, job, p_votes, s_votes, workflow_cfg):
        if not p_votes or not s_votes:
            return None
        return {
            "tp": p_votes[0][0],
            "ts": s_votes[0][0],
            "p_prob": p_votes[0][1],
            "s_prob": s_votes[0][1],
            "tp_std": 0.0,
            "ts_std": 0.0,
            "p_prob_std": 0.0,
            "s_prob_std": 0.0,
            "num_votes": len(p_votes),
        }


class _AnchorOnlyReassociator(object):
    def __init__(self):
        self.input_stations = None

    def associate(self, rows, verbose=False, unique_stations=False):
        assert unique_stations
        self.input_stations = {row["net_sta"] for row in rows}
        location = {
            "evt_ot": 1000.0, "evt_lat": 34.0, "evt_lon": -118.0,
            "evt_dep": 5.0, "res": 0.1, "mag": 1.0,
        }
        return [location], [rows.copy()]


@unittest.skipUnless(
    RUNTIME_AVAILABLE,
    "event repicker tests require the production PyTorch/ObsPy environment",
)
class EventRepickerBatchingTests(unittest.TestCase):
    def make_repicker(self):
        repicker = EventRepicker.__new__(EventRepicker)
        repicker.cfg = SimpleNamespace(
            repick_batch_size=4,
            repick_num_repeat=3,
            repick_min_window_vote_ratio=0.2,
            repick_random_seed=17,
            tp_dev=1.0,
            ts_dev=1.5,
            repick_group_min_picker_support=2,
            amp_win=[1.0, 6.0],
            num_chn=3,
        )
        return repicker

    def test_displacement_amplitude_requires_complete_three_component_window(self):
        repicker = self.make_repicker()
        start = UTCDateTime(1000.0)
        sampling_rate = 100.0
        times = np.arange(2001, dtype=np.float32) / sampling_rate
        stream = Stream([
            Trace(
                data=np.sin(2.0 * np.pi * frequency * times).astype(np.float32),
                header={"starttime": start, "sampling_rate": sampling_rate},
            )
            for frequency in (2.0, 3.0, 4.0)
        ])
        self.assertGreater(
            repicker._displacement_amplitude(stream, start + 2.0, start + 10.0),
            0.0,
        )
        self.assertEqual(
            repicker._displacement_amplitude(stream, start + 0.5, start + 10.0),
            -1.0,
        )

    def test_repicker_pick_defers_amplitude_until_reassociation(self):
        repicker = self.make_repicker()
        repicker._displacement_amplitude = lambda *args: self.fail(
            "candidate generation must not measure amplitude"
        )
        timing = {
            "tp": UTCDateTime(1002.0),
            "ts": UTCDateTime(1007.0),
            "p_prob": 0.9,
            "s_prob": 0.8,
            "tp_std": 0.1,
            "ts_std": 0.2,
            "p_prob_std": 0.01,
            "s_prob_std": 0.02,
            "num_support": 2,
            "members": {"SAR": {
                "num_votes": 4,
                "tp_std": 0.1,
                "ts_std": 0.2,
                "p_prob_std": 0.01,
                "s_prob_std": 0.02,
            }},
        }
        output = repicker._output_repicker_pick(
            {"station": "CI.TEST.HH", "epicentral_distance_km": 10.0},
            {"POS_NEG": timing},
            "pos_neg_only",
            ["POS_NEG"],
        )
        self.assertEqual(output["score"], -1.0)

    def test_p_energy_snr_is_reported_per_component(self):
        repicker = self.make_repicker()
        start = UTCDateTime(990.0)
        tp = UTCDateTime(1000.0)
        sampling_rate = 100.0
        npts = 2001
        streams = []
        for channel, signal_level in zip("ENZ", (2.0, 3.0, 4.0)):
            data = np.ones(npts, dtype=np.float32)
            signal_start = int((tp - start) * sampling_rate)
            data[signal_start:signal_start + int(0.8 * sampling_rate)] = (
                signal_level
            )
            streams.append(Trace(
                data=data,
                header={
                    "starttime": start,
                    "sampling_rate": sampling_rate,
                    "channel": "HH" + channel,
                },
            ))
        snr_e, snr_n, snr_z = repicker._p_energy_snr(
            Stream(streams), tp
        )
        self.assertGreater(snr_e, 3.0)
        self.assertGreater(snr_n, snr_e)
        self.assertGreater(snr_z, snr_n)

    def test_p_energy_snr_search_uses_picker_tp_dev(self):
        repicker = self.make_repicker()
        start = UTCDateTime(990.0)
        tp = UTCDateTime(1000.0)
        sampling_rate = 100.0
        data = np.ones(2001, dtype=np.float32)
        signal_start = int((tp - 0.8 - start) * sampling_rate)
        data[signal_start:signal_start + int(0.2 * sampling_rate)] = 5.0
        stream = Stream([Trace(
            data=data,
            header={
                "starttime": start,
                "sampling_rate": sampling_rate,
                "channel": "HHZ",
            },
        )])

        repicker.cfg.tp_dev = 0.5
        narrow_snr = repicker._p_energy_snr(stream, tp)[2]
        repicker.cfg.tp_dev = 1.0
        wide_snr = repicker._p_energy_snr(stream, tp)[2]

        self.assertGreater(wide_snr, narrow_snr)

    def test_models_on_one_device_reuse_full_batches(self):
        repicker = self.make_repicker()
        left = _FakeAdapter("left")
        right = _FakeAdapter("right")
        repicker.pickers = {"left": left, "right": right}
        windows = np.zeros((10, 3, 25), dtype=np.float32)
        owners = np.asarray([0] * 5 + [1] * 5, dtype=np.int32)
        starts = np.arange(10, dtype=np.float32)
        jobs = [{"meta": {}}, {"meta": {}}]

        results, seconds, _ = repicker._run_picker_device_groups(
            windows, owners, starts, jobs
        )

        self.assertEqual(left.batch_sizes, [4, 4, 2])
        self.assertEqual(right.batch_sizes, [4, 4, 2])
        self.assertEqual(set(results), {"left", "right"})
        self.assertEqual(len(results["left"]), 2)
        self.assertGreaterEqual(seconds["left"], 0.0)

    def test_random_starts_are_shared_and_deterministic(self):
        repicker = self.make_repicker()
        candidate = {"event_id": "event-1", "station": "CI.TEST.HH"}
        first = repicker._shared_random_starts(candidate, 4.0)
        second = repicker._shared_random_starts(candidate, 4.0)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 3)
        self.assertTrue(np.all(first >= 0.0))
        self.assertTrue(np.all(first <= 4.0))

    def test_missing_station_pair_is_not_gated_by_prediction_tolerance(self):
        adapter = PositivePickerAdapter.__new__(PositivePickerAdapter)
        context = UTCDateTime("2026-01-01T00:00:00Z")
        rows = [
            {"phase": "P", "pick_time": 8.0, "pick_prob": 0.9,
             "pick_time_std": 0.1, "pick_prob_std": 0.02,
             "num_votes": 4},
            {"phase": "S", "pick_time": 13.0, "pick_prob": 0.8,
             "pick_time_std": 0.2, "pick_prob_std": 0.03,
             "num_votes": 3},
        ]
        selected = adapter._select_pair(
            rows, context, context + 2.0, context + 4.0,
            1.0, 1.5,
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected["tp"], context + 8.0)
        self.assertEqual(selected["ts"], context + 13.0)

    def test_reassociation_uses_both_group_anchors_then_supplements(self):
        repicker = self.make_repicker()
        repicker.cfg.vp = 6.0
        repicker.cfg.vs = 3.45
        repicker.reassociation_params = {"min_sta": 2}
        repicker.reassociator = _AnchorOnlyReassociator()
        repicker.stations = {
            name: (34.0, -118.0, 0.0) for name in "ABCD"
        }
        origin = UTCDateTime(1000.0)
        tp_pred = origin + 5.0 / repicker.cfg.vp
        ts_pred = origin + 5.0 / repicker.cfg.vs
        picks = [{
            "sta": station,
            "p": (origin + 1.0 + index).datetime,
            "s": (origin + 2.0 + index).datetime,
            "score": 1.0,
            "pick_provenance": "both_groups",
        } for index, station in enumerate("AB")]
        picks.extend([
            {
                "sta": "C", "p": tp_pred.datetime, "s": ts_pred.datetime,
                "score": 1.0, "pick_provenance": "pos_only",
            },
            {
                "sta": "D", "p": (tp_pred + 5.0).datetime,
                "s": (ts_pred + 5.0).datetime, "score": 1.0,
                "pick_provenance": "pos_neg_only",
            },
        ])
        event = {
            "source": "test",
            "time": origin.datetime,
            "lat": 34.0,
            "lon": -118.0,
            "depth": 5.0,
            "mag": 1.0,
            "picks": picks,
        }

        selected = repicker._reassociate_event(event)

        self.assertEqual(repicker.reassociator.input_stations, {"A", "B"})
        self.assertEqual(len(selected), 1)
        self.assertEqual(
            {pick["sta"] for pick in selected[0]["picks"]},
            {"A", "B", "C"},
        )

    def test_existing_station_arrivals_do_not_gate_repicker_pair(self):
        adapter = PositivePickerAdapter.__new__(PositivePickerAdapter)
        context = UTCDateTime("2026-01-01T00:00:00Z")
        rows = [
            {"phase": "P", "pick_time": 8.0, "pick_prob": 0.9,
             "pick_time_std": 0.1, "pick_prob_std": 0.02,
             "num_votes": 4},
            {"phase": "S", "pick_time": 13.0, "pick_prob": 0.8,
             "pick_time_std": 0.2, "pick_prob_std": 0.03,
             "num_votes": 3},
        ]
        selected = adapter._select_pair(
            rows, context, context + 2.0, context + 4.0,
            1.0, 1.5,
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected["tp"], context + 8.0)
        self.assertEqual(selected["ts"], context + 13.0)

    def test_device_groups_are_scheduled_concurrently(self):
        repicker = self.make_repicker()
        repicker.pickers = {
            "first": SimpleNamespace(device="device-a"),
            "second": SimpleNamespace(device="device-b"),
        }

        def slow_group(members, windows, owners, starts, jobs):
            time.sleep(0.08)
            name = members[0][0]
            return {name: [None]}, {name: 0.08}, 0.0

        repicker._run_picker_device_group = slow_group
        started = time.perf_counter()
        repicker._run_picker_device_groups(
            np.zeros((1, 3, 25), dtype=np.float32),
            np.zeros(1, dtype=np.int32),
            np.zeros(1, dtype=np.float32),
            [{}],
        )
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 0.14)

    def test_empty_boundary_holder_is_skipped(self):
        repicker = self.make_repicker()
        repicker.cfg.num_chn = 3
        station = "CI.TEST.HH"
        next_day = UTCDateTime("2026-01-02T00:00:00Z")
        traces = []
        for channel in ("HHE", "HHN", "HHZ"):
            trace = Trace(np.ones(1000, dtype=np.float32))
            trace.stats.network = "CI"
            trace.stats.station = "TEST"
            trace.stats.channel = channel
            trace.stats.starttime = next_day
            trace.stats.sampling_rate = 100.0
            traces.append(trace)
        context = [
            {
                "start": UTCDateTime("2026-01-01T00:00:00Z"),
                "end": next_day,
                "waveforms": {
                    station: SimpleNamespace(stream=Stream())
                },
            },
            {
                "start": next_day,
                "end": next_day + 10.0,
                "waveforms": {
                    station: SimpleNamespace(stream=Stream(traces=traces))
                },
            },
        ]

        stream = repicker._merged_station_stream(
            context, station, next_day, next_day + 9.0
        )

        self.assertIsNotNone(stream)
        self.assertEqual(len(stream), 3)


if __name__ == "__main__":
    unittest.main()
