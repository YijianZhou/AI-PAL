"""Shared waveform quality-control helpers for native AI pickers."""

import numpy as np


def find_first_peak(data):
    """Return the first PAL turning-point offset."""
    data = np.asarray(data)
    if len(data) < 2:
        return 0
    delta = np.diff(data)
    negative = np.where(delta < 0)[0]
    positive = np.where(delta >= 0)[0]
    if not len(negative) or not len(positive):
        return 0
    return int(max(negative[0], positive[0]))


def find_second_peak(data):
    """Return the second PAL turning-point offset."""
    data = np.asarray(data)
    if len(data) < 2:
        return 0
    delta = np.diff(data)
    negative = np.where(delta < 0)[0]
    positive = np.where(delta >= 0)[0]
    if not len(negative) or not len(positive):
        return 0
    first = int(max(negative[0], positive[0]))
    later_negative = negative[negative > first]
    later_positive = positive[positive > first]
    if not len(later_negative) or not len(later_positive):
        return first
    return int(max(later_negative[0], later_positive[0]))


def _trace_key(trace):
    return trace.id


def _peak_to_peak_by_channel(stream, start_time, end_time):
    amplitudes = {}
    for trace in stream.slice(start_time, end_time):
        data = np.asarray(trace.data)
        if data.size:
            amplitudes[_trace_key(trace)] = float(np.ptp(data))
    return amplitudes


def _safe_ratio(numerator, denominator):
    if denominator > 0.0:
        return numerator / denominator
    return np.inf if numerator > 0.0 else 1.0


def calc_peak_amp_ratio(
    stream, win_peak_npts, find_first_peak, find_second_peak,
):
    """Return PAL peak/tail ratios for channels with complete data."""
    traces = [trace for trace in stream if len(trace.data)]
    if not traces:
        return []
    peak_trace = max(
        traces,
        key=lambda trace: np.max(np.abs(trace.data[:win_peak_npts])),
    )
    peak_search = np.asarray(peak_trace.data[:win_peak_npts])
    if not peak_search.size:
        return []
    idx0 = int(np.argmax(np.abs(peak_search)))
    idx1 = idx0 + find_first_peak(peak_trace.data[idx0:])
    idx0 -= find_second_peak(peak_trace.data[:idx0][::-1])
    idx1 += find_second_peak(peak_trace.data[idx1:]) + 1
    idx0 = max(0, int(idx0))
    idx1 = max(idx0 + 1, int(idx1))

    ratios = []
    for trace in traces:
        data = np.asarray(trace.data)
        peak = data[idx0:idx1]
        tail = data[idx1:2 * idx1 - idx0]
        if not peak.size or not tail.size:
            continue
        ratios.append(_safe_ratio(float(np.ptp(peak)), float(np.ptp(tail))))
    return ratios


def remove_glitch(
    stream, tp, ts, win_peak, win_peak_npts, amp_ratio_thresholds,
    find_first_peak, find_second_peak,
):
    """Apply PAL glitch rejection using only complete common channels."""
    for phase_time in (tp, ts):
        ratios = calc_peak_amp_ratio(
            stream.slice(phase_time, phase_time + win_peak * 3),
            win_peak_npts,
            find_first_peak,
            find_second_peak,
        )
        if ratios and np.min(ratios) > amp_ratio_thresholds[0]:
            return True

    half_ps = (ts - tp) / 2
    a1 = _peak_to_peak_by_channel(stream, tp, tp + half_ps)
    a2 = _peak_to_peak_by_channel(stream, tp + half_ps, ts)
    a3 = _peak_to_peak_by_channel(stream, ts, ts + half_ps)
    common_channels = set(a1) & set(a2) & set(a3)
    if not common_channels:
        return False
    a12 = min(_safe_ratio(a1[key], a2[key]) for key in common_channels)
    a13 = min(_safe_ratio(a1[key], a3[key]) for key in common_channels)
    return not (
        a12 < amp_ratio_thresholds[1]
        and a13 < amp_ratio_thresholds[2]
    )


def is_glitch(stream, tp, ts, cfg):
    """Apply the configured PAL glitch test to a filtered velocity stream."""
    sampling_rate = float(getattr(cfg, "samp_rate", 100.0))
    win_peak = float(getattr(cfg, "win_peak", 1.0))
    return remove_glitch(
        stream,
        tp,
        ts,
        win_peak,
        max(1, int(round(win_peak * sampling_rate))),
        list(getattr(cfg, "amp_ratio_thres", [5, 8, 3])),
        find_first_peak,
        find_second_peak,
    )


def displacement_amplitude(stream, tp, ts, amp_win, num_channels=3):
    """Measure PAL vector displacement amplitude from filtered velocity."""
    window_start = tp - float(amp_win[0])
    window_end = ts + float(amp_win[1])
    window = stream.slice(window_start, window_end)
    if len(window) != int(num_channels):
        return -1.0
    for trace in window:
        sampling_rate = float(trace.stats.sampling_rate)
        if not np.isfinite(sampling_rate) or sampling_rate <= 0:
            return -1.0
        tolerance = 0.51 / sampling_rate
        if (
            trace.stats.starttime > window_start + tolerance
            or trace.stats.endtime < window_end - tolerance
        ):
            return -1.0
    npts = min(len(trace.data) for trace in window)
    if npts < 2:
        return -1.0
    displacement = []
    for trace in window:
        velocity = np.asarray(trace.data[:npts], dtype=np.float64)
        if not np.all(np.isfinite(velocity)):
            return -1.0
        velocity = velocity - np.mean(velocity)
        displacement.append(
            np.cumsum(velocity) / float(trace.stats.sampling_rate)
        )
    displacement = np.asarray(displacement)
    amplitude = float(np.sqrt(np.max(np.sum(displacement ** 2, axis=0))))
    return amplitude if np.isfinite(amplitude) and amplitude > 0 else -1.0
