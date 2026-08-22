"""Shared preprocessed waveform and per-device tensor cache for AI pickers."""
import os
import pickle
from threading import Lock

import numpy as np
import torch

from data_pipeline import preprocess_picker_stream


def _drop_file_cache(file_object):
    """Tell Linux that a temporary spill file need not remain in page cache."""
    advise = getattr(os, "posix_fadvise", None)
    advice = getattr(os, "POSIX_FADV_DONTNEED", None)
    if advise is None or advice is None:
        return
    try:
        advise(file_object.fileno(), 0, 0, advice)
    except OSError:
        pass


def configure_torch_backends(_cfg=None):
    """Disable unsupported NNPACK dispatch before loading picker models."""
    nnpack = getattr(torch.backends, "nnpack", None)
    set_flags = getattr(nnpack, "set_flags", None)
    if set_flags is None:
        return False
    set_flags(False)
    print("PyTorch NNPACK backend disabled", flush=True)
    return True


class RetainedStationWaveform(object):
    """Filtered station stream retained after picker inference."""

    def __init__(self, net_sta, stream):
        self.net_sta = net_sta
        self._stream = stream
        self._spill_path = None

    @property
    def stream(self):
        """Load a realtime spill lazily only when repicking needs it."""
        if self._stream is None and self._spill_path is not None:
            with open(self._spill_path, "rb") as fp:
                self._stream = pickle.load(fp)
                _drop_file_cache(fp)
        return self._stream

    @stream.setter
    def stream(self, value):
        self._stream = value

    @classmethod
    def from_prepared(cls, prepared):
        return cls(prepared.net_sta, prepared.stream)

    def spill(self, path):
        """Persist the filtered stream and release its in-memory arrays."""
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        partial_path = path + ".partial"
        with open(partial_path, "wb") as fp:
            pickle.dump(self._stream, fp, protocol=pickle.HIGHEST_PROTOCOL)
            fp.flush()
            os.fsync(fp.fileno())
            _drop_file_cache(fp)
        os.replace(partial_path, path)
        self._spill_path = path
        self._stream = None
        return self

    def release(self):
        """Release loaded samples and remove any segment-local spill file."""
        self._stream = None
        if self._spill_path is not None:
            try:
                with open(self._spill_path, "rb") as fp:
                    _drop_file_cache(fp)
            except FileNotFoundError:
                pass
            try:
                os.remove(self._spill_path)
            except FileNotFoundError:
                pass
            self._spill_path = None

    def unload(self):
        """Drop a lazily loaded stream while retaining its spill file."""
        if self._spill_path is not None:
            self._stream = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    def trim(self, start_time, end_time):
        """Retain an owned time slice so the full-day arrays can be released."""
        self.stream = self.stream.slice(
            start_time, end_time, nearest_sample=True
        ).copy()
        return bool(self.stream)


class PreparedPickerStream(object):
    """Own one preprocessed station stream and one tensor copy per device."""

    def __init__(
        self,
        stream,
        raw_stream,
        sampling_rate,
        window_length_sec,
        window_stride_sec,
        taper_max_length_sec,
        num_channels=3,
    ):
        self.sampling_rate = float(sampling_rate)
        self.window_length_sec = float(window_length_sec)
        self.window_stride_sec = float(window_stride_sec)
        self.num_channels = int(num_channels)
        self.window_npts = int(self.window_length_sec * self.sampling_rate)
        self.stride_npts = int(self.window_stride_sec * self.sampling_rate)

        start_time = stream[0].stats.starttime + float(taper_max_length_sec)
        end_time = stream[0].stats.endtime - float(taper_max_length_sec)
        if end_time < start_time + self.window_length_sec:
            raise ValueError("preprocessed stream is too short after edge exclusion")
        self.start_time = start_time
        self.end_time = end_time
        self.stream = stream.slice(start_time, end_time)
        self.raw_stream = raw_stream.slice(start_time, end_time)
        # ObsPy filtering commonly promotes samples to float64. Native models
        # and positive repickers consume float32, so retaining float64 doubles
        # the dominant realtime waveform-cache allocation without adding useful
        # precision.
        for trace in self.stream:
            trace.data = np.asarray(trace.data, dtype=np.float32)
        self.net_sta = "{}.{}".format(
            self.stream[0].stats.network,
            self.stream[0].stats.station,
        )
        self.num_windows = int(
            (end_time - start_time - self.window_length_sec)
            / self.window_stride_sec
        ) + 1

        min_npts = min(len(trace) for trace in self.stream)
        data = np.asarray(
            [trace.data[:min_npts] for trace in self.stream],
            dtype=np.float32,
        )
        # Make ObsPy traces and the shared CPU tensor use one backing array.
        # Otherwise every active station owns two full filtered-day copies.
        for channel_index, trace in enumerate(self.stream):
            trace.data = data[channel_index]
        self._cpu_tensor = torch.from_numpy(data)
        self._device_tensors = {"cpu": self._cpu_tensor}
        self._device_lock = Lock()
        self.missing_channels = self._build_missing_channel_mask()
        # Raw samples are needed only for the missing-channel mask. Keeping an
        # hour-long raw copy for every station can exhaust host RAM while GPU
        # inference is serialized.
        self.raw_stream = None

    @classmethod
    def from_raw_stream(cls, stream, cfg):
        filtered, raw = preprocess_picker_stream(
            stream,
            num_channels=cfg.num_chn,
            sampling_rate=cfg.samp_rate,
            min_length_sec=cfg.win_len,
            frequency_band=cfg.freq_band,
            taper_max_length_sec=cfg.taper_max_length_sec,
        )
        if len(filtered) != cfg.num_chn:
            return None
        try:
            return cls(
                filtered,
                raw,
                cfg.samp_rate,
                cfg.win_len,
                cfg.win_stride,
                cfg.taper_max_length_sec,
                num_channels=cfg.num_chn,
            )
        except ValueError:
            return None

    def _build_missing_channel_mask(self):
        masks = []
        for index in range(self.num_windows):
            window_start = self.start_time + index * self.window_stride_sec
            channel_mask = []
            for trace in self.raw_stream:
                rate = float(trace.stats.sampling_rate)
                first = int(round((window_start - trace.stats.starttime) * rate))
                window_npts = int(round(rate * self.window_length_sec))
                last = first + window_npts
                if first < 0 or last > len(trace.data):
                    channel_mask.append(True)
                    continue
                data = trace.data[first:last]
                channel_mask.append(
                    len(data) < window_npts
                    or np.count_nonzero(data == 0) > window_npts / 4
                )
            masks.append(channel_mask)
        return np.asarray(masks, dtype=bool)

    def validate_layout(
        self,
        sampling_rate,
        window_length_sec,
        window_stride_sec,
        num_channels,
    ):
        actual = (
            self.sampling_rate,
            self.window_length_sec,
            self.window_stride_sec,
            self.num_channels,
        )
        expected = (
            float(sampling_rate),
            float(window_length_sec),
            float(window_stride_sec),
            int(num_channels),
        )
        if actual != expected:
            raise ValueError(
                "picker layout {} does not match prepared waveform {}".format(
                    expected, actual
                )
            )

    def tensor_for(self, device):
        device = torch.device(device)
        key = str(device)
        with self._device_lock:
            tensor = self._device_tensors.get(key)
            if tensor is None:
                tensor = self._cpu_tensor.to(device)
                self._device_tensors[key] = tensor
        return tensor

    def retained_waveform(self):
        """Drop device tensors and transfer stream ownership to postprocessing."""
        retained = RetainedStationWaveform.from_prepared(self)
        self.release()
        return retained

    def release(self):
        """Release station tensors after all requested pickers finish."""
        self._device_tensors.clear()
        self._cpu_tensor = None
        self.raw_stream = None
        self.stream = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
