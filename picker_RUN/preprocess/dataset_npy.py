"""RUN waveform preprocessing, NPY sharding, and Zarr dataset helpers."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import numpy as np
from torch.utils.data import Dataset
import sys

_PICKER_DIR = Path(__file__).resolve().parents[1]
if str(_PICKER_DIR) not in sys.path:
    sys.path.insert(0, str(_PICKER_DIR))
import config
cfg = config.Config()
samp_rate = cfg.samp_rate
num_chn = cfg.num_chn
win_len = int(cfg.win_len * samp_rate)
label_wid = int(cfg.label_wid * samp_rate)

def stream_to_sample(stream, tp_rel=-1.0, ts_rel=-1.0, win_npts=None):
    if win_npts is None:
        win_npts = min(len(trace.data) for trace in stream[:3])
    sample = np.zeros((3, int(win_npts) + 2), dtype=np.float32)
    sample[:, 0] = float(tp_rel)
    sample[:, 1] = float(ts_rel)
    for index, trace in enumerate(stream[:3]):
        data = np.asarray(trace.data, dtype=np.float32)
        npts = min(data.size, int(win_npts))
        sample[index, 2:2 + npts] = data[:npts]
    return sample


def write_split_shards(out_root, split, label, group_name, samples,
                       shard_size=1024):
    if not samples:
        return []
    split_dir = Path(out_root) / split / label
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for shard_index, start in enumerate(range(0, len(samples), int(shard_size))):
        shard = np.stack(samples[start:start + int(shard_size)]).astype(
            np.float32, copy=False
        )
        shard_path = split_dir / (
            "%s_shard_%06d.npy" % (group_name, shard_index)
        )
        np.save(shard_path, shard)
        rows.append((str(shard_path), str(shard.shape[0])))
    return rows


def write_cut_progress(out_root, stage, completed_items, total_items,
                       processed_attempts, planned_attempts,
                       generated_samples, finished=False):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(out_root)
    payload = {
        "stage": stage,
        "completed_items": int(completed_items),
        "total_items": int(total_items),
        "processed_attempts": int(processed_attempts),
        "planned_attempts": int(planned_attempts),
        "generated_samples": int(generated_samples),
        "bytes_per_sample": int(
            3 * (win_len + 2) * np.dtype(np.float32).itemsize
        ),
        "finished": bool(finished),
        "disk_total_bytes": int(disk.total),
        "disk_used_bytes": int(disk.used),
        "disk_free_bytes": int(disk.free),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path = out_root / ("cut_%s_progress.json" % stage)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.replace(path)


def save_shard_index(path, rows):
    index_path = Path(path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    root = index_path.resolve().parent
    portable_rows = []
    for shard_path, count in rows:
        shard_path = Path(shard_path).resolve()
        try:
            stored_path = shard_path.relative_to(root)
        except ValueError:
            stored_path = shard_path
        portable_rows.append((stored_path.as_posix(), str(count)))
    np.save(index_path, np.asarray(portable_rows, dtype=str))


def preprocess(stream, sample_rate, freq_band, max_gap=5.0):
    start_time = max(trace.stats.starttime for trace in stream)
    end_time = min(trace.stats.endtime for trace in stream)
    if start_time >= end_time:
        print("bad data!")
        return []
    st = stream.slice(start_time, end_time)
    for trace in st:
        trace.data[np.isnan(trace.data)] = 0
        trace.data[np.isinf(trace.data)] = 0
    max_gap_npts = int(max_gap * sample_rate)
    for trace in st:
        npts = len(trace.data)
        gap_idx = np.where(np.diff(trace.data) == 0)[0]
        gap_list = np.split(gap_idx, np.where(np.diff(gap_idx) != 1)[0] + 1)
        gap_list = [gap for gap in gap_list if len(gap) >= 3]
        for index, gap in enumerate(gap_list):
            idx0 = max(0, gap[0] - 1)
            idx1 = min(npts - 1, gap[-1] + 1)
            if index < len(gap_list) - 1:
                idx2 = min(
                    idx1 + (idx1 - idx0),
                    idx1 + max_gap_npts,
                    gap_list[index + 1][0],
                )
            else:
                idx2 = min(idx1 + (idx1 - idx0), idx1 + max_gap_npts, npts - 1)
            if idx1 == idx2:
                continue
            if idx2 == idx1 + (idx1 - idx0):
                trace.data[idx0:idx1] = trace.data[idx1:idx2]
            else:
                num_tile = int(np.ceil((idx1 - idx0) / (idx2 - idx1)))
                trace.data[idx0:idx1] = np.tile(
                    trace.data[idx1:idx2], num_tile
                )[:idx1 - idx0]
    st = st.detrend("demean").detrend("linear").taper(
        max_percentage=0.05, max_length=5.0
    )
    if st[0].stats.sampling_rate != sample_rate:
        st.resample(sample_rate)
    freq_min, freq_max = freq_band
    if freq_min and freq_max:
        return st.filter("bandpass", freqmin=freq_min, freqmax=freq_max)
    if freq_min:
        return st.filter("highpass", freq=freq_min)
    if freq_max:
        return st.filter("lowpass", freq=freq_max)
    print("filter type not supported!")
    return []




def get_phase_target(tp, ts):
    target_seq = np.zeros([3, win_len], dtype=np.float32)
    target_seq[0, :] = 1.
    if tp < 0 or ts < 0:
        return target_seq
    half_wid = label_wid // 2
    pick_lab = np.exp(-((np.arange(-half_wid, half_wid + 1))**2) / (2*(label_wid/5)**2)).astype(np.float32)

    def put_pick(phase_idx, pick_time):
        pick_idx = pick_time * samp_rate
        lab0 = int(pick_idx - half_wid)
        lab1 = int(pick_idx + half_wid + 1)
        out0 = max(lab0, 0)
        out1 = min(lab1, win_len)
        if out0 >= out1:
            return
        src0 = out0 - lab0
        src1 = src0 + (out1 - out0)
        target_seq[phase_idx, out0:out1] = pick_lab[src0:src1]

    put_pick(1, tp)
    put_pick(2, ts)
    target_seq[0, :] = np.maximum(0., 1. - target_seq[1, :] - target_seq[2, :])
    return target_seq


class NpyWindowShards(Dataset):
    def __init__(self, shard_list):
        self.shard_root = Path(shard_list).resolve().parent
        self.shard_rows = np.load(shard_list, allow_pickle=False)
        if self.shard_rows.ndim == 1:
            raise ValueError('Expected a shard index with shape (n_shards, 2): %s' % shard_list)
        self.num_samples = int(np.asarray(self.shard_rows[:, 1], dtype=np.int64).sum())

    def __len__(self):
        return self.shard_rows.shape[0]

    def __getitem__(self, index):
        shard_path = Path(str(self.shard_rows[index, 0]))
        if not shard_path.is_absolute():
            shard_path = self.shard_root / shard_path
        shard = np.load(shard_path, mmap_mode='r')
        count = int(self.shard_rows[index, 1])
        data = np.asarray(shard[:count, :num_chn, 2:2+win_len], dtype=np.float32)
        tp = np.asarray(shard[:count, 0, 0], dtype=np.float32)
        ts = np.asarray(shard[:count, 0, 1], dtype=np.float32)
        target = np.empty((count, 3, win_len), dtype=np.float32)
        for ii in range(count):
            target[ii] = get_phase_target(float(tp[ii]), float(ts[ii]))
        return data, target

