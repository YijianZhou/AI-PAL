"""Main function for offline stream picking with LoSAR.

The SAR model is loaded once in the main process.  Parallelism is handled with
threads so all workers share the same GPU-resident model parameters, matching
realtime picking behavior.
"""
import argparse
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import torch
from obspy import UTCDateTime

import config
import picker as picker_module

warnings.filterwarnings("ignore")

cfg = config.Config()
get_data_dict = cfg.get_data_dict
get_buffered_data_dict = cfg.get_buffered_data_dict
get_sta_dict = cfg.get_sta_dict
read_data = cfg.read_data


def pick_one_day(date, sar_picker, data_dir, sta_dict, out_root):
    pick_path = os.path.join(out_root, '%s.pick' % date.date)
    buffer_sec = float(cfg.data_buffer_sec)
    day_start, day_end = date, date + 86400
    data_dict = get_buffered_data_dict(date, data_dir, buffer_sec)

    with open(pick_path, 'w') as fout:
        for net_sta, data_paths in data_dict.items():
            if net_sta not in sta_dict:
                continue
            print('-' * 40)
            print('picking %s %s' % (net_sta, date.date))
            st = read_data(
                data_paths, sta_dict,
                start_time=day_start - buffer_sec,
                end_time=day_end + buffer_sec,
            )
            with torch.inference_mode():
                sar_picker.pick(
                    st, fout,
                    pick_start_time=day_start, pick_end_time=day_end,
                )

    return pick_path


def build_date_list(time_range):
    start_time, end_time = [UTCDateTime(time) for time in time_range.split('-')]
    num_days = int((end_time - start_time) / 86400)
    return [start_time + 86400 * day_idx for day_idx in range(num_days)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_idx', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--fsta', type=str)
    parser.add_argument('--out_root', type=str)
    parser.add_argument('--time_range', type=str)
    parser.add_argument('--ckpt_dir', type=str)
    parser.add_argument('--ckpt_idx', type=int)
    args = parser.parse_args()

    sar_picker = picker_module.SAR_Picker(args.ckpt_dir, args.ckpt_idx, args.gpu_idx)
    sta_dict = get_sta_dict(args.fsta)
    os.makedirs(args.out_root, exist_ok=True)

    date_list = build_date_list(args.time_range)
    if not date_list:
        print('no dates to process')
        return

    num_workers = max(1, int(args.num_workers))
    if num_workers == 1:
        for idx, date in enumerate(date_list, start=1):
            pick_path = pick_one_day(date, sar_picker, args.data_dir, sta_dict, args.out_root)
            print('%s / %s days done: %s' % (idx, len(date_list), pick_path))
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    pick_one_day,
                    date,
                    sar_picker,
                    args.data_dir,
                    sta_dict,
                    args.out_root,
                )
                for date in date_list
            ]
            for idx, future in enumerate(futures, start=1):
                pick_path = future.result()
                print('%s / %s days done: %s' % (idx, len(date_list), pick_path))


if __name__ == '__main__':
    main()