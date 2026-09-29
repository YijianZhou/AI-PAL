"""Publish compact final phases and a lossless per-picker QC companion."""
import csv
from pathlib import Path


FIELDS = ["phase_line", "event_ot", "event_lat", "event_lon", "station", "tp", "ts",
          "picker_group", "picker", "contribution_index", "p_prob_median", "s_prob_median",
          "tp_std", "ts_std", "p_prob_std", "s_prob_std", "window_vote_ratio",
          "ensemble_p_prob_std", "ensemble_s_prob_std"]
METRICS = {"pmed": "p_prob_median", "smed": "s_prob_median", "tp": "tp_std",
           "ts": "ts_std", "pp": "p_prob_std", "sp": "s_prob_std"}


def normalize_picker_group(value):
    return {'POS_NEG': 'Local', 'POS': 'Global'}.get(value, value)


def normalize_picker_details(value):
    """Read historical group-qualified source/metric strings under new names."""
    return '|'.join(
        normalize_picker_group(item.split(':', 1)[0]) + ':' + item.split(':', 1)[1]
        if ':' in item else item for item in value.split('|'))


def qc_path(phase_path):
    return Path(phase_path).with_suffix(".picker_qc.csv")


class QCDetails(dict):
    def __init__(self):
        super().__init__()
        self.summaries = {}
        self.sources = {}


def is_compact_phase_row(row):
    if len(row) != 13 or len(row[-1].split('|')) != 3:
        return False
    try:
        [float(value) for value in row[-1].split('|')]
    except ValueError:
        return False
    return True


def visible_picker(name):
    group, model = name.split(':', 1) if ':' in name else ('', name)
    if normalize_picker_group(group) == 'Global' and model in ('SAR', 'FT', 'PHN', 'RUN'):
        model += '_CEED'
    return model


def load_qc_details(phase_path):
    """Index detail rows by phase line and event/pick identity, not OT alone."""
    path = qc_path(phase_path)
    details = QCDetails()
    if not path.exists():
        return details
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["phase_line"]), row["event_ot"], row["event_lat"], row["event_lon"],
                   row["station"], row["tp"], row["ts"])
            group = normalize_picker_group(row["picker_group"])
            name = (group + ":" if group else "") + row["picker"]
            details.summaries[key] = [row.get('ensemble_p_prob_std') or 'nan',
                                      row.get('ensemble_s_prob_std') or 'nan']
            if row['picker']:
                details.sources.setdefault(key, {})[visible_picker(name)] = name
            values = [code + "=" + row[column] for code, column in METRICS.items() if row[column]]
            if values:
                details.setdefault(key, []).append(name + ":" + ";".join(values))
    for key, values in details.items():
        details[key] = '|'.join(values)
    return details


def expand_phase_row(codes, line_number, header, details):
    if is_compact_phase_row(codes):
        key = (line_number, *header[:3], *codes[:3])
        identities = getattr(details, 'sources', {}).get(key, {})
        ratios = []
        sources = []
        for item in filter(None, codes[10].split('|')):
            name, ratio = item.rsplit(':', 1)
            qualified = identities.get(name, ('Global:' if '_' in name else 'Local:') + name)
            sources.append(qualified)
            ratios.append(qualified + ':' + ratio)
        codes = (codes[:9] + getattr(details, 'summaries', {}).get(key, ['nan', 'nan'])
                 + [codes[9], '|'.join(sources), '|'.join(ratios), details.get(key, ''), codes[11]]
                 + codes[12].split('|'))
    if len(codes) == 18:
        key = (line_number, *header[:3], *codes[:3])
        codes = codes[:14] + [details.get(key, "")] + codes[14:]
    if len(codes) == 19:
        codes = list(codes)
        for index in (12, 13, 14):
            codes[index] = normalize_picker_details(codes[index])
        codes[15] = codes[15].replace('pos_neg_only', 'local_only').replace('pos_only', 'global_only')
    return codes


def export_phase_qc(phase_path):
    """Publish 13-field final rows and retain probability spreads in QC.

    Working phases remain rich. Already compact published files are unchanged,
    so rerunning cannot overwrite their retained QC with empty statistics.
    """
    phase_path = Path(phase_path)
    with phase_path.open(newline="", encoding="utf-8") as stream:
        compact = rich = False
        for row in csv.reader(stream):
            compact |= is_compact_phase_row(row)
            rich |= len(row) in (18, 19)
    if compact:
        if rich:
            raise ValueError("Mixed compact/rich phase rows: {}".format(phase_path))
        return qc_path(phase_path)
    details = load_qc_details(phase_path)
    phase_partial = phase_path.with_suffix(phase_path.suffix + ".qc.partial")
    companion = qc_path(phase_path)
    qc_partial = companion.with_suffix(companion.suffix + ".partial")
    header = None
    with phase_path.open(newline="", encoding="utf-8") as source, \
            phase_partial.open("w", newline="", encoding="utf-8") as phase_stream, \
            qc_partial.open("w", newline="", encoding="utf-8") as qc_stream:
        writer = csv.writer(phase_stream, lineterminator="\n")
        qc_writer = csv.DictWriter(qc_stream, fieldnames=FIELDS, lineterminator="\n")
        qc_writer.writeheader()
        for line_number, row in enumerate(csv.reader(source), 1):
            if len(row) == 5 and "T" in row[0]:
                header = row
            elif len(row) in (18, 19):
                if header is None:
                    raise ValueError("Phase row before event header")
                row = expand_phase_row(row, line_number, header, details)
                for index in (12, 13, 14):
                    row[index] = normalize_picker_details(row[index])
                row[15] = row[15].replace('pos_neg_only', 'local_only').replace('pos_only', 'global_only')
                ratios = dict(item.rsplit(":", 1) for item in row[13].split("|") if item)
                entries = {}
                for item in row[14].split("|"):
                    if item:
                        name, metrics = item.rsplit(":", 1)
                        entries.setdefault(name, []).append(dict(value.split("=", 1) for value in metrics.split(";")))
                for name in filter(None, row[12].split("|")):
                    entries.setdefault(name, [{}])
                for name in ratios:
                    entries.setdefault(name, [{}])
                if not entries:
                    entries[''] = [{}]
                for name, contributions in sorted(entries.items()):
                    group, model = name.split(":", 1) if ":" in name else ("", name)
                    for index, metrics in enumerate(contributions, 1):
                        detail = dict(phase_line=line_number, event_ot=header[0], event_lat=header[1],
                                      event_lon=header[2], station=row[0], tp=row[1], ts=row[2],
                                      picker_group=group, picker=model, contribution_index=index,
                                      window_vote_ratio=ratios.get(name, ""),
                                      ensemble_p_prob_std=row[9], ensemble_s_prob_std=row[10])
                        detail.update({column: metrics.get(code, "") for code, column in METRICS.items()})
                        qc_writer.writerow(detail)
                visible_ratios = '|'.join(visible_picker(name) + ':' + ratios.get(name, 'nan')
                                          for name in entries if name)
                row = row[:9] + [row[11], visible_ratios, row[15], '|'.join(row[16:19])]
            writer.writerow(row)
    qc_partial.replace(companion)
    phase_partial.replace(phase_path)
    return companion
