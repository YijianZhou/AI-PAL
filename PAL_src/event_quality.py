"""Final AI-PAL event acceptance using group agreement and pick quality."""
from numbers import Integral


def minimum_both_group_picks(cfg):
    value = getattr(cfg, 'final_event_min_both_group_picks', 2)
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError('final_event_min_both_group_picks must be a nonnegative integer')
    return int(value)


def filter_final_events(events, cfg):
    minimum = minimum_both_group_picks(cfg)
    accepted = []
    rejected = 0
    for event in events:
        both = sum(
            'both_groups' in {
                token.strip().lower()
                for token in str(pick.get('pick_provenance') or '').split('|')
            }
            for pick in event.get('picks', [])
        )
        if both >= minimum:
            accepted.append(event)
        else:
            rejected += 1
    return accepted, rejected


def minimum_quality0_picks(cfg):
    value = getattr(cfg, 'final_event_min_quality0_picks', 2)
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError('final_event_min_quality0_picks must be a nonnegative integer')
    return int(value)


def filter_quality0_events(events, cfg):
    """Apply after group-agreement QC; report additional rejected events."""
    minimum = minimum_quality0_picks(cfg)
    accepted = [event for event in events if sum(
        pick.get('quality') == 0 for pick in event.get('picks', [])
    ) >= minimum]
    return accepted, len(events) - len(accepted)
