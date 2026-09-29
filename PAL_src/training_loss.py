"""Negative-window weighting without changing within-window loss definitions."""
import math


def negative_loss_weight(cfg):
    value = getattr(cfg, 'negative_loss_weight', 1.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('negative_loss_weight must be finite and nonnegative')
    return float(value)


def describe_training_loss(cfg, bs_pos, bs_neg):
    weight = negative_loss_weight(cfg)
    denominator = bs_pos + weight * bs_neg
    print('training loss: negative multiplier={:g} | positive/negative coefficients={:.6f}/{:.6f}'.format(
        weight, bs_pos / denominator, weight * bs_neg / denominator), flush=True)


def weighted_window_loss(logits, target, num_pos, loss_fn, weight=1.0, details=None):
    """Average equal-length windows; positives precede negatives in the batch."""
    num_neg = len(target) - num_pos
    if not 0 < num_pos <= len(target):
        raise ValueError('Training requires at least one positive window')
    positive = loss_fn(logits[:num_pos], target[:num_pos])
    negative = loss_fn(logits[num_pos:], target[num_pos:]) if num_neg else None
    if details is not None:
        details.clear()
        details['loss/train_pos_loss'] = positive.detach()
        if negative is not None:
            details['loss/train_neg_loss'] = negative.detach()
    if not num_neg or weight == 0:
        return positive
    return (num_pos * positive + weight * num_neg * negative) / (num_pos + weight * num_neg)


def log_training_losses(details, writer, step):
    values = {key: value.item() for key, value in details.items()}
    for key, value in values.items():
        writer.add_scalar(key, value, step)
    print('   ' + ' | '.join('{} {:.4f}'.format(key.split('/')[-1], value)
                            for key, value in values.items()), flush=True)
    return values
