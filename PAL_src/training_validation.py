"""Exhaustive class-wise validation, independent of training batch ratios."""


def validation_due(step, total_steps, interval):
    """Steps are zero-based internally; validate after each interval and at end."""
    if int(interval) <= 0:
        raise ValueError('valid_step must be positive')
    return (step + 1) % int(interval) == 0 or step + 1 == total_steps


def validate_classes(loaders, evaluate, positive_diagnostic_only=False):
    """evaluate(data, target, kind) returns diagnostic, detection accuracy, loss."""
    means = {}
    diagnostic_sum = 0.0
    diagnostic_count = 0
    for kind, loader in loaders.items():
        count = 0
        loss_sum = accuracy_sum = 0.0
        for data, target in loader:
            size = len(data)
            diagnostic, accuracy, loss = evaluate(data, target, kind)
            count += size
            loss_sum += float(loss) * size
            accuracy_sum += float(accuracy) * size
            if not positive_diagnostic_only or kind == 'positive':
                diagnostic_sum += float(diagnostic) * size
                diagnostic_count += size
        if not count:
            raise ValueError('empty {} validation set'.format(kind))
        means[kind] = (loss_sum / count, accuracy_sum / count)
    if 'positive' not in means:
        raise ValueError('positive validation set is required')
    losses = {'loss/valid_pos_loss': means['positive'][0]}
    if 'negative' in means:
        losses['loss/valid_neg_loss'] = means['negative'][0]
    balanced_loss = sum(losses.values()) / len(losses)
    return (diagnostic_sum / diagnostic_count, means['positive'][1],
            means['negative'][1] if 'negative' in means else None,
            balanced_loss, losses)
