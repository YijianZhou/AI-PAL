"""Selection and identity for the combined continuous picker ensemble."""
from pick_ensemble import validate_picker_support


def picker_architecture(name):
    """Resolve SAR or SAR_<dataset> identities without duplicate config fields."""
    architecture, separator, dataset = str(name).partition('_')
    if architecture not in ('SAR', 'FT', 'PHN', 'RUN') or (separator and not dataset):
        raise ValueError('Unsupported picker name: {}'.format(name))
    return architecture


def continuous_specs(cfg, local, global_models):
    specs = {}
    for group, names, available in (
        ('Local', cfg.picker_local_group, local),
        ('Global', getattr(cfg, 'picker_global_group', []), global_models),
    ):
        for model in dict.fromkeys(names):
            if model not in available:
                raise ValueError('Missing {} continuous picker: {}'.format(group, model))
            runtime = group + '_' + model
            architecture = picker_architecture(model)
            specs[runtime] = dict(available[model], group=group, model=architecture, identity=model)
    validate_picker_support(cfg.picker_group_min_picker_support, specs)
    return specs
