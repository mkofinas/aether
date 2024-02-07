"""Factory for easily getting normalizers by name."""

from importlib import import_module

from experiments.utils.normalization.abstract_normalization import AbstractNormalization


def arg_to_class_name(x):
    class_names = {
        'no_norm': 'identity_normalization',
        'speed_norm': 'speed_normalization',
        'same_norm': 'same_norm_normalization',
        'min_max_norm': 'min_max_normalization',
    }
    print(x, class_names.get(x, x))
    return class_names.get(x, x)


class NormalizationFactory(object):
    @staticmethod
    def create(normalization_name, *args, **kwargs):
        print(args)
        print(kwargs)
        module_name = arg_to_class_name(normalization_name)
        class_name = ''.join([s.capitalize() for s in module_name.split('_')])

        try:
            normalization_module = import_module('experiments.utils.normalization.' + module_name)
            normalization_class = getattr(normalization_module, class_name)
            if not issubclass(normalization_class, AbstractNormalization):
                raise ImportError(f"{normalization_class} should be a subclass of {AbstractNormalization}")
            normalization_instance = normalization_class(*args, **kwargs)
        except (AttributeError, ImportError):
            raise

        return normalization_instance
