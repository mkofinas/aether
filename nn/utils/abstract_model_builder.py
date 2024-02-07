"""Model Builder for `seq2seq`  and `dynamicvars` settings"""

import os
from importlib import import_module


class NetworkFactory(object):
    @staticmethod
    def create(model_name, *args, **kwargs):
        module_name = '.'.join(model_name.split('.')[:-1])
        class_name = model_name.split('.')[-1]
        try:
            model_module = import_module(module_name)
            print('model_module', model_module)
            model_class = getattr(model_module, class_name)
            model_instance = model_class(*args, **kwargs)
        except (AttributeError, ImportError):
            raise

        return model_instance


def build_model(params):
    model = NetworkFactory.create(params['model_type'], params)
    print(f"{params['model_type']} MODEL: ", model)

    if params['load_best_model']:
        print("LOADING BEST MODEL")
        path = os.path.join(params['working_dir'], 'best_model')
        model.load(path)
    elif params['load_model']:
        print("LOADING MODEL FROM SPECIFIED PATH")
        path = os.path.join(params['working_dir'], params['load_model'])
        model.load(path)
    if params['gpu']:
        model.cuda()
    return model
