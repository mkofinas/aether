from experiments.utils.normalization.abstract_normalization import AbstractNormalization


class IdentityNormalization(AbstractNormalization):
    def __init__(self, data_path, params):
        pass

    def normalize(self, x):
        return x

    def unnormalize(self, x):
        return x.numpy()

    def torch_unnormalize(self, x):
        return x
