from functools import partial

from torch.utils.data.dataloader import default_collate


def collate_field(batch):
    elem = batch[0]
    if callable(elem) and isinstance(elem, partial):
        return partial(elem.func, sum([list(data.args) for data in batch], []))
    elif isinstance(elem, dict):
        return {k: collate_field([data[k] for data in batch]) for k in elem}
    else:
        return default_collate(batch)
