from typing import List
import numpy as np
import torch

from gia.utils.dicts import list_of_dicts_to_dict_of_lists, iterate_recursively


class TensorDict(dict):
    dict_key_type = str

    def __getitem__(self, key):
        if isinstance(key, self.dict_key_type):
            # if key is string assume we're accessing dict's interface
            return dict.__getitem__(self, key)
        else:
            # otherwise we want to index/slice into tensors themselves
            return self._index_func(self, key)

    def _index_func(self, x, indices):
        if isinstance(x, (dict, TensorDict)):
            res = TensorDict()
            for key, value in x.items():
                res[key] = self._index_func(value, indices)
            return res
        else:
            t = x[indices]
            return t

    def __setitem__(self, key, value):
        if isinstance(key, self.dict_key_type):
            dict.__setitem__(self, key, value)
        else:
            self._set_data_func(self, key, value)

    def _set_data_func(self, x, index, new_data):
        if isinstance(new_data, (dict, TensorDict)):
            for new_data_key, new_data_value in new_data.items():
                self._set_data_func(x.get(new_data_key), index, new_data_value)
        else:
            if torch.is_tensor(x):
                if isinstance(new_data, torch.Tensor):
                    t = new_data
                elif isinstance(new_data, np.ndarray):
                    t = torch.from_numpy(new_data)
                else:
                    raise ValueError(f"Type {type(new_data)} not supported in set_data_func")

                x[index].copy_(t)

            elif isinstance(x, np.ndarray):
                if isinstance(new_data, torch.Tensor):
                    n = new_data.cpu().numpy()
                elif isinstance(new_data, np.ndarray):
                    n = new_data
                else:
                    raise ValueError(f"Type {type(new_data)} not supported in set_data_func")

                x[index] = n


def cat_tensordicts(lst: List[TensorDict]) -> TensorDict:
    """
    Concatenates a list of tensordicts.
    """
    if not lst:
        return TensorDict()

    res = list_of_dicts_to_dict_of_lists(lst)
    # iterate res recursively and concatenate tensors
    for d, k, v in iterate_recursively(res):
        if isinstance(v[0], torch.Tensor):
            d[k] = torch.cat(v)
        elif isinstance(v[0], np.ndarray):
            d[k] = np.concatenate(v)
        else:
            raise ValueError(f"Type {type(v[0])} not supported in cat_tensordicts")

    return TensorDict(res)
