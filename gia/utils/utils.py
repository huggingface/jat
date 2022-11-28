import numpy as np
import torch


from gym.spaces import Box, Discrete, Dict

# numpy_to_torch_dtype_dict = {
#     np.bool       : torch.bool,
#     np.uint8      : torch.uint8,
#     np.int8       : torch.int8,
#     np.int16      : torch.int16,
#     np.int32      : torch.int32,
#     np.int64      : torch.int64,
#     np.float16    : torch.float16,
#     np.float32    : torch.float32,
#     np.float64    : torch.float64,
#     np.complex64  : torch.complex64,
#     np.complex128 : torch.complex128
# }
# torch_to_numpy_dtype_dict = {value : key for (key, value) in numpy_to_torch_dtype_dict.items()}


def to_torch_dtype(numpy_dtype):
    """from_numpy automatically infers type, so we leverage that."""
    x = np.zeros([1], dtype=numpy_dtype)
    t = torch.from_numpy(x)
    return t.dtype


def check_space_is_flat_dict(space: Dict):
    for k, v in space.items():
        assert isinstance(v, (Box, Discrete)), "An instance the thie space {space} is not flat {v}"


def _call_remote_method(method, rref, *args, **kwargs):
    r"""
    a helper function to call a method on the given RRef
    """
    return method(rref.local_value(), *args, **kwargs)


def lod_to_dol(lod):
    return {k: [dic[k] for dic in lod] for k in lod[0]}


def dol_to_lod(dol):
    return [dict(zip(dol, t)) for t in zip(*dol.values())]
