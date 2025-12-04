try:
    import cupy as xp
except:
    import numpy as xp


def address(array) -> int:
    import numpy
    if isinstance(array, numpy.ndarray):
        return array.__array_interface__['data'][0]
    else:
        return array.__cuda_array_interface__['data'][0]
