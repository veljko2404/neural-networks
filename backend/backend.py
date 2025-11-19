import numpy as np

def address(array) -> int:
    import numpy
    if isinstance(array, numpy.ndarray):
        return array.__array_interface__['data'][0]  # za numpy
    else:
        return array.__cuda_array_interface__['data'][0]