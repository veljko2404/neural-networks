from typing import Tuple

from backend.backend import xp


def im2col_map(Xcol_shape: tuple, f_h: int, f_w: int, stride: int, out_w: int) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray, xp.ndarray]:

    n, i, j = xp.indices(Xcol_shape)

    dl = i // (f_w * f_h)
    f_i = (i % (f_w * f_h)) // f_w
    f_j = (i % (f_w * f_h)) % f_w
    """
    What does the index 'j' tell us?
    The index 'j' runs over the dimension of size Oh * Ow, and the j-th column should contain exactly those
    input values that are used to compute the j-th pixel of the output feature map (if we flattened the map
    into a 1D vector).
    From 'j' we can recover the corresponding 2D indices in the output feature map for which this column is
    responsible. This is easy to do if we know the output map dimensions, in practice the width is sufficient.
    """
    i_out = j // out_w
    j_out = j % out_w
    """
    Now we almost know everything. The element at position output_flat_index must be an element from X
    belonging to sample batch_idx, coming from the input feature map with index dl (we know this because it is
    multiplied by the filter assigned to that map), and via the filter offset (f_i, f_j) it contributes to the
    activation at position (i_out, j_out). The only remaining step is to compute the exact index of that element
    in the input feature map.
    """
    i_in = i_out * stride + f_i
    j_in = j_out * stride + f_j

    return n, dl, i_in, j_in

def col2im(dEdXcol: xp.ndarray, X_shape: tuple, kernel_h: int, kernel_w: int, stride: int):
    dEdX = xp.zeros(X_shape, dtype=float)
    out_h = (X_shape[-2] - kernel_h) // stride + 1
    dEdX[im2col_map(dEdXcol.shape, kernel_h, kernel_w, stride, out_h)] += dEdXcol

    return dEdX


def im2col(X: xp.ndarray, kernel_h: int, kernel_w: int, stride: int) -> xp.ndarray:
    out_w = (X.shape[-1] - kernel_w) // stride + 1
    out_h = (X.shape[-2] - kernel_h) // stride + 1

    xcol_shape = (X.shape[0], X.shape[1] * kernel_h * kernel_w, out_h * out_w)
    return X[im2col_map(xcol_shape, kernel_h, kernel_w, stride, out_w)].reshape(xcol_shape)


