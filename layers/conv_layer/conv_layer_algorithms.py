from abc import ABC, abstractmethod
from typing import Tuple

from backend.backend import xp
from layers.conv_layer.transformations import im2col, col2im


class Conv2DAlgo(ABC):

    def __init__(self, padding: int = 0, stride: int = 1):
        self.pad = padding
        self.s = stride

    def _output_shape(self, X_shape: tuple, W_shape: tuple) -> tuple:
        from layers.conv_layer.convolution_layer import Convolution2D
        h = Convolution2D.get_height(X_shape[-2], W_shape[-2], self.pad, self.s)
        w = Convolution2D.get_width(X_shape[-1], W_shape[-1], self.pad, self.s)
        output_shape = (X_shape[0], W_shape[0], h, w)
        return output_shape

    def _add_padding(self, X: xp.ndarray) -> xp.ndarray:
        if self.pad == 0:
            return X
        return xp.pad(X, ((0, 0), (0, 0), (self.pad, self.pad),
                      (self.pad, self.pad)), mode='constant')

    def _remove_padding(self, X: xp.ndarray) -> xp.ndarray:
        if self.pad == 0:
            return X
        return X[:, :, self.pad: - self.pad, self.pad: - self.pad]

    def prepare(self, X: xp.ndarray, W: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray]:
        Y = xp.zeros(self._output_shape(X.shape, W.shape), dtype=float)
        return self._add_padding(X), Y

    @abstractmethod
    def __call__(self, X: xp.ndarray, W: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
        pass

    @abstractmethod
    def backward(self, dEdO: xp.ndarray, X: xp.ndarray, W: xp.ndarray) -> \
            Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
        pass


class FourForLoops(Conv2DAlgo):
    """
    The simplest algorithm. In the forward pass we use four nested loops that, for each sample in the batch,
    for each output feature map, and for each pixel, compute the activation potential.
    In the backward pass we follow the same structure with four nested loops, iterating through the output tensor
    and computing the required partial derivatives.
    """


    def __init__(self):
        super().__init__()

    def __call__(self, X: xp.ndarray, W: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
        X, Y = self.prepare(X, W)

        output_h, output_w = Y.shape[2], Y.shape[3]
        kernel_h, kernel_w = W.shape[2], W.shape[3]
        for n in range(X.shape[0]):
            for out_map_i in range(W.shape[0]):
                for i in range(output_h):
                    for j in range(output_w):
                        """
                        When we know the (i, j) position in the output feature map, we need to find the corresponding "slice" of the
                        input feature maps on which the filters must be applied to produce that specific output value at (i, j).
                        This slice lies within the bounds [input_i_start, input_i_end) along the height and [input_j_start, input_j_end) along the width.
                        """
                        in_i_start = i * self.s
                        in_j_start = j * self.s

                        in_i_end = in_i_start + kernel_h
                        in_j_end = in_j_start + kernel_w

                        Y[n, out_map_i, i, j] = xp.sum(X[n, :, in_i_start: in_i_end, in_j_start: in_j_end] * W[out_map_i]) + b[out_map_i]
        return Y

    def backward(self, dEdO: xp.ndarray, X: xp.ndarray, W: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
        X = self._add_padding(X)
        dEdX = xp.zeros_like(X)
        dEdW = xp.zeros_like(W)
        dEdb = xp.zeros((dEdO.shape[-3], ), dtype=float)

        output_w = dEdO.shape[-1]
        output_h = dEdO.shape[-2]

        for n in range(X.shape[0]):
            for out_map_i in range(W.shape[0]):
                dEdb[out_map_i] += xp.sum(dEdO[n, out_map_i, :, :])
                for i in range(output_h):
                    for j in range(output_w):
                        """
                        The indexing logic is the same as in the forward pass, except now we look at it from the opposite perspective:
                        "who influences whom."  
                        For an output at position (nb, out_map_idx, i, j), we determine which weights were used by the corresponding
                        slice of the input feature map—that is, how the weights influence the outputs and how the inputs influence the outputs.
                        When computing partial derivatives with respect to the biases, the situation is similar but even simpler:
                        the bias at index i influences the entire i-th output feature map for every sample in the batch.
                        """
                        in_i_start = i * self.s
                        in_j_start = j * self.s

                        in_i_end = in_i_start + W.shape[2]
                        in_j_end = in_j_start + W.shape[3]

                        dEdX[n, :, in_i_start: in_i_end, in_j_start: in_j_end] += dEdO[n, out_map_i, i, j] * W[out_map_i]

                        dEdW[out_map_i] += dEdO[n, out_map_i, i, j] * X[n, :, in_i_start: in_i_end, in_j_start: in_j_end]

        dEdX = self._remove_padding(dEdX)

        return dEdX, dEdW, dEdb


class ThreeForLoops(Conv2DAlgo):
    def __init__(self):
        super().__init__()

    def __call__(self, X: xp.ndarray, W: xp.ndarray,
                 b: xp.ndarray) -> xp.ndarray:
        X, Y = self.prepare(X, W)

        # For each output feature map, we iterate over every pixel, processing all samples in the batch simultaneously.
        output_h, output_w = Y.shape[2], Y.shape[3]
        kernel_h, kernel_w = W.shape[2], W.shape[3]

        for out_map_i in range(W.shape[0]):
            for i in range(output_h):
                for j in range(output_w):
                    """
                    When we know the (i, j) position in the output feature map, we must locate the corresponding slice of the
                    input feature maps on which the filters need to be applied to produce the value at (i, j).
                    This slice lies within the bounds [input_i_start, input_i_end) along the height and [input_j_start, input_j_end) along the width.
                    """

                    in_i_start = i * self.s
                    in_j_start = j * self.s

                    in_i_end = in_i_start + kernel_h
                    in_j_end = in_j_start + kernel_w

                    """
                    Consider the following line of code. From the padded_input tensor we extract a 4D slice by taking all elements
                    along the first two dimensions (every sample in the batch and every input feature map) and selecting only the
                    spatial region defined by input_i_start/input_i_end and input_j_start/input_j_end. This slice has shape
                    Nb × D_l × k_dim × k_dim, where Nb is the batch size, D_l the number of input feature maps, and k_dim the filter size.

                    We then multiply this slice by self._W[out_map_idx], a 3D tensor of shape D_l × k_dim × k_dim. NumPy broadcasts this
                    3D tensor so it behaves as if it were Nb × D_l × k_dim × k_dim, effectively applying the filter to every sample.

                    After obtaining this product (still Nb × D_l × k_dim × k_dim), we compute activation potentials for the given
                    output map at position (i, j) by summing over all dimensions except the first (the batch dimension). We achieve
                    this with xp.sum over axes (1, 2, 3).
                    """
                    xp.sum(X[:, :, in_i_start: in_i_end, in_j_start: in_j_end] * W[out_map_i],
                           axis=(1, 2, 3), out=Y[:, out_map_i, i, j])

                    xp.add(Y[:, out_map_i, i, j], b[out_map_i], out=Y[:, out_map_i, i, j])
        return Y

    def backward(self, dEdO: xp.ndarray, X: xp.ndarray, W: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
        X = self._add_padding(X)
        dEdX = xp.zeros_like(X)
        dEdW = xp.zeros_like(W)
        dEdb = xp.zeros((dEdO.shape[-3],), dtype=float)

        output_w = dEdO.shape[-1]
        output_h = dEdO.shape[-2]

        # Using three loops instead of four in the backward pass results in code that runs ~ 10x faster.

        for out_map_idx in range(W.shape[0]):
            dEdb[out_map_idx] = xp.sum(dEdO[:, out_map_idx, :, :])
            for i in range(output_h):
                for j in range(output_w):
                    in_i_start = i * self.s
                    in_j_start = j * self.s

                    in_i_end = in_i_start + W.shape[2]
                    in_j_end = in_j_start + W.shape[3]

                    # xp.einsum("i,jkl->ijkl", dEdO[:, out_map_idx, i, j], kernel_tensor[out_map_idx])
                    # We expect 4D output Nb x D x k x k
                    dEdX[:, :, in_i_start: in_i_end, in_j_start: in_j_end] += xp.einsum("i,jkl->ijkl", dEdO[:, out_map_idx, i, j], W[out_map_idx])
                    """
                    When computing the weight-gradient tensor, we need to sum the partial derivatives over all samples in the batch.
                    Each i-th element of the first array must be multiplied by the i-th 3D tensor of the second 4D array, and then
                    summed along the 0-th axis. The einsum expression "i,ijkl->jkl" performs exactly this: it produces a 3D output
                    (the indices after ->). The index i appears only on the left side, not on the right, which means that this
                    dimension is summed out.
                    """
                    dEdW[out_map_idx] += xp.einsum("i,ijkl->jkl", dEdO[:, out_map_idx, i, j],
                                                   X[:, :, in_i_start: in_i_end, in_j_start: in_j_end])

        dEdX = self._remove_padding(dEdX)
        return dEdX, dEdW, dEdb


class TwoForLoops(Conv2DAlgo):
    def __init__(self):
        super().__init__()

    def __call__(self, X: xp.ndarray, W: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
        X, Y = self.prepare(X, W)

        # For each output feature map and each sample in the batch, we compute the activation potential pixel by pixel, all at once.

        output_h, output_w = Y.shape[2], Y.shape[3]
        kernel_h, kernel_w = W.shape[2], W.shape[3]

        for i in range(output_h):
            for j in range(output_w):
                in_i_start = i * self.s
                in_j_start = j * self.s

                in_i_end = in_i_start + kernel_h
                in_j_end = in_j_start + kernel_w
                """
                Now we need to multiply two 4D slices in a way that produces a 2D slice. The multiplication must be
                element-wise, and the results in the last two dimensions must be reduced to a single number by summing.
                These last two dimensions correspond to a patch from the padded input (for each sample in the batch) and to the
                filter matrix from the 4D filter tensor.
                In the resulting 2D output of the einsum operation, the element at position (i, m) represents the value for the
                i-th sample in the batch and the m-th output feature map. That value is obtained by multiplying elements over
                indices j, k, and l component-wise and then summing over the dimensions associated with j, k, and l.
                """
                Y[:, :, i, j] = xp.einsum('ijkl,mjkl->im', X[:, :, in_i_start: in_i_end, in_j_start: in_j_end], W)
                Y[:, :, i, j] += b
        return Y

    def backward(self, dEdO: xp.ndarray, X: xp.ndarray, W: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
        X = self._add_padding(X)
        dEdX = xp.zeros_like(X)
        dEdW = xp.zeros_like(W)
        dEdb = xp.zeros((dEdO.shape[-3],), dtype=float)

        output_w = dEdO.shape[-1]
        output_h = dEdO.shape[-2]

        for i in range(output_h):
            for j in range(output_w):
                in_i_start = i * self.s
                in_j_start = j * self.s

                in_i_end = in_i_start + W.shape[2]
                in_j_end = in_j_start + W.shape[3]

                dEdX[:, :, in_i_start: in_i_end, in_j_start: in_j_end] += \
                    xp.einsum("ij,jklm->iklm", dEdO[:, :, i, j], W)

                dEdW += xp.einsum("ij,iklm->jklm", dEdO[:, :, i, j], X[:, :, in_i_start: in_i_end,
                                  in_j_start: in_j_end])
                dEdb += xp.sum(dEdO[:, :, i, j], axis=0)

        dEdX = self._remove_padding(dEdX)

        return dEdX, dEdW, dEdb


class Matmul(Conv2DAlgo):
    """
    The idea behind this algorithm (and the next one) is different: we aim to obtain the desired results using
    matrix multiplications—just one per sample in the forward pass and two in the backward pass. Since the needed
    behavior cannot be achieved by multiplying the raw input matrices directly, we instead construct appropriate
    matrices so that the result of the multiplication matches the quantities we want.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, X: xp.ndarray, W: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
        Y_shape = self._output_shape(X.shape, W.shape)
        X = self._add_padding(X)
        """
        We first need to construct a "special" matrix derived from the input such that multiplying it by the kernel
        (which will also be represented as a matrix) yields exactly the result we need. To create this matrix, we use
        the im2col transformation.
        """
        Xcol = im2col(X, W.shape[-2], W.shape[-1], self.s)

        # From kernel, we need to get metrix with dimensions: Dl+1 x (Dl * Kh * Kw)
        Wmat = W.reshape((W.shape[0], -1))

        """
        The im2col function produces a 3D tensor of shape Nb × (Dl * Kh * Kw) × (Oh * Ow), where Oh and Ow are the
        height and width of each output feature map.
        Multiplying a tensor of shape Dl+1 × (Dl * Kh * Kw) with a tensor of shape Nb × (Dl * Kh * Kw) × (Oh * Ow)
        via matmul yields Nb × Dl+1 × (Oh * Ow). When matmul receives a 3D tensor, it treats it as a batch of matrices
        and performs one matrix multiplication per batch element.
        Because the im2col tensor is carefully constructed, this product already contains the needed result. We then
        reshape it into a 4D tensor and split the last dimension into two spatial dimensions, followed by adding the bias.
        """
        Y = xp.matmul(Wmat, Xcol)
        Y = Y.reshape(Y_shape)

        for o_ch in range(Y.shape[1]):
            Y[:, o_ch, :, :] += b[o_ch]

        return Y

    def backward(self, dEdO: xp.ndarray, X: xp.ndarray, W: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
        X = self._add_padding(X)
        Xcol = im2col(X, W.shape[-2], W.shape[-1], self.s)
        dEdb = xp.zeros((dEdO.shape[-3],), dtype=float)

        xp.sum(dEdO, axis=(0, 2, 3), out=dEdb)

        WT = (W.reshape((W.shape[0], -1))).T
        dAmat = dEdO.reshape((dEdO.shape[0], dEdO.shape[1], -1))

        dXcol = xp.matmul(WT, dAmat)

        """
        The tensor dXcol, produced by multiplying WT and dAmat, is not yet the final gradient w.r.t. the inputs.
        Its shape is Nb × (Dl * Kh * Kw) × (Oh * Ow), while the true input-gradient tensor dEdX must be
        Nb × Dl × Ih × Iw (Ih and Iw are the input height and width).
        dXcol contains scattered pieces of the input gradients that need to be accumulated into dEdX. This
        reconstruction is performed by col2im. Details of how those pieces map back into the input space are handled
        inside the col2im function.
        """
        dEdX = col2im(dXcol, X.shape, W.shape[-2], W.shape[-1], self.s)
        """
        Computing the weight gradients is more straightforward: multiply the gradient matrix by the
        transpose of the im2col matrix from the forward pass, then sum the resulting products over all
        samples in the batch.
        """
        dEdW = xp.einsum('ijl,ikl->jk', dAmat, Xcol)
        dEdW = dEdW.reshape(W.shape)

        dEdX = self._remove_padding(dEdX)

        return dEdX, dEdW, dEdb
