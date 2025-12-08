from models.adaptive_object import AdaptiveObject
from layers.conv_layer.conv_layer_algorithms import *
from weight_initializers.random_initialize import rand_init


class Convolution2D(AdaptiveObject):
    """
    A convolutional layer is typically applied to three-dimensional inputs. The input consists of multiple
    two-dimensional feature maps. For the first convolutional layer these are usually image channels—three
    feature maps (red, green, blue), each one a matrix.
    We apply multiple filters to these inputs to produce output activation maps, which again form a 3D tensor
    made of several feature maps.
    To compute the i-th 3D activation tensor (the i-th output feature map), we apply one 2D filter (kernel) to
    each input feature map and then combine (sum) the results. Finally, we add a bias term for that output map.
    Each input–output feature-map pair has its own filter, so the total number of 2D filters is:
    desired_number_of_output_feature_maps × number_of_input_feature_maps.
    Because of this, the filter parameters form a 4D tensor of shape:
    desired_output_feature_maps × input_feature_maps × filter_size × filter_size,
    assuming square filters (though the implementation easily generalizes).
    """
    @staticmethod
    def get_height(input_height: int, kernel_height: int, padding: int, stride: int) -> int:
        return (input_height - kernel_height + 2 * padding) // stride + 1

    @staticmethod
    def get_width(input_width: int, kernel_width: int, padding: int, stride: int) -> int:
        return (input_width - kernel_width + 2 * padding) // stride + 1

    def __init__(self, in_maps_n: int, out_maps_n: int, kernel_size: int = 7,
                 padding: int = 0, stride: int = 1, algorithm: Conv2DAlgo = FourForLoops(),
                 name: str = 'Convolutional Layer'):
        super().__init__(name)
        self.in_maps_n = in_maps_n
        self.out_maps_n = out_maps_n

        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size

        w = self.in_maps_n * kernel_size * kernel_size
        W_shape = (out_maps_n, self.in_maps_n, kernel_size, kernel_size)
        self._W = rand_init(out_maps_n, w).reshape(W_shape)
        self._b = xp.zeros((out_maps_n,))

        self._dEdW = None
        self._dEdb = None
        """
        Since the forward and backward passes can be implemented in multiple ways, it's convenient to move that
        logic into a separate class hierarchy. When a specific computation is needed, we simply send a request to
        an object that “knows” how to compute it and get the result back.
        In strongly typed languages this object would belong to a type from which all concrete convolution
        algorithms inherit. Here, that class is called ConvolutionLayerAlgorithm and it is abstract (it cannot be
        instantiated directly and exists only for inheritance).
        Any object of this class or its subclasses can compute the forward and backward passes for a convolutional
        layer.
        """
        self.algo = algorithm
        self.algo.pad = self.padding
        self.algo.s = self.stride

    def __call__(self, X: xp.ndarray) -> xp.ndarray:
        return self.algo(X, self._W, self._b)

    def backward(self, dEdO: xp.ndarray) -> xp.ndarray:
        dEdX, self._dEdW, self._dEdb = self.algo.backward(dEdO, self._inputs, self._W)

        return dEdX

    @property
    def parameters(self) -> tuple:
        return self._W, self._b

    @parameters.setter
    def parameters(self, val: tuple):
        self._W, self._b = val

    def update_parameters(self):
        self._optimizer.update_parameters(self._W, self._dEdW)
        self._optimizer.update_parameters(self._b, self._dEdb)
        self._dEdW = None
        self._dEdb = None
