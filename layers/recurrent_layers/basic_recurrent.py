from typing import List

from models.adaptive_object import AdaptiveObject
from backend.backend import xp
from layers.activation_functions.activation_function import ActivationFunction
from layers.activation_functions.sigmoid import Sigmoid
from layers.activation_functions.tanh import Tanh
from weight_initializers.random_initialize import rand_init


class RecurrentLayer(AdaptiveObject):

    def __init__(self, in_units: int, hidden_units: int, out_units: int, out_act_f: ActivationFunction = Sigmoid(),
                 recurrent_act_f: ActivationFunction = Tanh(), name: str = "Recurrent layer", init_mode: str = "xavier_uniform"):
        super().__init__(name)
        """
        Inputs to the recurrent layer have shape Nb × T × m, where Nb is the batch size, T the sequence length, and m 
        the dimensionality of each timestep input. input[n, t] is the input vector of sample n at time t; input[:, t, :] 
        is an Nb × m matrix like in non-recurrent networks; input[n, :, :] is a T × m matrix containing the full sequence for sample n. 
        The layer outputs a tensor of shape Nb × T × output_units, and the design allows placing one or multiple recurrent layers anywhere in the network.
        """

        self._output_units = out_units
        self._hidden_units = hidden_units

        self._x0: xp.ndarray = None
        self._ay: List[xp.ndarray] = [None]  # saving activation potentials of output neurons
        self._ax: List[xp.ndarray] = [None]
        # Matrix of weights used to multiply the previous hidden state of the recurrent layer:
        self._Wx = rand_init(hidden_units, hidden_units, init_mode)
        # Matrix of weights used to multiply the new hidden state in order to produce the outputs (more precisely, the output activation potentials):
        self._Wy = rand_init(out_units, hidden_units, init_mode)
        # Matrix of weights used to multiply the inputs:
        self._Wu = rand_init(hidden_units, in_units, init_mode)  # We need the input size in order to initialize this matrix.
        self._bx = xp.zeros((1, hidden_units), dtype=float)
        self._by = xp.zeros((1, out_units), dtype=float)

        self._dEdWx: xp.ndarray = None
        self._dEdWy: xp.ndarray = None
        self._dEdWu: xp.ndarray = None
        self._dEdbx: xp.ndarray = None
        self._dEdby: xp.ndarray = None

        self.f = recurrent_act_f
        self.h = out_act_f

        self.reset_state = True

    def _pre_fw(self, inputs: xp.ndarray):
        super()._pre_fw(inputs)
        if self._x0 is None or self.reset_state or self._x0.shape[0] != inputs.shape[0]:
            self._x0 = xp.zeros((inputs.shape[0], self._hidden_units))

    @property
    def parameters(self) -> tuple:
        return self._Wx, self._Wy, self._Wu, self._bx, self._by

    @parameters.setter
    def parameters(self, val: tuple):
        self._Wx, self._Wy, self._Wu, self._bx, self._by = val

    def __call__(self, inputs: xp.ndarray) -> xp.ndarray:
        Nb = inputs.shape[0]
        T = inputs.shape[1]
        Y = xp.zeros((Nb, T, self._output_units), dtype=float)
        x = self._x0
        for k in range(1, T + 1):
            ax = xp.matmul(inputs[:, k - 1, :], self._Wu.T) + xp.matmul(x, self._Wx.T) + self._bx
            x = self.f(ax)
            ay = xp.matmul(x, self._Wy.T) + self._by
            y = self.h(ay)
            Y[:, k - 1, :] = y

            if self.training:
                self._ax.append(ax)
                self._ay.append(ay)
        if not self.reset_state:
            self._x0 = x

        return Y

    def backward(self, dEdO: xp.ndarray) -> xp.ndarray:
        T = dEdO.shape[1]

        self._dEdWx = xp.zeros_like(self._Wx)
        self._dEdWy = xp.zeros_like(self._Wy)
        self._dEdWu = xp.zeros_like(self._Wu)
        self._dEdbx = xp.zeros_like(self._bx)
        self._dEdby = xp.zeros_like(self._by)

        dEdU = xp.zeros_like(self._inputs)

        for k in reversed(range(1, T + 1)):

            dh = self.h.deriv(self._ay[k])
            self._ay.pop(-1)
            xk = self.f(self._ax[k])
            self._dEdWy += xp.matmul((dEdO[:, k - 1, :] * dh).T, xk)
            self._dEdby += xp.sum(dEdO[:, k - 1, :] * dh, 0)

            dEdXk = xp.matmul(dEdO[:, k - 1, :] * dh, self._Wy)
            if k == T:
                delta_x = dEdXk
            else:
                delta_x = xp.matmul(df * delta_x, self._Wx) + dEdXk

            df = self.f.deriv(self._ax[k])
            self._ax.pop(-1)
            if k == 1:
                x_prev = self._x0
            else:
                x_prev = self.f(self._ax[k - 1])
            self._dEdWx += xp.matmul((df * delta_x).T, x_prev)
            self._dEdWu += xp.matmul((df * delta_x).T, self._inputs[:, k - 1, :])
            self._dEdbx += xp.sum(df * delta_x, axis=0)

            dEdU[:, k - 1, :] = xp.matmul(df * delta_x, self._Wu)

        return dEdU

    def update_parameters(self):
        self._optimizer.update_parameters(self._Wx, self._dEdWx)
        self._optimizer.update_parameters(self._Wu, self._dEdWu)
        self._optimizer.update_parameters(self._Wy, self._dEdWy)
        self._optimizer.update_parameters(self._bx, self._dEdbx)
        self._optimizer.update_parameters(self._by, self._dEdby)

        self._dEdWx = None
        self._dEdWy = None
        self._dEdWu = None
        self._dEdbx = None
        self._dEdby = None