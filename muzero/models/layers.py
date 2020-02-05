from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from typing import Tuple, List

from tensorflow import Tensor
from tensorflow.math import add


class Layer:
    """
    Base class for the different layers
    """
    def __init__(self, input_layer: 'Layer' = None, input_tensor: Tensor = None):
        """
        :param input_layer: The prior layer that this layers gets its data from
        """
        if input_layer is not None and input_tensor is not None:
            raise Exception('MuZero Layer', 'Dont provide a input tensor and a input layer, but only one of them')
        elif input_layer is None and input_tensor is None:
            raise Exception('MuZero Layer', 'You have to define either the input layer or the input tensor')
        if input_layer is not None:
            self.input_layer = input_layer
            self.input = input_layer.output
        else:
            self.input = input_tensor
        self.output = self.determine_output()


class Input(Layer):
    """
    Input layer of a model
    """
    def __init__(self, input_tensor: Tensor):
        super().__init__(input_layer=None, input_tensor=input_tensor)

    @tf.function
    def determine_output(self) -> Tensor:
        return self.input


class Dense(Layer):
    """
    Densely connected layer
    """
    def __init__(self, input_layer: Layer, num: int, bias: float = 0.0):
        if len(input_layer.output.shape) != 2:
            raise Exception('MuZero Layer', 'The input of a dense layer must have two dimensions')

        self.num_neurons = num
        self.bias = bias
        self.weights = [tf.constant(0.1, shape=(num,)) for _ in range(len(input_layer.output.shape))]
        super().__init__(input_layer=input_layer)

    @tf.function
    def determine_output(self) -> Tensor:
        return add(tf.matmul(self.input, self.weights), self.bias)


class Conv2d(Layer):
    """
    2D Convolutional layer
    """
    def __init__(self, input_layer: Layer, filters: int = 1, kernel: Tuple[int] = (1, 1), stride: Tuple[int] = (1, 1)):
        """
        TODO: Add the implementation of a convolutional layer

        :param input_layer:
        :param filters:
        :param kernel:
        :param stride:
        """
        if len(input_layer.output.shape) != 2:
            raise Exception('MuZero Layer', 'The input of a conv2d layer must have three dimensions')

        super().__init__(input_layer=input_layer)


class Conv3d(Layer):
    pass


class Add(Layer):
    pass


class ReLu(Layer):
    pass


class BatchNormalization(Layer):
    pass


class TanH(Layer):
    pass
         

hidden_state = tf.constant([[5, 2]], dtype=float)
x = Input(hidden_state)
x = Dense(x, 3)
print(x.output)
