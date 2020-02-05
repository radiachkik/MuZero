from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from typing import Tuple, List

from tensorflow import Tensor


class PredictionModel():
    """
    Convolutional block:
    1. 2D Conv layer with 256 filters(kernel=3x3)  <- Input
    2. Batch normalization
    3. Relu activation function (clipping all negativ values to zero) -> Output

    Residual convolutional block:
    1. 2D Conv layer with 256 filters(kernel=3x3)  <- Input
    2. Batch normalization
    3. Relu activation function (clipping all negativ values to zero)
    4. 2D Conv layer with 256 filters(kernel=3x3)
    5. Batch normalization
    6. Skip connection where the original input is added to the output of the batch normalization  <- Input
    7. Relu activation function (clipping all negativ values to zero)  -> Output

    Value Head:
    1. 2D Conv layer with 1 filters(kernel=1x1)  <- Input
    2. Batch normalization
    3. Relu activation function (clipping all negativ values to zero)
    4. Dense layer (fully connected) with 256 neurons
    5. Relu activation function (clipping all negativ values to zero)
    6. Dense layer (fully connected) with 1 neuron
    7. TanH activation function giving an output between -1 and 1   ->  Output

    Policy Head:
    1. 2D Conv layer with 2 filters(kernel=1x1)  <- Input
    2. Batch normalization
    3. Relu activation function (clipping all negativ values to zero)
    4. Dense layer (fully connected) with one neuron for each possible action  ->  Output

    Loss Function:
    Loss function = value loss + policy loss + L2 regularization
    Value loss = mean squared error between the value predicted and the one returned by the MCTS
    Policy Loss = cross entropy between the predicted propability distribution and the one returned by the MCTS
    Regularization helps prevent over-fitting by adding a penalty if the weights within the actual network get to big

    Optimizer:
    SGD with momentum optimizer, momentum = 0.9
    Learning Rate: 10^-2 -> 10^-4 (after 600k training steps, no )
    """
    def __init__(self, hidden_state: Tuple[Tensor]):
        """
        Predict the value(discounted sum of future rewards) and policy(distribution of visit counts for all possible
        actions) from the history of states (to evaluate the current state and policy distribution)

        :param hidden_state:
            The current hidden state, representing a 17 layers thick stack of 19x19 Go boards. The fist 8 layers are
            the current state on top and the 7 previous states below. Each of them has a 1 where the black stones are
            and a 0 where they dont are .The next 8 layers have the same structure as the first 8, except that there is
            a 1 when there is white stone and a 0 if there is none. The last (17.th) layer is either filled with 1 or
            with 0, depending on which players turn it currently is.
        :return:
            The value of the given state (the final reward) and the probability distribution for all possible actions
        """
        self.input = hidden_state


