from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, Flatten
from tensorflow.keras.activations import tanh

from typing import Tuple


class ConvBlock(tf.keras.Model):
    def __init__(self, filters: int, kernel_size: Tuple, name: str, strides: Tuple = (1, 1), padding: str = 'same'):
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
        super(ConvBlock, self).__init__(name=name)
        # The input has the shape of an hidden state
        self.conv2d_layer = Conv2D(filters, kernel_size, padding=padding, strides=strides)
        # Batch Normalization in order to allow each row to learn more independently, as its guarantees
        # each layer to constantly have approximately the same distribution of input values
        self.batch_norm_layer = BatchNormalization()

    def call(self, input_tensor, training = False):
        x = self.conv2d_layer(input_tensor)
        x = self.batch_norm_layer(x, training=training)
        # Take ReLu as the activation function as the slope does'nt plateau when the input gets bigger.
        # This helps avoiding the vanishing gradients problem
        return tf.nn.relu(x)


class ResConvBlock(tf.keras.Model):
    def __init__(self, filters: int, kernel_size: Tuple, name: str, strides: Tuple = (1, 1), padding: str = 'same'):
        super(ResConvBlock, self).__init__(name=name)
        self.conv_block = ConvBlock(filters, kernel_size, padding=padding, strides=strides, name='ResInput')
        self.conv2d_layer_2 = Conv2D(filters, kernel_size, padding=padding, strides=strides)
        self.batch_norm_layer_2 = BatchNormalization()

    def call(self, input_tensor, training = False):
        x = self.conv_block(input_tensor, training=training)
        x = self.conv2d_layer_2(x)
        x = self.batch_norm_layer_2(x, training=training)
        x += input_tensor
        return tf.nn.relu(x)


class ValueHead(tf.keras.Model):
    """
    The value head of the prediction model
    """
    def __init__(self):
        """

        """
        super(ValueHead, self).__init__(name='ValueHead')
        self.conv2d_layer = Conv2D(1, (1, 1))
        self.batch_norm_layer = BatchNormalization()
        self.flatten_layer = Flatten()
        self.dense_layer_1 = Dense(256)
        self.dense_layer_2 = Dense(1)

    def call(self, input_tensor, training = False):
        """

        :param input_tensor: The output of the last residual convolutional block
        :param training:
            True means that the layer will normalize the inputs using the the data of the current batch
            False means that the layer will normalize using the mean and variance learned during training
        :return: The predicted value of this state (discounted sum of future rewards within a specific amount of steps)
        """
        x = self.conv2d_layer(input_tensor)
        x = self.batch_norm_layer(x, training=training)
        x = tf.nn.relu(x)
        x = self.flatten_layer(x)
        x = self.dense_layer_1(x)
        x = tf.nn.relu(x)
        x = self.dense_layer_2(x)
        return tanh(x)


class PolicyHead(tf.keras.Model):
    """
    The policy head of the prediction model
    """
    def __init__(self, num_actions: int):
        """
        :param num_actions: The number of different actions
        """
        super(PolicyHead, self).__init__(name='PolicyHead')
        self.conv2d_layer = Conv2D(2, (1, 1))
        self.batch_norm_layer = BatchNormalization()
        self.flatten_layer = Flatten()
        self.dense_layer = Dense(num_actions)

    def call(self, input_tensor, training = False):
        """

        :param input_tensor: The output of the last residual convolutional block
        :param training: see ValueHead above
        :return: The predicted probability distribution of the policy
        """
        x = self.conv2d_layer(input_tensor)
        x = self.batch_norm_layer(x, training=training)
        x = tf.nn.relu(x)
        x = self.flatten_layer(x)
        x = self.dense_layer(x)
        return x


