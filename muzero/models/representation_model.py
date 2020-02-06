from muzero.models.layer_blocks import ConvBlock, ResConvBlock

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import AveragePooling2D


class AtariDownSampler(Model):
    """
    Because the observations of Atari games have large spatial resolution, we have to downsample it first.
    """

    def __init__(self):
        super(AtariDownSampler, self).__init__(name='AtariObservationDownSampler')

        self.conv_block_1 = ConvBlock(filters=128, kernel_size=(3, 3), strides=(2, 2), name='AtariDownsamplerConv1')
        self.residual_block_1 = ResConvBlock(filters=128, kernel_size=(3, 3), name='AtariDownsamplerRes1')
        self.residual_block_2 = ResConvBlock(filters=128, kernel_size=(3, 3), name='AtariDownsamplerRes2')
        self.conv_block_2 = ConvBlock(filters=256, kernel_size=(3, 3), strides=(2, 2), name='AtariDownsamplerConv2')
        self.residual_block_3 = ResConvBlock(filters=256, kernel_size=(3, 3), name='AtariDownsamplerRes3')
        self.residual_block_4 = ResConvBlock(filters=256, kernel_size=(3, 3), name='AtariDownsamplerRes4')
        self.residual_block_5 = ResConvBlock(filters=256, kernel_size=(3, 3), name='AtariDownsamplerRes5')
        self.avg_pooling_layer_1 = AveragePooling2D()
        self.residual_block_6 = ResConvBlock(filters=256, kernel_size=(3, 3), name='AtariDownsamplerRes6')
        self.residual_block_7 = ResConvBlock(filters=256, kernel_size=(3, 3), name='AtariDownsamplerRes7')
        self.residual_block_8 = ResConvBlock(filters=256, kernel_size=(3, 3), name='AtariDownsamplerRes8')
        self.avg_pooling_layer_2 = AveragePooling2D()

    def call(self, input_tensor, training=False):
        """
        :param input_tensor:
            Starting with an input observation of resolution 96x96 and 128 planes (32 history frames of 3 color channels
            each, concatenated with the corresponding32 actions broadcast to planes)
        :param training:
        :return: The hidden state containing all essential information
        """
        # Resolution: 96x96
        x = self.conv_block_1(input_tensor, training=training)
        # Resolution: 48x48
        x = self.residual_block_1(x, training=training)
        x = self.residual_block_2(x, training=training)
        x = self.conv_block_2(x, training=training)
        # Resolution: 24x24
        x = self.residual_block_3(x, training=training)
        x = self.residual_block_4(x, training=training)
        x = self.residual_block_5(x, training=training)
        x = self.avg_pooling_layer_1(x)
        # Resolution: 12x12
        x = self.residual_block_6(x, training=training)
        x = self.residual_block_7(x, training=training)
        x = self.residual_block_8(x, training=training)
        x = self.avg_pooling_layer_2(x)
        # Resolution: 6x6
        return x


class BoardGameDownSampler(Model):
    """
    Because the observations of Atari games have large spatial resolution, we have to downsample it first.
    """

    def __init__(self):
        super(BoardGameDownSampler, self).__init__(name='AtariObservationDownSampler')

        self.conv_block_1 = ConvBlock(filters=256, kernel_size=(3, 3), name='BoardGameInput')

    def call(self, input_tensor, training=False):
        """
        :param input_tensor:
            Starting with an input observation of resolution 96x96 and 128 planes (32 history frames of 3 color channels
            each, concatenated with the corresponding32 actions broadcast to planes)
        :param training:
        :return: The hidden state containing all essential information
        """
        x = self.conv_block_1(input_tensor, training=training)
        return x


resolution_sampler_dict = {
    'Atari': AtariDownSampler,
    'BoardGame': BoardGameDownSampler
}


class RepresentationModel(Model):
    """
    The model representing the dynamics function
    """

    def __init__(self, game_mode: str = 'BoardGame'):
        super(RepresentationModel, self).__init__(name='RepresentationModel')

        self.representation_input = resolution_sampler_dict[game_mode]()

        self.res_block_1 = ResConvBlock(filters=256, kernel_size=(3, 3), name='DynamicsResLayer1')
        self.res_block_2 = ResConvBlock(filters=256, kernel_size=(3, 3), name='DynamicsResLayer2')
        self.res_block_3 = ResConvBlock(filters=256, kernel_size=(3, 3), name='DynamicsResLayer3')
        self.res_block_4 = ResConvBlock(filters=256, kernel_size=(3, 3), name='DynamicsResLayer4')
        self.res_block_5 = ResConvBlock(filters=256, kernel_size=(3, 3), name='DynamicsResLayer5')
        self.res_block_6 = ResConvBlock(filters=256, kernel_size=(3, 3), name='DynamicsResLayer6')
        self.res_block_7 = ResConvBlock(filters=256, kernel_size=(3, 3), name='DynamicsResLayer7')
        self.res_block_8 = ResConvBlock(filters=256, kernel_size=(3, 3), name='DynamicsResLayer8')
        self.res_block_9 = ResConvBlock(filters=256, kernel_size=(3, 3), name='DynamicsResLayer9')
        self.res_block_10 = ResConvBlock(filters=256, kernel_size=(3, 3), name='DynamicsResLayer10')

        self.res_block_11 = ResConvBlock(filters=256, kernel_size=(3, 3), name='DynamicsResLayer11')
        self.res_block_12 = ResConvBlock(filters=256, kernel_size=(3, 3), name='DynamicsResLayer12')
        self.res_block_13 = ResConvBlock(filters=256, kernel_size=(3, 3), name='DynamicsResLayer13')
        self.res_block_14 = ResConvBlock(filters=256, kernel_size=(3, 3), name='DynamicsResLayer14')
        self.res_block_15 = ResConvBlock(filters=256, kernel_size=(3, 3), name='DynamicsResLayer15')
        self.res_block_16 = ResConvBlock(filters=256, kernel_size=(3, 3), name='DynamicsResLayer16')

        self.conv_block_output = ConvBlock(filters=1, kernel_size=(1, 1), name='DynamicsModelInput')

    def call(self, input_tensor, training = False):
        """

        :param input_tensor: the observation sampled to the resolution of 6x6
        :param training: bool
        :return: The hidden state
        """
        x = self.representation_input(input_tensor, training=training)

        x = self.res_block_1(x, training=training)
        x = self.res_block_2(x, training=training)
        x = self.res_block_3(x, training=training)
        x = self.res_block_4(x, training=training)
        x = self.res_block_5(x, training=training)
        x = self.res_block_6(x, training=training)
        x = self.res_block_7(x, training=training)
        x = self.res_block_8(x, training=training)
        x = self.res_block_9(x, training=training)
        x = self.res_block_10(x, training=training)

        x = self.res_block_11(x, training=training)
        x = self.res_block_12(x, training=training)
        x = self.res_block_13(x, training=training)
        x = self.res_block_14(x, training=training)
        x = self.res_block_15(x, training=training)
        x = self.res_block_16(x, training=training)

        x = self.conv_block_output(x, training=training)

        return x


if __name__ == '__main__':
    # Atari
    rep = RepresentationModel(game_mode='Atari')
    hidden_state = rep(tf.ones([32, 96, 96, 128]))
    assert hidden_state.shape == (32, 6, 6, 1)
    print("Hidden state Shape: ", hidden_state.shape)

    # TicTocToe
    rep = RepresentationModel(game_mode='BoardGame')
    hidden_state = rep(tf.ones([32, 3, 3, 17]))
    assert hidden_state.shape == (32, 3, 3, 1)
    print("Hidden state Shape: ", hidden_state.shape)
