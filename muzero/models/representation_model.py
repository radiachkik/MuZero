from muzero.models.layer_blocks import ConvBlock, ResConvBlock, AtariDownSampler, BoardGameDownSampler

from tensorflow.keras import Model


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
