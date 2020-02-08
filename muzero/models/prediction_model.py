from muzero.models.layer_blocks import ConvBlock, ResConvBlock, ValueHead, PolicyHead

import tensorflow as tf

class PredictionModel(tf.keras.Model):
    """
    The model representing the prediction function
    """

    def __init__(self, num_actions: int):
        """
        Loss Function:
        Loss function = value loss + policy loss + L2 regularization
        Value loss = mean squared error between the value predicted and the one returned by the MCTS
        Policy Loss = cross entropy between the predicted propability distribution and the one returned by the MCTS
        Regularization helps prevent over-fitting by adding a penalty if the weights within the actual network get to big

        Optimizer:
        SGD with momentum optimizer, momentum = 0.9
        Learning Rate: 10^-2 -> 10^-4 (after 600k training steps, no )

        :param num_actions: The number of different actions to predict the probability for
        """
        super(PredictionModel, self).__init__(name='PredictionModel')
        self.conv_block = ConvBlock(filters=256, kernel_size=(3, 3), name='PredictionModelInput')

        self.res_block_1 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer1')
        self.res_block_2 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer2')
        self.res_block_3 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer3')
        self.res_block_4 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer4')
        self.res_block_5 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer5')
        self.res_block_6 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer6')
        self.res_block_7 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer7')
        self.res_block_8 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer8')
        self.res_block_9 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer9')
        self.res_block_10 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer10')

        self.res_block_11 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer11')
        self.res_block_12 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer12')
        self.res_block_13 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer13')
        self.res_block_14 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer14')
        self.res_block_15 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer15')
        self.res_block_16 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer16')
        self.res_block_17 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer17')
        self.res_block_18 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer18')
        self.res_block_19 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer19')
        self.res_block_20 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer20')

        self.res_block_21 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer21')
        self.res_block_22 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer22')
        self.res_block_23 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer23')
        self.res_block_24 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer24')
        self.res_block_25 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer25')
        self.res_block_26 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer26')
        self.res_block_27 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer27')
        self.res_block_28 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer28')
        self.res_block_29 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer29')
        self.res_block_30 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer30')

        self.res_block_31 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer31')
        self.res_block_32 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer32')
        self.res_block_33 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer33')
        self.res_block_34 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer34')
        self.res_block_35 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer35')
        self.res_block_36 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer36')
        self.res_block_37 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer37')
        self.res_block_38 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer38')
        self.res_block_39 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer39')
        self.res_block_40 = ResConvBlock(filters=256, kernel_size=(3, 3), name='PredictionResLayer40')

        self.value_head = ValueHead()
        self.policy_head = PolicyHead(num_actions=num_actions)

    def call(self, input_tensor, training = False):
        """
        :param input_tensor: The current hidden state
        :param training: bool
        :return: The predicted probability distribution of the policy
        """
        x = self.conv_block(input_tensor, training=training)

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
        x = self.res_block_17(x, training=training)
        x = self.res_block_18(x, training=training)
        x = self.res_block_19(x, training=training)
        x = self.res_block_20(x, training=training)

        value = self.value_head(x, training=training)
        policy = self.policy_head(x, training=training)

        return value, policy

