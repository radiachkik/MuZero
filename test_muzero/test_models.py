import unittest


class TestDynamicsModel(unittest.TestCase):

    def setUp(self):
        from muzero.models.dynamics_model import DynamicsModel
        from muzero.environment.action import Action
        import tensorflow as tf
        self.dynamics_model = DynamicsModel()
        self.batch_of_hidden_states = tf.ones([4, 3, 3, 1])
        self.default_action = Action(0)

    def test_output_shape(self):
        import tensorflow as tf
        import numpy as np
        input_tensor = None
        state_count = 0
        for state in self.batch_of_hidden_states:
            self.assertEqual(state.shape, (3, 3, 1))
            action = tf.ones([3, 3, 1]) * self.default_action.action_id
            state_with_action = tf.concat([state, action], axis=2)
            if input_tensor is None:
                input_tensor = np.array([state_with_action])
            else:
                input_tensor = tf.concat([input_tensor, [state_with_action]], axis=0)
            state_count += 1
            self.assertEqual(input_tensor.shape, (state_count, 3, 3, 2))
        self.assertEqual(input_tensor.shape, (4, 3, 3, 2))
        hidden_states, reward = self.dynamics_model(input_tensor, training=True)
        assert hidden_states.shape == (4, 3, 3, 1)


class TestResentationModel(unittest.TestCase):

    def test_output_shape(self):
        from muzero.models.representation_model import RepresentationModel
        import tensorflow as tf
        # Atari
        rep = RepresentationModel(game_mode='Atari')
        hidden_state = rep(tf.ones([4, 96, 96, 128]))
        self.assertEqual(hidden_state.shape, (4, 6, 6, 1))

        # TicTocToe
        rep = RepresentationModel(game_mode='BoardGame')
        hidden_state = rep(tf.ones([4, 8, 8, 17]))
        self.assertEqual(hidden_state.shape, (4, 8, 8, 1))


class TestPredictionModel(unittest.TestCase):

    def test_output_shape(self):
        from muzero.models.prediction_model import PredictionModel
        import tensorflow as tf

        num_action = 10

        pred = PredictionModel(num_action)
        value, policy = pred(tf.ones([3, 24, 24, 5]))
        self.assertEqual(value.shape, (3, 1))
        self.assertEqual(policy.shape, (3, num_action))
