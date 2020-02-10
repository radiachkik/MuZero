from muzero.models.representation_model import RepresentationModel
from muzero.models.dynamics_model import DynamicsModel
from muzero.models.prediction_model import PredictionModel

import tensorflow as tf

game_mode_dict = {
    'Atari': 'Atari',
    'BoardGame': 'BoardGame'
}


class Network:
	"""
	This class represents the network structure consisting of the representation, dynamics and prediction model
	"""
	def __init__(self, num_action: int, game_mode: str):
		self.representation_model = RepresentationModel(game_mode)
		self.dynamics_model = DynamicsModel()
		self.prediction_model = PredictionModel(num_actions=num_action)
		self.train_step = 0

	def initial_inference(self, image, training: bool = True):
		"""
		Execute the representation function in order to get the current hidden state, then execute the prediction function

		:param image: The history of the last states
		:param training: whether the batch normalization should depend on the current batch (not overall statistics)
		:return: NetworkOutput (value, reward, policy_logits, hidden_state)
		"""
		hidden_state = self.representation_model(image, training=training)

		value, policy_distribution = self.prediction_model(hidden_state, training=training)
		reward = tf.constant(0, shape=(1, 1), dtype=float, name='zero_reward')

		return tf.identity(value, 'value'), tf.identity(reward, 'reward'), tf.identity(policy_distribution, 'policy'), tf.identity(hidden_state, 'hidden_state')

	def recurrent_inference(self, hidden_state_with_action, training: bool = True):
		"""
		First execute the dynamics function to get the next hidden state, then execute the prediction function

		:param hidden_state_with_action: The current hidden state combined with the chosen action
		:param training: bool
		:return: NetworkOutput
		"""
		next_hidden_state, reward = self.dynamics_model(hidden_state_with_action)
		value, policy_distribution = self.prediction_model(next_hidden_state, training=training)

		return tf.identity(value, 'value'), tf.identity(reward, 'reward'), tf.identity(policy_distribution, 'policy'), tf.identity(next_hidden_state, 'next_hidden_state')

	def get_weights_callback(self):
		"""
		:return: A callback returning the weights of this network.
		"""
		def get_weights():
			models = (self.representation_model, self.prediction_model, self.dynamics_model)
			return [weights for weight_list in map(lambda n: n.weights, models) for weights in weight_list]

		return get_weights

	def training_steps(self) -> int:
		"""
		:return: How many steps / batches the network has been trained for.
		"""
		return self.train_step
