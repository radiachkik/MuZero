from muzero.network.network_output import NetworkOutput
from muzero.models.representation_model import RepresentationModel
from muzero.models.dynamics_model import DynamicsModel
from muzero.models.prediction_model import PredictionModel

from datetime import datetime
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

	@tf.function
	def minimize_loss(self, loss):
		pass

	@tf.function
	def initial_inference(self, image, training: bool = True) -> NetworkOutput:
		"""
		Execute the representation function in order to get the current hidden state, then execute the prediction function

		:param image: The history of the last states
		:param training: whether the batch normalization should depend on the current batch (not overall statistics)
		:return: NetworkOutput (value, reward, policy_logits, hidden_state)
		"""
		input_tensor = tf.convert_to_tensor(image, dtype=float)

		hidden_state = self.representation_model(input_tensor, training=training)

		value, policy_distribution = self.prediction_model(hidden_state, training=training)

		return NetworkOutput(value=value, reward=0.0, policy_logits=policy_distribution, hidden_state=hidden_state)

	@tf.function
	def recurrent_inference(self, hidden_state, action, training: bool = True) -> NetworkOutput:
		"""
		First execute the dynamics function to get the next hidden state, then execute the prediction function

		:param hidden_state: The current hidden state
		:param action: The chosen action
		:param training: bool
		:return: NetworkOutput
		"""
		action_layer = tf.ones(hidden_state.shape) * action
		hidden_state_with_action = tf.concat([hidden_state, action_layer], axis=3)

		next_hidden_state, reward = self.dynamics_model(hidden_state_with_action)
		value, policy_distribution = self.prediction_model(next_hidden_state, training=training)

		return NetworkOutput(value=value, reward=reward, policy_logits=policy_distribution, hidden_state=hidden_state)

	def get_weights(self):
		"""
		:return: The weights of this network.
		"""
		return []

	def training_steps(self) -> int:
		"""
		:return: How many steps / batches the network has been trained for.
		"""
		return self.train_step
