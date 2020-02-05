from gym import core
from typing import List

from muzero.environment.action import Action
from muzero.environment.player import Player
from muzero.mcts.node import Node


class Game(object):
	"""
	A single episode of interaction with the environment.
	"""

	def __init__(self, environment: core.Env, number_players: int, discount: float):
		"""
		:param environment: The gym environment to interact with
		:param number_players: The number of players alternating in this environment
		:param discount: The discount to apply to future rewards when calculating the value target
		"""
		self.environment = environment
		self.action_space_size = environment.action_space.n
		self.players = [Player(i) for i in range(number_players)]
		self.step = 0
		self.discount = discount

		self.action_history = [Action(0)]
		self.reward_history = [0]
		self.root_values = []
		self.probability_distributions = []
		self.observation_history = []
		self.environment = environment
		self.observation_history.append(self.environment.reset())
		self.opponent_reward = 0
		self.player_reward = 0
		self.child_visits = []

		self.done = False

		if number_players not in [1, 2]:
			raise Exception('Game init', 'Valid number_player-values are: 1 or 2')

	def terminal(self) -> bool:
		"""
		:return: A bool indicating whether this game has ended or not
		"""
		return self.done

	def legal_actions(self) -> List[Action]:
		"""
		:return: A list of all legal actions in this environment
		"""
		action_list = []
		for i in range(self.action_space_size):
			action_list.append(Action(i))
		return action_list

	def apply(self, action: Action):
		"""
		Applies a action on the environment and saves the action as well as the observed the next state and reward.

		:param action: The action to execute in the environment
		"""
		observation, reward, self.done, _ = self.environment.step(action.action_id)
		self.observation_history.append(observation)
		self.reward_history.append(reward if self.to_play() == Player(0) else -reward)
		self.action_history.append(action)
		self.step += 1

	def store_search_statistics(self, root: Node):
		"""
		Stores the results of the MCTS from a given root node into the history memory

		:param root: The root node to collect the policy distribution from
		"""
		self.probability_distributions.append([
			child.visit_count / root.visit_count for child in root.child_nodes
		])
		self.root_values.append(root.get_value_mean())

	def make_image(self, state_index: int) -> List:
		"""
		Get the observation of a specific step of the game

		:param state_index: The state index representing a specific time step of the game (similar to index in history buffer)
		:return: The initial observation of the specified time step
		"""
		return self.observation_history[state_index]

	def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int) -> List:
		"""
		The value target is the discounted root value of the search tree N steps into the future, plus the discounted
		sum of all rewards until then. The sum is the predicted reward from the very beginning until n-steps in the future

		:param state_index: The time step to make the target functions from
		:param num_unroll_steps: How further time steps should be included in the targets
		:param td_steps: How many time steps should be included in determining the value target
		:return: A List containing the values of the target functions from num_unroll_states subsequent states
		"""
		target_values = []
		# Add target value for each step from state_index till state_index + num_unroll_steps
		for current_index in range(state_index, state_index + num_unroll_steps + 1):
			bootstrap_index = current_index + td_steps
			if bootstrap_index < len(self.root_values):
				# First part of target value is the discounted value of current_index + td_steps (0 if game already terminated)
				# This is equivalent to the discounted sum of all rewards gained after bootstrap_index
				value = self.root_values[bootstrap_index] * self.discount ** td_steps
				# Second part of target value or is the discounted sum of rewards between current_index and bootstrap index
				for i, reward in enumerate(self.reward_history[current_index:bootstrap_index]):
					value += reward * self.discount ** i
				# Add the current target value to the list, if the game hasn't terminated yet
				if current_index < len(self.root_values):
					target_values.append(
						(value, self.reward_history[current_index], self.probability_distributions[current_index]))
			else:
				# States past the end of games are treated as absorbing states.
				target_values.append((0, 0, []))
		return target_values

	def to_play(self) -> Player:
		"""
		:return: A bool indicating whether the player which is currently on turn
		"""
		return Player(self.step % len(self.players))
