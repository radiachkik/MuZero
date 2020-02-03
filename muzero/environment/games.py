from gym import core
from typing import List

from muzero.environment.action import Action
from muzero.environment.player import Player
from muzero.mcts.node import Node

"""
A single episode of interaction with the environment.
"""


class Game(object):

	def __init__(self, environment: core.Env, action_space_size: int, number_players: int, discount: float):
		self.environment = environment
		self.action_space_size = action_space_size
		self.players = [Player(i) for i in range(number_players)]
		self.step = 0
		self.discount = discount

		self.action_history = []
		self.reward_history = []
		self.observation_history = []
		self.opponent_reward = 0
		self.player_reward = 0

		self.probability_distributions = []
		self.root_values = []
		self.child_visits = []

		self.environment = environment
		self.observation_history.append(self.environment.reset())
		self.done = False

	"""
	Return whether this game has ended or not
	"""

	def terminal(self) -> bool:
		return self.done

	"""
	Returns all legal actions in this game
	"""

	def legal_actions(self) -> List[Action]:
		action_list = []
		for i in range(self.action_space_size):
			action_list.append(Action(i))
		return action_list

	""" 
	Applies a action on the environment and observes the next state and reward. 
	Additionally the gained reward and the chosen action are stored in the history memory.
	"""

	def apply(self, action: Action):
		# Only one player
		if len(self.players) == 1:
			observation, reward, self.done, _ = self.environment.step(action)
			self.observation_history.append(observation)
			self.reward_history.append(reward)
			self.action_history.append(action)
		# Multiple players
		elif len(self.players) == 2:
			# Player
			if self.to_play() == self.players[0]:
				observation, self.player_reward, self.done, _ = self.environment.step(action)
				self.action_history.append(action)
			# Opponent
			else:
				observation, reward, self.done, _ = self.environment.step(action)
				self.observation_history.append(observation)
				self.reward_history.append(self.player_reward - reward)
		else:
			raise Exception('MuZero Games', 'Only games with two players are implemented')
		self.step += 1

	"""
	Stores the results of the MCTS from a given root node into the history memory
	"""

	def store_search_statistics(self, root: Node):
		self.probability_distributions.append([
			child.visit_count / root.visit_count for child in root.child_nodes
		])
		self.root_values.append(root.get_value_mean())

	"""
	Get the observation of a specific step of the game 
	"""

	def make_image(self, state_index: int):
		return self.observation_history[state_index]

	"""
	The value target is the discounted root value of the search tree N steps
	into the future, plus the discounted sum of all rewards until then. 
	The sum is the predicted reward from the very beginning untill n-steps in the future 
	"""

	def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int):
		target_values = []
		# Add target value for each step from state_index till state_index + num_unroll_steps
		for current_index in range(state_index, state_index + num_unroll_steps + 1):
			bootstrap_index = current_index + td_steps
			# First part of target value is the discounted value of current_index + td_steps (0 if game already terminated)
			# This is equivalent to the discounted sum of all rewards gained after bootstrap_index
			if bootstrap_index < len(self.root_values):
				value = self.root_values[bootstrap_index] * self.discount ** td_steps
			else:
				value = 0
			# Second part of target value or is the discounted sum of rewards between current_index and bootstrap index
			for i, reward in enumerate(self.reward_history[current_index:bootstrap_index]):
				value += reward * self.discount ** i
			# Add the current target value to the list, if the game hasn't terminated yet
			if current_index < len(self.root_values):
				target_values.append((value, self.reward_history[current_index], self.probability_distributions[current_index]))
			# States past the end of games are treated as absorbing states.
			else:
				target_values.append((0, 0, []))
		return target_values

	"""
	Returns the player which is currently on turn
	"""

	def to_play(self) -> Player:
		return Player(self.step % len(self.players))
