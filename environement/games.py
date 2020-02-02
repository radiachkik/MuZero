from gym import core
from typing import List

from action import Action
from node import Node
from tree import Tree

node = Node()


"""
A single episode of interaction with the environment.
"""
class Game(object):
	def __init__(self, environment: core.Env, mcts:Tree, action_space_size: int, discount: float):
		self.action_space_size = action_space_size                                              # The size of the action space (~ the number of possible actions)
		self.discount = discount                                                                # ??

		self.action_history = []                                                                # History of all actions taken in this episode (1. action -> 1. reward)
		self.reward_history = []                                                                # History of all rewards gained in this episode 
                                                                                                
		self.propability_distributions = []                                                     # History of all propability distributions returned by the MCTS 
		self.root_values = []                                                                   # History of all root values in this episode returned by the MCTS

		self.environment = environment                                                          # The Environemnt to interact with
		self.environment.reset()                                                                # Reset the environment before executing any actions on it
		self.observation = None																	# The current observation returned by the environment
		self.done = False																		# Wether this game has ended or not

	"""
	Return wether this game has ended or not
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
	Applies a action on the environemt and observes the next state and reward. 
	Addtionaly the gained reward and the chosen action are stored in the history memory.
	"""
	def apply(self, action: Action):
		self.observation, reward, done, _ = self.environment.step(action)
		self.rewards.append(reward)
		self.history.append(action)

	"""
	Stores the results of the MCTS from a given root node into the history memory
	"""
	def store_search_statistics(self, root: Node):
		propability_distribution = []
		for child in root.child_nodes:
			propability_distribution.append(child.visit_count / root.visit_count)
		self.propability_distributions.append(propability_distributions)
		self.root_values.append(root.value_mean())

	# TODO: Muss erstellt werden (es sollen alle bisherigen)
	def make_image(self, state_index: int):
		# Game specific feature planes.
		return []

	# TODO: Muss angepasst werden
	"""
	The value target is the discounted root value of the search tree N steps
	into the future, plus the discounted sum of all rewards until then. 
	The sum is the predicted reward from the very beginning untill n-steps in the future 
	"""
	def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int, to_play: Player):
		targets = []
		for current_index in range(state_index, state_index + num_unroll_steps + 1):
			bootstrap_index = current_index + td_steps
			if bootstrap_index < len(self.root_values):
				value = self.root_values[bootstrap_index] * self.discount**td_steps
			else:
				value = 0
			for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
				value += reward * self.discount**i  # pytype: disable=unsupported-operands

			if current_index < len(self.root_values):
				targets.append((value, self.rewards[current_index],
				self.child_visits[current_index]))
			else:
				# States past the end of games are treated as absorbing states.
				targets.append((0, 0, []))
		return targets

	# TODO: Muss angepasst werden
	def to_play(self) -> Player:
		return Player()

	# TODO: Muss angepasst werden 
	def action_history(self) -> ActionHistory:
		return ActionHistory(self.history, self.action_space_size)