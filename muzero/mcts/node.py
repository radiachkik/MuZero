from muzero.muzero.network import Network
from muzero.mcts.min_max_stats import MinMaxStats
from muzero.environment.player import Player
from muzero.environment.action import Action

from typing import Optional, List
import collections
import math

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

"""
Class recursive class implementing the structure of the Monte Carlo Search Tree with UCT.

Before initialising any Nodes, you have to set the static parameters with the 'init_tree()' function

This class only works on environments with only one none-zero reward at the end of an episode. The final result has to be exctracted out of the reward. 
A reward of 1 means a win, reward of -1 means a loss and everything else is treated as a draw
"""


class Node(object):
    network = None

    @staticmethod
    def initialize_static_parameters(network: Network):
        Node.network = network

    def __init__(self,
                 value: float,
                 reward: int,
                 policy_logits,
                 hidden_state,
                 action,
                 to_play: Player,
                 parent_node: Optional['Node'] = None):

        self.parent_node = parent_node
        self.to_play = to_play
        self.action = action
        self.value_sum = value
        self.reward = reward
        self.hidden_state = hidden_state
        self.policy_logits = policy_logits

        self.to_play = None
        self.visit_count = 1
        self.child_nodes = []

    """
    First checks that the policy and value prediction is not done twice on the same node and that it isn't done on a terminal node.
    Afterwards the prediction model predicts the policy distribution and the value that would be returned by the MCTS when this node was the root.
    Additionally it updates the node to be no leaf any more and increment the simulation count.
    At the end all unexplored child leafs are added (one for every possible action)
    """

    def expand(self, to_play: Player, legal_actions: List[Action], min_max_stats: MinMaxStats):
        if self.expanded():  # Check if the method call is valid
            raise Exception('MCTS Node', 'Cant explore the same node twice')

        for action in legal_actions:  # Only add child nodes if the maximum search depth is not reached
            value, reward, policy_logits, hidden_state = Node.network.recurrent_inference(self.hidden_state, action)
            min_max_stats.update(value)
            value = min_max_stats.normalize(value)

            # the next hidden state of this action
            leaf = Node(reward=reward,
                        value=value,
                        policy_logits=policy_logits,
                        hidden_state=hidden_state,
                        action=action,
                        to_play=to_play,
                        parent_node=self)

            self.child_nodes.append(leaf)

    """
    Returns a leaf found with the UCT algorithm or None if none was found
    """

    def select(self, to_play: Player, exploration_weight: float):
        # Add nodes resulting in a dead end to this list, in order to prevent an endless search
        forbidden_nodes = []

        # Retry to search another path if the chosen one was leading to a dead end
        for _ in range(len(self.child_nodes)):
            # Get the child with highest UCT value except of the dead ends
            best_uct_node = self.get_highest_uct_node_except(forbidden_nodes, exploration_weight)
            # If the above method return None, it means that we reached a dead end (no child nodes for current player)
            if best_uct_node is None:
                return None

            elif not best_uct_node.expanded():
                if self.to_play == to_play:
                    return best_uct_node
                else:
                    forbidden_nodes.append(best_uct_node)
                    continue
            else:
                new_best_uct_node = best_uct_node.select(to_play=to_play, exploration_weight=exploration_weight)
                if new_best_uct_node is not None:
                    return new_best_uct_node
                else:
                    forbidden_nodes.append(best_uct_node)
                    continue

        return None

    """
    Returns the child node with the highest UCT value for the next exploration (except those in the forbidden_nodes array) or None if none has been found
    """

    def get_highest_uct_node_except(self, forbidden_nodes, exploration_weight):

        maximum_uct_value = -100
        maximum_uct_node = None

        # Compares the UCT value for each child node to find the best path to choose
        for child in self.child_nodes:
            if child in forbidden_nodes:
                continue

            unknown_indicator = math.sqrt(self.visit_count) / (1 + child.visit_count)
            # Is higher the less the chosen node is explored(decreases while searching)
            exploration_summand = exploration_weight * child.action_propability * unknown_indicator
            # Is higher the more this node is expected to be beneficial
            exploitation_summand = child.get_value_mean()
            uct_value = exploitation_summand + exploration_summand

            if uct_value > maximum_uct_value:  # Updates the maximum if a new one is found
                maximum_uct_value = uct_value
                maximum_uct_node = child

        return maximum_uct_node

    """
    Update parent nodes and roll back to root position
    """

    def backup(self, to_play: Player, min_max_stats: MinMaxStats, discount: float):

        parent_node = self.parent_node
        value = self.reward + self.value_sum * discount

        while parent_node is not None:  # Only the root has None as parent node
            parent_node.visit_count += 1  # Increment their visit counts
            min_max_stats.update(parent_node.value())

            if parent_node.to_play == to_play:
                parent_node.value_sum += value  # Add the own value to all the parent nodes values
            else:
                parent_node.value_sum -= value

            value = parent_node.reward + value * discount
            parent_node = parent_node.parent_node  # Repeat until the root is reached

    """
    Returns whether this node is a leaf (not expanded)
    """

    def expanded(self):
        return len(self.child_nodes) > 0

    """
    Returns the mean value of this node
    """

    def get_value_mean(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
