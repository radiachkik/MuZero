from src.mcts.node import Node
from src.muzero.network import Network
from src.muzero.muzero_config import MuZeroConfig
from src.environement.action import Action
from src.environement.player import Player
from src.mcts.min_max_stats import MinMaxStats

import numpy as np
from typing import List


class Tree(object):

    def __init__(self, action_list, config: MuZeroConfig, network: Network, player_list: List[Player], discount):
        self.network = network
        self.max_sims = config.num_simulations
        self.root = None
        self.history_trees = []
        self.legal_actions = action_list
        self.num_players = len(player_list)
        self.exploration_weight = config.root_exploration_fraction
        self.min_max_stats = MinMaxStats(config.known_bounds)
        self.discount = discount

        Node.initialize_static_parameters(network=network)

    """
    Reset and initialize the tree
    """

    def reset(self, value: float, reward: int, policy_logits, hidden_state):
        self.history_trees = []
        self.root = Node(reward=reward,
                         value=value,
                         policy_logits=policy_logits,
                         hidden_state=hidden_state,
                         action=None,
                         to_play=Player(0))

    """
    Start a Monte Carlo tree search based on the given observation and return the recommended action.
        * 1.st: roll out the search tree for the maximum possible and allowed number of simulations 
        * 2.nd: Return the most promising action
            # On training mode, the returned action is sampled based on the policy distribution
            # On evaluation mode, the returned action is chosen deterministically (highest visit count)
    """

    def get_action(self, evaluation=False):
        self.rollout()
        if evaluation:
            return self.get_action_with_highest_visit_count()
        else:
            probability_distribution = self.get_probability_distribution()
            return np.random.choice(a=len(probability_distribution), size=1, p=probability_distribution)

    """
    Rollout the search tree with the following schema (consists of a repeating 3-step procedure)
        # 1. If there are still nodes to explore, SELECT the leaf that seems to be the best to explore (UTC)
        # 2. EXPLORE the leaf. Adding all actions as child nodes (leafs) and calculate the value and policy distribution
        # 3. Update (BACKUP) all the visit counts and values on the way to the explored node accordingly 
    """

    def rollout(self):
        player_counter = 0
        for simulation in range(self.max_sims):
            to_play = Player(player_counter % self.num_players)
            player_counter += 1
            next_to_play = Player(player_counter % self.num_players)
            leaf = self.root.select(to_play=to_play, exploration_weight=self.exploration_weight)
            # End rollout if no valid leaf can be found
            if leaf is None:
                break
            # Explore node (predict value and policy distribution and add child nodes as unexplored leafs)
            leaf.expand(to_play=next_to_play, legal_actions=self.legal_actions, min_max_stats=self.min_max_stats)
            # Update all the nodes above accordingly
            leaf.backup(to_play=to_play, min_max_stats=self.min_max_stats, discount=self.discount)

    """
    Returns the action with the highest visit count
    """

    def get_action_with_highest_visit_count(self) -> Action:
        # Get the action with the highest visit count for evaluation
        maximum_visit_count = 0
        best_action = None

        for possible_action in self.root.child_nodes:
            if possible_action.visit_count > maximum_visit_count:
                maximum_visit_count = possible_action.visit_count
                best_action = possible_action.action

        return best_action

    """
    Returns the probability distribution resulting through the MCTS (stochastic / exploration)
    """

    def get_probability_distribution(self):
        policy_distribution = []

        for possible_action in self.root.child_nodes:
            action_probability = possible_action.visit_count / self.root.visit_count
            policy_distribution.append(action_probability)

        return policy_distribution
