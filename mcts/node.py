import numpy as np
import math

"""
Class recursive class implementing the structure of the Monte Carlo Search Tree with UCT.

Before initialising any Nodes, you have to set the static parameters with the 'init_tree()' function

This class only works on environments with only one none-zero reward at the end of an episode. The final result has to be exctracted out of the reward. 
A reward of 1 means a win, reward of -1 means a loss and everything else is treated as a draw
"""
class Node(object):

    exploration_weight = 0                                                              # Exploration factor 
    num_simulations = 0                                                                 # Maximum numer of simulations (exploration) per search
    max_depth = 0                                                                       # Maximum depth to look for
    action_list = []                                                                    # A 1-Dimensional list of all possible actions
    dynamics_model = None                                                               # The model predicting the game dynamics for planning
    prediction_model = None                                                             # The model predicting the policy distribution and value for planning

    @staticmethod
    def initialize_static_parameters(self, max_depth, action_list, dynamics_model, prediction_model, exploration_weight):
        Node.max_depth = max_depth
        Node.action_list = action_list
        Node.dynamics_model = dynamics_model
        Node.prediction_model = prediction_model
        Node.exploration_weight = exploration_weight

    @staticmethod
    def reset_num_simulations(self):
        Node.num_simulations = 0

    def _init_(self, parent_node, reward, hidden_state, depth, action, action_propability):
        self.parent_node = parent_node                                                  # The parent node of the actually created
        self.action = action                                                            # The action taken to get to this node/state
        self.action_propability = action_propability                                    # The prior propability of taking this action
        self.reward = reward                                                            # The reward obtained by taking this action in the last node/state
        self.hidden_state = hidden_state                                                # The new hidden state observed by taking this action
        self.depth = depth                                                              # The depth of this node
                                        
        self.policy_distribution = None                                                 # The policy distribution predicted on the current state
        self.visit_count = 1                                                            # How often this state has been visited while the current tree search
        self.child_nodes = []                                                           # References to all child nodes/next moves
        self.value_sum = 0                                                            # The total value of this node (sum of the values of all child LEAFS)
        self.terminal = False                                                           # Wether this node is terminal (the game ends with this step)

        if reward != 0:                                                                 # Every none-zero reward means that the match has terminated
            self.terminal = True                              
            if reward == 1:                                                             # A reward of 1 means that the transition resulted in a theoretical win
                self.value_sum = 1
            elif reward == -1:                                                          # A reward of -1 means that the transition resulted in a theoretical lose
                self.value_sum = -1
            else:                                                                       # Other none-zero reward means that the game ended in a theoretical draw
                self.value_sum = 0

    """
    First checks that the policy and value prediction is not done twice on the same node and that it isnt done on a terminal node.
    Afterwards the prediction model predicts the policy distribution and the value that would be returned by the MCTS when this node was the root.
    Additionaly it updates the node to be no leaf any more and increment the simulation count.
    At the end all unexplored child leafs are added (one for every possible action)
    """
    def explore(self):
        if self.is_leaf() == False:                                                                              # Check if the method call is valid
            raise Exception('MCTS Node', 'Cant explore the same node twice')
        elif self.terminal == True:
            raise Exception('MCTS Node', 'Cant predict the policy distribution of a terminal state')
        
        self.policy_distribution, self.value_sum = Node.prediction_model.predict(self.hidden_state)  # Predict the value and the policy distribution 

        Node.num_simulations += 1                          

        if self.policy_distribution.shape != self.action_list.shape:
            raise Exception('MCTS Node', 'Shape of policy distribution doesnt match the action list')

        for action in range(len(self.action_list)):                                                         # Only add child nodes if the maximum search depth is not reached
            self.add_child_node(
                action=self.action_list[action], 
                action_propability=self.policy_distribution[action]
                )


    """
    Add a child leaf for a chosen action below the current node 
    """
    def add_child_leaf(self, action, action_propability):
        next_reward, next_hidden_state = Node.dynamics_model.predict(self.hidden_state, action)    # Make the dynamics model predict the immediate reward and 
                                                                                                        # the next hidden state of this action
        leaf = Node( reward=next_reward,                                                           # Create and append the child leaf
                            hidden_state=next_hidden_state, 
                            depth=self.depth + 1, 
                            action=action, 
                            action_propability=action_propability, 
                            parent_node=self)

        self.child_nodes.append(leaf)

    """
    Returns a leaf found with the UCT algorithm or None if none was found
    """
    def select():
        if self.terminal == True:                                                           # Check that select is not called on a terminal node
            raise Exception('MCTS Node', 'Cant call select on a terminal node')

        best_uct_node = None
        forbidden_nodes = []

        for action in Node.action_list:                                                     # Retry to search another path if the chosen one was leading to a dead end
            best_uct_node = self.get_highest_uct_node_except(forbidden_nodes)               # Get the child with highest UCT value except of the dead ends

            if best_uct_node == None:                                                       # If the above method return None, it means that we either reached the
                return None                                                                 # maximum depth, all childs are terminal or are ending in a dead end
            
            if best_uct_node.is_leaf() == True:                                               # We found an unexplored leaf so the search is over 
                return maximum_uct_node                                                     # and the node has to be recursivly passed towards the root

            else:                                                                           # We found the best node to continue the search
                new_best_uct_node = best_uct_node.select()
                if new_best_uct_node != None:                                               # If the select method returns a node, it has to be passed towards the root 
                    return new_best_uct_node                                                # (the only situation where a node is returned is when a leaf has been found)

                else:                                                                       # Two reasons for None: all childs of the maximum_utc_node 
                    forbidden_nodes.append(best_uct_node)                                   # are terminal or it reached the maximum depth 
                    continue

        return None

    """
    Returns the child node with the highest UCT value for the next exploration (except those in the forbidden_nodes array) or None if none has been found
    """
    def get_highest_uct_node_except(self, forbidden_nodes):

        maximum_uct_value = -100
        maximum_uct_node = None

        if self.depth < Node.max_depth:

            for child in self.child_nodes:                                                                    # Compare all child nodes for the best path to choose
                if child.terminal == True or child in forbiden_nodes:                                          # Only compare none-terminal states
                    continue

                unknown_indicator = math.sqrt(self.visit_count) / (1 + child.visit_count)                     
                exploration_summand = Node.exploration_weight * child.action_propability * unknown_indicator  # Is higher the less the chosen node is explored(decreases while searching)
                exploitation_summand = child.get_value_mean()                                                       # Is higher the more this node is expected to be beneficial 
                uct_value = exploitation_summand + exploration_summand                                        # Calculates the UCT value for each child node

                if uct_value > maximum_uct_value:                                                             # Updates the maximum if a new one is found
                    maximum_uct_value = uct_value
                    maximum_uct_node = child
        
        return maximum_uct_node

    """
    Update parent nodes and roll back to root position
    """
    def backup():

        parent_node = self.parent_node

        while parent_node != None:                                                                              # Only the root has None as parent node
            parent_node.visit_count += 1                                                                        # Increment their visit counts
            parent_node.value_sum += self.value_sum                                                             # Add the own value to all the parent nodes values
            parent_node = parent_node.parent_node                                                               # Repeat untill the root is reached

    """
    Returns wether this node is a leaf (not expl)
    """
    def is_leaf(self):
        return len(self.child_nodes) > 0

    """
    Returns the mean value of this node
    """
    def get_value_mean(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count



