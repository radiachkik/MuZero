import numpy as np
from node import Node

class Tree(object):
    """
    All models need to have the predict function, which takes in the input (state, hidden state, action,...) 
    and returns the predicted value (next hidden state, reward)
    """
    def _init_(self, action_list, dynamics_model, prediction_model, max_depth = 5, max_sims=50, exploration_weight=1.0):
        self.max_sims = max_sims
        self.tree = None
        self.history_trees = []
        self.num_actions = len(action_list)

        Node.initialize_static_parameters( max_depth=max_depth, 
                                                action_list=action_list, 
                                                dynamics_model = dynamics_model, 
                                                prediction_model=prediction_model, 
                                                exploration_weight = exploration_weight
                                               )

    """
    Reset and initialize the tree
    """
    def reset(self, hidden_state):
        Node.reset_num_simulations()
        self.history_trees = []
        self.tree = Node(parent_node=None, 
                              reward=0, 
                              hidden_state=np.array(self.hidden_state), 
                              depth=0, 
                              action=None, 
                              action_propability=0
                            )
    
    """
    Start a Monte Carlo tree search based on the given observation and return the recommended action.
        * 1.st: roll out the search tree for the maximum possible and allowed number of simulations 
        * 2.nd: Return the most promising action
            # On training mode, the returned action is sampled based on the policy distribution
            # On evaluation mode, the returned action is chosen deterministically by taken the action with the highest visit count
    """
    def get_action(self, evaluation=False):
        self.rollout()

        

    """
    Rollout the search tree with the following schema (consists of a repeating 3-step procedure)
        # 1. If there are still nodes to explore, SELECT the leaf that seems to be the best to explore (Best is defined by UCT (Upper Confidence bounds applied to Trees))
        # 2. EXPLORE the leaf. That means adding all actions as child nodes (leafs) and calculating the value and policy distribution
        # 3. Update (BACKUP) all the visit counts and values on the way to the explored node accordingly (see more in the MCTS_Node class)
    """
    def rollout(self):
        while Node.num_simulations < self.max_sims:
            leaf = self.tree.select()
            # End rollout if no valid leaf can be found
            if leaf == None:
                break
            # Because we ensured that select will only return leaf nodes suitable for the exploration, we don't have to check anything
            leaf.explore()
            # Update all the nodes above accordingly
            leaf.backup()  

    """
    Returns the action with the highest visit count
    """
    def get_action_with_highest_value(self):
        # Get the action with the highest visit count for evaluation
        maximum_visit_count = 0
        best_action = None

        for possible_action in self.tree.child_nodes:
            if possible_action.visit_count > maximum_visit_count:
                maximum_visit_count = possible_action.visit_count
                best_action = possible_action.action
        
        return np.array(best_action)

    """
    Returns the propability distribution resulting through the MCTS (stochastic / exploration)
    """
    def get_propability_distribution(self): 
        # Add the current tree to the history (for the backpropagation when the predicted timesteps have occured)
        self.history_trees.append(self.tree) 
        policy_distribution = []

        for possible_action in self.tree.child_nodes:
            action_propability = possible_action.visit_count / self.tree.visit_count
            policy_distribution.append(action_propability)   

        return policy_distribution                                       






    
