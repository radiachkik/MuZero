import unittest
from unittest.mock import MagicMock


class TestAction(unittest.TestCase):

    def test_hash(self):
        from muzero.environment.action import Action
        import random
        for _ in range(100):
            action_id = random.randint(0, 10000)
            action = Action(action_id)
            self.assertEqual(action.__hash__(), action_id,
                             'The hash of an action should be equal to the id')

    def test_eq(self):
        from muzero.environment.action import Action
        import random
        action_list = [Action(action_id) for action_id in range(100)]
        for _ in range(100):
            action_id_one = random.randint(0, 99)
            action_id_two = random.randint(0, 99)
            if action_id_one != action_id_two:
                self.assertNotEqual(action_list[action_id_one], action_list[action_id_two],
                                    "Two actions with different ids must not be equal in comparison")
            else:
                self.assertEqual(action_list[action_id_one], action_list[action_id_two],
                                 "Two actions with the same ids have to be equal in comparison")


class TestGames(unittest.TestCase):
    def setUp(self):
        from muzero.environment.games import Game
        from muzero.environment.action import Action
        from muzero.environment.player import Player
        from muzero.mcts.node import Node
        import gym
        self.env = gym.make('CartPole-v0')
        self.game = Game(environment=self.env, discount=0.995, number_players=1, max_moves=50)
        self.default_action = Action(0)
        self.default_player = Player(0)
        self.default_root_node = Node(value=1,
                                      action=self.default_action,
                                      hidden_state=0,
                                      policy_logits=[0],
                                      to_play=self.default_player,
                                      reward=0)
        # Add two child nodes for both possible action
        leaf_one = Node(value=1,
                        action=self.default_action,
                        hidden_state=0,
                        policy_logits=[0],
                        to_play=self.default_player,
                        reward=0)
        leaf_two = Node(value=1,
                        action=self.default_action,
                        hidden_state=0,
                        policy_logits=[0],
                        to_play=self.default_player,
                        reward=0)
        leaf_one.visit_count += 1
        leaf_two.visit_count += 1
        self.default_root_node.child_nodes.append(leaf_one)
        self.default_root_node.child_nodes.append(leaf_two)
        self.default_root_node.visit_count += 3

    def test_game_init_raises_exception_on_invalid_player_number(self):
        from muzero.environment.games import Game
        self.assertRaises(Exception, Game, environment=self.env, discount=0.995, number_players=0, max_moves=50)
        Game(environment=self.env, discount=0.995, number_players=1, max_moves=50)
        Game(environment=self.env, discount=0.995, number_players=2, max_moves=50)
        self.assertRaises(Exception, Game, environment=self.env, discount=0.995, number_players=3, max_moves=50)

    # Check that the list of valid actions contains Action objects and no duplicates
    def test_legal_actions(self):
        from muzero.environment.action import Action
        legal_actions = self.game.legal_actions()
        self.assertEqual(len(legal_actions), self.env.action_space.n,
                         'The number of legal actions should match the size of the action space of the gym environment')
        action_history = {}
        for action in legal_actions:
            self.assertTrue(isinstance(action, Action),
                            'Every action returned in the lega_actions list has to be an instance of the Action class')
            self.assertFalse(action in action_history,
                             'The must not be a duplicates in the list of legal actions')
            action_history[action] = 0

    def test_apply_changes_player_on_turn(self):
        from muzero.environment.games import Game
        from muzero.environment.player import Player
        game_one_player = self.game
        game_two_players = Game(environment=self.env, discount=0.995, number_players=2, max_moves=50)
        to_play_one = game_one_player.to_play()
        to_play_two = game_two_players.to_play()
        game_one_player.apply(self.default_action, self.default_player)
        game_two_players.apply(self.default_action, self.default_player)
        self.assertEqual(to_play_one, game_one_player.to_play(),
                         'The player on turn must not change in single agent domains')
        self.assertNotEqual(to_play_two, game_two_players.to_play(),
                            'The player on turn has to rotate in two agent domains')
        game_two_players.apply(self.default_action, Player(1))
        self.assertEqual(to_play_two, game_two_players.to_play(),
                         'The player on turn has to rotate in two agent domains')

    def test_apply_saves_observations_to_history(self):
        state_index = 0
        while not self.game.terminal():
            state_index += 1
            self.game.apply(self.default_action, self.default_player)
            self.assertEqual(len(self.game.observation_history), state_index + 1,
                             'The observations returned by the environment are not saved')
            self.assertEqual(len(self.game.reward_history), state_index + 1,
                             'The rewards returned by the environment are not saved')
            self.assertEqual(len(self.game.action_history), state_index + 1,
                             'The actions returned by the environment are not saved')

    def test_terminal(self):
        step_count = 0
        while not self.game.terminal():
            self.game.apply(self.default_action, self.default_player)
            step_count += 1
        self.assertNotEqual(step_count, 0, 'Game starts with a terminal state')

    def test_make_image(self):
        state_index = 0
        while not self.game.terminal():
            # Skip initial observation
            state_index += 1
            self.game.apply(self.default_action, self.default_player)
            image = self.game.make_image(state_index)
            self.assertEqual(image.all(), self.game.observation_history[state_index].all(),
                             'The make image function has to return the observation with the given state index')

    def test_store_search_statistics(self):
        import tensorflow as tf
        inserted_mean_value = 7
        self.default_root_node.get_value_mean = MagicMock(return_value=inserted_mean_value)
        self.assertEqual(len(self.game.root_values), 0,
                         'The game should start with an empty value history')
        self.assertEqual(len(self.game.probability_distributions), 0,
                         'The game should start with an empty probability distribution history')
        self.game.store_search_statistics(self.default_root_node)
        self.assertEqual(len(self.game.root_values), 1,
                         'The first added root value is missing (root value for initial observation)')
        self.assertTrue(tf.math.equal(self.game.root_values[0], tf.convert_to_tensor(inserted_mean_value, dtype=float)),
                        'The saved root value is not the one returned by the root node')
        self.assertEqual(len(self.game.probability_distributions), 1,
                         'The first added policy distribution is missing (policy distribution for initial observation)')

    def test_make_target(self):
        """
        TODO: Check whether the value returned by the function is correctly calculated
        """
        import tensorflow as tf
        while not self.game.terminal():
            self.game.store_search_statistics(self.default_root_node)
            self.game.apply(self.default_action, self.default_player)
        state_index = 0
        num_steps = len(self.game.observation_history) - 1
        td_steps = 2
        targets = self.game.make_target(state_index=state_index, num_unroll_steps=num_steps, td_steps=td_steps)
        self.assertEqual(len(targets), num_steps + 1,
                         'The returned list should have a target for the initial step + one for every step to unroll')
        self.assertEqual(len(targets[state_index]), 3,
                         'Each target should contain three values: value, reward and probability distribution')

        for target_index in range(len(targets)):
            value, reward, probability_distribution = targets[target_index]
            if state_index + target_index + td_steps < len(self.game.root_values):
                self.assertEqual(reward, self.game.reward_history[target_index],
                                 'The reward returned must have the correct position in the reward history')
                self.assertNotEqual(probability_distribution, [[0.0 for _ in range(self.game.action_space_size)]],
                                    'All "legal" targets must have a non empty probability distribution')
                self.assertNotEqual(value, 0,
                                    "The value (discounted sum of future rewards) can't be 0 before the game terminated")
            else:
                self.assertTrue(tf.math.equal(reward, tf.convert_to_tensor(0, dtype=float)),
                                'All target values of time steps where the bootstrapped value cannot be calculated '
                                'because the game ended before the reward needed is observed, should be zero')
                # Have to iterate the tensors because we cannot compare two eager tensors directly
                for (probability_value, probability_test) in zip(probability_distribution, tf.convert_to_tensor([0.0 for _ in range(self.game.action_space_size)])):
                    self.assertTrue(tf.math.equal(probability_value, probability_test),
                                    'All target values of time steps where the bootstrapped value cannot be calculated '
                                    'because the game ended before the reward needed is observed, should be zero')
                self.assertTrue(tf.math.equal(value, tf.convert_to_tensor(0, dtype=float)),
                                 'All target values of time steps where the bootstrapped value cannot be calculated '
                                 'because the game ended before the reward needed is observed, should be zero')


class TestPlayer(unittest.TestCase):
    def test_eq(self):
        from muzero.environment.player import Player
        import random
        player_list = [Player(player_id) for player_id in range(50)]
        for _ in range(50):
            player_id_one = random.randint(0, 49)
            player_id_two = random.randint(0, 49)
            if player_id_one != player_id_two:
                self.assertNotEqual(player_list[player_id_one], player_list[player_id_two],
                                    'Two players with different ids must not be equal in comparison')
            else:
                self.assertEqual(player_list[player_id_one], player_list[player_id_two],
                                 'Two players with the same ids have to be equal in comparison')


if __name__ == '__main__':
    unittest.main()
