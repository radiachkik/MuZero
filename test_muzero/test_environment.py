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
        import gym
        self.env = gym.make('CartPole-v0')
        self.game = Game(environment=self.env, discount=0.995, number_players=1)
        self.default_action = Action(0)

    def test_game_init_raises_exception_on_invalid_player_number(self):
        from muzero.environment.games import Game
        self.assertRaises(Exception, Game, environment=self.env, discount=0.995, number_players=0)
        self.assertRaises(Exception, Game, environment=self.env, discount=0.995, number_players=3)

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
        game_one_player = self.game
        game_two_players = Game(environment=self.env, discount=0.995, number_players=2)
        to_play_one = game_one_player.to_play()
        to_play_two = game_two_players.to_play()
        game_one_player.apply(self.default_action)
        game_two_players.apply(self.default_action)
        self.assertEqual(to_play_one, game_one_player.to_play(),
                         'The player on turn must not change in single agent domains')
        self.assertNotEqual(to_play_two, game_two_players.to_play(),
                            'The player on turn has to rotate in two agent domains')
        game_two_players.apply(self.default_action)
        self.assertEqual(to_play_two, game_two_players.to_play(),
                         'The player on turn has to rotate in two agent domains')

    def test_apply_saves_observations_to_history(self):
        state_index = 0
        while not self.game.terminal():
            state_index += 1
            self.game.apply(self.default_action)
            self.assertEqual(len(self.game.observation_history), state_index + 1,
                             'The observations returned by the environment are not saved')
            self.assertEqual(len(self.game.reward_history), state_index + 1,
                             'The rewards returned by the environment are not saved')
            self.assertEqual(len(self.game.action_history), state_index + 1,
                             'The actions returned by the environment are not saved')

    def test_terminal(self):
        from muzero.environment.games import Game
        step_count = 0
        while not self.game.terminal():
            self.game.apply(self.default_action)
            step_count += 1
        self.assertNotEqual(step_count, 0, 'Game starts with a terminal state')

    def test_make_image(self):
        state_index = 0
        while not self.game.terminal():
            # Skip initial observation
            state_index += 1
            self.game.apply(self.default_action)
            image = self.game.make_image(state_index)
            self.assertEqual(image.all(), self.game.observation_history[state_index].all(),
                             'The make image function has to return the observation with the given state index')

    def test_make_target(self):
        pass


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
