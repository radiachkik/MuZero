from threading import Lock

from muzero.network.muzero_config import MuZeroConfig
from muzero.environment.games import Game

import multiprocessing
import pickle
import random
import numpy as np


class ReplayBuffer:

    def __init__(self, config: MuZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.save_path = config.buffer_save_path
        self.buffer = []

    """
    Save game to the replay buffer
    """
    def save_game(self, game: Game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self, num_unroll_steps: int, td_steps: int):
        """
        Sample a batch of game positions

        :param num_unroll_steps:
        :param td_steps: How many time steps (rewards) should be summed in order to approximate the target value function
        :return: The target values for the positions sampled
        """
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        return [(g.make_image(step),                                                # observation on chosen step
                np.array(g.action_history[step:step + num_unroll_steps]),           # num_unroll_steps next actions
                g.make_target(step, num_unroll_steps, td_steps))                    # target values for steps in MCTS
                for (g, step) in game_pos]

    def sample_game(self) -> Game:
        """
        TODO: Add some prioritizing the the choice
        :return: return a random game from the buffer
        """
        # Sample game from buffer either uniformly or according to some priority.
        return random.choice(self.buffer)

    def sample_position(self, game: Game) -> int:
        """
        TODO: Add some prioritizing the the choice
        :param game: the sampled game to sample a step from
        :return: the sampled step of the game
        """
        return random.randint(0, len(game.observation_history) - 1)

    def save(self):
        proc = multiprocessing.Process(target=self._save)
        proc.start()

    def _save(self):
        print("saving replay buffer...")
        with open("%s.npz" % self.save_path, "wb") as f:
            pickle.dump(self.buffer, f)
        print("...done saving.")

    def load(self):
        print("Loading replay buffer (may take a while...)")
        with open("%s.npz" % self.save_path, 'rb') as f:
            self.buffer = pickle.load(f)

    def __len__(self):
        return len(self.buffer)
