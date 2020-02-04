from muzero.network.muzero_config import MuZeroConfig
from muzero.environment.games import Game

import random


class ReplayBuffer(object):

    def __init__(self, config: MuZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    """
    Save game to the replay buffer
    """
    def save_game(self, game: Game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    """
    Sample a batch of game positions
    """
    def sample_batch(self, num_unroll_steps: int, td_steps: int):
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        return [(g.make_image(step),                                                # observation on chosen step
                g.action_history[step:step + num_unroll_steps],                     # num_unroll_steps next actions
                g.make_target(step, num_unroll_steps, td_steps))                    # target values for steps in MCTS
                for (g, step) in game_pos]

    """
    Sample a random game from the memory
    """
    # TODO: Add some prioritizing the the choice
    def sample_game(self) -> Game:
        # Sample game from buffer either uniformly or according to some priority.
        return random.choice(self.buffer)

    """
    Sample a random position from the game chosen 
    """
    # TODO: Add some prioritizing the the choice
    def sample_position(self, game: Game) -> int:
        return random.randint(0, len(game.observation_history))