from muzero.environment.games import Game

from typing import Optional
from gym import core
import collections

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

"""
Containing all values relevant for setting up the whole structure of MuZero and the Monte Carlo Tree Search
"""


class MuZeroConfig:

    def __init__(self,
                 environment: core.Env,
                 action_space_size: int,
                 number_players: int,
                 max_moves: int,
                 discount: float,
                 dirichlet_alpha: float,
                 num_simulations: int,
                 batch_size: int,
                 td_steps: int,
                 lr_init: float,
                 lr_decay_steps: float,
                 visit_softmax_temperature_fn,
                 known_bounds: Optional[KnownBounds] = None):
        ### Self-Play
        self.environment = environment
        self.action_space_size = action_space_size
        self.number_players = number_players

        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # If we already have some information about which values occur in the
        # environment, we can use them to initialize the rescaling.
        # This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.known_bounds = known_bounds

        ### Training
        self.training_steps = int(1000e3)
        self.checkpoint_interval = int(1e3)
        self.window_size = int(1e6)
        self.batch_size = batch_size
        self.num_unroll_steps = 2
        self.td_steps = td_steps

        self.weight_decay = 1e-4
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = lr_decay_steps

        self.buffer_save_game_interval = 1
        self.buffer_save_path = 'replay_buffer/serialized_replay_buffer'

    def new_game(self):
        return Game(self.environment, self.number_players, self.discount, self.max_moves)


class MuZeroBoardConfig(MuZeroConfig):
    """
    TODO: Check the temperature function
    """
    @staticmethod
    def visit_softmax_temperature(num_moves, training_steps):
        if num_moves < 30:
            return 1.0
        else:
            return 0.0  # Play according to the max.

    def __init__(self, environment: core.Env, action_space_size: int, max_moves: int, dirichlet_alpha: float, lr_init: float):
        super().__init__(
            environment=environment,
            action_space_size=action_space_size,
            number_players=2,
            max_moves=max_moves,
            discount=1.0,
            dirichlet_alpha=dirichlet_alpha,
            num_simulations=800,
            batch_size=2048,
            td_steps=max_moves,  # Always use Monte Carlo return.
            lr_init=lr_init,
            lr_decay_steps=400e3,
            visit_softmax_temperature_fn=self.visit_softmax_temperature,
            known_bounds=KnownBounds(-1, 1),
        )


class MuZeroGoConfig(MuZeroBoardConfig):
    def __init__(self):
        super().__init__(action_space_size=362, max_moves=722, dirichlet_alpha=0.03, lr_init=0.01)


class MuZeroChessConfig(MuZeroBoardConfig):
    def __init__(self):
        super().__init__(action_space_size=4672, max_moves=512, dirichlet_alpha=0.3, lr_init=0.1)


class MuZeroShogiConfig(MuZeroBoardConfig):
    def __init__(self):
        super().__init__(action_space_size=11259, max_moves=512, dirichlet_alpha=0.15, lr_init=0.1)


class MuZeroAtariConfig(MuZeroConfig):
    """
    TODO: Ich verstehe diese definition nicht
    """
    @staticmethod
    def visit_softmax_temperature(num_moves, training_steps):
        if training_steps < 500e3:
            return 1.0
        elif training_steps < 750e3:
            return 0.5
        else:
            return 0.25

    def __init__(self, environment: core.Env):
        super().__init__(
            environment=environment,
            action_space_size=environment.action_space.n,
            number_players=1,
            max_moves=3,  # Half an hour at action repeat 4.
            discount=0.975,
            dirichlet_alpha=0.25,
            num_simulations=2,
            batch_size=1,
            td_steps=1,
            lr_init=0.1,
            lr_decay_steps=35000,
            visit_softmax_temperature_fn=self.visit_softmax_temperature
        )
