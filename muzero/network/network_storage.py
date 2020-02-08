from muzero.network.network import Network
from muzero.network.muzero_config import MuZeroConfig

"""
Storage for all saved versions of the network
"""


class NetworkStorage:

    def __init__(self, config: MuZeroConfig):
        self._networks = {}
        self.config = config

    def latest_network(self) -> Network:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return Network(num_action=self.config.action_space_size,
                           game_mode='Atari',
                           lr_init=self.config.lr_init,
                           lr_decay_rate=self.config.lr_decay_rate,
                           lr_decay_steps=self.config.lr_decay_steps,
                           momentum=self.config.momentum)

    def save_network(self, step: int, network: Network):
        self._networks[step] = network
