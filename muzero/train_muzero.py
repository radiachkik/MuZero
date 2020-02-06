from muzero.network.muzero import MuZero, MuZeroAtariConfig

import gym

import asyncio


if __name__ == '__main__':
    environment = gym.make('Breakout-v0')
    muzero_config = MuZeroAtariConfig(environment=environment)
    muzero = MuZero(muzero_config)

    muzero.start_training()

