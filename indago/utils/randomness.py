import random
from typing import List

import numpy as np

from indago.envs.donkey.track_generator.unity.command import Command


def get_random_float(low: int = 0, high: int = 1) -> float:
    return low + (high - low) * np.random.rand()


def get_random_int(low: int, high: int) -> int:
    return int(np.random.randint(low=low, high=high))


def get_randint_sample(low: int, high: int, count: int) -> List[int]:
    return random.sample(population=range(low, high), k=count)


def get_random_command(excluded_commands: List[Command]) -> Command:
    return random.choices(
        population=[c for c in Command if c not in excluded_commands]
    )[0]
