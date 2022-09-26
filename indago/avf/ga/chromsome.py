# Wrapper around EnvConfiguration
import copy
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from indago.avf.env_configuration import EnvConfiguration


class Chromosome:
    def __init__(self, env_config: EnvConfiguration):
        self.fitness = None
        self.fitness_info = None
        self.env_config = env_config
        self.length = self.env_config.get_length()

    def set_env_config(self, env_config: EnvConfiguration):
        assert env_config is not None, "Env config must have a value"
        self.env_config = copy.deepcopy(env_config)

    def compute_fitness(self, fitness_fn: Callable[["Chromosome"], Tuple[float, dict]] = None,) -> None:
        if self.fitness is not None:
            # fitness already computed
            pass
        if fitness_fn is not None:
            self.fitness, self.fitness_info = fitness_fn(self)
        else:
            raise RuntimeError("Fitness function is none")

    def mutate(self) -> Optional["Chromosome"]:
        new_env_config = self.env_config.mutate()
        if new_env_config is not None:
            return Chromosome(env_config=new_env_config)
        return None

    def mutate_hot(self, attributions: np.ndarray, mapping: Dict) -> Optional["Chromosome"]:
        new_env_config = self.env_config.mutate_hot(attributions=attributions, mapping=mapping)
        if new_env_config is not None:
            return Chromosome(env_config=new_env_config)
        return None

    def crossover(self, c: "Chromosome", pos1: int, pos2: int) -> Optional["Chromosome"]:
        new_env_config = self.env_config.crossover(other_env_config=c.env_config, pos1=pos1, pos2=pos2)
        if new_env_config is not None:
            return Chromosome(env_config=new_env_config)
        return None
