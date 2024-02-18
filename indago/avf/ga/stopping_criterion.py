from abc import ABC, abstractmethod

from indago.avf.ga.chromosome import Chromosome


class StoppingCriterion(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def stop(self, best_chromosome: Chromosome) -> bool:
        pass
