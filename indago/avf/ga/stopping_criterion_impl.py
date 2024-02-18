from indago.avf.ga.chromosome import Chromosome
from indago.avf.ga.stopping_criterion import StoppingCriterion


class StoppingCriterionImpl(StoppingCriterion):
    def __init__(self, target_fitness: float):
        super(StoppingCriterionImpl, self).__init__()
        self.target_fitness = target_fitness

    def stop(self, best_chromosome: Chromosome) -> bool:
        assert (
            best_chromosome.fitness is not None
        ), "Fitness of best chromosome not computed"
        return best_chromosome.fitness <= self.target_fitness
