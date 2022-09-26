from typing import List

from indago.avf.ga.chromsome import Chromosome
from indago.utils import randomness


def roulette_wheel_selection(population: List[Chromosome], miminize: bool = True) -> Chromosome:
    sum_of_fitnesses = (
        sum([c.fitness for c in population]) if not miminize else sum([(1 / (c.fitness + 1)) for c in population])
    )

    if sum_of_fitnesses == 0.0:
        return population[randomness.get_random_int(low=0, high=len(population) - 1)]

    rnd = randomness.get_random_float(low=0, high=1) * sum_of_fitnesses

    for i in range(len(population)):
        fitness = population[i].fitness

        if miminize:
            fitness = 1 / (fitness + 1)

        if fitness >= rnd:
            return population[i]
        rnd = rnd - fitness

    return population[randomness.get_random_int(low=0, high=len(population))]
