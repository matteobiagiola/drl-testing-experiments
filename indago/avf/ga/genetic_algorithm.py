import copy
import math
import random
import time
from typing import Callable, List, Optional, Tuple

import torch
from captum.attr import Saliency

from indago.avf.avf_policy import AvfPolicy
from indago.avf.config import ELITISM_PERCENTAGE
from indago.avf.dataset import Dataset
from indago.avf.ga.chromosome import Chromosome
from indago.avf.ga.crossover import single_point_fixed_crossover
from indago.avf.ga.selection import roulette_wheel_selection
from indago.avf.ga.stopping_criterion import StoppingCriterion
from indago.utils import randomness
from indago.utils.torch_utils import to_numpy
from log import Log


def fitness_sorting(chromosome: Chromosome):
    return chromosome.fitness


def keep_offspring(
    parent_1: Chromosome,
    parent_2: Chromosome,
    offspring_1: Chromosome,
    offspring_2: Chromosome,
    minimize: bool = True,
) -> bool:

    assert parent_1.fitness is not None, "First parent fitness not computed"
    assert parent_2.fitness is not None, "Second parent fitness not computed"
    assert offspring_1.fitness is not None, "First offspring fitness not computed"
    assert offspring_2.fitness is not None, "Second offspring fitness not computed"

    if minimize:
        return (
            compare_best_offspring_to_best_parent(
                parent_1=parent_1,
                parent_2=parent_2,
                offspring_1=offspring_1,
                offspring_2=offspring_2,
                minimize=minimize,
            )
            <= 0
        )
    return (
        compare_best_offspring_to_best_parent(
            parent_1=parent_1,
            parent_2=parent_2,
            offspring_1=offspring_1,
            offspring_2=offspring_2,
            minimize=minimize,
        )
        >= 0
    )


def compare_best_offspring_to_best_parent(
    parent_1: Chromosome,
    parent_2: Chromosome,
    offspring_1: Chromosome,
    offspring_2: Chromosome,
    minimize: bool = True,
) -> int:

    best_parent = get_best(c1=parent_1, c2=parent_2, minimize=minimize)
    best_offspring = get_best(c1=offspring_1, c2=offspring_2, minimize=minimize)

    if best_offspring.fitness > best_parent.fitness:
        return 1
    if best_offspring.fitness < best_parent.fitness:
        return -1
    return 0


def get_best(c1: Chromosome, c2: Chromosome, minimize: bool = True) -> Chromosome:
    if minimize:
        return c1 if c1.fitness < c2.fitness else c2
    return c1 if c1.fitness < c2.fitness else c2


class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int,
        avf_test_policy: str,
        stopping_criterion_factory: Callable[[], StoppingCriterion],
        chromosome_factory: Callable[[], Chromosome],
        preprocessed_dataset: Dataset,
        trained_avf_policy: AvfPolicy,
        avf_train_policy: str,
        regression: bool,
        crossover_rate: float,
        minimize: bool = True,
        minimize_attribution: bool = False,
    ):
        self.population_size = population_size
        self.minimize = minimize
        self.minimize_attribution = minimize_attribution
        self.population: List[Chromosome] = []
        self.fitness_values = []
        self.num_generations = 0
        self.fitness_fn = None
        self.fitness_evaluations = 0
        self.chromosome_factory = chromosome_factory
        self.stopping_criterion_factory = stopping_criterion_factory
        self.avf_test_policy = avf_test_policy

        self.preprocessed_dataset = preprocessed_dataset
        self.avf_train_policy = avf_train_policy
        self.trained_avf_policy = trained_avf_policy
        self.regression = regression

        self.crossover_rate = crossover_rate

        self.failure_env_configurations_pop = []

        self.logger = Log("GeneticAlgorithm")

    def generate_population(
        self,
        population_size: int,
        start_time: float,
        budget: int,
        constraint_fn: Callable[[Chromosome], bool] = lambda c: True,
    ) -> List[Chromosome]:
        population: List[Chromosome] = []
        while len(population) < population_size:
            chromosome = self.chromosome_factory()
            if constraint_fn(chromosome):
                population.append(chromosome)
            if budget != -1 and time.perf_counter() - start_time >= budget:
                break

        return population

    def generate(
        self,
        num_generations: int,
        only_initial_population: bool = False,
        only_best: bool = False,
        constraint_fn: Callable[[Chromosome], bool] = lambda c: True,
        fitness_fn: Callable[[Chromosome], Tuple[float, dict]] = None,
        population_size: int = None,
        budget: int = -1,
    ) -> Optional[Chromosome]:

        # save initial population of failure env configurations
        if "failure" in self.avf_test_policy:
            self.failure_env_configurations_pop = copy.deepcopy(self.population)

        self.fitness_fn = fitness_fn

        if population_size is not None:
            population_size_value = population_size
        else:
            population_size_value = self.population_size

        start_time = time.perf_counter()
        self.logger.debug("Generating population")
        if len(self.population) == 0:
            self.population = self.generate_population(
                population_size=population_size_value,
                constraint_fn=constraint_fn,
                start_time=start_time,
                budget=budget,
            )
        elif len(self.population) < population_size_value:
            self.population.extend(
                self.generate_population(
                    population_size=population_size_value - len(self.population),
                    constraint_fn=constraint_fn,
                    start_time=start_time,
                    budget=budget,
                )
            )

        assert (
            len(self.population) == population_size_value
        ), f"Initial population must be of size {population_size_value}. Found: {len(self.population)}"

        if only_initial_population:
            return None

        if budget != -1 and time.perf_counter() - start_time >= budget:
            _ = self.get_best_fitness()
            self.sort_population()

        stagnation_counter = 0
        i = 0
        condition = (
            i < num_generations
            if budget == -1
            else time.perf_counter() - start_time < budget
        )
        while condition:
            best_fitness_before_evolution = self.get_best_fitness()

            if i == 0:
                self.fitness_values.append(best_fitness_before_evolution)

            self.evolve(start_time=start_time, budget=budget)
            self.sort_population()

            best_fitness_after_evolution = self.population[0].fitness
            if len(self.fitness_values) > 0:
                self.logger.debug(
                    "Current best fitness: {:.4f}. Previous best fitness: {:.4f}".format(
                        best_fitness_after_evolution, self.fitness_values[-1]
                    )
                )
            if len(self.fitness_values) > 0 and math.isclose(
                best_fitness_after_evolution, self.fitness_values[-1], abs_tol=0.005
            ):
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            assert (
                best_fitness_after_evolution <= best_fitness_before_evolution
                if self.minimize
                else best_fitness_after_evolution >= best_fitness_before_evolution
            ), "Best fitness after evolution {} is worse than best fitness before evolution {}".format(
                best_fitness_after_evolution, best_fitness_before_evolution
            )
            self.fitness_values.append(best_fitness_after_evolution)
            self.logger.debug("Current iteration: {}".format(i))
            self.logger.debug("Stagnation counter: {}".format(stagnation_counter))
            self.logger.debug(
                "Best individual has fitness: {:.4f}".format(self.population[0].fitness)
            )
            self.logger.debug(
                "Worst individual has fitness: {:.4f}".format(
                    self.population[-1].fitness
                )
            )
            # print('Archive length: {}'.format(self.archive.length()))
            self.num_generations += 1
            if self.stopping_criterion_factory().stop(
                best_chromosome=self.get_population()[0]
            ):
                if only_best:
                    return self.get_population()[0]
                else:
                    break

            if budget == -1 and stagnation_counter > int(num_generations * 20 / 100):
                self.logger.debug("Stopping generation. Stagnation reached!")
                break

            if stagnation_counter > int(num_generations * 10 / 100):
                if "failure" in self.avf_test_policy:
                    # reseed the initial failure env configurations
                    self.logger.debug(
                        "Reseed the population with 5 random failure env configurations"
                    )
                    for _ in range(5):
                        chromosome = random.choices(
                            population=self.failure_env_configurations_pop
                        )[0]
                        if chromosome.fitness is None:
                            self.fitness_evaluations += 1

                        chromosome.compute_fitness(fitness_fn=self.fitness_fn)

                        self.population.append(chromosome)
                    self.sort_population()
                    self.logger.debug("Removing worst 5 individuals")
                    for _ in range(5):
                        chromosome_removed = self.population.pop()
                        self.logger.debug(
                            "Removing individual with fitness: {}".format(
                                chromosome_removed.fitness
                            )
                        )
                else:
                    self.logger.debug("Adding 5 new individuals")
                    current_population_length = len(self.population)
                    while len(self.population) != current_population_length + 10:
                        chromosome = self.chromosome_factory()
                        if constraint_fn(chromosome):
                            if chromosome.fitness is None:
                                self.fitness_evaluations += 1
                            chromosome.compute_fitness(fitness_fn=self.fitness_fn)
                            self.population.append(chromosome)

                        if budget != -1 and time.perf_counter() - start_time >= budget:
                            break

                    self.sort_population()
                    self.logger.debug("Removing worst 5 individuals")
                    for _ in range(5):
                        chromosome_removed = self.population.pop()
                        self.logger.debug(
                            "Removing individual with fitness: {}".format(
                                chromosome_removed.fitness
                            )
                        )

            i += 1

            condition = (
                i < num_generations
                if budget == -1
                else time.perf_counter() - start_time < budget
            )

        # return best chromosome
        if only_best:
            return None

        return self.population[0]

    def get_num_generations(self) -> float:
        return self.num_generations

    def get_fitness_values_over_time(self) -> List[float]:
        return self.fitness_values

    def get_best_fitness(self) -> float:
        # does not assume that population is sorted
        fitnesses = []
        for ce in self.population:
            if ce.fitness is None:
                self.fitness_evaluations += 1
            ce.compute_fitness(fitness_fn=self.fitness_fn)
            fitnesses.append(ce.fitness)
        return min(fitnesses) if self.minimize else max(fitnesses)

    def sort_population(self) -> None:
        self.population.sort(key=fitness_sorting)

    def get_population(self) -> List[Chromosome]:
        return self.population

    def evolve(self, start_time: float, budget: int) -> None:
        # /*
        #  * Copyright (C) 2010-2018 Gordon Fraser, Andrea Arcuri and EvoSuite
        #  * contributors
        #  *
        #  * This file is part of EvoSuite.
        #  *
        #  * EvoSuite is free software: you can redistribute it and/or modify it
        #  * under the terms of the GNU Lesser General Public License as published
        #  * by the Free Software Foundation, either version 3.0 of the License, or
        #  * (at your option) any later version.
        #  *
        #  * EvoSuite is distributed in the hope that it will be useful, but
        #  * WITHOUT ANY WARRANTY; without even the implied warranty of
        #  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
        #  * Lesser Public License for more details.
        #  *
        #  * You should have received a copy of the GNU Lesser General Public
        #  * License along with EvoSuite. If not, see <http://www.gnu.org/licenses/>.
        #  */
        new_generation: List[Chromosome] = self.elitism()
        if ELITISM_PERCENTAGE > 0:
            assert len(new_generation) > 0, (
                f"No elitism was carried out: population size too small {len(self.population)} "
                f"or elitism percentage too small {ELITISM_PERCENTAGE}"
            )

        if budget != -1 and time.perf_counter() - start_time >= budget:
            return

        while len(new_generation) < len(self.population):
            parent_1 = roulette_wheel_selection(
                population=self.population, miminize=self.minimize
            )
            parent_2 = roulette_wheel_selection(
                population=self.population, miminize=self.minimize
            )

            offspring_1 = copy.deepcopy(parent_1)
            offspring_2 = copy.deepcopy(parent_2)
            if offspring_1.fitness is None:
                self.fitness_evaluations += 1
            if offspring_2.fitness is None:
                self.fitness_evaluations += 1
            offspring_1.compute_fitness(fitness_fn=self.fitness_fn)
            offspring_2.compute_fitness(fitness_fn=self.fitness_fn)

            if randomness.get_random_float(low=0, high=1) < self.crossover_rate:

                if budget != -1 and time.perf_counter() - start_time >= budget:
                    return

                crossover = single_point_fixed_crossover(c1=offspring_1, c2=offspring_2)
                if crossover:
                    if offspring_1.fitness is None:
                        self.fitness_evaluations += 1
                    if offspring_2.fitness is None:
                        self.fitness_evaluations += 1
                    offspring_1.compute_fitness(fitness_fn=self.fitness_fn)
                    offspring_2.compute_fitness(fitness_fn=self.fitness_fn)

            # elif randomness.get_random_float(low=0, high=1) < self.mutation_rate:
            if budget != -1 and time.perf_counter() - start_time >= budget:
                return

            if "saliency" in self.avf_test_policy:
                mutated_offsprings = []
                for offspring in [offspring_1, offspring_2]:

                    env_config_transformed = (
                        self.preprocessed_dataset.transform_env_configuration(
                            env_configuration=offspring.env_config,
                            policy=self.avf_train_policy,
                        )
                    )
                    saliency = Saliency(
                        forward_func=self.trained_avf_policy.get_model().forward
                    )
                    env_config_tensor = torch.tensor(
                        env_config_transformed, dtype=torch.float32, requires_grad=True
                    )
                    env_config_tensor = env_config_tensor.view(1, -1)
                    if not self.regression:
                        attributions = saliency.attribute(
                            env_config_tensor, abs=False, target=1
                        )
                    else:
                        attributions = saliency.attribute(env_config_tensor, abs=False)
                    mapping = self.preprocessed_dataset.get_mapping_transformed(
                        env_configuration=offspring.env_config
                    )
                    attributions = to_numpy(attributions).squeeze()
                    mutated_offsprings.append(
                        offspring.mutate_hot(
                            attributions=attributions,
                            mapping=mapping,
                            minimize=self.minimize_attribution,
                        )
                    )

                mutated_offspring_1 = mutated_offsprings[0]
                mutated_offspring_2 = mutated_offsprings[1]
            else:
                mutated_offspring_1 = offspring_1.mutate()
                mutated_offspring_2 = offspring_2.mutate()

            if mutated_offspring_1 is not None:
                offspring_1.set_env_config(env_config=mutated_offspring_1.env_config)
                if offspring_1.fitness is None:
                    self.fitness_evaluations += 1
                offspring_1.compute_fitness(fitness_fn=self.fitness_fn)
            if mutated_offspring_2 is not None:
                offspring_2.set_env_config(env_config=mutated_offspring_2.env_config)
                if offspring_2.fitness is None:
                    self.fitness_evaluations += 1
                offspring_2.compute_fitness(fitness_fn=self.fitness_fn)

            # The two offsprings replace the parents if and only if one of the
            # offspring is not worse than the best parent.
            if keep_offspring(
                parent_1=parent_1,
                parent_2=parent_2,
                offspring_1=offspring_1,
                offspring_2=offspring_2,
                minimize=self.minimize,
            ):
                new_generation.append(offspring_1)
                new_generation.append(offspring_2)
            else:
                new_generation.append(parent_1)
                new_generation.append(parent_2)

        self.population = copy.deepcopy(new_generation)

    def elitism(self) -> List[Chromosome]:
        self.sort_population()
        new_population: List[Chromosome] = []
        for i in range(int(len(self.population) * ELITISM_PERCENTAGE / 100)):
            new_population.append(self.population[i])
        return new_population
