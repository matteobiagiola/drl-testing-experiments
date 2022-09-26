import copy
import json
import math
import os
import random
import time
from functools import reduce
from typing import Any, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import Saliency
from joblib import Parallel, delayed, dump, load
from PIL import Image
from sklearn.metrics import mean_absolute_error, precision_score, r2_score, recall_score, roc_auc_score, roc_curve
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from indago.avf.avf_policy import AvfPolicy
from indago.avf.config import AVF_DNN_POLICIES, AVF_TEST_POLICIES_WITH_DNN, FILTER_FAILURE_BASED_APPROACHES, NUM_GENERATIONS
from indago.avf.dataset import Data, Dataset, TorchDataset
from indago.avf.env_configuration import EnvConfiguration
from indago.avf.factories import get_avf_policy, get_chromosome_factory, get_stopping_criterion_factory
from indago.avf.ga.chromsome import Chromosome
from indago.avf.ga.genetic_algorithm import GeneticAlgorithm
from indago.avf.preprocessor import preprocess_data
from indago.avf.training_logs import TrainingLogs
from indago.config import DONKEY_ENV_NAME, ENV_NAMES, HUMANOID_ENV_NAME, PARK_ENV_NAME
from indago.envs.donkey.donkey_env_configuration import DonkeyEnvConfiguration
from indago.envs.humanoid.humanoid_env_configuration import HumanoidEnvConfiguration
from indago.envs.park.parking_env_configuration import ParkingEnvConfiguration
from indago.utils.file_utils import parse_experiment_file
from indago.utils.torch_utils import to_numpy
from log import Log


class Avf:
    def __init__(
        self,
        env_name: str,
        log_path: str,
        storing_logs: bool,
        seed: int = None,
        regression: bool = False,
        num_envs: int = 1,
        exp_file: str = None,
        exp_files: List[str] = None,
    ):
        assert num_envs == 1, "Num envs must be 1"
        self.logger = Log("Avf")
        self.regression = regression
        self.env_name = env_name
        assert self.env_name in ENV_NAMES, "Env not supported: {}".format(env_name)
        self.log_path = log_path
        self.storing_logs = storing_logs
        self.seed = seed
        # to take into account multiple environments
        self.file_num_first = 0
        self.is_training = self.storing_logs

        self.num_episodes: int = None
        self.training_progress_filter: int = None
        self.avf_train_policy: str = None
        self.avf_test_policy: str = None
        self.layers: int = 4
        self.hc_counter: int = 100
        self.oversample_minority_class_percentage: float = None
        self.dnn_sampling: str = None
        self.sampling_size: int = None
        self.failure_prob_dist: bool = None
        self.parallelize_fp_dist: bool = False
        self.num_runs_each_env_config: int = None
        self.testing_strategy_name: str = None
        self.neighborhood_size: int = None
        self.stagnation_tolerance: float = None

        # testing variables
        self.preprocessed_dataset: Dataset = None
        self.failed_data_items: List[Data] = []
        self.sorted_failed_data_items: List[Data] = []
        self.idx_data: int = 0
        self.indices_data_selected: List[int] = []
        self.weights_data: List[float] = []
        self.trained_avf_policy: AvfPolicy = None
        self.current_env_config: EnvConfiguration = None
        self.failure_probability_env_configurations: List[EnvConfiguration] = []
        self.num_trials: int = -1
        self.num_episodes: int = -1

        self.budget = -1
        self.population_size = -1
        self.crossover_rate = -1.0

        self.failure_test_env_configs = []
        self.all_configurations = []
        self.exp_file = None
        if exp_file is not None:
            self.exp_file = os.path.join(self.log_path, exp_file)
            assert os.path.exists(self.exp_file), "Exp file not found: {}".format(self.exp_file)
            self.failure_test_env_configs = parse_experiment_file(exp_file=self.exp_file, env_name=env_name)

        if exp_files is not None and len(exp_files) != 0:
            assert exp_file is None, "exp_file must be None"
            for exp_file in exp_files:
                ith_failure_test_env_configs = parse_experiment_file(exp_file=exp_file, env_name=env_name)
                self.failure_test_env_configs.extend(ith_failure_test_env_configs)
            self.logger.info("Replaying {} failures".format(len(self.failure_test_env_configs)))
            self.weights_data = np.ones(shape=len(self.failure_test_env_configs,))

        self.predictions = []

    def update_state_variables_to_enable_testing_mode(
        self,
        num_episodes: int,
        training_progress_filter: int,
        avf_train_policy: str,
        avf_test_policy: str,
        oversample_minority_class_percentage: float,
        layers: int,
        dnn_sampling: str,
        sampling_size: int,
        failure_prob_dist: bool,
        num_runs_each_env_config: int,
        testing_strategy_name: str,
        neighborhood_size: int,
        stagnation_tolerance: float,
        hc_counter: int,
        parallelize_fp_dist: bool,
        budget: int,
        population_size: int,
        crossover_rate: float,
    ) -> None:
        self.num_episodes = num_episodes
        self.training_progress_filter = training_progress_filter
        self.avf_train_policy = avf_train_policy
        self.avf_test_policy = avf_test_policy
        self.oversample_minority_class_percentage = oversample_minority_class_percentage
        self.layers = layers
        self.dnn_sampling = dnn_sampling
        self.sampling_size = sampling_size
        self.failure_prob_dist = failure_prob_dist
        self.num_runs_each_env_config = num_runs_each_env_config
        self.testing_strategy_name = testing_strategy_name
        self.neighborhood_size = neighborhood_size
        self.stagnation_tolerance = stagnation_tolerance
        self.hc_counter = hc_counter
        self.parallelize_fp_dist = parallelize_fp_dist
        self.budget = budget
        self.all_configurations = []
        self.population_size = population_size
        self.crossover_rate = crossover_rate

    def load_avf_policy(self, avf_train_policy: str, input_size: int, load_path: str, layers: int = 4,) -> AvfPolicy:

        if avf_train_policy in AVF_DNN_POLICIES:
            save_paths = None
            avf_policy = get_avf_policy(
                env_name=self.env_name,
                policy=avf_train_policy,
                input_size=input_size,
                regression=self.regression,
                layers=layers,
            )
            self.logger.info("Loading avf model: {}".format(load_path))
            avf_policy.load(filepath=load_path, save_paths=save_paths)
            self.logger.info("Model architecture: {}".format(avf_policy.get_model()))
            return avf_policy

        raise NotImplementedError("Train policy {} not supported".format(avf_train_policy))

    def sample_test_env_configuration(
        self,
        avf_train_policy: str,
        avf_test_policy: str,
        dnn_sampling_policy: str = "original",
        sampling_size: int = 1000,
        neighborhood_size: int = 50,
        hc_counter: int = 100,
        stagnation_tolerance: float = 0.005,
        index_parallel: int = -1,
        population_size: int = -1,
        crossover_rate: float = -1.0,
    ) -> Union[EnvConfiguration, Tuple[EnvConfiguration, float, float], None]:

        if avf_test_policy == "random":
            return self.generate_random_env_configuration()

        if avf_test_policy == "replay_test_failure":
            if len(self.failure_test_env_configs) == 0:
                return None
            return self.replay_test_failures()

        start_time = time.perf_counter()
        if avf_test_policy == "prioritized_replay":
            env_config, max_prediction = self.sample_test_env_configuration_prioritized()
        elif avf_test_policy == "nn":
            env_config, max_prediction = self.sample_test_env_configuration_nn(
                avf_train_policy=avf_train_policy, dnn_sampling_policy=dnn_sampling_policy, sampling_size=sampling_size,
            )
        elif "hc" in avf_test_policy:
            env_config, max_prediction = self.sample_test_env_configuration_hc(
                avf_train_policy=avf_train_policy,
                avf_test_policy=avf_test_policy,
                neighborhood_size=neighborhood_size,
                hc_counter=hc_counter,
                stagnation_tolerance=stagnation_tolerance,
                index_failure=index_parallel,
            )
        elif "ga" in avf_test_policy:
            env_config, max_prediction = self.sample_test_env_configuration_ga(
                avf_train_policy=avf_train_policy,
                avf_test_policy=avf_test_policy,
                population_size=population_size,
                crossover_rate=crossover_rate,
            )
        else:
            raise NotImplementedError("Unknown test policy: {}".format(avf_test_policy))

        return env_config, time.perf_counter() - start_time, max_prediction

    def replay_test_failures(self) -> EnvConfiguration:
        idx_data = random.choices(population=np.arange(0, len(self.failure_test_env_configs)))[0]

        if not self.is_training and len(self.indices_data_selected) == len(self.failure_test_env_configs):
            raise RuntimeError("Cannot select more failure test env configurations")

        if not self.is_training:
            while idx_data in self.indices_data_selected:
                idx_data = random.choices(population=np.arange(0, len(self.failure_test_env_configs)))[0]
        else:
            idx_data = random.choices(
                population=np.arange(0, len(self.failure_test_env_configs)), weights=self.weights_data
            )[0]
        self.indices_data_selected.append(idx_data)

        self.current_env_config = self.failure_test_env_configs[idx_data]
        self.logger.info("Test env configuration fail: {}".format(self.current_env_config.get_str()))
        return self.current_env_config

    def sample_test_env_configuration_prioritized(self) -> Tuple[EnvConfiguration, float]:
        if len(self.failed_data_items) == 0:
            self.failed_data_items = [data_item for data_item in self.preprocessed_dataset.get() if data_item.label == 1]
            self.weights_data = [data_item.training_progress for data_item in self.failed_data_items]

        idx_data = random.choices(population=np.arange(0, len(self.failed_data_items)), weights=self.weights_data)[0]

        if len(self.indices_data_selected) == len(self.failed_data_items):
            self.logger.warn("Fallback to random generation")
            self.current_env_config = self.generate_random_env_configuration()
            self.logger.info("Env configuration: {}".format(self.current_env_config.get_str()))
            return self.current_env_config, 0.0

        while idx_data in self.indices_data_selected:
            idx_data = random.choices(population=np.arange(0, len(self.failed_data_items)), weights=self.weights_data)[0]

        self.logger.debug(
            "Index data: {}. {}/{}".format(idx_data, len(self.indices_data_selected), len(self.failed_data_items))
        )
        self.indices_data_selected.append(idx_data)

        self.current_env_config = self.failed_data_items[idx_data].training_logs.get_config()
        self.logger.info(
            "Env configuration: {}; Training progress: {}".format(
                self.current_env_config.get_str(), self.failed_data_items[idx_data].training_logs.get_training_progress()
            )
        )
        return self.current_env_config, 0.0

    def _get_env_config_for_hc(
        self, avf_test_policy: str, index_failure: int = -1, reuse_previous_failure_index: bool = False,
    ) -> EnvConfiguration:

        if (
            avf_test_policy == "hc_failure"
            or avf_test_policy == "hc_saliency_failure"
            or avf_test_policy == "hc_importance_failure"
        ):
            if index_failure != -1:
                if index_failure >= len(self.failed_data_items) and not reuse_previous_failure_index:
                    self.logger.warn(
                        "Fallback to random generation index {}/{}".format(index_failure, len(self.failed_data_items))
                    )
                    env_config = self.generate_random_env_configuration()
                else:
                    if reuse_previous_failure_index:
                        self.logger.debug("Reusing configuration with index {}".format(index_failure))
                    self.logger.debug(
                        "Index data: {}. {}/{}".format(
                            index_failure, len(self.indices_data_selected), len(self.failed_data_items)
                        )
                    )
                    env_config = self.failed_data_items[index_failure].training_logs.get_config()
            else:
                if reuse_previous_failure_index and len(self.indices_data_selected) > 0:
                    self.logger.debug("Reusing configuration with index {}".format(self.indices_data_selected[-1]))
                    env_config = self.failed_data_items[self.indices_data_selected[-1]].training_logs.get_config()
                else:
                    idx_data = random.choices(population=np.arange(0, len(self.failed_data_items)), weights=self.weights_data)[
                        0
                    ]
                    if len(self.indices_data_selected) == len(self.failed_data_items):
                        self.logger.warn("Fallback to random generation")
                        env_config = self.generate_random_env_configuration()
                    else:
                        while idx_data in self.indices_data_selected:
                            idx_data = random.choices(
                                population=np.arange(0, len(self.failed_data_items)), weights=self.weights_data
                            )[0]
                        self.indices_data_selected.append(idx_data)
                        self.logger.debug(
                            "Index data selected: {}. {}/{}".format(
                                idx_data, len(self.indices_data_selected), len(self.failed_data_items)
                            )
                        )
                        env_config = self.failed_data_items[idx_data].training_logs.get_config()
        else:
            env_config = self.generate_random_env_configuration()

        return env_config

    def sample_test_env_configuration_hc(
        self,
        avf_train_policy: str,
        avf_test_policy: str,
        neighborhood_size: int = 50,
        hc_counter: int = 100,
        stagnation_tolerance: float = 0.005,
        index_failure: int = -1,
        stochastic: bool = False,
    ) -> Tuple[EnvConfiguration, float]:

        assert hc_counter > 0, "hc_counter must be > 0. Found: {}".format(hc_counter)
        assert stagnation_tolerance > 0.0, "stagnation_tolerance counter must be > 0.0. Found: {}".format(stagnation_tolerance)
        assert neighborhood_size > 0, "neighborhood_size counter must be > 0. Found: {}".format(neighborhood_size)

        if len(self.failed_data_items) == 0:
            self.failed_data_items = [data_item for data_item in self.preprocessed_dataset.get() if data_item.label == 1]
            self.weights_data = [data_item.training_progress for data_item in self.failed_data_items]

        env_config = self._get_env_config_for_hc(
            avf_test_policy=avf_test_policy, index_failure=index_failure, reuse_previous_failure_index=False
        )

        hill_climbing_restart_counter = hc_counter
        max_stagnation_counter = 20
        stagnation_counter = 0
        max_prediction = -1.0

        current_num_predictions = 0

        initial_env_config_transformed = self.preprocessed_dataset.transform_env_configuration(
            env_configuration=env_config, policy=avf_train_policy
        )
        initial_env_config_prediction = self.trained_avf_policy.get_failure_class_prediction(
            env_config_transformed=initial_env_config_transformed, dataset=self.preprocessed_dataset
        )
        self.logger.info("Initial env config: {}, prediction: {}".format(env_config.get_str(), initial_env_config_prediction))

        start_time = time.perf_counter()
        condition = hill_climbing_restart_counter > 0 if self.budget == -1 else time.perf_counter() - start_time < self.budget

        env_config_with_max_prediction = copy.deepcopy(env_config)

        while condition:
            # generate neighborhood
            neighborhood = [env_config]

            if avf_test_policy == "hc_saliency_rnd" or avf_test_policy == "hc_saliency_failure":
                env_config_transformed = self.preprocessed_dataset.transform_env_configuration(
                    env_configuration=env_config, policy=avf_train_policy,
                )
                saliency = Saliency(forward_func=self.trained_avf_policy.get_model().forward)
                env_config_tensor = torch.tensor(env_config_transformed, dtype=torch.float32, requires_grad=True)
                env_config_tensor = env_config_tensor.view(1, -1)
                if not self.regression:
                    attributions = saliency.attribute(env_config_tensor, abs=False, target=1)
                else:
                    attributions = saliency.attribute(env_config_tensor, abs=False)
                mapping = self.preprocessed_dataset.get_mapping_transformed(env_configuration=env_config)

            # max time (in seconds) given to generate the neighbourhood
            counter = 10000
            neighborhood_condition = (
                len(neighborhood) != neighborhood_size
                if self.budget == -1
                else len(neighborhood) != neighborhood_size and time.perf_counter() - start_time < self.budget
            )
            while neighborhood_condition:
                if avf_test_policy == "hc_saliency_rnd" or avf_test_policy == "hc_saliency_failure":
                    mutated_env_config = env_config.mutate_hot(attributions=to_numpy(attributions).squeeze(), mapping=mapping)
                else:
                    mutated_env_config = env_config.mutate()

                if mutated_env_config is not None:
                    neighborhood.append(mutated_env_config)

                counter -= 1
                if counter == 0:
                    self.logger.warn("Timeout reached when generating neighborhood")
                    break

                neighborhood_condition = (
                    len(neighborhood) != neighborhood_size
                    if self.budget == -1
                    else len(neighborhood) != neighborhood_size and time.perf_counter() - start_time < self.budget
                )

            if len(neighborhood) != neighborhood_size:
                self.logger.warn(
                    "Not possible to generate neighbourhood for env configuration {}".format(env_config.get_str())
                )
                if max_prediction == -1 and len(neighborhood) == 0:
                    env_config_transformed = self.preprocessed_dataset.transform_env_configuration(
                        env_configuration=env_config, policy=avf_train_policy
                    )
                    max_prediction = self.trained_avf_policy.get_failure_class_prediction(
                        env_config_transformed=env_config_transformed, dataset=self.preprocessed_dataset
                    )

            env_configs_transformed = []
            for neighbor in neighborhood:
                env_configs_transformed.append(
                    self.preprocessed_dataset.transform_env_configuration(env_configuration=neighbor, policy=avf_train_policy)
                )

            predictions = []
            for env_configuration_transformed in env_configs_transformed:
                prediction = self.trained_avf_policy.get_failure_class_prediction(
                    env_config_transformed=env_configuration_transformed, dataset=self.preprocessed_dataset
                )
                predictions.append(prediction)

                current_num_predictions += 1

            if not stochastic:
                chosen_idx = np.asarray(predictions).argmax()
            else:
                predictions = np.asarray(predictions)
                indices_greater_prediction = np.where(predictions >= max_prediction)[0]
                chosen_idx = random.choices(
                    population=indices_greater_prediction, weights=predictions[indices_greater_prediction]
                )[0]

            prediction = predictions[chosen_idx]
            if prediction < max_prediction or math.isclose(prediction, max_prediction, abs_tol=stagnation_tolerance):
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            # previous_prediction = prediction
            max_env_config = neighborhood[chosen_idx]

            if prediction > max_prediction:
                env_config = copy.deepcopy(max_env_config)
                max_prediction = max(max_prediction, prediction)
                env_config_with_max_prediction = copy.deepcopy(max_env_config)
                # print(max_prediction)

            self.logger.debug(
                "Prediction: {}, Counter: {}, Stagnation: {}".format(
                    max_prediction, hill_climbing_restart_counter, stagnation_counter
                )
            )

            if stagnation_counter == max_stagnation_counter:
                self.logger.debug("Stagnation reached!")
                if self.budget == -1:
                    break
                elif time.perf_counter() - start_time < self.budget:
                    # random restart
                    hill_climbing_restart_counter = hc_counter + 1  # because there is a -1 at the end of the loop
                    stagnation_counter = 0
                    env_config = self._get_env_config_for_hc(
                        avf_test_policy=avf_test_policy, index_failure=index_failure, reuse_previous_failure_index=True
                    )

            hill_climbing_restart_counter -= 1
            condition = (
                hill_climbing_restart_counter > 0 if self.budget == -1 else time.perf_counter() - start_time < self.budget
            )

        self.current_env_config = copy.deepcopy(env_config_with_max_prediction)
        self.logger.info("Env configuration: {}; max prediction: {}".format(self.current_env_config.get_str(), max_prediction))
        self.predictions.append(max_prediction)

        return self.current_env_config, max_prediction

    def sample_test_env_configuration_ga(
        self, avf_train_policy: str, avf_test_policy: str, population_size: int, crossover_rate: float,
    ) -> Tuple[EnvConfiguration, float]:

        assert population_size > 0, "Population size must be > 0. Found: {}".format(population_size)
        assert crossover_rate > 0.0, "Mutation rate size must be > 0.0. Found: {}".format(crossover_rate)

        if len(self.failed_data_items) == 0:
            self.failed_data_items = [data_item for data_item in self.preprocessed_dataset.get() if data_item.label == 1]
            self.weights_data = [data_item.training_progress for data_item in self.failed_data_items]

        preprocessed_dataset_closure = self.preprocessed_dataset
        trained_avf_policy_closure = self.trained_avf_policy

        def fitness_fn(c: Chromosome) -> Tuple[float, dict]:

            env_configuration_transformed = preprocessed_dataset_closure.transform_env_configuration(
                env_configuration=c.env_config, policy=avf_train_policy
            )

            prediction = trained_avf_policy_closure.get_failure_class_prediction(
                env_config_transformed=env_configuration_transformed, dataset=self.preprocessed_dataset
            )
            fitness = abs(1 - prediction)

            return fitness, {}

        stopping_criterion_factory = get_stopping_criterion_factory(target_fitness=0.0)
        chromosome_factory = get_chromosome_factory(generate_env_config_fn=self.generate_random_env_configuration)
        ga = GeneticAlgorithm(
            population_size=population_size,
            stopping_criterion_factory=stopping_criterion_factory,
            chromosome_factory=chromosome_factory,
            minimize=True,
            avf_test_policy=avf_test_policy,
            avf_train_policy=avf_train_policy,
            preprocessed_dataset=self.preprocessed_dataset,
            regression=self.regression,
            trained_avf_policy=self.trained_avf_policy,
            crossover_rate=crossover_rate,
        )

        if "failure" in avf_test_policy:
            while len(ga.population) < len(self.failed_data_items) and len(ga.population) < population_size:
                idx_data = random.choices(population=np.arange(0, len(self.failed_data_items)), weights=self.weights_data)[0]
                while idx_data in self.indices_data_selected:
                    idx_data = random.choices(population=np.arange(0, len(self.failed_data_items)), weights=self.weights_data)[
                        0
                    ]
                self.indices_data_selected.append(idx_data)
                failed_data_item = self.failed_data_items[idx_data]
                ga.population.append(Chromosome(env_config=failed_data_item.training_logs.get_config()))

        # possible to introduce the same failures in the population across different runs
        self.indices_data_selected.clear()

        chromosome = ga.generate(num_generations=NUM_GENERATIONS, only_best=True, fitness_fn=fitness_fn, budget=self.budget,)
        if chromosome is None:
            chromosome = ga.population[0]

        self.current_env_config = copy.deepcopy(chromosome.env_config)

        optimal_env_configuration_transformed = self.preprocessed_dataset.transform_env_configuration(
            env_configuration=chromosome.env_config, policy=self.avf_train_policy,
        )
        max_prediction = self.trained_avf_policy.get_failure_class_prediction(
            env_config_transformed=optimal_env_configuration_transformed, dataset=self.preprocessed_dataset
        )

        self.trained_avf_policy.reset_current_num_evaluation_predictions()

        self.logger.info("Env configuration: {}; max prediction: {}".format(self.current_env_config.get_str(), max_prediction))
        self.predictions.append(max_prediction)

        return self.current_env_config, max_prediction

    def sample_test_env_configuration_nn(
        self, avf_train_policy: str, dnn_sampling_policy: str, sampling_size: int,
    ) -> Tuple[EnvConfiguration, float]:

        if dnn_sampling_policy == "original":
            assert sampling_size > 0, "Sampling size must be > 0"

            if len(self.all_configurations) == 0:
                num_generated = 0
                predictions = []
                env_configurations = []
                start_time = time.perf_counter()
                current_num_predictions = 0
                condition = (
                    num_generated != sampling_size if self.budget == -1 else time.perf_counter() - start_time < self.budget
                )
                while condition:
                    env_configuration = self.generate_random_env_configuration()
                    env_configurations.append(env_configuration)
                    env_configuration_transformed = self.preprocessed_dataset.transform_env_configuration(
                        env_configuration=env_configuration, policy=avf_train_policy
                    )

                    prediction = self.trained_avf_policy.get_failure_class_prediction(
                        env_config_transformed=env_configuration_transformed, dataset=self.preprocessed_dataset
                    )

                    predictions.append(prediction)
                    current_num_predictions += 1

                    num_generated += 1
                    condition = (
                        num_generated != sampling_size if self.budget == -1 else time.perf_counter() - start_time < self.budget
                    )

                max_prediction_idx = np.argmax(predictions)
                max_env_config = env_configurations[max_prediction_idx]
                max_prediction = predictions[max_prediction_idx]

                self.current_env_config = max_env_config
                self.logger.info(
                    "Env configuration: {}; max prediction: {}".format(self.current_env_config.get_str(), max_prediction)
                )
                self.predictions.append(max_prediction)

                return self.current_env_config, max_prediction

            assert len(self.all_configurations) > 0, "Run out of configurations"
            max_env_config, max_prediction = self.all_configurations.pop(0)
            self.current_env_config = max_env_config
            self.logger.info(
                "Env configuration: {}; max prediction: {}".format(self.current_env_config.get_str(), max_prediction)
            )
            self.predictions.append(max_prediction)
            return self.current_env_config, max_prediction

        raise NotImplementedError("Unknown dnn sampling policy: {}".format(dnn_sampling_policy))

    def generate_random_env_configuration(self) -> EnvConfiguration:
        if self.env_name == PARK_ENV_NAME:
            env_config = ParkingEnvConfiguration().generate_configuration()
        elif self.env_name == HUMANOID_ENV_NAME:
            env_config = HumanoidEnvConfiguration().generate_configuration()
        elif self.env_name == DONKEY_ENV_NAME:
            env_config = DonkeyEnvConfiguration().generate_configuration()
        else:
            raise NotImplementedError("Environment {} not supported".format(self.env_name))

        if self.avf_test_policy == "random":
            self.current_env_config = env_config
            self.logger.info("Env configuration: {}".format(self.current_env_config.get_str()))

        return env_config

    def generate_env_configuration(self) -> EnvConfiguration:
        if self.is_training:
            if len(self.failure_test_env_configs) > 0:
                return self.replay_test_failures()
            return self.generate_random_env_configuration()
        if not self.is_training:

            if self.avf_test_policy != "random" and self.avf_test_policy != "replay_test_failure":
                if self.preprocessed_dataset is None:

                    if "failure" in self.avf_test_policy:
                        training_progress_filter = FILTER_FAILURE_BASED_APPROACHES
                        self.logger.info(
                            "Progress filter for failure based approach {} is {}".format(
                                self.avf_test_policy, training_progress_filter
                            )
                        )
                    else:
                        training_progress_filter = self.training_progress_filter

                    self.preprocessed_dataset = preprocess_data(
                        env_name=self.env_name,
                        log_path=self.log_path,
                        training_progress_filter=training_progress_filter,
                        policy=self.avf_train_policy,
                    )

                    # load data scalers computed during training of the model
                    if self.avf_test_policy in AVF_TEST_POLICIES_WITH_DNN:
                        output_scaler_path = self.get_scaler_path(
                            output=True,
                            training_progress_filter=self.training_progress_filter,
                            avf_train_policy=self.avf_train_policy,
                            oversample_minority_class_percentage=self.oversample_minority_class_percentage,
                            layers=self.layers,
                        )
                        input_scaler_path = self.get_scaler_path(
                            output=False,
                            training_progress_filter=self.training_progress_filter,
                            avf_train_policy=self.avf_train_policy,
                            oversample_minority_class_percentage=self.oversample_minority_class_percentage,
                            layers=self.layers,
                        )
                        if os.path.exists(output_scaler_path):
                            self.logger.info("Loading output scaler: {}".format(output_scaler_path))
                            self.preprocessed_dataset.output_scaler = load(output_scaler_path)
                        if os.path.exists(input_scaler_path):
                            self.logger.info("Loading input scaler: {}".format(input_scaler_path))
                            self.preprocessed_dataset.input_scaler = load(input_scaler_path)

                if self.avf_test_policy != "prioritized_replay" and self.trained_avf_policy is None:
                    save_path = self.get_avf_save_path(
                        training_progress_filter=self.training_progress_filter,
                        avf_train_policy=self.avf_train_policy,
                        oversample_minority_class_percentage=self.oversample_minority_class_percentage,
                        layers=self.layers,
                    )

                    input_size = self.preprocessed_dataset.get_num_features()

                    self.trained_avf_policy = self.load_avf_policy(
                        avf_train_policy=self.avf_train_policy, input_size=input_size, load_path=save_path, layers=self.layers,
                    )

            if not self.failure_prob_dist:
                res = self.sample_test_env_configuration(
                    avf_train_policy=self.avf_train_policy,
                    avf_test_policy=self.avf_test_policy,
                    dnn_sampling_policy=self.dnn_sampling,
                    sampling_size=self.sampling_size,
                    neighborhood_size=self.neighborhood_size,
                    hc_counter=self.hc_counter,
                    stagnation_tolerance=self.stagnation_tolerance,
                    population_size=self.population_size,
                    crossover_rate=self.crossover_rate,
                )
                if type(res) == tuple:
                    return res[0]
                return res
            if len(self.failure_probability_env_configurations) == 0:
                assert (
                    self.num_runs_each_env_config >= 1
                ), "Num runs for each env configuration must be >= 1. Found: {}".format(self.num_runs_each_env_config)
                self.logger.info(
                    "Generating {} env configurations for failure probability distribution".format(
                        self.num_runs_each_env_config
                    )
                )
                times_elapsed = []
                if (
                    self.parallelize_fp_dist
                    and self.avf_test_policy != "prioritized_replay"
                    and self.avf_test_policy != "random"
                ):
                    num_failures_dataset = sum([data.label for data in self.preprocessed_dataset.get() if data.label == 1])
                    if (
                        "failure" in self.avf_test_policy
                        and "hc" in self.avf_test_policy
                        and self.num_episodes < num_failures_dataset
                    ):
                        raise NotImplementedError(
                            "Parallel mode not supported for {} and {} episodes since they are less than the number of failures {}".format(
                                self.avf_test_policy, self.num_episodes, num_failures_dataset
                            )
                        )

                    with Parallel(n_jobs=-1, batch_size="auto", backend="loky") as parallel:
                        res = parallel(
                            (
                                delayed(self.sample_test_env_configuration)(
                                    avf_train_policy=self.avf_train_policy,
                                    avf_test_policy=self.avf_test_policy,
                                    dnn_sampling_policy=self.dnn_sampling,
                                    sampling_size=self.sampling_size,
                                    neighborhood_size=self.neighborhood_size,
                                    hc_counter=self.hc_counter,
                                    stagnation_tolerance=self.stagnation_tolerance,
                                    index_parallel=i,
                                    population_size=self.population_size,
                                    crossover_rate=self.crossover_rate,
                                )
                                for i in range(self.num_episodes)
                            ),
                        )
                    for env_config, time_elapsed, max_prediction in res:
                        self.logger.info(
                            "Env configuration: {}; max prediction: {}".format(env_config.get_str(), max_prediction)
                        )
                        self.logger.info("Times elapsed: {} s".format(round(time_elapsed, 2)))
                        times_elapsed.append(time_elapsed)
                        for _ in range(self.num_runs_each_env_config):
                            self.failure_probability_env_configurations.append(env_config)
                else:
                    for i in range(self.num_episodes):
                        start_time = time.perf_counter()
                        res = self.sample_test_env_configuration(
                            avf_train_policy=self.avf_train_policy,
                            avf_test_policy=self.avf_test_policy,
                            dnn_sampling_policy=self.dnn_sampling,
                            sampling_size=self.sampling_size,
                            neighborhood_size=self.neighborhood_size,
                            hc_counter=self.hc_counter,
                            stagnation_tolerance=self.stagnation_tolerance,
                            population_size=self.population_size,
                            crossover_rate=self.crossover_rate,
                        )

                        if type(res) == tuple:
                            env_config = res[0]
                        else:
                            env_config = res

                        self.logger.info("Times elapsed: {} s".format(round(time.perf_counter() - start_time, 2)))
                        times_elapsed.append(round(time.perf_counter() - start_time, 2))
                        for j in range(self.num_runs_each_env_config):
                            self.failure_probability_env_configurations.append(env_config)
                self.logger.info(
                    "Times elapsed (s): {}, Mean: {:.2f}, Std: {:.2f}, Min: {}, Max: {}".format(
                        times_elapsed,
                        np.mean(times_elapsed),
                        np.std(times_elapsed),
                        np.min(times_elapsed),
                        np.max(times_elapsed),
                    )
                )
                # return the first configuration; in any case it will not be used, since the reset will be called
                # immediately after and a new configuration (below) will be taken
                self.current_env_config = self.failure_probability_env_configurations[self.idx_data]
                self.idx_data += 1
                self.logger.info("{}/{}".format(self.idx_data, len(self.failure_probability_env_configurations)))
                return self.current_env_config
            self.current_env_config = self.failure_probability_env_configurations[self.idx_data]
            self.idx_data += 1
            self.logger.info("{}/{}".format(self.idx_data, len(self.failure_probability_env_configurations)))
            return self.current_env_config

        raise RuntimeError("Cannot generate env configuration")

    def store_testing_logs(self, training_logs: TrainingLogs) -> None:
        if not self.is_training:
            assert self.testing_strategy_name is not None, "Testing strategy name is not assigned"
            assert self.num_episodes != -1, "Num episodes not assigned"
            if self.num_trials != -1:
                filepath = os.path.join(self.log_path, self.testing_strategy_name, "trial-{}".format(self.num_trials))
                os.makedirs(filepath, exist_ok=True)
            else:
                filepath = os.path.join(self.log_path, self.testing_strategy_name)
                os.makedirs(filepath, exist_ok=True)

            env_image_array = training_logs.get_testing_image()
            if env_image_array is not None:
                env_image = Image.fromarray(env_image_array.astype(np.uint8))
                filepath_image = os.path.join(
                    filepath, "{}-log-image-{}.png".format(self.num_episodes, int(training_logs.get_label()))
                )
                env_image.save(fp=filepath_image)

            filepath_json = os.path.join(filepath, "{}-log-{}.json".format(self.num_episodes, int(training_logs.get_label())))
            if self.exp_file is not None:
                json_string = json.dumps(training_logs.to_dict_test(), indent=4)
            else:
                json_string = json.dumps(training_logs.to_dict(), indent=4)
            with open(filepath_json, "w+", encoding="utf-8") as f:
                f.write(json_string)

    def store_training_logs(self, training_logs: TrainingLogs) -> None:
        if self.storing_logs:
            file_num_last = 0
            log_path = os.path.join(self.log_path, "avf_log_{}_{}.json".format(self.file_num_first, file_num_last))
            while os.path.exists(log_path):
                file_num_last += 1
                log_path = os.path.join(self.log_path, "avf_log_{}_{}.json".format(self.file_num_first, file_num_last))

            json_string = json.dumps(training_logs.to_dict(), indent=4)
            with open(log_path, "w+", encoding="utf-8") as f:
                f.write(json_string)

    def clear_state(self) -> None:
        self.indices_data_selected.clear()
        self.idx_data = 0

    def get_current_env_config(self) -> EnvConfiguration:
        return self.current_env_config

    def train_dnn(
        self,
        avf_train_policy: str,
        dataset: Dataset,
        seed: int,
        n_epochs: int = 20,
        oversample_minority_class_percentage: float = 0.0,
        test_split: float = 0.2,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        patience: int = 20,
        batch_size: int = 64,
        training_progress_filter: int = 0,
        weight_loss: bool = False,
        train_dataset_: TorchDataset = None,
        validation_dataset_: TorchDataset = None,
        test_dataset_: TorchDataset = None,
        no_test_set: bool = False,
        layers: int = 4,
        save_model: bool = False,
    ) -> Tuple[float, float, float, float, float, float, float]:

        powers_of_two = [2, 4, 8, 16, 32, 64, 128]

        if train_dataset_ is not None and validation_dataset_ is not None and test_dataset_ is not None:
            train_dataset = train_dataset_
            validation_dataset = validation_dataset_
            test_dataset = test_dataset_

            self.logger.info("Train data labels proportion: {}".format(train_dataset.labels.mean()))
            self.logger.info("Validation data labels proportion: {}".format(validation_dataset.labels.mean()))
            self.logger.info("Test data labels proportion: {}".format(test_dataset.labels.mean()))

            if not self.regression:
                failures_train_dataset = reduce(lambda a, b: a + b, filter(lambda label: label == 1, train_dataset.labels))
                failures_validation_dataset = reduce(
                    lambda a, b: a + b, filter(lambda label: label == 1, validation_dataset.labels)
                )
                failures_test_dataset = reduce(lambda a, b: a + b, filter(lambda label: label == 1, test_dataset.labels))

                non_failures_train_dataset = len(train_dataset.labels) - failures_train_dataset
                non_failures_validation_dataset = len(validation_dataset.labels) - failures_validation_dataset
                non_failures_test_dataset = len(test_dataset.labels) - failures_test_dataset

                self.logger.info(
                    "Failures train dataset: {}/{} non failures".format(failures_train_dataset, non_failures_train_dataset)
                )
                self.logger.info(
                    "Failures validation dataset: {}/{} non failures".format(
                        failures_validation_dataset, non_failures_validation_dataset
                    )
                )
                self.logger.info(
                    "Failures test dataset: {}/{} non failures".format(failures_test_dataset, non_failures_test_dataset)
                )

                # otherwise impossible to compute precision and recall on the failure class
                assert failures_test_dataset > 0, "Failures in test dataset must be > 0. Found: {}".format(
                    failures_test_dataset
                )

            if len(train_dataset_.data) < batch_size:
                # find first power of two smaller than validation_set length
                batch_size = list(filter(lambda n: n < len(train_dataset_.data), powers_of_two))[-1]
                self.logger.info("New batch size for training dataset: {}".format(batch_size))

            if len(validation_dataset_.data) < batch_size:
                # find first power of two smaller than validation_set length
                batch_size = list(filter(lambda n: n < len(validation_dataset_.data), powers_of_two))[-1]
                self.logger.info("New batch size for validation dataset: {}".format(batch_size))
        else:
            train_dataset, test_validation_dataset = dataset.transform_data(
                test_split=test_split,
                oversample_minority_class_percentage=oversample_minority_class_percentage,
                regression=self.regression,
                seed=seed,
                weight_loss=weight_loss,
            )
            if not no_test_set and len(test_validation_dataset.data) > 0:
                test_data, test_labels, validation_data, validation_labels = Dataset.split_train_test(
                    test_split=0.5,
                    data=test_validation_dataset.data,
                    labels=test_validation_dataset.labels,
                    regression=self.regression,
                    oversample_minority_class_percentage=0.0,
                    seed=seed,
                )
            else:
                validation_data, validation_labels = test_validation_dataset.data, test_validation_dataset.labels

            if len(train_dataset.data) < batch_size:
                # find first power of two smaller than validation_set length
                batch_size = list(filter(lambda n: n < len(train_dataset.data), powers_of_two))[-1]
                self.logger.info("New batch size for training dataset: {}".format(batch_size))

            if 0 < len(validation_data) < batch_size:
                # find first power of two smaller than validation_set length
                batch_size = list(filter(lambda n: n < len(validation_data), powers_of_two))[-1]
                self.logger.info("New batch size for validation dataset: {}".format(batch_size))

            self.logger.info("Train data labels proportion: {}".format(train_dataset.labels.mean()))
            if len(validation_labels) > 0:
                self.logger.info("Validation data labels proportion: {}".format(validation_labels.mean()))
            if not no_test_set:
                self.logger.info("Test data labels proportion: {}".format(test_labels.mean()))

            if len(validation_data) > 0:
                validation_dataset = TorchDataset(
                    data=validation_data, labels=validation_labels, regression=self.regression, weight_loss=weight_loss,
                )
                if not no_test_set:
                    test_dataset = TorchDataset(
                        data=test_data, labels=test_labels, regression=self.regression, weight_loss=weight_loss,
                    )
                else:
                    test_dataset = TorchDataset(
                        data=np.asarray([]), labels=np.asarray([]), regression=self.regression, weight_loss=weight_loss,
                    )
            else:
                validation_dataset = TorchDataset(
                    data=np.asarray([]), labels=np.asarray([]), regression=self.regression, weight_loss=weight_loss,
                )
                test_dataset = TorchDataset(
                    data=np.asarray([]), labels=np.asarray([]), regression=self.regression, weight_loss=weight_loss,
                )

        self.logger.info(
            "Train dataset size: {}, Validation dataset size: {}, Test dataset size: {}".format(
                len(train_dataset.data), len(validation_dataset.data), len(test_dataset.data)
            )
        )

        learning_rate = learning_rate
        patience = patience
        n_val_epochs_no_improve = 0

        save_path = self.get_avf_save_path(
            training_progress_filter=training_progress_filter,
            avf_train_policy=avf_train_policy,
            oversample_minority_class_percentage=oversample_minority_class_percentage,
            layers=layers,
            seed=seed,
        )

        input_size = dataset.get_num_features()

        avf_policy = get_avf_policy(
            env_name=self.env_name,
            policy=avf_train_policy,
            input_size=input_size,
            regression=self.regression,
            layers=layers,
            learning_rate=learning_rate,
        )
        self.logger.info("Model architecture: {}".format(avf_policy.get_model()))

        optimizer = optim.Adam(params=avf_policy.parameters(), lr=learning_rate, weight_decay=weight_decay)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        best_val_loss = np.inf
        train_accuracies = []
        train_losses = []
        val_accuracies = []
        val_losses = []

        best_avf_policy = None
        best_val_precision = -1.0
        best_val_regression_score = -math.inf
        best_val_mae = math.inf
        test_loss: float = None
        test_precision: float = None
        test_recall: float = None
        auc_roc: float = None
        test_mae: float = None
        test_accuracy: float = None
        best_epochs: float = -1

        for epoch in range(n_epochs):
            if epoch > 0:  # test untrained net first
                avf_policy.train()
                train_accuracy = 0
                train_loss = 0
                train_mae = 0.0
                train_batches = 0
                with tqdm(train_dataloader, unit="batch") as train_epoch:
                    for train_data, train_target, weights in train_epoch:
                        # ===================forward=====================
                        loss, predictions = avf_policy.forward_and_loss(data=train_data, target=train_target, weights=weights)

                        if predictions is not None:
                            if self.regression:
                                train_accuracy += r2_score(y_true=to_numpy(train_target), y_pred=to_numpy(predictions))
                                train_mae += mean_absolute_error(y_true=to_numpy(train_target), y_pred=to_numpy(predictions))
                            else:
                                correct = (predictions == train_target).sum().item()
                                accuracy = correct / len(train_target)
                                train_accuracy += accuracy

                        train_loss += loss.item()
                        train_batches += 1
                        # ===================backward====================
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        train_epoch.set_postfix(epoch=epoch)

                train_loss /= train_batches
                train_accuracy /= train_batches
                if self.regression:
                    train_mae /= train_batches
                train_losses.append(train_loss)
                train_accuracies.append(train_accuracy)

                if not self.regression:
                    self.logger.info(
                        "Train loss: {:.2f}, Train accuracy: {:.2f}, Epoch: {}".format(train_loss, train_accuracy, epoch)
                    )
                else:
                    self.logger.info(
                        "Train loss: {:.2f}, Train accuracy: {:.2f}, Train mae: {:.2f} Epoch: {}".format(
                            train_loss, train_accuracy, train_mae, epoch
                        )
                    )

            if (epoch == 0 or epoch % 5 == 0) and len(validation_dataset.data) > 0:
                # calculate accuracy on validation set
                val_loss = 0
                precision = 0.0
                val_accuracy = 0
                precision_batch = 0
                recall_batch = 0
                val_mae = 0.0
                with torch.no_grad():
                    # switch model to evaluation mode
                    avf_policy.eval()
                    with tqdm(validation_dataloader, unit="batch") as validation_epoch:
                        validation_batches = 0
                        for validation_data, validation_target, weights in validation_epoch:
                            # ===================forward=====================
                            loss, predictions = avf_policy.forward_and_loss(
                                data=validation_data, target=validation_target, weights=weights
                            )

                            if predictions is not None:
                                if self.regression:
                                    val_accuracy += r2_score(y_true=to_numpy(validation_target), y_pred=to_numpy(predictions))
                                    val_mae += mean_absolute_error(
                                        y_true=to_numpy(validation_target), y_pred=to_numpy(predictions)
                                    )
                                else:
                                    correct = (predictions == validation_target).sum().item()
                                    accuracy = correct / len(validation_target)
                                    val_accuracy += accuracy

                            validation_epoch.set_postfix(epoch=epoch)
                            val_loss += loss.item()
                            validation_batches += 1
                            if not self.regression:
                                # precision and recall failure class
                                precision_batch += precision_score(
                                    y_true=to_numpy(validation_target), y_pred=to_numpy(predictions), pos_label=1
                                )
                                recall_batch += recall_score(
                                    y_true=to_numpy(validation_target), y_pred=to_numpy(predictions), pos_label=1
                                )

                val_loss /= validation_batches
                val_accuracy /= validation_batches
                if not self.regression:
                    precision_batch /= validation_batches
                    recall_batch /= validation_batches
                    precision = precision_batch
                else:
                    val_mae /= validation_batches

                if not self.regression:
                    self.logger.info(
                        "Validation loss: {:.2f}, Validation accuracy: {:.2f}, Precision failure class: {:.2f}, Recall failure class: {:.2f}, Epoch: {}".format(
                            val_loss, val_accuracy, precision_batch, recall_batch, epoch
                        )
                    )
                else:
                    self.logger.info(
                        "Validation loss: {:.2f}, Validation accuracy: {:.2f}, Validation mae: {:.2f}, Epoch: {}".format(
                            val_loss, val_accuracy, val_mae, epoch
                        )
                    )

                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)

                # if val_loss < best_val_loss:
                if not self.regression:
                    if precision > best_val_precision:
                        best_val_loss = val_loss
                        best_val_precision = precision
                        best_avf_policy = copy.deepcopy(avf_policy)
                        best_epochs = epoch
                        self.logger.info(
                            "New best validation precision: {}. Saving model to path: {}".format(best_val_precision, save_path)
                        )
                        self.logger.info("Corresponding validation loss: {}".format(best_val_loss))
                        if (train_dataset_ is None and validation_dataset_ is None and test_dataset_ is None) or save_model:
                            # no cross validation -> we can save the model
                            avf_policy.save(filepath=save_path)
                        n_val_epochs_no_improve = 0
                    else:
                        n_val_epochs_no_improve += 1
                else:
                    if val_mae < best_val_mae:
                        best_val_loss = val_loss
                        best_val_mae = val_mae
                        best_avf_policy = copy.deepcopy(avf_policy)
                        best_epochs = epoch
                        self.logger.info("New best mae: {}. Saving model to path: {}".format(best_val_mae, save_path))
                        self.logger.info("Corresponding R2 score: {}".format(val_accuracy))
                        self.logger.info("Corresponding validation loss: {}".format(best_val_loss))
                        if (train_dataset_ is None and validation_dataset_ is None and test_dataset_ is None) or save_model:
                            # no cross validation -> we can save the model
                            avf_policy.save(filepath=save_path)
                        n_val_epochs_no_improve = 0
                    else:
                        n_val_epochs_no_improve += 1
            else:
                if len(val_accuracies) > 0:
                    val_accuracies.append(val_accuracies[-1])
                    val_losses.append(val_losses[-1])
                else:
                    val_accuracies.append(0)
                    val_losses.append(train_losses[0])

            if n_val_epochs_no_improve == patience:
                self.logger.info(
                    "Early stopping! No improvement in validation loss for {} epochs".format(n_val_epochs_no_improve)
                )
                break

        if len(validation_dataset.data) == 0:
            self.logger.info("Saving model to path: {}".format(save_path))
            avf_policy.save(filepath=save_path)

        if len(test_dataset.data) > 0 or (no_test_set and len(validation_dataset.data) > 0):

            if no_test_set:
                # plot auc-roc on validation dataset
                test_dataset = validation_dataset

            # loading best model and evaluate it on test set
            with torch.no_grad():

                test_dataloader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset.data), shuffle=False)
                test_data, test_target, weights = next(iter(test_dataloader))

                test_loss, _ = avf_policy.forward_and_loss(data=test_data, target=test_target, weights=weights, training=True,)
                test_loss = test_loss.item()

                if self.regression:
                    predictions = avf_policy.forward_and_loss(data=test_data, target=test_target, training=False,)
                    np_predictions = to_numpy(predictions)
                    np_target = to_numpy(test_target)
                    # print(np_predictions[:5], np_target[:5])
                    test_mae = mean_absolute_error(y_true=np_target, y_pred=np_predictions)
                    test_accuracy = r2_score(y_true=np_target, y_pred=np_predictions)
                    self.logger.info("R2 score on test set: {:.2f}".format(test_accuracy))
                    self.logger.info("MAE score on test set: {:.2f}".format(test_mae))
                else:
                    logits, predictions = avf_policy.forward_and_loss(data=test_data, target=test_target, training=False,)
                    scores = avf_policy.compute_score(logits=logits)

                    correct = (predictions == test_target).sum().item()
                    self.logger.info("Accuracy test set: {}".format(correct / len(test_target)))

                if not self.regression:
                    auc = roc_auc_score(y_true=to_numpy(test_target), y_score=to_numpy(scores))
                    precision = precision_score(y_true=to_numpy(test_target), y_pred=to_numpy(predictions), pos_label=1)
                    recall = recall_score(y_true=to_numpy(test_target), y_pred=to_numpy(predictions), pos_label=1)
                    self.logger.info("Target: {}".format(to_numpy(test_target)))
                    self.logger.info("Predictions: {}".format(to_numpy(predictions)))
                    self.logger.info("Precision: {:.2f}".format(precision))
                    self.logger.info("Recall: {:.2f}".format(recall))
                    self.logger.info(
                        "F-measure: {:.2f}".format(
                            2 * (precision * recall) / (precision + recall) if precision + recall > 0.0 else 0.0
                        )
                    )
                    self.logger.info("AUROC: {:.2f}".format(auc))
                    test_precision = precision
                    test_recall = recall
                    auc_roc = auc
                    if (train_dataset_ is None and validation_dataset_ is None and test_dataset_ is None) or save_model:
                        # no cross-validation so plot results
                        fpr, tpr, thresholds = roc_curve(y_true=to_numpy(test_target), y_score=to_numpy(scores))
                        self.plot_roc_curve(fpr=fpr, tpr=tpr)
                        if not no_test_set:
                            plt.savefig(
                                os.path.join(
                                    self.log_path,
                                    "avf-{}-{}-{}-roc-auc.pdf".format(
                                        avf_train_policy, training_progress_filter, oversample_minority_class_percentage
                                    ),
                                ),
                                format="pdf",
                            )
                        else:
                            plt.savefig(
                                os.path.join(
                                    self.log_path,
                                    "avf-{}-{}-{}-roc-auc-validation.pdf".format(
                                        avf_train_policy, training_progress_filter, oversample_minority_class_percentage
                                    ),
                                ),
                                format="pdf",
                            )

        if (train_dataset_ is None and validation_dataset_ is None and test_dataset_ is None) or save_model:
            # no cross-validation so plot results
            if len(test_dataset.data) > 0 or no_test_set:
                plt.figure()
                plt.plot(train_accuracies, label="Train accuracy")
                plt.plot(val_accuracies, label="Validation accuracy")
                plt.xlabel("# Epochs")
                plt.ylabel("Accuracy")
                plt.legend()

                plt.figure()
                plt.plot(train_losses, label="Train loss")
                plt.plot(val_losses, label="Validation loss")
                plt.xlabel("# Epochs")
                plt.ylabel("Loss")
                plt.legend()
                plt.savefig(
                    os.path.join(
                        self.log_path,
                        "avf-{}-{}-{}-{}-{}-loss.pdf".format(
                            avf_train_policy,
                            training_progress_filter,
                            oversample_minority_class_percentage,
                            layers,
                            "cls" if not self.regression else "rgr",
                        ),
                    ),
                    format="pdf",
                )

        if (train_dataset_ is None and validation_dataset_ is None and test_dataset_ is None) or save_model:
            # no cross-validation so save input and output scalers
            if dataset.input_scaler is not None:
                input_scaler_path = self.get_scaler_path(
                    output=False,
                    training_progress_filter=training_progress_filter,
                    avf_train_policy=avf_train_policy,
                    oversample_minority_class_percentage=oversample_minority_class_percentage,
                    layers=layers,
                )
                self.logger.info("Saving input scaler in {}".format(input_scaler_path))
                dump(value=dataset.input_scaler, filename=input_scaler_path)

            if dataset.output_scaler is not None:
                output_scaler_path = self.get_scaler_path(
                    output=True,
                    training_progress_filter=training_progress_filter,
                    avf_train_policy=avf_train_policy,
                    oversample_minority_class_percentage=oversample_minority_class_percentage,
                    layers=layers,
                )
                self.logger.info("Saving output scaler in {}".format(output_scaler_path))
                dump(value=dataset.output_scaler, filename=output_scaler_path)

        if train_dataset_ is not None and validation_dataset_ is not None and test_dataset_ is not None:
            assert test_loss is not None, "Test loss cannot be None"
            assert best_epochs >= 0, "Best epochs must be >= 0: {}".format(best_epochs)
            if not self.regression:
                assert test_precision is not None, "Test precision cannot be None"
                assert test_recall is not None, "Test recall cannot be None"
            else:
                assert test_mae is not None, "Test mae cannot be None"
                assert test_accuracy is not None, "Test accuracy cannot be None"

        return test_loss, test_precision, test_recall, best_epochs, auc_roc, test_mae, test_accuracy

    @staticmethod
    def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray):
        plt.figure()
        plt.plot(fpr, tpr, color="orange", label="ROC")
        plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend()

    def get_scaler_path(
        self,
        output: bool,
        training_progress_filter: int,
        avf_train_policy: str,
        oversample_minority_class_percentage: float = 0.0,
        layers: int = 4,
    ) -> str:
        save_path = self.get_avf_save_path(
            training_progress_filter=training_progress_filter,
            avf_train_policy=avf_train_policy,
            oversample_minority_class_percentage=oversample_minority_class_percentage,
            layers=layers,
        )
        save_path_without_ext = save_path[: save_path.rindex(".")]
        if output:
            return "{}-output-scaler.joblib".format(save_path_without_ext)
        return "{}-input-scaler.joblib".format(save_path_without_ext)

    @staticmethod
    def remove_extension_from_file(filename_with_extension: str) -> str:
        return filename_with_extension[: filename_with_extension.rindex(".")]

    def get_avf_save_path(
        self,
        training_progress_filter: int,
        avf_train_policy: str,
        oversample_minority_class_percentage: float = 0.0,
        layers: int = 4,
        seed: int = None,
    ) -> str:

        if avf_train_policy in AVF_DNN_POLICIES:
            if training_progress_filter is not None:
                save_path = os.path.join(
                    self.log_path,
                    "best-avf-{}-{}-{}-{}.pkl".format(
                        avf_train_policy, training_progress_filter, oversample_minority_class_percentage, layers
                    ),
                )
            else:
                save_path = os.path.join(
                    self.log_path,
                    "best-avf-{}-{}-{}.pkl".format(avf_train_policy, oversample_minority_class_percentage, layers),
                )

            if self.regression:
                save_path = save_path[: save_path.rindex(".")] + "-rgr.pkl"

        else:
            raise NotImplementedError("Train policy: {} not supported".format(avf_train_policy))

        return save_path
