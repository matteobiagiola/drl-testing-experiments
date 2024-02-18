import time
from typing import Tuple, List

import gym
import numpy as np

from indago.avf.config import AVF_DNN_POLICIES, FILTER_FAILURE_BASED_APPROACHES
from indago.avf.env_configuration import EnvConfiguration
from indago.config import DONKEY_ENV_NAME
from indago.env_wrapper import EnvWrapper
from indago.utils.dummy_c_vec_env import DummyCVecEnv
from indago.utils.policy_utils import instantiate_trained_policy
from log import Log


class ExpConfigurator:
    def __init__(
        self,
        env_name: str,
        env_id: str,
        seed: int = -1,
        regression: bool = False,
        minimize: bool = False,
        num_envs: int = 1,
        folder: str = "logs",
        algo: str = "sac",
        exp_id: int = 0,
        avf_train_policy: str = "mlp",
        avf_test_policy: str = "nn",
        dnn_sampling: str = "original",
        sampling_size: int = 1,
        neighborhood_size: int = 50,
        hc_counter: int = 100,
        stagnation_tolerance: float = 0.005,
        model_checkpoint: int = -1,
        training_progress_filter: int = None,
        layers: int = 4,
        oversample_minority_class_percentage: float = 0.0,
        failure_prob_dist: bool = True,
        num_episodes: int = 1000,
        num_runs_each_env_config: int = 10,
        exp_name: str = None,
        vae_path: str = None,
        add_to_port: int = 1,
        simulation_mul: int = 1,
        z_size: int = 64,
        exe_path: str = None,
        exp_file: str = None,
        parallelize_fp_dist: bool = False,
        budget: int = -1,
        population_size: int = -1,
        crossover_rate: float = -1.0,
        resume_dir: str = None,
        remove_road_constraints: bool = False,
    ):

        self.logger = Log("ExpConfigurator")

        self.env_name = env_name
        self.num_envs = num_envs
        assert self.num_envs == 1, "Num envs must be = 1. Found: {}".format(
            self.num_envs
        )
        self.folder = folder
        self.algo = algo
        self.env_id = env_id
        self.exp_id = exp_id
        self.seed = seed
        self.avf_train_policy = avf_train_policy
        self.avf_test_policy = avf_test_policy
        self.dnn_sampling = dnn_sampling
        self.sampling_size = sampling_size
        self.neighborhood_size = neighborhood_size
        self.oversample_minority_class_percentage = oversample_minority_class_percentage
        self.layers = layers
        self.stagnation_tolerance = stagnation_tolerance
        self.hc_counter = hc_counter
        self.model_checkpoint = model_checkpoint
        self.training_progress_filter = training_progress_filter
        self.failure_prob_dist = failure_prob_dist
        self.num_episodes = num_episodes
        self.num_runs_each_env_config = num_runs_each_env_config
        self.regression = regression
        self.minimize = minimize
        self.exp_name = None
        self.other_exp_name = None

        if exp_name is None or len(exp_name.split("-")) == 2:
            self.exp_name = exp_name
        elif exp_name is not None and len(exp_name.split("-")) > 2:
            # {}-{}-trial
            self.other_exp_name = exp_name.split("-")[0]
            self.exp_name = exp_name[exp_name.index("-") + 1 :]

        self.vae_path = vae_path
        self.add_to_port = add_to_port
        self.simulation_mul = simulation_mul
        self.z_size = z_size
        self.exe_path = exe_path
        self.exp_file = exp_file
        self.parallelize_fp_dist = parallelize_fp_dist
        self.budget = budget
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.resume_dir = resume_dir
        self.remove_road_constraints = remove_road_constraints

        self.testing_strategy = self.avf_test_policy
        if self.avf_test_policy == "random":
            self.testing_strategy = "random"
        elif self.avf_test_policy == "prioritized_replay":
            self.testing_strategy = "{}-{}".format(
                self.avf_test_policy, self.training_progress_filter
            )
        elif self.avf_test_policy == "nn":
            self.testing_strategy = "{}-{}-{}-{}-{}-{}-{}".format(
                self.avf_train_policy,
                self.avf_test_policy,
                self.dnn_sampling,
                self.training_progress_filter,
                self.oversample_minority_class_percentage,
                self.layers,
                self.sampling_size,
            )
        elif "hc" in self.avf_test_policy:

            if "hc_saliency" in self.avf_test_policy:
                assert (
                    self.avf_train_policy in AVF_DNN_POLICIES
                ), "Saliency is supported only with these policies: {}, found {}".format(
                    AVF_DNN_POLICIES, self.avf_train_policy
                )

            self.testing_strategy = "{}-{}-{}-{}-{}-{}-{}-{}".format(
                self.avf_train_policy,
                self.avf_test_policy,
                (
                    self.training_progress_filter
                    if "failure" not in self.avf_test_policy
                    else FILTER_FAILURE_BASED_APPROACHES
                ),
                self.oversample_minority_class_percentage,
                self.layers,
                self.neighborhood_size,
                self.hc_counter,
                self.stagnation_tolerance,
            )

        elif "ga" in self.avf_test_policy:
            self.testing_strategy = "{}-{}-{}-{}-{}-{}-{}".format(
                self.avf_train_policy,
                self.avf_test_policy,
                (
                    self.training_progress_filter
                    if "failure" not in self.avf_test_policy
                    else FILTER_FAILURE_BASED_APPROACHES
                ),
                self.oversample_minority_class_percentage,
                self.layers,
                self.population_size,
                self.crossover_rate,
            )

        if self.regression:
            self.testing_strategy += "-regression"
            if self.minimize:
                self.testing_strategy += "-minimize"

        if self.other_exp_name is not None:
            self.testing_strategy += f"-{self.other_exp_name}"

        if self.failure_prob_dist:
            if self.exp_file is None:
                self.testing_strategy += "-failure-prob-dist-{}-{}".format(
                    self.num_episodes, self.num_runs_each_env_config
                )
            if self.parallelize_fp_dist:
                self.testing_strategy += "-parallel"

        if self.avf_test_policy == "test":
            self.testing_strategy += "-test"

        if self.budget != -1:
            self.testing_strategy += "-budget-{}".format(self.budget)

        if self.model_checkpoint != -1:
            self.testing_strategy += f"-{model_checkpoint}"

        if self.exp_name is not None:
            self.testing_strategy += "-{}".format(self.exp_name)

        if self.resume_dir is not None:
            self.testing_strategy += "-resume"

        if self.remove_road_constraints and self.env_name == DONKEY_ENV_NAME:
            assert (
                self.avf_test_policy == "random"
                or self.avf_test_policy == "hc"
                or "nn" in self.avf_test_policy
                or "rnd" in self.avf_test_policy
            ), f"Policy {self.avf_test_policy} not supported. Only random policies are supported."
            self.testing_strategy += "-no-constraints"

        self._initialize()

    def _initialize(self):

        (
            self.model,
            self.avf,
            self.env,
            self.log_path,
            testing_strategy,
        ) = instantiate_trained_policy(
            env_name=self.env_name,
            algo=self.algo,
            folder=self.folder,
            seed=self.seed,
            env_id=self.env_id,
            exp_id=self.exp_id,
            model_checkpoint=self.model_checkpoint,
            headless=True,
            enjoy_mode=False,
            testing_strategy=self.testing_strategy,
            regression=self.regression,
            minimize=self.minimize,
            vae_path=self.vae_path,
            add_to_port=self.add_to_port,
            simulation_mul=self.simulation_mul,
            z_size=self.z_size,
            exe_path=self.exe_path,
            exp_file=self.exp_file,
            resume_dir=self.resume_dir,
        )

        self.num_episodes = (
            self.num_episodes
            if self.exp_file is None
            else len(self.avf.failure_test_env_configs)
        )

        self.avf.update_state_variables_to_enable_testing_mode(
            num_episodes=self.num_episodes,
            training_progress_filter=self.training_progress_filter,
            avf_train_policy=self.avf_train_policy,
            avf_test_policy=self.avf_test_policy,
            oversample_minority_class_percentage=self.oversample_minority_class_percentage,
            layers=self.layers,
            stagnation_tolerance=self.stagnation_tolerance,
            dnn_sampling=self.dnn_sampling,
            sampling_size=self.sampling_size,
            failure_prob_dist=self.failure_prob_dist,
            num_runs_each_env_config=self.num_runs_each_env_config,
            testing_strategy_name=testing_strategy,
            neighborhood_size=self.neighborhood_size,
            hc_counter=self.hc_counter,
            parallelize_fp_dist=self.parallelize_fp_dist,
            budget=self.budget,
            population_size=self.population_size,
            crossover_rate=self.crossover_rate,
            remove_road_constraints=self.remove_road_constraints,
        )

        self.deterministic = True

    def test(
        self,
        max_n_episodes: int = 100,
        num_trials: int = -1,
    ) -> Tuple[int, float, int]:

        start_time = time.time()
        self.avf.clear_state()
        self.avf.num_trials = num_trials
        self.avf.num_episodes = 0

        for i in range(max_n_episodes):
            obs = self.env.reset()
            done, state = False, None
            episode_reward = 0.0
            episode_length = 0

            while not done:
                action, state = self.model.predict(
                    obs, state=state, deterministic=self.deterministic
                )
                # Clip Action to avoid out of bound errors
                if isinstance(self.env.action_space, gym.spaces.Box):
                    action = np.clip(
                        action, self.env.action_space.low, self.env.action_space.high
                    )
                obs, reward, done, _info = self.env.step(action)
                episode_reward += reward[0]
                episode_length += 1
                if done:
                    self.avf.num_episodes += 1
                    # NOTE: for env using VecNormalize, the mean reward
                    # is a normalized reward when `--norm_reward` flag is passed
                    self.logger.debug("Episode #{}".format(i + 1))
                    self.logger.debug("Episode Reward: {:.2f}".format(episode_reward))
                    self.logger.debug("Episode Length: {}".format(episode_length))
                    is_success = _info[0].get("is_success", None)
                    if is_success is not None:
                        if is_success == 0:
                            return i + 1, time.time() - start_time, episode_length

        return max_n_episodes, time.time() - start_time, 0

    def test_single_episode(
        self, episode_num: int, num_trials: int = -1
    ) -> Tuple[bool, EnvConfiguration, List[float]]:
        obs = self.env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        fitness_values = []
        if episode_num != -1:
            self.avf.num_episodes = episode_num
        if num_trials != -1:
            self.avf.num_trials = num_trials

        while not done:
            action, state = self.model.predict(
                obs, state=state, deterministic=self.deterministic
            )
            # Clip Action to avoid out of bound errors
            if isinstance(self.env.action_space, gym.spaces.Box):
                action = np.clip(
                    action, self.env.action_space.low, self.env.action_space.high
                )
            obs, reward, done, _info = self.env.step(action)
            if _info[0].get("fitness", None) is not None:
                fitness_values.append(_info[0]["fitness"])

            episode_reward += reward[0]
            episode_length += 1
            if done:
                self.avf.num_episodes += 1
                # NOTE: for env using VecNormalize, the mean reward
                # is a normalized reward when `--norm_reward` flag is passed
                self.logger.debug("Episode #{}".format(episode_num + 1))
                self.logger.debug("Episode Reward: {:.2f}".format(episode_reward))
                self.logger.debug("Episode Length: {}".format(episode_length))
                if episode_length < 5:
                    self.logger.warn("Very short episode")
                is_success = _info[0].get("is_success", None)
                if is_success is not None:
                    self.logger.debug("Failure: {}".format(not is_success))
                    if is_success == 0:
                        return True, self.avf.get_current_env_config(), fitness_values
                    return False, self.avf.get_current_env_config(), fitness_values

    def close_env(self):

        assert self.num_envs == 1, "Num envs must be = 1. Found: {}".format(
            self.num_envs
        )

        env_unwrapped = self.env.unwrapped
        while not isinstance(env_unwrapped, DummyCVecEnv):
            env_unwrapped = env_unwrapped.unwrapped

        assert isinstance(
            env_unwrapped.envs[0], EnvWrapper
        ), "{} is not an instance of EnvWrapper".format(type(env_unwrapped.envs[0]))
        if hasattr(env_unwrapped.envs[0], "env"):
            env_unwrapped.envs[0].env.close()
        else:
            # it can be that the wrapper does not contain an instance of the actual environment
            # but rather that the wrapper is the environment itself
            if self.env_name == DONKEY_ENV_NAME:
                env_unwrapped.envs[0].exit_scene()
                time.sleep(5)
                env_unwrapped.envs[0].close_connection()
            env_unwrapped.envs[0].close()

    def get_num_evaluation_predictions(self) -> int:
        if self.avf.trained_avf_policy is not None:
            return self.avf.trained_avf_policy.get_num_evaluation_predictions()
        return 0
