"""
The MIT License

Copyright (c) 2019 Antonin Raffin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import os
from typing import Dict, Union

import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import X_EPISODES, X_TIMESTEPS
from stable_baselines3.common.vec_env import VecEnv

from indago.avf.avf import Avf
from indago.utils.results_plotter import X_SUCCESS, ts2xy
from log import Log


class SaveBestModelCallback(BaseCallback):
    def __init__(
        self,
        log_interval: int,
        save_checkpoint_interval: int,
        log_dir: str,
        verbose: int = 0,
        num_envs: int = 1,
        eval_env: Union[gym.Env, VecEnv] = None,
        normalize_kwargs: Dict = None,
        n_eval_episodes: int = 1,
        deterministic: bool = True,
        avf: Avf = None
    ):
        super(SaveBestModelCallback, self).__init__(verbose)

        self._logger = Log("SaveBestModelCallback")

        self.log_interval = log_interval // num_envs
        self.save_checkpoint_interval = save_checkpoint_interval
        self.count_log_interval = 0
        self.log_dir = log_dir

        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.best_mean_success_rate = 0.0
        self.best_mean_reward_eval = -np.inf
        self.eval_env = eval_env
        self.eval_env_unnormalized = eval_env
        self.normalize_kwargs = normalize_kwargs
        self.num_successful_episodes = 0

        self.avf = avf
        self.indices_failing_configurations = set()

        if self.eval_env:
            assert n_eval_episodes > 1, "Num eval episodes must be > 1"

        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self.log_interval == 0:

            self.count_log_interval += 1

            # Retrieve training reward
            x_rewards, y_rewards = ts2xy(load_results(self.log_dir), X_TIMESTEPS)
            x_ep_lengths, y_ep_lengths = ts2xy(load_results(self.log_dir), X_EPISODES)

            x_success_rates, y_success_rates = [], []
            x_success_rates, y_success_rates = ts2xy(load_results(self.log_dir), X_SUCCESS)

            # self._logger.debug('x_rewards: {}, y_rewards: {}'.format(x_rewards, y_rewards))
            # self._logger.debug('x_ep_lengths: {}, y_ep_lengths: {}'.format(x_ep_lengths, y_ep_lengths))

            mean_reward = np.mean(y_rewards[-100:])
            if self.verbose > 0:
                self._logger.debug("Num timesteps: {}".format(self.num_timesteps))
                self._logger.debug(
                    "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                        self.best_mean_reward, mean_reward
                    )
                )
                mean_success_rate = np.mean(y_success_rates[-100:])
                self._logger.debug(
                    "Best mean success rate: {:.2f} "
                    "- Last mean success rate per episode: {:.2f}".format(self.best_mean_success_rate, mean_success_rate)
                )

            mean_success_rate = np.mean(y_success_rates[-100:])
            # assuming that the agent keeps improving with time
            if mean_success_rate >= self.best_mean_success_rate or mean_reward > self.best_mean_reward:
                self.best_mean_success_rate = mean_success_rate
                if self.verbose > 0:
                    self._logger.debug("Saving new best model to {}".format(self.save_path))
                self.model.save(self.save_path)

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward

            if self.save_checkpoint_interval == self.count_log_interval:
                self.count_log_interval = 0
                save_path = os.path.join(self.log_dir, "model-checkpoint-{}".format(self.num_timesteps))
                self._logger.info("Saving best model checkpoint: {}".format(save_path))
                self.model.save(save_path)

            if self.avf is not None:
                all_indices_failure_configs = list(np.arange(0, len(self.avf.failure_test_env_configs)))
                set_indices_selected = set(self.avf.indices_data_selected)
                indices_episode_successful = set()
                # Some algorithms have a warmup start in which they only carry out random actions. Such episodes
                # are not logged in the monitor file (i.e. y_success_rates). However, they are counted in the
                # self.avf.indices_data_selected. In order to consider only the episodes logged the first diff elements
                # of self.avf.indices_data_selected have to be ignored.
                diff = len(self.avf.indices_data_selected) - len(y_success_rates)

                for i, index in enumerate(self.avf.indices_data_selected[diff:]):
                    if y_success_rates[i]:
                        indices_episode_successful.add(index)
                        # making it less likely for this index to get selected in the future
                        # this makes the training faster by not selecting "easy" configurations very often; on
                        # the other hand it makes it possible for "difficult" configurations to be learned by
                        # feeding the agent also "easier" configuration in the late stages of training (i.e. when
                        # only "difficult" configurations are yet to be learned)
                        self.avf.weights_data[index] = 0.2

                self.indices_failing_configurations = \
                    set(all_indices_failure_configs).difference(indices_episode_successful)

                self._logger.info("Number of successful configurations {}/{}".format(
                    len(indices_episode_successful), len(all_indices_failure_configs))
                )
                self._logger.info("Number of failing configurations {}/{}".format(
                    len(self.indices_failing_configurations), len(all_indices_failure_configs))
                )

                if len(indices_episode_successful) > self.num_successful_episodes:
                    if self.verbose > 0:
                        self._logger.debug("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)
                    self.num_successful_episodes = len(indices_episode_successful)

                if len(set_indices_selected.intersection(set(all_indices_failure_configs))) \
                        == len(all_indices_failure_configs):

                    if len(indices_episode_successful) == len(all_indices_failure_configs):
                        self._logger.info(
                            "In all configurations the agent is successful. Aborting training."
                        )
                        return False

        return True
