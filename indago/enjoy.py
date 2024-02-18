# Author: Antonin Raffin

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

import argparse
import os
import time
from typing import List

import cv2
import gym
import numpy as np
from stable_baselines3.common.vec_env import VecFrameStack

from indago.config import DONKEY_ENV_NAME, ENV_IDS, ENV_NAMES, PARK_ENV_NAME
from indago.utils.env_utils import ALGOS
from indago.utils.policy_utils import instantiate_trained_policy
from log import Log

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", help="Log folder", type=str, default="logs")
parser.add_argument(
    "--algo",
    help="RL Algorithm",
    default="sac",
    type=str,
    required=False,
    choices=list(ALGOS.keys()),
)
parser.add_argument(
    "-n", "--n-episodes", help="number of episodes", default=5, type=int
)
parser.add_argument("--exp-id", help="Experiment ID (0: latest)", default=0, type=int)
parser.add_argument(
    "--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int
)
parser.add_argument(
    "--env-name", help="Env name", type=str, choices=ENV_NAMES, default="park"
)
parser.add_argument(
    "--env-id", help="Env id", type=str, choices=ENV_IDS, default="parking-v0"
)
parser.add_argument(
    "--deterministic",
    action="store_true",
    default=False,
    help="Use deterministic actions",
)
parser.add_argument("--seed", help="Random generator seed", type=int, default=-1)
parser.add_argument(
    "--model-checkpoint", help="Model checkpoint to load", type=int, default=-1
)
parser.add_argument(
    "--headless",
    help="Whether to run the simulator headless or not (for Parking the frames are recorded on disk)",
    action="store_true",
    default=False,
)

# DonkeyCar parameters
parser.add_argument(
    "-vae", "--vae-path", help="Path to saved VAE", type=str, default=None
)
parser.add_argument(
    "--add-to-port",
    help="Adding to default port 9091 in order to execute more simulators in parallel",
    type=int,
    default=-1,
)
parser.add_argument(
    "--simulation-mul",
    help="Speed up DonkeyCar simulation by at most 5x",
    type=int,
    default=1,
)
parser.add_argument(
    "--z-size",
    help="Latent space size. Needs to match the latent space of the trained VAE",
    type=int,
    default=64,
)
parser.add_argument(
    "--exe-path", help="DonkeyCar simulator execution path", type=str, default=None
)

args = parser.parse_args()

if __name__ == "__main__":

    logger = Log("enjoy")
    logger.info("Args: {}".format(args))

    model, avf, env, log_path, _ = instantiate_trained_policy(
        env_name=args.env_name,
        algo=args.algo,
        folder=args.folder,
        seed=args.seed,
        env_id=args.env_id,
        exp_id=args.exp_id,
        model_checkpoint=args.model_checkpoint,
        headless=args.headless,
        vae_path=args.vae_path,
        add_to_port=args.add_to_port,
        simulation_mul=args.simulation_mul,
        z_size=args.z_size,
        exe_path=args.exe_path,
    )
    avf.update_state_variables_to_enable_testing_mode(
        num_episodes=0,
        training_progress_filter=0,
        avf_train_policy="dummy",
        avf_test_policy="random",
        oversample_minority_class_percentage=0.0,
        layers=4,
        dnn_sampling="original",
        sampling_size=0,
        failure_prob_dist=False,
        num_runs_each_env_config=0,
        testing_strategy_name="enjoy",
        neighborhood_size=0,
        hc_counter=100,
        stagnation_tolerance=0.005,
        parallelize_fp_dist=False,
        budget=0,
        population_size=0,
        crossover_rate=0.0,
        remove_road_constraints=False,
    )

    # Force deterministic for SAC and DDPG
    deterministic = args.deterministic or args.algo in ["ddpg", "sac"]
    logger.info("Deterministic actions: {}".format(deterministic))

    episode_rewards: List[float] = []
    episode_lengths: List[float] = []
    episodes_info: List = []

    start_time = time.time()
    num_failures = 0

    for i in range(args.n_episodes):
        obs = env.reset()

        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        episode_info = None

        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            # Clip Action to avoid out of bound errors
            if isinstance(env.action_space, gym.spaces.Box):
                action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, done, _info = env.step(action)
            episode_reward += reward[0]
            episode_length += 1
            if not args.headless:
                if args.env_name == PARK_ENV_NAME:
                    image = env.render("rgb_array")
                    cv2.imwrite(
                        filename=os.path.join(
                            log_path, "{}_{}.png".format(i, episode_length)
                        ),
                        img=image,
                    )
                else:
                    env.render()
            if done:
                avf.num_episodes += 1
                logger.debug("Episode #{}".format(i + 1))
                logger.debug("Episode Reward: {:.2f}".format(episode_reward))
                logger.debug("Episode Length: {}".format(episode_length))
                is_success = _info[0].get("is_success", None)
                if is_success is not None:
                    if is_success == 0:
                        num_failures += 1
                        logger.info(
                            "Failure #{} found after {} episodes. Time elapsed: {}".format(
                                num_failures, i + 1, time.time() - start_time
                            )
                        )
                episode_info = _info

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episodes_info.append(episode_info)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    logger.info("Avg Reward: {:.2f} +- {:.2f}".format(mean_reward, std_reward))

    success_rates = []
    for info in episodes_info:
        is_success = info[0].get("is_success", None)
        assert (
            is_success is not None
        ), "Info object does not contain the field is_success. Found: {}".format(info)
        if is_success == 0:
            success_rates.append(0)
        elif is_success == 1:
            success_rates.append(1)
        else:
            logger.warn("Warning done_reason is {}".format(is_success))
    if len(success_rates) > 0:
        logger.info("Success rates: {}".format(np.mean(success_rates)))

    logger.info("Time elapsed: {}s".format(time.time() - start_time))

    # Close the connection properly
    env.reset()
    if isinstance(env, VecFrameStack):
        env = env.venv

    # HACK to bypass Monitor wrapper
    if args.env_name == DONKEY_ENV_NAME:
        env.envs[0].exit_scene()
        time.sleep(5)
        env.envs[0].close_connection()
        if args.exe_path:
            env.envs[0].close()
    else:
        env.close()
