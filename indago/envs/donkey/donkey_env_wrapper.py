# Original author: Roma Sokolkov
# Edited by Antonin Raffin

"""
MIT License

Copyright (c) 2018 Roma Sokolkov
Copyright (c) 2018 Antonin Raffin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import platform
from typing import Dict

import gym
import numpy as np
import torch
from gym import spaces

from indago.avf.avf import Avf
from indago.config import (
    BASE_PORT,
    BASE_SOCKET_LOCAL_ADDRESS,
    MAX_STEERING,
    MAX_STEERING_DIFF,
    MAX_THROTTLE,
    MIN_STEERING,
    MIN_THROTTLE,
    N_COMMAND_HISTORY,
    ROI,
)
from indago.env_wrapper import EnvWrapper
from indago.envs.donkey.core.donkey_proc import DonkeyUnityProcess
from indago.envs.donkey.core.donkey_sim import DonkeyUnitySimController
from indago.envs.donkey.scenes.simulator_scenes import SimulatorScene
from indago.envs.donkey.vae.data_loader import preprocess_image
from indago.envs.donkey.vae.vae import VAE, reparameterize
from indago.utils.torch_utils import from_numpy_no_device, to_numpy
from log import Log


class DonkeyEnvWrapper(gym.Env, EnvWrapper):
    """
    Gym interface for DonkeyCar with support for using
    a VAE encoded observation instead of raw pixels if needed.
    """

    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        vae: VAE,
        n_stack: int,
        seed: int,
        exe_path: str,
        add_to_port: int,
        avf: Avf,
        simulation_mul: int,
        simulator_scene: SimulatorScene,
        headless: bool,
    ):

        super(DonkeyEnvWrapper, self).__init__(avf=avf)

        self.logger = Log("DonkeyWrapper")

        if platform.system().lower() == "windows":
            self.logger.warn("Headless mode is disabled in Windows")
            headless = False

        self.vae = vae
        self.z_size = vae.z_size

        self.min_throttle = MIN_THROTTLE
        self.max_throttle = MAX_THROTTLE
        self.exe_path = exe_path
        self.avf = avf

        # Save last n commands (throttle + steering)
        self.n_commands = 2
        self.n_command_history = N_COMMAND_HISTORY
        self.command_history = np.zeros((1, self.n_commands * self.n_command_history))

        # Custom frame-stack
        self.n_stack = n_stack
        self.stacked_obs = None

        # TCP port for communicating with simulation
        if add_to_port == -1:
            port = int(os.environ.get("DONKEY_SIM_PORT", 9091))
            socket_local_address = int(
                os.environ.get("BASE_SOCKET_LOCAL_ADDRESS", 52804)
            )
        else:
            port = BASE_PORT + add_to_port
            socket_local_address = BASE_SOCKET_LOCAL_ADDRESS + port

        self.logger.debug("Simulator port: {}".format(port))

        self.unity_process = None
        self.logger.info("Starting DonkeyGym env")
        assert os.path.exists(self.exe_path), "Path {} does not exist".format(
            self.exe_path
        )
        # Start Unity simulation subprocess if needed
        self.unity_process = DonkeyUnityProcess()
        # headless = os.environ.get('DONKEY_SIM_HEADLESS', False) == '1'
        self.unity_process.start(
            sim_path=self.exe_path,
            headless=headless,
            port=port,
            simulation_mul=simulation_mul,
        )

        # start simulation com
        self.viewer = DonkeyUnitySimController(
            socket_local_address=socket_local_address,
            port=port,
            seed=seed,
            avf=avf,
            simulation_mul=simulation_mul,
            vae=self.vae,
            simulator_scene=simulator_scene,
        )

        # steering + throttle, action space must be symmetric
        self.action_space = spaces.Box(
            low=np.array([-MAX_STEERING, -1]),
            high=np.array([MAX_STEERING, 1]),
            dtype=np.float32,
        )

        # z latent vector from the VAE (encoded input image)
        self.observation_space = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(self.z_size + self.n_commands * self.n_command_history,),
            dtype=np.float32,
        )

        # Frame-stacking with teleoperation
        if self.n_stack > 1:
            obs_space = self.observation_space
            low = np.repeat(obs_space.low, self.n_stack, axis=-1)
            high = np.repeat(obs_space.high, self.n_stack, axis=-1)
            self.stacked_obs = np.zeros(low.shape, low.dtype)
            self.observation_space = spaces.Box(
                low=low, high=high, dtype=obs_space.dtype
            )

        self.seed(seed)
        # wait until loaded
        self.viewer.wait_until_loaded()

    def unwrap(self):
        return self

    def close_connection(self):
        return self.viewer.close_connection()

    def exit_scene(self):
        self.viewer.handler.send_exit_scene()

    def postprocessing_step(
        self,
        action: np.ndarray,
        observation: np.ndarray,
        reward: float,
        done: bool,
        info: Dict,
    ):
        """
        Update the reward (add jerk_penalty if needed), the command history
        and stack new observation (when using frame-stacking).

        :param action: ([float])
        :param observation: (np.ndarray)
        :param reward: (float)
        :param done: (bool)
        :param info: (dict)
        :return: (np.ndarray, float, bool, dict)
        """
        # Update command history
        if self.n_command_history > 0 and self.vae:
            self.command_history = np.roll(
                self.command_history, shift=-self.n_commands, axis=-1
            )
            self.command_history[..., -self.n_commands :] = action
            observation = np.concatenate((observation, self.command_history), axis=-1)

        # jerk_penalty = self.jerk_penalty()
        # # Cancel reward if the continuity constrain is violated
        # if jerk_penalty > 0 and reward > 0:
        #     reward = 0
        # reward -= jerk_penalty

        if self.n_stack > 1:
            self.stacked_obs = np.roll(
                self.stacked_obs, shift=-observation.shape[-1], axis=-1
            )
            if done:
                self.stacked_obs[...] = 0
            self.stacked_obs[..., -observation.shape[-1] :] = observation
            return self.stacked_obs, reward, done, info

        if self.vae:
            return observation.flatten(), reward, done, info
        return observation, reward, done, info

    def clip_steering_diff(self, steering):
        """
        :param steering: (float)
        :return: (foat)
        """
        prev_steering = self.command_history[0, -2]
        max_diff = (MAX_STEERING_DIFF - 1e-5) * (MAX_STEERING - MIN_STEERING)
        diff = np.clip(steering - prev_steering, -max_diff, max_diff)
        return prev_steering + diff

    def convert_throttle_to_donkey(self, throttle):
        """
        :param throttle: (float)
        :return: (float)
        """
        # Convert from [-1, 1] to [0, 1]
        t = (throttle + 1) / 2
        # Convert from [0, 1] to [min, max]
        return (1 - t) * self.min_throttle + self.max_throttle * t

    def step(self, action: np.ndarray):
        """
        :param action: (np.ndarray)
        :return: (np.ndarray, float, bool, dict)
        """
        # action[0] is the steering angle
        # action[1] is the throttle
        action[1] = self.convert_throttle_to_donkey(action[1])

        # Clip steering angle rate to enforce continuity
        if self.n_command_history > 0:
            action[0] = self.clip_steering_diff(action[0])

        self.viewer.take_action(action)
        observation, reward, done, info = self.observe()

        return self.postprocessing_step(action, observation, reward, done, info)

    def reset(self, end_of_episode: bool = False):

        self.viewer.reset(end_of_episode=end_of_episode)
        self.command_history = np.zeros((1, self.n_commands * self.n_command_history))
        observation, reward, done, info = self.observe()

        if self.n_command_history > 0 and self.vae:
            observation = np.concatenate((observation, self.command_history), axis=-1)

        if self.n_stack > 1:
            self.stacked_obs[...] = 0
            self.stacked_obs[..., -observation.shape[-1] :] = observation
            return self.stacked_obs

        if self.vae:
            return observation.flatten()
        return observation

    def render(self, mode="human"):
        """
        :param mode: (str)
        """
        if mode == "rgb_array":
            return self.viewer.handler.original_image
        return None

    def observe(self):
        """
        Encode the observation using VAE if needed.

        :return: (np.ndarray, float, bool, dict)
        """
        observation, reward, done, info = self.viewer.observe()
        # print('reward: {}, done: {}, info: {}'.format(reward, done, info))
        # Learn from Pixels
        if self.vae is None:
            return observation, reward, done, info

        with torch.no_grad():
            observation = preprocess_image(image=observation, roi=False)
            observation = observation.reshape((1,) + observation.shape)
            image_tensor = from_numpy_no_device(observation)
            mu, log_var = self.vae.encode(image_tensor)
            z = reparameterize(mu, log_var)
            encoded_image = z

            # obs_predicted = self.vae.decode(z=encoded_image)
            # loss = self.vae.loss_function(obs_predicted, image_tensor, mu, log_var)['loss']

        return encoded_image, reward, done, info

    def close(self):
        if self.unity_process is not None:
            self.unity_process.quit()
        self.viewer.quit()

    def pause_simulation(self):
        self.viewer.handler.send_pause_simulation()

    def restart_simulation(self):
        self.viewer.handler.send_restart_simulation()

    def send_agent_state(self, agent_state: Dict):
        self.viewer.handler.send_agent_state(agent_state=agent_state)

    def seed(self, seed=None):
        self.viewer.seed(seed)

    def set_vae(self, vae):
        """
        :param vae: (VAEController object)
        """
        self.vae = vae
