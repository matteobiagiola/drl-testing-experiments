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
import glob
import os
import socket
import time
from typing import Callable, Union

from stable_baselines3.common.monitor import Monitor

from indago.algos.dqn_wrapper import DQNWrapper
from indago.algos.her_wrapper import HERWrapper
from indago.algos.ppo_wrapper import PPOWrapper
from indago.algos.sac_wrapper import SACWrapper
from indago.algos.tqc_wrapper import TQCWrapper
from indago.avf.avf import Avf
from indago.config import (
    BASE_PORT,
    DONKEY_ENV_NAME,
    HUMANOID_ENV_NAME,
    PARK_ENV_NAME,
    CARTPOLE_ENV_NAME,
)
from indago.envs.cartpole.cartpole_env_wrapper import CartPoleEnvWrapper
from indago.envs.donkey.donkey_env_wrapper import DonkeyEnvWrapper
from indago.envs.donkey.scenes.simulator_scenes import SimulatorScene
from indago.envs.donkey.vae.vae import VAE
from indago.envs.humanoid.humanoid_env_wrapper import HumanoidEnvWrapper
from indago.envs.park.parking_env_wrapper import ParkingEnvWrapper
from indago.utils.torch_utils import DEVICE

ALGOS = {
    "her": HERWrapper,
    "sac": SACWrapper,
    "tqc": TQCWrapper,
    "dqn": DQNWrapper,
    "ppo": PPOWrapper,
}


def check_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func


def constfn(val):
    """
    Create a function that returns a constant
    It is useful for learning rate schedule (to avoid code duplication)

    :param val: (float)
    :return: (function)
    """

    def func(_):
        return val

    return func


def get_latest_run_id(log_path: str, env_id: str) -> int:
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.
    :param log_path: path to log folder
    :param env_id:
    :return: latest run number
    """
    max_run_id = 0
    for path in glob.glob(log_path + os.sep + f"{env_id}_[0-9]*"):
        file_name = path.split(os.sep)[-1]
        ext = file_name.split("_")[-1]
        # to take into account files with extension
        if "." in ext:
            ext = ext[: ext.rindex(".")]
        if (
            env_id == "_".join(file_name.split("_")[:-1])
            and ext.isdigit()
            and int(ext) > max_run_id
        ):
            max_run_id = int(ext)
    return max_run_id


def make_env_fn(
    env_name: str,
    seed: int,
    avf: Avf,
    record_video: bool = False,
    save_path: str = None,
    vae_path: str = None,
    add_to_port: int = 0,
    simulation_mul: int = 1,
    z_size: int = 64,
    n_stack: int = 1,
    exe_path: str = None,
    simulator_scene: SimulatorScene = None,
    headless: bool = False,
) -> Callable:
    def _cartpole_env_init():
        env_ = CartPoleEnvWrapper(avf=avf, headless=headless)
        env_.seed(seed)
        env_.action_space.seed(seed)
        info_keywords = ("is_success",)

        if save_path is not None:
            return Monitor(
                env_, save_path, allow_early_resets=True, info_keywords=info_keywords
            )
        return env_

    def _park_env_init():
        env_ = ParkingEnvWrapper(avf=avf)
        env_.seed(seed)
        env_.action_space.seed(seed)
        info_keywords = ("is_success",)

        if not record_video:
            env_.unwrap().config["offscreen_rendering"] = True
        else:
            env_.unwrap().config["show_trajectories"] = True

        if save_path is not None:
            return Monitor(
                env_, save_path, allow_early_resets=True, info_keywords=info_keywords
            )
        return env_

    def _humanoid_env_init():
        env_ = HumanoidEnvWrapper(avf=avf)
        env_.seed(seed)
        env_.action_space.seed(seed)
        info_keywords = ("is_success",)
        if save_path is not None:
            return Monitor(
                env_, save_path, allow_early_resets=True, info_keywords=info_keywords
            )
        return env_

    def _donkey_env_init():
        # not a very nice solution since some simulator instances may have been already started and
        # there is no way to stop them from this point
        count = 0
        while check_port_in_use(port=BASE_PORT + add_to_port) and count < 10:
            time.sleep(1)
            count += 1

        assert not check_port_in_use(
            port=BASE_PORT + add_to_port
        ), "Port {} is in use".format(BASE_PORT + add_to_port)

        vae = VAE(in_channels=3, latent_dim=z_size).to(DEVICE)
        vae.load(vae_path)

        env_ = DonkeyEnvWrapper(
            vae=vae,
            n_stack=n_stack,
            seed=seed,
            exe_path=exe_path,
            add_to_port=add_to_port,
            avf=avf,
            simulation_mul=simulation_mul,
            simulator_scene=simulator_scene,
            headless=headless,
        )

        env_.seed(seed)
        env_.action_space.seed(seed)

        info_keywords = ("is_success",)
        if save_path is not None:
            return Monitor(
                env_, save_path, allow_early_resets=True, info_keywords=info_keywords
            )

        return env_

    if env_name == CARTPOLE_ENV_NAME:
        return _cartpole_env_init
    if env_name == PARK_ENV_NAME:
        return _park_env_init
    if env_name == HUMANOID_ENV_NAME:
        return _humanoid_env_init
    if env_name == DONKEY_ENV_NAME:
        assert vae_path is not None, "Vae path is not set"
        assert exe_path is not None, "Exe path is not set"

        return _donkey_env_init

    raise NotImplementedError("Unknown env name: {}".format(env_name))
