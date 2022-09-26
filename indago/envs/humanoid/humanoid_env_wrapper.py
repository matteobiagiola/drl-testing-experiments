"""
The MIT License

Copyright (c) 2017 OpenAI (http://openai.com)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import base64
import platform
from io import BytesIO
from typing import Dict

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from PIL import Image

import indago.env_wrapper
from indago.avf.env_configuration import EnvConfiguration
from indago.env_wrapper import EnvWrapper
from indago.envs.humanoid.humanoid_training_logs import HumanoidTrainingLogs

MAX_TIMESTEPS = 300  # new training


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


class HumanoidEnvWrapper(mujoco_env.MujocoEnv, utils.EzPickle, EnvWrapper):

    # no typing to avoid circular inputs when called from main
    def __init__(self, avf=None):

        self.max_timesteps = MAX_TIMESTEPS
        self.first_frame_string = None
        self.actions = []
        self.rewards = []
        self.obs = []
        self.abdomen_trajectory = []

        mujoco_env.MujocoEnv.__init__(self, "humanoid.xml", 5)
        utils.EzPickle.__init__(self)
        indago.env_wrapper.EnvWrapper.__init__(self, avf=avf)

        self.configuration: EnvConfiguration = None
        self.qpos = self.init_qpos
        self.qvel = self.init_qvel

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate(
            [
                data.qpos.flat[2:],
                data.qvel.flat,
                data.cinert.flat,
                data.cvel.flat,
                data.qfrc_actuator.flat,
                data.cfrc_ext.flat,
            ]
        )

    def unwrap(self):
        return self

    def step(self, a):
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 1.25 * (pos_after - pos_before) / self.dt
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = 0.5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done_fall = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        done_time = self.max_timesteps <= 0
        state = self._get_obs()
        info = dict(
            reward_linvel=lin_vel_cost,
            reward_quadctrl=-quad_ctrl_cost,
            reward_alive=alive_bonus,
            reward_impact=-quad_impact_cost,
            is_success=done_time,
        )
        actions = list(map(lambda action: float(action), a))
        self.actions.append(actions)
        self.rewards.append(float(reward))
        self.abdomen_trajectory.append(float(qpos[2]))
        self.max_timesteps -= 1
        done = done_fall or done_time
        if done:
            # assert self.first_frame_string is not None, 'First frame not yet encoded'
            humanoid_training_logs = HumanoidTrainingLogs(
                is_success=int(info["is_success"]),
                agent_state=self.agent_state,
                first_frame_string=self.first_frame_string,
                config=self.configuration,
                actions=self.actions,
                rewards=self.rewards,
                state=state,
                abdomen_trajectory=self.abdomen_trajectory,
            )
            self.avf.store_training_logs(training_logs=humanoid_training_logs)
            self.avf.store_testing_logs(training_logs=humanoid_training_logs)
            self.actions.clear()
            self.rewards.clear()
            self.abdomen_trajectory.clear()

        return state, reward, done, info

    def reset(self, end_of_episode: bool = False):
        return self.reset_model(end_of_episode=end_of_episode)

    def reset_model(self, end_of_episode: bool = False):
        if not end_of_episode:
            self.configuration = self.avf.generate_env_configuration()
            self.configuration.update_implementation(
                qpos=self.configuration.qpos, qvel=self.configuration.qvel,
            )
            self.qpos = self.configuration.qpos
            self.qvel = self.configuration.qvel

        self.max_timesteps = MAX_TIMESTEPS
        self.set_state(qpos=self.qpos, qvel=self.qvel)

        # On Windows there is an issue with rendering mujoco environments
        if platform.system() == "windows":
            image = self.render("rgb_array")
            buffered = BytesIO()
            pil_image = Image.fromarray(image)
            pil_image.save(buffered, optimize=True, format="PNG", quality=95)
            self.first_frame_string = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20

    def send_agent_state(self, agent_state: Dict) -> None:
        self.agent_state = agent_state
