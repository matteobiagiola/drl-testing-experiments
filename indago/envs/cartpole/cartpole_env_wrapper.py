import base64
import numpy as np
from io import BytesIO
from typing import Dict, Optional, Tuple
from PIL import Image

from highway_env.envs.common.abstract import Observation
from indago.avf.env_configuration import EnvConfiguration

from indago.env_wrapper import EnvWrapper
from indago.envs.cartpole.cartpole_env import CartPoleEnv
from indago.envs.cartpole.cartpole_training_logs import CartPoleTrainingLogs


class CartPoleEnvWrapper(EnvWrapper):

    # no typing to avoid circular inputs when called from main
    def __init__(self, avf, headless: bool = False):
        super(CartPoleEnvWrapper, self).__init__(avf=avf)
        self.env: CartPoleEnv = CartPoleEnv()
        self.configuration: EnvConfiguration = None
        self.agent_state = None
        self.first_frame_string = None
        self.fitness_values = []
        self.actions = []
        self.rewards = []
        self.speeds = []
        self.cart_positions = []
        self.cart_angles = []
        self.headless = headless

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.metadata = self.env.metadata
        self.spec = self.env.spec
        self.seed = self.env.seed

    def unwrap(self):
        return self.env

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = self.env.step(action=action)
        if self.env.discrete_action_space:
            actions = int(action)
        else:
            actions = float(action)

        self.actions.append(actions)
        self.rewards.append(reward)

        if info.get("fitness", None) is not None:
            self.fitness_values.append(info["fitness"])

        self.cart_positions.append(self.env.state[0])
        self.cart_angles.append(self.env.state[2])
        if done:
            cartpole_training_logs = CartPoleTrainingLogs(
                is_success=int(info["is_success"]),
                fitness_values=self.fitness_values,
                agent_state=self.agent_state,
                first_frame_string=self.first_frame_string,
                config=self.configuration,
                actions=self.actions,
                rewards=self.rewards,
                cart_positions=self.cart_positions,
                cart_angles=self.cart_angles,
            )
            self.avf.store_training_logs(training_logs=cartpole_training_logs)
            self.avf.store_testing_logs(training_logs=cartpole_training_logs)
            self.actions.clear()
            self.rewards.clear()
            self.cart_positions.clear()
            self.cart_angles.clear()
            self.fitness_values.clear()

        return obs, reward, done, info

    def reset(self, end_of_episode: bool = False) -> Observation:
        if not end_of_episode:
            self.configuration = self.avf.generate_env_configuration()
            self.configuration.update_implementation(
                x=self.configuration.x,
                x_dot=self.configuration.x_dot,
                theta=self.configuration.theta,
                theta_dot=self.configuration.theta_dot,
            )

            self.env.x = self.configuration.x
            self.env.x_dot = self.configuration.x_dot
            self.env.theta = self.configuration.theta
            self.env.theta_dot = self.configuration.theta_dot

        obs_reset = self.env.reset()
        if not self.headless:
            image = self.env.render(mode="rgb_array")
            buffered = BytesIO()
            pil_image = Image.fromarray(image)
            pil_image.save(buffered, optimize=True, format="PNG", quality=95)
            self.first_frame_string = base64.b64encode(buffered.getvalue()).decode(
                "utf-8"
            )

        return obs_reset

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        return self.env.render(mode=mode)

    def close(self) -> None:
        self.env.close()

    def send_agent_state(self, agent_state: Dict) -> None:
        self.agent_state = agent_state
