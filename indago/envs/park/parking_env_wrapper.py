import base64
from io import BytesIO
from PIL import Image
from typing import Dict, Optional, Tuple

from highway_env.envs import Action
from highway_env.envs.common.abstract import Observation
import numpy as np
from indago.avf.env_configuration import EnvConfiguration

from indago.env_wrapper import EnvWrapper
from indago.envs.park.parking_env import ParkingEnv
from indago.envs.park.parking_training_logs import ParkingTrainingLogs


class ParkingEnvWrapper(EnvWrapper):

    # no typing to avoid circular inputs when called from main
    def __init__(self, avf):
        super(ParkingEnvWrapper, self).__init__(avf=avf)
        self.env = ParkingEnv()
        self.configuration: EnvConfiguration = None
        self.agent_state = None
        self.first_frame_string = None
        self.actions = []
        self.rewards = []
        self.speeds = []
        self.fitness_values = []
        self.car_trajectory = []

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.spec = self.env.spec
        self.seed = self.env.seed

    def unwrap(self):
        return self.env

    # abstract method of ParkingEnv
    def _cost(self, action: Action) -> float:
        return self.env._cost(action=action)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = self.env.step(action=action)
        actions = list(map(lambda a: float(a), action))
        self.actions.append(actions)
        self.rewards.append(reward)
        self.speeds.append(info["speed"])
        if info.get("fitness", None) is not None:
            self.fitness_values.append(info["fitness"])

        self.car_trajectory.append(
            list(map(lambda a: float(a), list(info["vehicle_position"])))
        )
        if done:
            assert self.first_frame_string is not None, "First frame not yet encoded"
            goal_position = info.get("goal_position", None)
            parked_vehicle_positions = info.get("parked_vehicles_positions", None)
            parking_training_logs = ParkingTrainingLogs(
                is_success=int(info["is_success"]),
                fitness_values=self.fitness_values,
                agent_state=self.agent_state,
                first_frame_string=self.first_frame_string,
                config=self.configuration,
                actions=self.actions,
                rewards=self.rewards,
                speeds=self.speeds,
                car_trajectory=self.car_trajectory,
                goal_position=goal_position,
                parked_vehicle_positions=parked_vehicle_positions,
            )
            self.avf.store_training_logs(training_logs=parking_training_logs)
            self.avf.store_testing_logs(training_logs=parking_training_logs)
            self.actions.clear()
            self.rewards.clear()
            self.speeds.clear()
            self.car_trajectory.clear()
            self.fitness_values.clear()

        return obs, reward, done, info

    def reset(self, end_of_episode: bool = False) -> Observation:
        if not end_of_episode:
            self.configuration = self.avf.generate_env_configuration()
            self.configuration.update_implementation(
                num_lanes=self.configuration.num_lanes,
                goal_lane_idx=self.configuration.goal_lane_idx,
                heading_ego=self.configuration.heading_ego,
                parked_vehicles_lane_indices=list(
                    set(
                        sorted(
                            map(
                                lambda num: int(num),
                                self.configuration.parked_vehicles_lane_indices,
                            )
                        )
                    )
                ),
                position_ego=(
                    self.configuration.position_ego[0],
                    self.configuration.position_ego[1],
                ),
            )

            self.env.num_lanes = self.configuration.num_lanes
            self.env.goal_lane_idx = self.configuration.goal_lane_idx
            self.env.heading_ego = self.configuration.heading_ego
            self.env.parked_vehicles_lane_indices = (
                self.configuration.parked_vehicles_lane_indices
            )
            self.env.position_ego = self.configuration.position_ego
        obs_reset = self.env.reset()
        image = self.env.render("rgb_array")
        buffered = BytesIO()
        pil_image = Image.fromarray(image)
        pil_image.save(buffered, optimize=True, format="PNG", quality=95)
        self.first_frame_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return obs_reset

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        return self.env.render(mode=mode)

    def close(self) -> None:
        self.env.close()

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
        p: float = 0.5,
    ) -> float:
        return self.env.compute_reward(
            achieved_goal=achieved_goal, desired_goal=desired_goal, info=info, p=p
        )

    def send_agent_state(self, agent_state: Dict) -> None:
        self.agent_state = agent_state
