import base64
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from indago.avf.env_configuration import EnvConfiguration
from indago.avf.training_logs import TrainingLogs


class ParkingTrainingLogs(TrainingLogs):
    def __init__(
        self,
        is_success: int,
        agent_state: Dict,
        first_frame_string: str,
        fitness_values: List[float],
        actions: List[Tuple[float, float]],
        rewards: List[float],
        speeds: List[float],
        car_trajectory: List[List[float]],
        config: EnvConfiguration,
        goal_position: Tuple[float] = None,
        parked_vehicle_positions: List = None,
        is_success_testing: int = -1,
    ):
        super().__init__(agent_state=agent_state, config=config)
        self.is_success = is_success
        self.is_success_testing = is_success_testing
        self.first_frame_string = first_frame_string
        self.config = config
        self.actions = actions
        self.rewards = rewards
        self.speeds = speeds
        self.fitness_values = fitness_values
        self.car_trajectory = car_trajectory
        # episode length
        self.regression_value = (
            len(rewards) if fitness_values is None else np.min(fitness_values)
        )
        self.goal_position = goal_position
        self.parked_vehicle_positions = parked_vehicle_positions

    def to_dict(self) -> Dict:
        return {
            "is_success": self.is_success,
            "agent_state": self.agent_state,
            "first_frame_string": self.first_frame_string,
            "env_config": self.config.impl,
            "fitness_values": self.fitness_values,
            "actions": self.actions,
            "rewards": self.rewards,
            "speeds": self.speeds,
            "car_trajectory": self.car_trajectory,
        }

    def get_label(self) -> int:
        # failure class is 1
        return not self.is_success

    def get_label_testing(self) -> int:
        return self.is_success_testing

    def is_agent_state_empty(self) -> bool:
        return len(self.agent_state.keys()) == 0

    def get_training_progress(self) -> float:
        if self.is_agent_state_empty():
            return 0.0
        assert (
            "training_progress" in self.agent_state
        ), "Key training_progress not present in agent state: {}".format(
            self.agent_state
        )
        return self.agent_state["training_progress"]

    def get_image(self) -> np.ndarray:
        image_decoded = Image.open(BytesIO(base64.b64decode(self.first_frame_string)))
        return np.asarray(image_decoded)

    def get_testing_image(self) -> np.ndarray:
        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        ax.axis("off")
        ax.scatter(
            self.config.impl["position_ego"][0],
            self.config.impl["position_ego"][1],
            c="black",
            s=200,
        )
        car_trajectory_x = [
            car_trajectory_item[0] for car_trajectory_item in self.car_trajectory
        ]
        car_trajectory_y = [
            car_trajectory_item[1] for car_trajectory_item in self.car_trajectory
        ]
        ax.scatter(car_trajectory_x, car_trajectory_y, c="blue", s=50)
        if self.goal_position is not None:
            ax.scatter(self.goal_position[0], self.goal_position[1], c="red", s=200)
        if self.parked_vehicle_positions is not None:
            for i in range(len(self.parked_vehicle_positions)):
                ax.scatter(
                    self.parked_vehicle_positions[i][0],
                    self.parked_vehicle_positions[i][1],
                    c="green",
                    s=200,
                )

        canvas.draw()
        buf = canvas.buffer_rgba()
        position_trajectory_image_array = np.asarray(buf).astype(np.uint8)

        return position_trajectory_image_array

    def get_dynamic_episode_info(self) -> np.ndarray:
        return np.asarray(self.car_trajectory)

    def get_regression_value(self) -> float:
        assert self.regression_value is not None, "Regression value not set yet"
        return self.regression_value

    def is_regression_value_set(self) -> bool:
        return self.regression_value is not None

    def get_exploration_coefficient(self) -> float:
        if self.is_agent_state_empty():
            return 0.0
        assert (
            "ent_coef" in self.agent_state
        ), "Key ent_coef not present in agent state: {}".format(self.agent_state)
        return self.agent_state["ent_coef"]

    def get_config(self) -> EnvConfiguration:
        return self.config
