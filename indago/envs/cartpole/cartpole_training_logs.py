import base64
from io import BytesIO
from typing import Dict, List, Union, Optional

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from indago.avf.env_configuration import EnvConfiguration
from indago.avf.training_logs import TrainingLogs


class CartPoleTrainingLogs(TrainingLogs):
    def __init__(
        self,
        is_success: int,
        fitness_values: List[float],
        agent_state: Dict,
        first_frame_string: str,
        actions: List[Union[float, int]],
        rewards: List[float],
        cart_positions: List[float],
        cart_angles: List[float],
        config: EnvConfiguration,
        is_success_testing: int = -1,
    ):
        super().__init__(agent_state=agent_state, config=config)
        self.is_success = is_success
        self.is_success_testing = is_success_testing
        self.first_frame_string = first_frame_string
        self.config = config
        self.actions = actions
        self.rewards = rewards
        self.cart_positions = cart_positions
        self.cart_angles = cart_angles
        self.fitness_values = fitness_values
        if len(fitness_values) > 0:
            fitness_values_arr = np.asarray(fitness_values)
            num_zeros = len(fitness_values_arr[fitness_values_arr == 0.0])
            assert num_zeros <= 1, (
                f"Number of 0.0 in fitness values cannot be > 1. Found: {num_zeros}. "
                f"Fitness values: {fitness_values}"
            )

        self.regression_value = (
            len(rewards) if len(self.fitness_values) == 0 else min(self.fitness_values)
        )

    def to_dict(self) -> Dict:
        return {
            "is_success": self.is_success,
            "fitness_values": self.fitness_values
            if len(self.fitness_values) > 0
            else None,
            "agent_state": self.agent_state,
            "first_frame_string": self.first_frame_string,
            "env_config": self.config.impl,
            "actions": self.actions,
            "rewards": self.rewards,
            "cart_positions": self.cart_positions,
            "cart_angles": self.cart_angles,
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

    def get_image(self) -> Optional[np.ndarray]:
        if self.first_frame_string is not None:
            image_decoded = Image.open(
                BytesIO(base64.b64decode(self.first_frame_string))
            )
            return np.asarray(image_decoded)
        return None

    def get_testing_image(self) -> np.ndarray:
        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        ax_1 = fig.add_subplot(1, 2, 1)

        ax_1.plot(self.cart_positions, c="black", label="Cart Position", linewidth=2)
        ax_1.plot(np.ones(shape=len(self.cart_positions)) * 2.4, c="blue", linewidth=3)
        ax_1.plot(np.ones(shape=len(self.cart_positions)) * -2.4, c="blue", linewidth=3)
        ax_1.legend()

        ax_2 = fig.add_subplot(1, 2, 2)

        ax_2.plot(
            np.degrees(self.cart_angles), c="red", label="Cart Angle", linewidth=2
        )
        ax_2.plot(np.ones(shape=len(self.cart_angles)) * 12, c="blue", linewidth=3)
        ax_2.plot(np.ones(shape=len(self.cart_angles)) * -12, c="blue", linewidth=3)
        ax_2.legend()

        canvas.draw()
        buf = canvas.buffer_rgba()
        image_array = np.asarray(buf)
        image_array = image_array.astype(np.float32)

        return image_array

    def get_dynamic_episode_info(self) -> np.ndarray:
        return np.asarray(self.cart_positions)

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
