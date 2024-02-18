import base64
from io import BytesIO
from typing import Dict, List, Tuple, cast

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from indago.avf.env_configuration import EnvConfiguration
from indago.avf.training_logs import TrainingLogs


class HumanoidTrainingLogs(TrainingLogs):
    def __init__(
        self,
        is_success: int,
        agent_state: Dict,
        config: EnvConfiguration,
        fitness_values: List[float] = None,
        state: np.ndarray = None,
        abdomen_trajectory: List[float] = None,
        first_frame_string: str = None,
        is_success_testing: int = -1,
    ):
        super().__init__(agent_state=agent_state, config=config)
        self.is_success = is_success
        self.is_success_testing = is_success_testing
        # FIXME: if first_frame_string is None then get it by instantiating the environment
        self.first_frame_string = first_frame_string
        self.config = config
        self.fitness_values = fitness_values
        self.state = state
        self.abdomen_trajectory = abdomen_trajectory
        # episode length
        self.regression_value = (
            None
            if fitness_values is None or len(fitness_values) == 0
            else min(fitness_values)
        )

    def to_dict(self) -> Dict:
        config = dict()
        config["qpos"] = self.config.impl["qpos"].tolist()
        config["qvel"] = self.config.impl["qvel"].tolist()
        return {
            "is_success": self.is_success,
            "fitness_values": (
                self.fitness_values if self.fitness_values is not None else []
            ),
            "agent_state": self.agent_state,
            "env_config": config,
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
        if self.first_frame_string is not None:
            image_decoded = Image.open(
                BytesIO(base64.b64decode(self.first_frame_string))
            )
            return np.asarray(image_decoded)
        return None

    def get_testing_image(self) -> np.ndarray:
        assert (
            len(self.abdomen_trajectory) > 1
        ), "Number of points in trajectory must be > 1. Found: {}".format(
            len(self.abdomen_trajectory)
        )
        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        ax.plot(np.ones(shape=len(self.abdomen_trajectory)), c="blue")
        ax.plot(self.abdomen_trajectory, c="black")
        ax.plot(np.ones(shape=len(self.abdomen_trajectory)) * 2, c="blue")

        canvas.draw()
        buf = canvas.buffer_rgba()
        image_array = np.asarray(buf)
        image_array = image_array.astype(np.float32)

        return image_array

    def get_dynamic_episode_info(self) -> np.ndarray:
        return np.asarray(self.abdomen_trajectory)

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
