from typing import Dict, List, Tuple, cast

import numpy as np

from indago.avf.env_configuration import EnvConfiguration
from indago.avf.training_logs import TrainingLogs
from indago.envs.donkey.donkey_env_configuration import DonkeyEnvConfiguration


class DonkeyTrainingLogs(TrainingLogs):
    def __init__(
        self,
        is_success: int,
        agent_state: Dict,
        actions: List[Tuple[float, float]],
        rewards: List[float],
        config: EnvConfiguration,
        car_trajectory: List[Tuple[float, float]] = None,
        images: List[str] = None,
        reconstruction_losses: List[float] = None,
        speeds: List[float] = None,
        steerings: List[float] = None,
        ctes: List[float] = None,
        is_success_testing: int = -1,
    ):
        super().__init__(agent_state=agent_state, config=config)
        self.is_success = is_success
        self.is_success_testing = is_success_testing
        self.config = config
        self.actions = actions
        self.rewards = rewards
        self.is_success = is_success
        self.car_trajectory = car_trajectory
        self.images = images
        self.reconstruction_losses = reconstruction_losses
        self.speeds = speeds
        self.steerings = steerings
        self.ctes = ctes
        # episode length
        self.regression_value = len(rewards)

    def to_dict(self) -> Dict:
        return {
            "is_success": self.is_success,
            "agent_state": self.agent_state,
            "env_config": self.config.get_str(),
            "actions": self.actions,
            "rewards": self.rewards,
            "speeds": self.speeds,
            "reconstruction_losses": self.reconstruction_losses,
            "steerings": self.steerings,
            "ctes": self.ctes,
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
        assert "training_progress" in self.agent_state, "Key training_progress not present in agent state: {}".format(
            self.agent_state
        )
        return self.agent_state["training_progress"]

    def get_image(self) -> np.ndarray:
        return self.config.get_image()

    def get_testing_image(self) -> np.ndarray:
        donkey_env_configuration = cast(DonkeyEnvConfiguration, self.config)
        assert len(self.car_trajectory) > 1, "Number of points in trajectory must be > 1. Found: {}".format(
            len(self.car_trajectory)
        )
        return donkey_env_configuration.get_testing_image(car_trajectory=self.car_trajectory)

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
        assert "ent_coef" in self.agent_state, "Key ent_coef not present in agent state: {}".format(self.agent_state)
        return self.agent_state["ent_coef"]

    def get_config(self) -> EnvConfiguration:
        return self.config
