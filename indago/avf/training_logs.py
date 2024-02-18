from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from indago.avf.env_configuration import EnvConfiguration
from indago.avf.testing_logs import TestingLogs


class TrainingLogs(ABC):
    def __init__(self, agent_state: Dict, config: EnvConfiguration):
        self.agent_state = agent_state
        self.config = config

    @abstractmethod
    def to_dict(self) -> Dict:
        assert NotImplementedError("Not implemented error")

    def to_dict_test(self) -> Dict:
        return TestingLogs(
            config=self.config, dynamic_info=self.get_dynamic_episode_info().tolist()
        ).to_dict()

    @abstractmethod
    def is_agent_state_empty(self) -> bool:
        assert NotImplementedError("Not implemented error")

    @abstractmethod
    def get_label(self) -> int:
        assert NotImplementedError("Not implemented error")

    @abstractmethod
    def get_label_testing(self) -> int:
        assert NotImplementedError("Not implemented error")

    @abstractmethod
    def get_training_progress(self) -> float:
        assert NotImplementedError("Not implemented error")

    @abstractmethod
    def get_exploration_coefficient(self) -> float:
        assert NotImplementedError("Not implemented error")

    @abstractmethod
    def get_image(self) -> np.ndarray:
        assert NotImplementedError("Not implemented error")

    @abstractmethod
    def get_testing_image(self) -> np.ndarray:
        assert NotImplementedError("Not implemented error")

    @abstractmethod
    def get_dynamic_episode_info(self) -> np.ndarray:
        assert NotImplementedError("Not implemented error")

    @abstractmethod
    def get_regression_value(self) -> float:
        assert NotImplementedError("Not implemented error")

    @abstractmethod
    def is_regression_value_set(self) -> bool:
        assert NotImplementedError("Not implemented error")

    @abstractmethod
    def get_config(self) -> EnvConfiguration:
        assert NotImplementedError("Not implemented error")
