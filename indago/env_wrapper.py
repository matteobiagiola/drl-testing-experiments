from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np


class EnvWrapper(ABC):

    # no typing to avoid circular inputs when called from main
    def __init__(self, avf):
        self.avf = avf
        self.agent_state = None

    @abstractmethod
    def unwrap(self):
        assert NotImplementedError("Not implemented error")

    @abstractmethod
    def step(self, action: np.ndarray):
        assert NotImplementedError("Not implemented error")

    @abstractmethod
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        assert NotImplementedError("Not implemented error")

    @abstractmethod
    def reset(self, end_of_episode: bool = False):
        assert NotImplementedError("Not implemented error")

    @abstractmethod
    def send_agent_state(self, agent_state: Dict) -> None:
        assert NotImplementedError("Not implemented error")

    @abstractmethod
    def close(self) -> None:
        assert NotImplementedError("Not implemented error")
