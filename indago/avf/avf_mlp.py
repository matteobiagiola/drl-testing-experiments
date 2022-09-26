import torch.nn as nn
from torch import Tensor

from indago.avf.avf_policy import AvfPolicy
from indago.config import DONKEY_ENV_NAME, HUMANOID_ENV_NAME, PARK_ENV_NAME
from indago.utils.torch_utils import DEVICE


class AvfMlpPolicy(AvfPolicy):
    def __init__(
        self, env_name: str, input_size: int, regression: bool = False, layers: int = 4, learning_rate: float = 3e-4
    ) -> None:
        super(AvfMlpPolicy, self).__init__(
            loss_type="classification",
            regression=regression,
            learning_rate=learning_rate,
            input_size=input_size,
            layers=layers,
            avf_policy="mlp",
        )
        if env_name == PARK_ENV_NAME:
            self.model = self._get_park_env_mlp_architecture(input_size=input_size, regression=regression, layers=layers)
        elif env_name == HUMANOID_ENV_NAME:
            self.model = self._get_humanoid_env_mlp_architecture(input_size=input_size, regression=regression, layers=layers)
        elif env_name == DONKEY_ENV_NAME:
            self.model = self._get_donkey_env_mlp_architecture(input_size=input_size, regression=regression, layers=layers)
        else:
            raise NotImplementedError("Unknown env name: {}".format(env_name))
        self.model.to(device=DEVICE)

    def get_model(self) -> nn.Module:
        return self.model

    def generate(self, data: Tensor, training: bool = True) -> Tensor:
        raise NotImplementedError("Generate not implemented")

    def _get_park_env_mlp_architecture(self, input_size: int, regression: bool = False, layers: int = 0) -> nn.Module:
        return self.get_model_architecture(input_size=input_size, layers=layers, avf_policy="mlp", regression=regression)()

    def _get_humanoid_env_mlp_architecture(self, input_size: int, regression: bool = False, layers: int = 4) -> nn.Module:
        return self.get_model_architecture(input_size=input_size, layers=layers, avf_policy="mlp", regression=regression)()

    def _get_donkey_env_mlp_architecture(self, input_size: int, regression: bool = False, layers: int = 4) -> nn.Module:
        return self.get_model_architecture(input_size=input_size, layers=layers, avf_policy="mlp", regression=regression)()
