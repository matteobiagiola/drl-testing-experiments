from typing import Dict, List, Optional, Tuple, cast

import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from indago.avf.dataset import Dataset
from indago.avf.env_configuration import EnvConfiguration
from indago.config import COMMAND_NAME_VALUE_DICT, VALUE_COMMAND_NAME_DICT
from indago.envs.donkey.donkey_env_configuration import DonkeyEnvConfiguration
from indago.envs.donkey.track_generator.track_elem import TrackElem
from indago.envs.donkey.track_generator.unity.command import Command, parse_command
from indago.type_aliases import Scaler


class DonkeyDataset(Dataset):
    def __init__(self, policy: str):
        super(DonkeyDataset, self).__init__(policy=policy)

    def get_mapping_transformed(self, env_configuration: EnvConfiguration) -> Dict:
        mapping = dict()
        donkey_env_configuration = cast(DonkeyEnvConfiguration, env_configuration)
        mapping["commands"] = list(range(0, donkey_env_configuration.get_length()))
        mapping["values"] = list(range(donkey_env_configuration.get_length(), donkey_env_configuration.get_length() * 2))
        return mapping

    @staticmethod
    def transform_mlp(env_configuration: EnvConfiguration) -> np.ndarray:
        donkey_env_configuration = cast(DonkeyEnvConfiguration, env_configuration)
        track_elements = donkey_env_configuration.track_elements
        commands = np.zeros(shape=(donkey_env_configuration.get_length(),))
        values = np.zeros(shape=(donkey_env_configuration.get_length(),))
        for i in range(len(track_elements)):
            ce = track_elements[i]
            assert ce.command.name in COMMAND_NAME_VALUE_DICT, "Unknown command {}".format(ce.command.name)
            commands[i] = COMMAND_NAME_VALUE_DICT[ce.command.name]
            values[i] = ce.value
        return np.concatenate((commands, values), axis=0)

    def get_feature_names(self) -> List[str]:
        # bound to transform_mlp method, in particular the order
        data = self.dataset[0]
        env_config = cast(DonkeyEnvConfiguration, data.training_logs.get_config())
        res = ["command_{}".format(i) for i in range(len(env_config.track_elements))]
        res.extend(["value_{}".format(i) for i in range(len(env_config.track_elements))])
        return res

    @staticmethod
    def get_scalers_for_data(
        data: np.ndarray, labels: np.ndarray, regression: bool
    ) -> Tuple[Optional[Scaler], Optional[Scaler]]:
        if regression:
            # input_scaler = MinMaxScaler()
            # input_scaler.fit(X=data)
            output_scaler = MinMaxScaler()
            output_scaler.fit(X=labels)
            return None, output_scaler

        input_scaler = StandardScaler()
        input_scaler.fit(X=data)
        # return input_scaler, None
        return None, None

    def get_original_env_configuration(self, env_config_transformed: np.ndarray) -> EnvConfiguration:
        donkey_env_configuration = DonkeyEnvConfiguration()
        commands = env_config_transformed[: donkey_env_configuration.get_length()]
        values = env_config_transformed[donkey_env_configuration.get_length() : donkey_env_configuration.get_length() * 2]
        track_elements = []
        for i in range(donkey_env_configuration.get_length()):
            ce = track_elements[i]
            command, _ = parse_command(command_name=VALUE_COMMAND_NAME_DICT[commands[i]], command_value="dummy")
            value = values[i]
            track_elements.append(TrackElem(command=command, value=value))
        return DonkeyEnvConfiguration(track_elements=track_elements)

    def compute_distance(self, env_config_1: EnvConfiguration, env_config_2: EnvConfiguration) -> float:
        env_config_1_transformed = self.transform_env_configuration(env_configuration=env_config_1, policy=self.policy)
        env_config_2_transformed = self.transform_env_configuration(env_configuration=env_config_2, policy=self.policy)
        return euclidean(u=env_config_1_transformed, v=env_config_2_transformed)
