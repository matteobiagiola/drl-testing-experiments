from typing import Dict, List, Optional, Tuple, cast

import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from indago.avf.dataset import Dataset
from indago.avf.env_configuration import EnvConfiguration
from indago.envs.humanoid.humanoid_env_configuration import HumanoidEnvConfiguration
from indago.type_aliases import Scaler


class HumanoidDataset(Dataset):
    def __init__(self, policy: str):
        super(HumanoidDataset, self).__init__(policy=policy)

    def get_mapping_transformed(self, env_configuration: EnvConfiguration) -> Dict:
        mapping = dict()
        humanoid_env_configuration = cast(HumanoidEnvConfiguration, env_configuration)
        mapping["qpos"] = list(range(0, len(humanoid_env_configuration.qpos)))
        mapping["qvel"] = list(
            range(
                len(humanoid_env_configuration.qpos),
                len(humanoid_env_configuration.qpos) + len(humanoid_env_configuration.qvel),
            )
        )
        return mapping

    @staticmethod
    def transform_mlp(env_configuration: EnvConfiguration) -> np.ndarray:
        humanoid_env_configuration = cast(HumanoidEnvConfiguration, env_configuration)
        return np.concatenate((humanoid_env_configuration.qpos, humanoid_env_configuration.qvel), axis=0)

    def get_feature_names(self) -> List[str]:
        # bound to transform_mlp method, in particular the order
        data = self.dataset[0]
        env_config = cast(HumanoidEnvConfiguration, data.training_logs.get_config())
        res = ["qpos_{}".format(i) for i in range(len(env_config.qpos))]
        res.extend(["qvel_{}".format(i) for i in range(len(env_config.qvel))])
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
        humanoid_env_configuration = HumanoidEnvConfiguration()
        humanoid_env_configuration.qpos = env_config_transformed[: len(humanoid_env_configuration.init_qpos)]
        humanoid_env_configuration.qvel = env_config_transformed[
            len(humanoid_env_configuration.init_qpos) : len(humanoid_env_configuration.init_qvel)
        ]
        return humanoid_env_configuration

    def compute_distance(self, env_config_1: EnvConfiguration, env_config_2: EnvConfiguration) -> float:
        env_config_1_transformed = self.transform_env_configuration(env_configuration=env_config_1, policy=self.policy)
        env_config_2_transformed = self.transform_env_configuration(env_configuration=env_config_2, policy=self.policy)
        return euclidean(u=env_config_1_transformed, v=env_config_2_transformed)
