from typing import Dict, List, Optional, Tuple, cast

import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler

from indago.avf.dataset import Dataset
from indago.avf.env_configuration import EnvConfiguration
from indago.envs.park.parking_env_configuration import ParkingEnvConfiguration
from indago.type_aliases import Scaler


class ParkingDataset(Dataset):
    def __init__(self, policy: str):
        super(ParkingDataset, self).__init__(policy=policy)

    def get_mapping_transformed(self, env_configuration: EnvConfiguration) -> Dict:
        mapping = dict()
        mapping = dict()
        mapping["goal_lane_idx"] = [0]
        mapping["heading_ego"] = [1]
        mapping["position_ego"] = [2, 3]
        return mapping

    @staticmethod
    def transform_mlp(env_configuration: EnvConfiguration) -> np.ndarray:
        parking_env_configuration = cast(ParkingEnvConfiguration, env_configuration)

        transformed = np.asarray(
            [
                parking_env_configuration.goal_lane_idx,
                parking_env_configuration.heading_ego,
                parking_env_configuration.position_ego[0],
                parking_env_configuration.position_ego[1],
            ]
        )
        return transformed

    def get_feature_names(self) -> List[str]:
        res = ["goal_lane", "h", "pos_x", "pos_y"]
        return res

    @staticmethod
    def get_scalers_for_data(
        data: np.ndarray, labels: np.ndarray, regression: bool
    ) -> Tuple[Optional[Scaler], Optional[Scaler]]:
        if regression:
            output_scaler = MinMaxScaler()
            output_scaler.fit(X=labels)
            return None, output_scaler

        input_scaler = MinMaxScaler()
        input_scaler.fit(X=data)
        return None, None

    def get_original_env_configuration(
        self, env_config_transformed: np.ndarray
    ) -> EnvConfiguration:
        parking_env_configuration = ParkingEnvConfiguration()
        parking_env_configuration.position_ego = (
            env_config_transformed[-2],
            env_config_transformed[-1],
        )
        parking_env_configuration.heading_ego = env_config_transformed[-3]
        parking_env_configuration.goal_lane_idx = env_config_transformed[-4]
        return parking_env_configuration

    def compute_distance(
        self, env_config_1: EnvConfiguration, env_config_2: EnvConfiguration
    ) -> float:
        env_config_1_transformed = self.transform_env_configuration(
            env_configuration=env_config_1, policy=self.policy
        )
        env_config_2_transformed = self.transform_env_configuration(
            env_configuration=env_config_2, policy=self.policy
        )
        return euclidean(u=env_config_1_transformed, v=env_config_2_transformed)
