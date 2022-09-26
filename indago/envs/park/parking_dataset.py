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
        parking_env_configuration = cast(ParkingEnvConfiguration, env_configuration)
        parked_vehicles_one_hot = list(range(0, 2 * parking_env_configuration.num_lanes))
        mapping["parked_vehicles_lane_indices"] = parked_vehicles_one_hot
        mapping["goal_lane_idx"] = [len(parked_vehicles_one_hot)]
        mapping["heading_ego"] = [len(parked_vehicles_one_hot) + 1]
        mapping["position_ego"] = [len(parked_vehicles_one_hot) + 2, len(parked_vehicles_one_hot) + 3]
        return mapping

    @staticmethod
    def transform_mlp(env_configuration: EnvConfiguration) -> np.ndarray:
        parking_env_configuration = cast(ParkingEnvConfiguration, env_configuration)

        one_hot_parked_vehicles = np.zeros(shape=(2 * parking_env_configuration.num_lanes,))
        for i in range(len(parking_env_configuration.parked_vehicles_lane_indices)):
            idx = parking_env_configuration.parked_vehicles_lane_indices[i]
            one_hot_parked_vehicles[idx] = 1

        transformed = np.asarray(
            [
                *one_hot_parked_vehicles,
                parking_env_configuration.goal_lane_idx,
                parking_env_configuration.heading_ego,
                parking_env_configuration.position_ego[0],
                parking_env_configuration.position_ego[1],
            ]
        )
        return transformed

    def get_feature_names(self) -> List[str]:
        # bound to transform_mlp method, in particular the order
        data = self.dataset[0]
        env_config = cast(ParkingEnvConfiguration, data.training_logs.get_config())
        res = ["pv_{}".format(i) for i in range(2 * env_config.num_lanes)]
        res.append("goal_lane")
        res.append("h")
        res.append("pos_x")
        res.append("pos_y")
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

        input_scaler = MinMaxScaler()
        input_scaler.fit(X=data)
        # return input_scaler, None
        return None, None

    def get_original_env_configuration(self, env_config_transformed: np.ndarray) -> EnvConfiguration:
        parking_env_configuration = ParkingEnvConfiguration()
        parking_env_configuration.position_ego = (env_config_transformed[-2], env_config_transformed[-1])
        parking_env_configuration.heading_ego = env_config_transformed[-3]
        parking_env_configuration.goal_lane_idx = env_config_transformed[-4]
        one_hot_parked_vehicles = env_config_transformed[:-4]
        for i in range(len(one_hot_parked_vehicles)):
            if one_hot_parked_vehicles[i] == 1:
                parking_env_configuration.parked_vehicles_lane_indices.append(i)
        return parking_env_configuration

    def compute_distance(self, env_config_1: EnvConfiguration, env_config_2: EnvConfiguration) -> float:
        env_config_1_transformed = self.transform_env_configuration(env_configuration=env_config_1, policy=self.policy)
        env_config_2_transformed = self.transform_env_configuration(env_configuration=env_config_2, policy=self.policy)
        return euclidean(u=env_config_1_transformed, v=env_config_2_transformed)
