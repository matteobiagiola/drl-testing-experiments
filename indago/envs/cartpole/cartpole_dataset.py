from typing import Dict, List, Optional, Tuple, cast

import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler

from indago.avf.dataset import Dataset
from indago.avf.env_configuration import EnvConfiguration
from indago.envs.cartpole.cartpole_env_configuration import CartPoleEnvConfiguration
from indago.type_aliases import Scaler


class CartPoleDataset(Dataset):
    def __init__(self, policy: str):
        super(CartPoleDataset, self).__init__(policy=policy)

    def get_mapping_transformed(self, env_configuration: EnvConfiguration) -> Dict:
        mapping = dict()
        mapping["x"] = [0]
        mapping["x_dot"] = [1]
        mapping["theta"] = [2]
        mapping["theta_dot"] = [3]
        return mapping

    @staticmethod
    def transform_mlp(env_configuration: EnvConfiguration) -> np.ndarray:
        cartpole_env_configuration = cast(CartPoleEnvConfiguration, env_configuration)

        # normalize
        transformed = np.asarray(
            [
                (cartpole_env_configuration.x - cartpole_env_configuration.low)
                / (cartpole_env_configuration.high - cartpole_env_configuration.low),
                (cartpole_env_configuration.x_dot - cartpole_env_configuration.low)
                / (cartpole_env_configuration.high - cartpole_env_configuration.low),
                (cartpole_env_configuration.theta - cartpole_env_configuration.low)
                / (cartpole_env_configuration.high - cartpole_env_configuration.low),
                (cartpole_env_configuration.theta_dot - cartpole_env_configuration.low)
                / (cartpole_env_configuration.high - cartpole_env_configuration.low),
            ]
        )

        return transformed

    def get_feature_names(self) -> List[str]:
        return ["masscart", "masspole", "length"]

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
        # unnormalize
        # x_norm = (x - x_min) / (x_max - x_min) ->
        # (x_max - x_min) * x_norm = x - x_min -> x = (x_max - x_min) * x_norm + x_min
        cartpole_env_configuration = CartPoleEnvConfiguration()
        cartpole_env_configuration.x = (
            env_config_transformed[0]
            * (cartpole_env_configuration.low - cartpole_env_configuration.high)
            + cartpole_env_configuration.low
        )
        cartpole_env_configuration.x_dot = (
            env_config_transformed[1]
            * (cartpole_env_configuration.low - cartpole_env_configuration.high)
            + cartpole_env_configuration.low
        )
        cartpole_env_configuration.theta = (
            env_config_transformed[2]
            * (cartpole_env_configuration.low - cartpole_env_configuration.high)
            + cartpole_env_configuration.low
        )
        cartpole_env_configuration.theta_dot = (
            env_config_transformed[3]
            * (cartpole_env_configuration.low - cartpole_env_configuration.high)
            + cartpole_env_configuration.low
        )
        return cartpole_env_configuration

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
