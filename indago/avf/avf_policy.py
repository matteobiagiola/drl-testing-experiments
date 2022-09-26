import abc
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from indago.avf.dataset import Dataset
from indago.utils.torch_utils import DEVICE, to_numpy


def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = th.mean(loss)
    return loss


def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction="none")
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = th.mean(loss)
    return loss


def weighted_focal_mse_loss(inputs, targets, activate="sigmoid", beta=0.2, gamma=1, weights=None):
    loss = (inputs - targets) ** 2
    loss *= (
        (th.tanh(beta * th.abs(inputs - targets))) ** gamma
        if activate == "tanh"
        else (2 * th.sigmoid(beta * th.abs(inputs - targets)) - 1) ** gamma
    )
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = th.mean(loss)
    return loss


def weighted_focal_l1_loss(inputs, targets, activate="sigmoid", beta=0.2, gamma=1, weights=None):
    loss = F.l1_loss(inputs, targets, reduction="none")
    loss *= (
        (th.tanh(beta * th.abs(inputs - targets))) ** gamma
        if activate == "tanh"
        else (2 * th.sigmoid(beta * th.abs(inputs - targets)) - 1) ** gamma
    )
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = th.mean(loss)
    return loss


def weighted_huber_loss(inputs, targets, beta=1.0, weights=None):
    l1_loss = th.abs(inputs - targets)
    cond = l1_loss < beta
    loss = th.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = th.mean(loss)
    return loss


class AvfPolicy(nn.Module, metaclass=abc.ABCMeta):
    def __init__(
        self,
        loss_type: str,
        input_size: int,
        layers: int,
        avf_policy: str,
        regression: bool = False,
        learning_rate: float = 3e-4,
    ):
        super(AvfPolicy, self).__init__()
        self.loss_type = loss_type
        self.regression = regression
        self.num_evaluation_predictions = 0
        self.current_evaluation_predictions = 0
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.layers = layers
        self.avf_policy = avf_policy

        assert (
            self.loss_type == "classification" or self.loss_type == "anomaly_detection"
        ), "Loss type {} not supported".format(self.loss_type)

    @staticmethod
    def get_model_architecture(
        input_size: int, layers: int, avf_policy: str, regression: bool = False
    ) -> Callable[[], nn.Module]:
        def __init_model() -> nn.Module:
            if avf_policy == "mlp":
                models = [
                    # 1 layer nn
                    nn.Sequential(
                        nn.Linear(in_features=input_size, out_features=32),
                        nn.BatchNorm1d(num_features=32),
                        nn.ReLU(),
                        nn.Linear(in_features=32, out_features=2)
                        if not regression
                        else nn.Linear(in_features=32, out_features=1),
                    ),
                    # 2 layers nn
                    nn.Sequential(
                        nn.Linear(in_features=input_size, out_features=32),
                        nn.BatchNorm1d(num_features=32),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(in_features=32, out_features=32),
                        nn.BatchNorm1d(num_features=32),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(in_features=32, out_features=2)
                        if not regression
                        else nn.Linear(in_features=32, out_features=1),
                    ),
                    # 3 layers nn
                    nn.Sequential(
                        nn.Linear(in_features=input_size, out_features=32),
                        nn.BatchNorm1d(num_features=32),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(in_features=32, out_features=32),
                        nn.BatchNorm1d(num_features=32),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(in_features=32, out_features=32),
                        nn.BatchNorm1d(num_features=32),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(in_features=32, out_features=2)
                        if not regression
                        else nn.Linear(in_features=32, out_features=1),
                    ),
                    # 4 layers nn
                    nn.Sequential(
                        nn.Linear(in_features=input_size, out_features=32),
                        nn.BatchNorm1d(num_features=32),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(in_features=32, out_features=32),
                        nn.BatchNorm1d(num_features=32),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(in_features=32, out_features=32),
                        nn.BatchNorm1d(num_features=32),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(in_features=32, out_features=32),
                        nn.BatchNorm1d(num_features=32),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(in_features=32, out_features=2)
                        if not regression
                        else nn.Linear(in_features=32, out_features=1),
                    ),
                ]
                assert 0 <= layers - 1 < len(models)
                return models[layers - 1]
            raise NotImplementedError("avf_policy {} not supported".format(avf_policy))

        return __init_model

    @staticmethod
    def loss_function(input: Tensor, target: Tensor, regression: bool = False, weights: Tensor = None) -> Tensor:
        if not regression:
            if weights is not None:
                weights = weights[0].view(len(weights[0]))
            return F.cross_entropy(input=input, target=target, weight=weights)
        if weights is not None:
            weights = weights.view(len(weights))
            return weighted_mse_loss(inputs=input, targets=target, weights=weights)
        return F.mse_loss(input=input, target=target)

    def save(self, filepath: str) -> None:
        th.save(self.state_dict(), filepath)

    def load(self, filepath: str, load_on_device: bool = False, save_paths: List[str] = None) -> None:
        if load_on_device:
            self.load_state_dict(th.load(filepath, map_location=th.device(DEVICE)))
        else:
            self.load_state_dict(th.load(filepath, map_location=th.device("cpu")))
        self.eval()

    @abc.abstractmethod
    def get_model(self) -> nn.Module:
        raise NotImplementedError("Get model not implemented")

    def generate(self, data: Tensor, training: bool = True) -> Tensor:
        raise NotImplementedError("Generate not implemented")

    @staticmethod
    def compute_score(logits: Tensor) -> Tensor:
        # select failure class
        return F.softmax(logits, dim=1).squeeze()[:, 1]

    def forward(self, data):
        return self.model.forward(data)

    def forward_and_loss(
        self, data: Tensor, target: Tensor, training: bool = True, weights: Tensor = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if not self.regression:
            if self.loss_type == "classification":
                output = self.forward(data)
                predictions = F.softmax(output, dim=1).detach().argmax(dim=1, keepdim=True).squeeze()
                if training:
                    return self.loss_function(input=output, target=target, weights=weights), predictions
                return output.detach(), predictions
            else:
                raise NotImplementedError("Loss type {} is not implemented".format(self.loss_type))

        predictions = th.squeeze(self.forward(data))
        if training:
            return self.loss_function(input=predictions, target=target, weights=weights, regression=True), predictions
        else:
            return predictions

    def get_failure_class_prediction(
        self, env_config_transformed: np.ndarray, dataset: Dataset, count_num_evaluation: bool = True,
    ) -> float:

        if count_num_evaluation:
            self.num_evaluation_predictions += 1
            self.current_evaluation_predictions += 1

        output = self.forward(th.tensor(env_config_transformed, dtype=th.float32).view(1, -1))

        if self.regression:
            output = to_numpy(output).squeeze()
            if dataset.output_scaler is not None:
                output = dataset.output_scaler.transform(X=output.reshape(1, -1)).squeeze()
            return output

        output = to_numpy(F.softmax(output, dim=1))
        if dataset.output_scaler is not None:
            output = dataset.output_scaler.transform(X=output)
        return output.squeeze()[1]

    def get_num_evaluation_predictions(self) -> int:
        return self.num_evaluation_predictions

    def get_current_num_evaluation_predictions(self) -> int:
        return self.current_evaluation_predictions

    def reset_current_num_evaluation_predictions(self) -> None:
        self.current_evaluation_predictions = 0
