from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from torch.utils.data import Dataset as ThDataset

from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang

from indago.avf.env_configuration import EnvConfiguration
from indago.avf.training_logs import TrainingLogs
from indago.type_aliases import Scaler
from indago.utils.torch_utils import DEVICE
from log import Log


class Data:
    def __init__(
        self, filename: str, training_logs: TrainingLogs, regression: bool = False
    ):
        self.filename = filename
        self.training_logs = training_logs
        self.training_progress = self.training_logs.get_training_progress()
        self.exploration_coef = self.training_logs.get_exploration_coefficient()
        self.label = self.training_logs.get_label()
        self.regression_value = None
        if regression:
            assert (
                self.training_logs.is_regression_value_set()
            ), f"Regression value for {training_logs} not set"
            self.regression_value = self.training_logs.get_regression_value()

    def __lt__(self, other: "Data"):
        # return np.max(self.reconstruction_losses) < np.max(other.reconstruction_losses)
        return self.training_progress < other.training_progress


# defining the Dataset class
class TorchDataset(ThDataset):
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        regression: bool = False,
        weight_loss: bool = False,
    ):
        self.data = data
        self.labels = labels
        self.regression = regression
        self.weight_loss = weight_loss
        self.weights = None
        self.logger = Log("TorchDataset")
        if len(self.labels) > 0:
            if not self.regression:
                if self.weight_loss:
                    hist = dict(Counter(self.labels))
                    n_classes = len(hist)
                    self.weights = list(
                        compute_class_weight(
                            class_weight="balanced",
                            classes=np.arange(n_classes),
                            y=self.labels,
                        )
                    )
                else:
                    hist = dict(Counter(self.labels))
                    self.weights = [np.float32(1.0) for _ in range(len(hist.keys()))]

                self.logger.info(f"Classification weights: {self.weights}")
            else:
                if self.weight_loss:
                    # FIXME: add check to verify that the fitness goes from 0 to 1
                    self.weights = self.compute_weights_regression(
                        labels=self.labels, ks=1, sigma=1
                    )
                else:
                    self.weights = [np.float32(1.0) for _ in range(len(self.labels))]

                self.logger.info(f"Regression weights: {self.weights}")

    @staticmethod
    def get_bin_edges(num_bins: int = 10) -> np.ndarray:
        return np.linspace(start=0.0, stop=1.0, num=num_bins + 1)

    @staticmethod
    def get_bin_indexes(labels: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
        assert (
            labels.dtype == "float64" or labels.dtype == "float32"
        ), f"Type {labels.dtype} not supported"
        return np.digitize(x=labels, bins=bin_edges).flatten() - 1

    @staticmethod
    def get_custom_bin_indexes(labels: np.ndarray) -> np.ndarray:
        """
        # TODO: make it more efficient
        returns indices based on fitness, i.e., failures and near-failures on one bin while the others on another bin
        """
        assert (
            labels.dtype == "float64" or labels.dtype == "float32"
        ), f"Type {labels.dtype} not supported"
        assert (
            min(labels) >= 0.0 and max(labels) <= 1.0
        ), "Labels not supported. Values should be in [0, 1]"
        # I am assuming that the labels passed here are split evenly between train and validation/test set
        near_failure_percentage = 10 * max(labels) / 100
        return np.asarray(
            [0 if label <= near_failure_percentage else 1 for label in labels]
        )

    @staticmethod
    def get_lds_kernel_window(kernel: str, ks: float, sigma: float) -> List:
        assert kernel in ["gaussian", "triang", "laplace"]
        half_ks = (ks - 1) // 2
        if kernel == "gaussian":
            base_kernel = [0.0] * half_ks + [1.0] + [0.0] * half_ks
            kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(
                gaussian_filter1d(base_kernel, sigma=sigma)
            )
        elif kernel == "triang":
            kernel_window = triang(ks)
        else:

            def laplace(x):
                return np.exp(-abs(x) / sigma) / (2.0 * sigma)

            kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(
                map(laplace, np.arange(-half_ks, half_ks + 1))
            )

        return kernel_window

    def compute_weights_regression(
        self, labels: np.ndarray, ks: int = 5, sigma: int = 2
    ) -> np.ndarray:

        # assign each label to its corresponding bin (start from 0)
        # with your defined get_bin_idx(), return bin_index_per_label: [Ns,]
        bin_index_per_label = self.get_bin_indexes(
            labels=labels, bin_edges=self.get_bin_edges()
        )
        if (
            len(
                list(
                    filter(
                        lambda value: value == 1,
                        list(Counter(bin_index_per_label).values()),
                    )
                )
            )
            > 0
        ):
            print(
                "WARN: there is at least one bin with one sample only. Falling back to two bins."
            )
            bin_index_per_label = self.get_custom_bin_indexes(labels=labels)

        # calculate empirical (original) label distribution: [Nb,]
        # "Nb" is the number of bins
        Nb = max(bin_index_per_label) + 1
        num_samples_of_bins = dict(Counter(bin_index_per_label))

        if len(list(num_samples_of_bins.keys())) == 2:
            self.logger.warn(
                f"Since there only two bins, overriding ks {ks} and sigma {sigma} to 1, i.e., no smoothing"
            )
            ks = 1
            sigma = 1

        emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]

        # lds_kernel_window: [ks,], here for example, we use gaussian, ks=5, sigma=2
        lds_kernel_window = self.get_lds_kernel_window(
            kernel="gaussian", ks=ks, sigma=sigma
        )
        # calculate effective label distribution: [Nb,]
        effective_label_distribution = convolve1d(
            np.array(emp_label_dist), weights=lds_kernel_window, mode="constant"
        )
        assert (
            max(effective_label_distribution) > 0
        ), f"The most representative bin cannot have an effective label distribution of 0: {effective_label_distribution}"
        # we take the inverse to have the weights of each bin
        return 1 / (effective_label_distribution / max(effective_label_distribution))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if not self.regression:
            return (
                th.tensor(self.data[index], dtype=th.float32).to(DEVICE),
                th.tensor(self.labels[index], dtype=th.long).to(DEVICE),
                th.tensor(self.weights, dtype=th.float32).to(DEVICE),
            )

        if self.weight_loss:
            return (
                th.tensor(self.data[index], dtype=th.float32).to(DEVICE),
                th.tensor(self.labels[index], dtype=th.float32).to(DEVICE),
                th.tensor(self.weights, dtype=th.float32).to(DEVICE),
            )

        weights = np.asarray([self.weights[index]]).astype("float32")

        return (
            th.tensor(self.data[index], dtype=th.float32).to(DEVICE),
            th.tensor(self.labels[index], dtype=th.float32).to(DEVICE),
            th.tensor(weights, dtype=th.float32).to(DEVICE),
        )


class Dataset(ABC):
    def __init__(self, policy: str = None):
        self.dataset: List[Data] = []
        self.input_scaler: Scaler = None
        self.output_scaler: Scaler = None
        self.policy = policy
        self.logger = Log("dataset")

    def add(self, data: Data) -> None:
        self.dataset.append(data)

    def get(self) -> List[Data]:
        return self.dataset

    def get_num_failures(self) -> int:
        assert (
            len(self.dataset) != 0
        ), "Cannot compute num failures since the dataset is empty"
        return sum([1 if data_item.label == 1 else 0 for data_item in self.get()])

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        raise NotImplementedError("Not implemented")

    def get_num_features(self) -> int:
        assert (
            len(self.dataset) > 0
        ), "Not possible to infer num features since there is no data point"
        assert self.policy is not None, "Policy not instantiated"
        if self.policy == "mlp":
            data_item = self.dataset[0]
            return len(
                self.transform_mlp(
                    env_configuration=data_item.training_logs.get_config()
                )
            )
        # TODO: add cnn, i.e. number of channels
        raise NotImplementedError("Unknown policy: {}".format(self.policy))

    def transform_data_item(self, data_item: Data) -> np.ndarray:
        return self.transform_env_configuration(
            env_configuration=data_item.training_logs.get_config(), policy=self.policy
        )

    def transform_env_configuration(
        self,
        env_configuration: EnvConfiguration,
        policy: str,
    ) -> np.ndarray:
        assert self.policy is not None, "Policy not instantiated"
        if policy == "mlp":
            transformed = self.transform_mlp(env_configuration=env_configuration)
            if self.input_scaler is not None:
                transformed = self.input_scaler.transform(
                    X=transformed.reshape(1, -1)
                ).squeeze()
            return transformed
        raise NotImplementedError("Unknown policy: {}".format(policy))

    @staticmethod
    def transform_mlp(env_configuration: EnvConfiguration) -> np.ndarray:
        raise NotImplementedError("Transform mlp not implemented")

    @abstractmethod
    def get_mapping_transformed(self, env_configuration: EnvConfiguration) -> Dict:
        raise NotImplementedError("Get mapping transformed not implemented")

    @abstractmethod
    def get_original_env_configuration(
        self, env_config_transformed: np.ndarray
    ) -> EnvConfiguration:
        raise NotImplementedError("Get original env configuration not implemented")

    @staticmethod
    def get_scalers_for_data(
        data: np.ndarray, labels: np.ndarray, regression: bool
    ) -> Tuple[Optional[Scaler], Optional[Scaler]]:
        raise NotImplementedError("Get scalers for data not implemented")

    @abstractmethod
    def compute_distance(
        self, env_config_1: EnvConfiguration, env_config_2: EnvConfiguration
    ) -> float:
        raise NotImplementedError("Compute distance not implemented")

    @staticmethod
    def sampling(
        data: np.ndarray,
        labels: np.ndarray,
        seed: int,
        under: bool = False,
        sampling_percentage: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:

        logger = Log("sampling")

        if sampling_percentage > 0.0:
            if under:
                sampler = RandomUnderSampler(
                    sampling_strategy=sampling_percentage, random_state=seed
                )
            else:
                sampler = RandomOverSampler(
                    sampling_strategy=sampling_percentage, random_state=seed
                )

            logger.info("Label proportions before sampling: {}".format(labels.mean()))
            sampled_data, sampled_labels = sampler.fit_resample(X=data, y=labels)
            logger.info(
                "Label proportions after sampling: {}".format(sampled_labels.mean())
            )

            return sampled_data, sampled_labels

        return data, labels

    @staticmethod
    def get_regression_labels_from_split(
        split_data: np.ndarray,
        whole_data: np.ndarray,
        continuous_labels: np.ndarray,
    ) -> np.ndarray:

        labels = np.zeros(shape=(len(split_data), 1), dtype=np.float32)

        # TODO: find way to do it more efficiently, with numpy-only operations
        for i, split_data_item in enumerate(split_data):
            # Use boolean indexing to find the row(s) in the matrix that match the input_array
            matching_rows = (whole_data == split_data_item).all(axis=1)

            assert (
                len(matching_rows[matching_rows is True]) == 1
            ), f"There should be only one match for {split_data_item} in {whole_data}. Found: {len(matching_rows[matching_rows == True])}"

            # Get the index of the matching row
            row_index = np.where(matching_rows)[0]

            labels[i] = continuous_labels[row_index]

        return labels

    # @staticmethod
    def split_train_test(
        self,
        test_split: float,
        data: np.ndarray,
        labels: np.ndarray,
        seed: int,
        oversample_minority_class_percentage: float = 0.0,
        regression: bool = False,
        shuffle: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        logger = Log("split_train_test")

        if test_split > 0.0:
            if not regression:
                train_data, test_data, train_labels, test_labels = train_test_split(
                    data,
                    labels,
                    test_size=test_split,
                    shuffle=shuffle,
                    stratify=labels,
                    random_state=seed,
                )
            else:
                regression_values_between_zero_and_one = (
                    min(labels) >= 0.0 and max(labels) <= 1.0
                )
                if regression_values_between_zero_and_one:

                    value_error = True
                    binned_labels = TorchDataset.get_bin_indexes(
                        labels=labels, bin_edges=TorchDataset.get_bin_edges()
                    )

                    logger.info(f"Current binning: {Counter(binned_labels)}")

                    while value_error:

                        try:
                            (
                                train_data,
                                test_data,
                                train_labels,
                                test_labels,
                            ) = train_test_split(
                                data,
                                binned_labels,
                                test_size=test_split,
                                shuffle=shuffle,
                                stratify=binned_labels,
                                random_state=seed,
                            )
                            value_error = False
                        except ValueError as _:
                            logger.warn(
                                "Try to increase the batch size if the least populated class "
                                f"has only one sample, or decrease the binning. Original binning: "
                                f"{Counter(binned_labels)}"
                            )
                            value_error = True
                            logger.warn("Falling back to two bins")
                            binned_labels = TorchDataset.get_custom_bin_indexes(
                                labels=labels
                            )
                            logger.info(
                                f"New binning with two bins: {Counter(binned_labels)}"
                            )
                            assert (
                                list(Counter(binned_labels).values())[0] > 1
                                and list(Counter(binned_labels).values())[1] > 1
                            ), f"Too few samples: {Counter(binned_labels)}"

                    train_labels = Dataset.get_regression_labels_from_split(
                        split_data=train_data, whole_data=data, continuous_labels=labels
                    )
                    test_labels = Dataset.get_regression_labels_from_split(
                        split_data=test_data, whole_data=data, continuous_labels=labels
                    )

                else:
                    logger.warn(
                        "Ignoring stratify of the continuous values as they are not between 0 and 1"
                    )
                    train_data, test_data, train_labels, test_labels = train_test_split(
                        data,
                        labels,
                        test_size=test_split,
                        shuffle=shuffle,
                        random_state=seed,
                    )
        else:
            train_data, train_labels, test_data, test_labels = (
                data,
                labels,
                np.asarray([]),
                np.asarray([]),
            )
            if shuffle:
                np.random.shuffle(train_data)
                np.random.shuffle(train_labels)

        if not regression and oversample_minority_class_percentage > 0.0:
            undersampler = RandomUnderSampler(
                sampling_strategy=oversample_minority_class_percentage
            )
            logger.info(
                "Label proportions before undersampling: {}".format(labels.mean())
            )
            previous_shape = None
            # FIXME
            if len(data.shape) > 2:
                previous_shape = data.shape[1:]
                data = data.reshape(data.shape[0], -1)
            oversampled_data, oversampled_labels = undersampler.fit_resample(
                X=train_data, y=train_labels
            )
            logger.info(
                "Label proportions after undersampling: {}".format(
                    oversampled_labels.mean()
                )
            )

            train_data, train_labels = oversampled_data, oversampled_labels

            if previous_shape is not None and test_split > 0.0:
                train_data = train_data.reshape(-1, *previous_shape)
                test_data = test_data.reshape(-1, *previous_shape)
                logger.info(
                    "Train data shape: {}, Test data shape: {}".format(
                        train_data.shape, test_data.shape
                    )
                )
            elif previous_shape is not None:
                train_data = train_data.reshape(-1, *previous_shape)
                logger.info("Train data shape: {}".format(train_data.shape))

        return train_data, train_labels.reshape(-1), test_data, test_labels.reshape(-1)

    def preprocess_test_data(
        self,
        test_data: np.ndarray,
        test_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(test_data) > 0:
            if self.input_scaler is not None:
                test_data = self.input_scaler.transform(X=test_data)
            if self.output_scaler is not None:
                test_labels = self.output_scaler.transform(
                    X=test_labels.reshape(len(test_labels), 1)
                ).reshape(len(test_labels))
        return test_data, test_labels

    # also assigns input and output scalers
    def preprocess_train_and_test_data(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        test_data: np.ndarray,
        test_labels: np.ndarray,
        regression: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        logger = Log("preprocess_train_and_test_data")
        # The statistics required for the transformation (e.g., the mean) are estimated
        # from the training set and are applied to all data sets (e.g., the test set or new samples)
        self.input_scaler, self.output_scaler = self.get_scalers_for_data(
            data=train_data,
            labels=train_labels.reshape(len(train_labels), 1),
            regression=regression,
        )
        if self.input_scaler is not None:
            logger.info("Preprocessing input data")
            train_data = self.input_scaler.transform(X=train_data)
        if self.output_scaler is not None:
            logger.info("Preprocessing output data")
            train_labels = self.output_scaler.transform(
                X=train_labels.reshape(len(train_labels), 1)
            ).reshape(len(train_labels))
        if len(test_data) > 0:
            if self.input_scaler is not None:
                test_data = self.input_scaler.transform(X=test_data)
            if self.output_scaler is not None:
                test_labels = self.output_scaler.transform(
                    X=test_labels.reshape(len(test_labels), 1)
                ).reshape(len(test_labels))
        return train_data, train_labels, test_data, test_labels

    def transform_data(
        self,
        seed: int,
        test_split: float = 0.2,
        oversample_minority_class_percentage: float = 0.0,
        regression: bool = False,
        weight_loss: bool = False,
        preprocess: bool = False,
        training_progress: bool = False,
        shuffle: bool = True,
        dnn: bool = True,
    ) -> Union[
        Tuple[TorchDataset, TorchDataset],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:

        num_features = self.get_num_features()

        if training_progress:
            # + 1 is the index of the data in the original dataset
            num_features += 1

        data = np.zeros(shape=(len(self.dataset), num_features))
        labels = np.zeros(shape=(len(self.dataset), 1))

        for idx in range(len(self.dataset)):
            data_item = self.dataset[idx]
            if not regression:
                labels[idx] = data_item.label
            else:
                assert data_item.regression_value is not None
                labels[idx] = data_item.regression_value

            a = self.transform_data_item(data_item=data_item)

            if training_progress:
                data[idx] = np.append(a, idx)
            else:
                data[idx] = a

        train_data, train_labels, test_data, test_labels = self.split_train_test(
            test_split=test_split,
            data=data,
            labels=labels,
            oversample_minority_class_percentage=oversample_minority_class_percentage,
            regression=regression,
            seed=seed,
            shuffle=shuffle,
        )

        if training_progress:
            if regression and preprocess:
                (
                    train_data,
                    train_labels,
                    test_data,
                    test_labels,
                ) = self.preprocess_train_and_test_data(
                    train_data=train_data,
                    train_labels=train_labels,
                    test_data=test_data,
                    test_labels=test_labels,
                    regression=regression,
                )
                assert (
                    self.input_scaler is None
                ), "Not possible to scale input features when distinguishing train and test dataset"
                assert (
                    self.output_scaler is not None
                ), "Output scaler must be assigned in regression problems"
            return train_data, train_labels, test_data, test_labels

        if preprocess:
            (
                train_data,
                train_labels,
                test_data,
                test_labels,
            ) = self.preprocess_train_and_test_data(
                train_data=train_data,
                train_labels=train_labels,
                test_data=test_data,
                test_labels=test_labels,
                regression=regression,
            )

        if dnn:
            return (
                TorchDataset(
                    data=train_data,
                    labels=train_labels,
                    regression=regression,
                    weight_loss=weight_loss,
                ),
                TorchDataset(
                    data=test_data,
                    labels=test_labels,
                    regression=regression,
                    weight_loss=weight_loss,
                ),
            )
        return train_data, train_labels, test_data, test_labels
