import argparse
import logging
import os
import time
from functools import reduce
from typing import List, Tuple

import numpy as np
from stable_baselines3.common.utils import set_random_seed

from indago.avf.avf import Avf
from indago.avf.config import AVF_DNN_POLICIES, AVF_TRAIN_POLICIES, CLASSIFIER_LAYERS
from indago.avf.dataset import Dataset, TorchDataset
from indago.avf.preprocessor import preprocess_data
from indago.config import DONKEY_ENV_NAME, ENV_IDS, ENV_NAMES, HUMANOID_ENV_NAME
from indago.envs.donkey.scenes.simulator_scenes import SIMULATOR_SCENES_DICT
from indago.utils.env_utils import ALGOS, get_latest_run_id
from indago.utils.file_utils import get_training_logs_path
from log import Log

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", help="Log folder", type=str, default="logs")
parser.add_argument(
    "--algo",
    help="RL Algorithm",
    default="sac",
    type=str,
    required=False,
    choices=list(ALGOS.keys()),
)
parser.add_argument(
    "--env-name", help="Env name", type=str, choices=ENV_NAMES, default="park"
)
parser.add_argument(
    "--env-id", help="Env id", type=str, choices=ENV_IDS, default="parking-v0"
)
parser.add_argument("--exp-id", help="Experiment ID (0: latest)", default=0, type=int)
parser.add_argument("--seed", help="Random generator seed", type=int, default=-1)
parser.add_argument(
    "--test-split",
    help="Percentage of data reserved for testing",
    type=float,
    default=0.2,
)
parser.add_argument(
    "--avf-policy", help="Avf policy training", type=str, choices=AVF_TRAIN_POLICIES
)
parser.add_argument(
    "--oversample",
    help="Percentage of oversampling of the minority class for the classification problem",
    type=float,
    default=0.0,
)
parser.add_argument(
    "--training-progress-filter",
    help="Percentage of training to filter",
    type=int,
    default=None,
)
parser.add_argument(
    "--n-epochs", help="Number of epochs to train AVF DNN", type=int, default=20
)
parser.add_argument(
    "--learning-rate", help="Learning rate to train AVF DNN", type=float, default=3e-4
)
parser.add_argument(
    "--batch-size", help="Batch size to train AVF DNN", type=int, default=64
)
parser.add_argument(
    "--weight-decay",
    help="Weight decay for optimizer when training AVF DNN",
    type=float,
    default=0.0,
)
parser.add_argument(
    "--weight-loss",
    help="Whether to use a simple weight loss scheme",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--patience",
    help="Early stopping patience (# of epochs of no improvement) when training AVF DNN",
    type=int,
    default=20,
)
parser.add_argument("--save-data", action="store_true", default=False)
parser.add_argument(
    "--preprocess", help="Apply scaler to data", action="store_true", default=False
)
parser.add_argument(
    "--layers",
    help="Num layers architecture",
    type=int,
    choices=CLASSIFIER_LAYERS,
    default=1,
)
parser.add_argument("--no-test-set", action="store_true", default=False)
parser.add_argument("--build-heldout-test", action="store_true", default=False)
parser.add_argument("--heldout-test-file", type=str, default=None)
parser.add_argument(
    "--regression",
    help="Train a regression network",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--training-stability", help="Evaluate training stability", type=int, default=1
)
args = parser.parse_args()


def failures_non_failures_set(ls: np.ndarray) -> Tuple[int, int]:
    failures = int(reduce(lambda a, b: a + b, filter(lambda label: label == 1, ls)))
    non_failures = len(ls) - failures
    return failures, non_failures


def filter_train_test_sets(
    d: Dataset,
    training_progress_filter: float,
    train_d: np.ndarray,
    train_l: np.ndarray,
    test_d: np.ndarray,
    test_l: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    train_data_filtered = []
    train_labels_filtered = []
    test_data_new = []
    test_labels_new = []

    indices_test = [int(idx) for idx in test_d[:, -1]]
    indices_train = [int(idx) for idx in train_d[:, -1]]

    for index in range(len(d.get())):
        d_item = d.get()[index]

        if index not in indices_test:
            idx = indices_train.index(index)
            if d_item.training_progress >= training_progress_filter:
                train_d_item = train_d[idx]
                assert (
                    int(train_d_item[-1]) == index
                ), "The two indices do not match: {} != {}".format(
                    int(train_d_item[-1]), index
                )
                train_l_item = train_l[idx]
                train_data_filtered.append(list(train_d_item[:-1]))
                train_labels_filtered.append(train_l_item)
        else:
            idx = indices_test.index(index)
            test_d_item = test_d[idx]
            assert (
                int(test_d_item[-1]) == index
            ), "The two indices do not match: {} != {}".format(
                int(test_d_item[-1]), index
            )
            test_l_item = test_l[idx]
            test_data_new.append(list(test_d_item[:-1]))
            test_labels_new.append(test_l_item)

    assert len(train_labels_filtered) > 0, "Train set cannot be empty"

    assert len(test_data_new) == len(
        test_d
    ), "Test data lengths do not match. {} != {}".format(
        len(test_data_new), len(test_d)
    )
    assert len(test_labels_new) == len(
        test_l
    ), "Test data lengths do not match. {} != {}".format(
        len(test_labels_new), len(test_l)
    )
    assert len(test_data_new) == len(
        test_labels_new
    ), "Test data and test labels lengths do not match. {} != {}".format(
        len(test_data_new), len(test_labels_new)
    )

    assert (
        np.asarray(test_data_new).shape[1] == test_d.shape[1] - 1
    ), "Test data shape do not match. {} != {}".format(
        np.asarray(test_data_new).shape[1], test_d.shape[1] - 1
    )
    assert (
        np.asarray(train_data_filtered).shape[1] == train_d.shape[1] - 1
    ), "Train data shape do not match. {} != {}".format(
        np.asarray(train_data_filtered).shape[1], train_d.shape[1] - 1
    )

    assert (
        np.asarray(train_data_filtered).shape[0] < train_d.shape[0]
    ), "Train data length after filtering must be < than the original one. {} > {}".format(
        np.asarray(train_data_filtered).shape[0], train_d.shape[0]
    )
    assert (
        np.asarray(train_data_filtered).shape[0] < train_d.shape[0]
    ), "Train labels length after filtering must be < than the original one. {} > {}".format(
        np.asarray(train_data_filtered).shape[0], train_l.shape[0]
    )

    assert len(train_data_filtered) == len(
        train_labels_filtered
    ), "Train data and train labels lengths do not match. {} != {}".format(
        len(train_data_filtered), len(train_labels_filtered)
    )

    return (
        np.asarray(train_data_filtered),
        np.asarray(train_labels_filtered),
        np.asarray(test_data_new),
        np.asarray(test_labels_new),
    )


def print_training_summary(training_metrics: List[Tuple], log: Log) -> None:
    test_losses = [training_metric[0] for training_metric in training_metrics]
    test_precisions = [
        training_metric[1] if training_metric[1] is not None else 0.0
        for training_metric in training_metrics
    ]
    test_recalls = [
        training_metric[2] if training_metric[2] is not None else 0.0
        for training_metric in training_metrics
    ]
    best_epochs_nums = [
        training_metric[3] if training_metric[3] is not None else 0.0
        for training_metric in training_metrics
    ]
    auc_rocs = [
        training_metric[4] if training_metric[4] is not None else 0.0
        for training_metric in training_metrics
    ]
    f_measures = [
        training_metric[5] if training_metric[5] is not None else 0.0
        for training_metric in training_metrics
    ]
    test_maes = [
        (
            training_metric[6]
            if len(training_metric) > 6 and training_metric[6] is not None
            else 0.0
        )
        for training_metric in training_metrics
    ]
    test_r2s = [
        (
            training_metric[7]
            if len(training_metric) > 7 and training_metric[7] is not None
            else 0.0
        )
        for training_metric in training_metrics
    ]

    log.info(
        "Training metrics test losses. Mean: {:.2f}, Std: {:.2f} ({}%), Min: {:.2f}, Max: {:.2f}".format(
            np.mean(test_losses),
            np.std(test_losses),
            (
                int((100 * np.std(test_losses) / np.mean(test_losses)))
                if np.mean(test_losses) > 0.0
                else 0.0
            ),
            np.min(test_losses) if np.mean(test_losses) > 0.0 else 0.0,
            np.max(test_losses) if np.mean(test_losses) > 0.0 else 0.0,
        )
    )
    log.info(
        "Training metrics best epochs. Mean: {:.2f}, Std: {:.2f} ({}%), Min: {:.2f}, Max: {:.2f}".format(
            np.mean(best_epochs_nums),
            np.std(best_epochs_nums),
            (
                int((100 * np.std(best_epochs_nums) / np.mean(best_epochs_nums)))
                if np.mean(best_epochs_nums) > 0.0
                else 0.0
            ),
            np.min(best_epochs_nums) if np.mean(best_epochs_nums) > 0.0 else 0.0,
            np.max(best_epochs_nums) if np.mean(best_epochs_nums) > 0.0 else 0.0,
        )
    )
    if test_precisions[0] is not None:
        log.info(
            "Training metrics test precisions. Mean: {:.2f}, Std: {:.2f} ({}%), Min: {:.2f}, Max: {:.2f}".format(
                np.mean(test_precisions),
                np.std(test_precisions),
                (
                    int((100 * np.std(test_precisions) / np.mean(test_precisions)))
                    if np.mean(test_precisions) > 0.0
                    else 0.0
                ),
                np.min(test_precisions),
                np.max(test_precisions),
            )
        )
        log.info(
            "Training metrics test recalls. Mean: {:.2f}, Std: {:.2f} ({}%), Min: {:.2f}, Max: {:.2f}".format(
                np.mean(test_recalls),
                np.std(test_recalls),
                (
                    int((100 * np.std(test_recalls) / np.mean(test_recalls)))
                    if np.mean(test_recalls) > 0.0
                    else 0.0
                ),
                np.min(test_recalls),
                np.max(test_recalls),
            )
        )
        log.info(
            "Training metrics f-measures. Mean: {:.2f}, Std: {:.2f} ({}%), Min: {:.2f}, Max: {:.2f}".format(
                np.mean(f_measures),
                np.std(f_measures),
                (
                    int((100 * np.std(f_measures) / np.mean(f_measures)))
                    if np.mean(f_measures) > 0.0
                    else 0.0
                ),
                np.min(f_measures),
                np.max(f_measures),
            )
        )
        log.info(
            "Training metrics auc_rocs. Mean: {:.2f}, Std: {:.2f} ({}%), Min: {:.2f}, Max: {:.2f}".format(
                np.mean(auc_rocs),
                np.std(auc_rocs),
                (
                    int((100 * np.std(auc_rocs) / np.mean(auc_rocs)))
                    if np.mean(auc_rocs) > 0.0
                    else 0.0
                ),
                np.min(auc_rocs),
                np.max(auc_rocs),
            )
        )
    if len(test_maes) > 0:
        log.info(
            "Training metrics test maes. Mean: {:.2f}, Std: {:.2f} ({}%), Min: {:.2f}, Max: {:.2f}".format(
                np.mean(test_maes),
                np.std(test_maes),
                (
                    int((100 * np.std(test_maes) / np.mean(test_maes)))
                    if np.mean(test_maes) > 0.0
                    else 0.0
                ),
                np.min(test_maes),
                np.max(test_maes),
            )
        )

    if len(test_r2s) > 0:
        log.info(
            "Training metrics test r2s. Mean: {:.2f}, Std: {:.2f} ({}%), Min: {:.2f}, Max: {:.2f}".format(
                np.mean(test_r2s),
                np.std(test_r2s),
                (
                    int((100 * np.std(test_r2s) / np.mean(test_r2s)))
                    if np.mean(test_maes) > 0.0
                    else 0.0
                ),
                np.min(test_r2s),
                np.max(test_r2s),
            )
        )


if __name__ == "__main__":

    algo = args.algo
    folder = args.folder

    if args.seed == -1:
        args.seed = np.random.randint(2**32 - 1)

    set_random_seed(args.seed)

    ENV_ID = args.env_id
    assert args.exp_id >= 0, "exp_id should be >= 0: {}".format(args.exp_id)

    simulator_scene = SIMULATOR_SCENES_DICT["generated_track"]
    if args.env_name == DONKEY_ENV_NAME:
        assert ENV_ID == "DonkeyVAE-v0", "env_id must be DonkeyVAE, found: {}".format(
            ENV_ID
        )
        ENV_ID = "DonkeyVAE-v0-scene-{}".format(simulator_scene.get_scene_name())

    if args.exp_id == 0:
        args.exp_id = get_latest_run_id(os.path.join(folder, algo), ENV_ID)

    log_path = get_training_logs_path(
        folder=folder, algo=algo, env_id=ENV_ID, exp_id=args.exp_id
    )

    avf = Avf(
        env_name=args.env_name,
        log_path=log_path,
        storing_logs=False,
        seed=args.seed,
        regression=args.regression,
    )

    logger = Log("avf_train")
    log_filename = "avf-{}".format(args.avf_policy)

    if args.build_heldout_test:
        log_filename += "-build-heldout-test-set"
        if args.regression:
            log_filename += "-rgr"
        else:
            log_filename += "-cls"
    else:
        if args.training_stability > 1:
            log_filename += "-training-stability-{}".format(args.training_stability)
        else:
            log_filename += "-seed-{}".format(args.seed)

        if args.avf_policy in AVF_DNN_POLICIES:
            log_filename += "-{}-{}-{}-{}".format(
                args.training_progress_filter,
                args.oversample,
                args.layers,
                "cls" if not args.regression else "rgr",
            )

    if args.preprocess:
        log_filename += "-preprocess"

    logging.basicConfig(
        filename=os.path.join(log_path, "{}.txt".format(log_filename)),
        filemode="w",
        level=logging.DEBUG,
    )

    logger.info("Args: {}".format(args))
    start_time = time.time()

    if args.build_heldout_test:

        training_progress_filter = 5 if args.env_name != HUMANOID_ENV_NAME else 20

        dataset = preprocess_data(
            env_name=args.env_name,
            log_path=log_path,
            training_progress_filter=training_progress_filter,
            policy=args.avf_policy,
            regression=args.regression,
        )

        train_data, train_labels, test_data, test_labels = dataset.transform_data(
            test_split=args.test_split,
            oversample_minority_class_percentage=0.0,
            regression=args.regression,
            weight_loss=args.weight_loss,
            preprocess=args.preprocess,
            training_progress=True,
            seed=args.seed,
        )

        numpy_dict = {
            "test_data": test_data,
            "test_labels": (
                np.asarray(test_labels, dtype=np.int)
                if not args.regression
                else test_labels
            ),
        }
        logger.info("Test set of size: {}".format(len(test_data)))

        if not args.regression:
            failures_train_dataset = int(
                reduce(
                    lambda a, b: a + b, filter(lambda label: label == 1, train_labels)
                )
            )
            failures_test_dataset = int(
                reduce(
                    lambda a, b: a + b, filter(lambda label: label == 1, test_labels)
                )
            )
            non_failures_train_dataset = len(train_labels) - failures_train_dataset
            non_failures_test_dataset = len(test_labels) - failures_test_dataset

            logger.info(
                "Failures/Non-failures train set: {}/{}. Proportion: {:.2f}".format(
                    failures_train_dataset,
                    non_failures_train_dataset,
                    np.mean(train_labels),
                )
            )
            logger.info(
                "Failures/Non-failures test set: {}/{}. Proportion: {:.2f}".format(
                    failures_test_dataset,
                    non_failures_test_dataset,
                    np.mean(test_labels),
                )
            )

        filename = "heldout-set-seed-{}-{}-split-{}-filter".format(
            args.seed, args.test_split, training_progress_filter
        )

        if args.regression:
            filename += "-rgr"
        else:
            filename += "-cls"

        if args.preprocess:
            filename += "-preprocess"

        filename += ".npz"
        np.savez(os.path.join(log_path, filename), **numpy_dict)

    else:
        save_model = True
        # if heldout test file is present the training progress filtering happens below,
        # in the filter_train_test_sets function
        dataset = preprocess_data(
            env_name=args.env_name,
            log_path=log_path,
            training_progress_filter=(
                args.training_progress_filter if args.heldout_test_file is None else 0.0
            ),
            policy=args.avf_policy,
            regression=args.regression,
        )

        training_stability_metrics = []

        if args.heldout_test_file is not None:
            assert os.path.exists(
                os.path.join(log_path, args.heldout_test_file)
            ), "Heldout test file {}".format(
                os.path.join(log_path, args.heldout_test_file)
            )

            numpy_dict = np.load(os.path.join(log_path, args.heldout_test_file))
            test_data = numpy_dict["test_data"]
            test_labels = numpy_dict["test_labels"]

            # # transform dataset, keep index of each datapoint at the end
            # # of its array (training_progress = True)
            (
                train_validation_data,
                train_validation_labels,
                _,
                _,
            ) = dataset.transform_data(
                seed=args.seed,
                test_split=0.0,
                oversample_minority_class_percentage=0.0,
                regression=args.regression,
                weight_loss=args.weight_loss,
                preprocess=args.preprocess,
                training_progress=True,
                shuffle=False,
            )

            # remove datapoints in train_validation set that are part of the test set;
            # remove index of datapoint from test set
            (
                train_validation_data,
                train_validation_labels,
                test_data,
                test_labels,
            ) = filter_train_test_sets(
                d=dataset,
                training_progress_filter=args.training_progress_filter,
                train_d=train_validation_data,
                train_l=train_validation_labels,
                test_d=test_data,
                test_l=test_labels,
            )

        for i in range(args.training_stability):

            train_dataset = None
            validation_dataset = None
            test_dataset = None

            if args.training_stability > 1:
                args.seed = np.random.randint(2**32 - 1)
                set_random_seed(args.seed)
                logger.info("Training stability run {} seed {}".format(i, args.seed))

            if args.heldout_test_file is not None:

                # split train_validation data into train and validation data using the split provided
                (
                    train_data,
                    train_labels,
                    validation_data,
                    validation_labels,
                ) = Dataset.split_train_test(
                    seed=args.seed,
                    test_split=args.test_split,
                    data=train_validation_data,
                    labels=train_validation_labels,
                    oversample_minority_class_percentage=0.0,
                    regression=args.regression,
                )

                if args.oversample > 0.0:
                    logger.info("Train data sampling")
                    logger.info("Fixing seed for over/under sampling: 0")
                    train_data, train_labels = Dataset.sampling(
                        data=train_data,
                        labels=train_labels,
                        under=True,
                        sampling_percentage=args.oversample,
                        seed=0,
                    )
                    train_failures, train_non_failures = failures_non_failures_set(
                        ls=train_labels
                    )
                    logger.info(
                        "Train data size: {}, {}/{}".format(
                            len(train_data), train_failures, train_non_failures
                        )
                    )
                    to_zip = list(zip(train_data, train_labels))
                    np.random.shuffle(to_zip)
                    train_data = np.asarray([to_zip_item[0] for to_zip_item in to_zip])
                    train_labels = np.asarray(
                        [to_zip_item[1] for to_zip_item in to_zip]
                    )

                    logger.info("Validation data sampling")
                    validation_data, validation_labels = Dataset.sampling(
                        data=validation_data,
                        labels=validation_labels,
                        under=True,
                        sampling_percentage=args.oversample,
                        seed=0,
                    )
                    (
                        validation_failures,
                        validation_non_failures,
                    ) = failures_non_failures_set(ls=validation_labels)
                    logger.info(
                        "Validation data size: {}, {}/{}".format(
                            len(validation_data),
                            validation_failures,
                            validation_non_failures,
                        )
                    )
                    to_zip = list(zip(validation_data, validation_labels))
                    np.random.shuffle(to_zip)
                    validation_data = np.asarray(
                        [to_zip_item[0] for to_zip_item in to_zip]
                    )
                    validation_labels = np.asarray(
                        [to_zip_item[1] for to_zip_item in to_zip]
                    )

                train_dataset = TorchDataset(
                    data=train_data,
                    labels=train_labels,
                    regression=args.regression,
                    weight_loss=args.weight_loss,
                )

                validation_dataset = TorchDataset(
                    data=validation_data,
                    labels=validation_labels,
                    regression=args.regression,
                    weight_loss=args.weight_loss,
                )

                test_dataset = TorchDataset(
                    data=test_data,
                    labels=test_labels,
                    regression=args.regression,
                    weight_loss=args.weight_loss,
                )

            if args.avf_policy in AVF_DNN_POLICIES:
                (
                    test_loss,
                    test_precision,
                    test_recall,
                    best_epochs,
                    auc_roc,
                    test_mae,
                    test_r2,
                ) = avf.train_dnn(
                    seed=args.seed,
                    n_epochs=args.n_epochs,
                    dataset=dataset,
                    avf_train_policy=args.avf_policy,
                    layers=args.layers,
                    oversample_minority_class_percentage=args.oversample,
                    test_split=args.test_split,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    patience=args.patience,
                    batch_size=args.batch_size,
                    training_progress_filter=args.training_progress_filter,
                    weight_loss=args.weight_loss,
                    no_test_set=args.no_test_set,
                    train_dataset_=train_dataset,
                    validation_dataset_=validation_dataset,
                    test_dataset_=test_dataset,
                    save_model=save_model,
                )
                if test_precision is not None and test_recall is not None:
                    fmeasure = (
                        2
                        * (test_precision * test_recall)
                        / (test_precision + test_recall)
                        if test_precision + test_recall > 0.0
                        else 0.0
                    )
                else:
                    fmeasure = None
                training_stability_metrics.append(
                    (
                        test_loss,
                        test_precision,
                        test_recall,
                        best_epochs,
                        auc_roc,
                        fmeasure,
                        test_mae,
                        test_r2,
                    )
                )
            else:
                raise NotImplementedError(
                    "avf_policy {} not supported".format(args.avf_policy)
                )

        print_training_summary(training_metrics=training_stability_metrics, log=logger)

    logger.info("Time elapsed: {}s".format(time.time() - start_time))
