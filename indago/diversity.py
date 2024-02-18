import argparse
import glob
import logging
import math
import os
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from stable_baselines3.common.utils import set_random_seed

from indago.avf.config import AVF_TRAIN_POLICIES
from indago.clustering import cluster_data
from indago.config import DONKEY_ENV_NAME, ENV_NAMES, HUMANOID_ENV_NAME, PARK_ENV_NAME
from indago.envs.donkey.donkey_dataset import DonkeyDataset
from indago.envs.humanoid.humanoid_dataset import HumanoidDataset
from indago.envs.park.parking_dataset import ParkingDataset
from indago.input_diversity import input_diversity
from indago.output_diversity import output_diversity
from indago.stats.effect_size import cohend, vargha_delaney_unpaired
from indago.stats.power_analysis import parametric_power_analysis
from indago.stats.stat_tests import mannwhitney_test
from log import Log

parser = argparse.ArgumentParser()
parser.add_argument(
    "--folder", help="Parent folder where logs are", type=str, required=True
)
parser.add_argument("--seed", help="Random generator seed", type=int, default=-1)
parser.add_argument(
    "--avf-train-policy",
    help="Avf train policy",
    type=str,
    choices=AVF_TRAIN_POLICIES,
    default="mlp",
)
parser.add_argument(
    "--env-name", help="Env name", type=str, choices=ENV_NAMES, default=None
)
parser.add_argument(
    "--max-num-clusters", help="Max number of clusters", type=int, default=None
)
parser.add_argument(
    "--visualize", help="Visualize clusters", action="store_true", default=False
)
parser.add_argument(
    "--visualize-name",
    help="Valid only if visualize = True. Plot only points of the specified technique",
    type=str,
    default=None,
)
parser.add_argument(
    "--annotate", help="Annotate points with ids", action="store_true", default=False
)
parser.add_argument(
    "--show-ids",
    help="Show ids of each method associated with respective cluster labels",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--directories", nargs="+", help="List of files to analyze", default=[]
)
parser.add_argument(
    "--pattern",
    help="Whether or not to use patter to find directories",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--names", nargs="+", help="Names associated to files", required=True
)
parser.add_argument(
    "--type",
    help="Type of diversity to compute (input/output)",
    type=str,
    choices=["input", "output"],
    default="input",
)
parser.add_argument(
    "--output-dir-name",
    help="Append to input_diversity or output_diversity directory name to save output in a different directory than the default one",
    type=str,
    default=None,
)
parser.add_argument(
    "--pca",
    help="Whether to apply pca before clustering to reduce the dimensionality to only the most important ones (i.e. the ones with more variance)",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--num-runs-clustering",
    help="How many times to run clustering",
    type=int,
    default=1,
)
parser.add_argument(
    "--sil-threshold",
    help="Silhouette threshold to overcome in order to deem a higher silhouette score an improvement",
    type=int,
    default=-1,
)
# params for statistical analysis
parser.add_argument(
    "--alpha",
    help="Statistical significance level for statistical tests",
    type=float,
    default=0.05,
)
parser.add_argument(
    "--beta", help="Power level for statistical tests", type=float, default=0.8
)
parser.add_argument(
    "--adjust",
    help="Adjust p-values when multiple comparisons",
    action="store_true",
    default=False,
)

args = parser.parse_args()


def sort_directory(d: str) -> int:
    return int(d.split("-")[-2])


def compute_coverage_entropy(
    logger: Log,
    seed: int,
    folder: str,
    dirs: List[str],
    names: List[str],
    visualize: bool,
    visualize_name: str,
    tp: str,
    output_dir_name: str,
    avf_train_policy: str,
    env_name: str,
    max_num_clusters: int,
    show_ids: bool,
    annotate: bool,
    pca: bool,
    silhouette_threshold: int,
) -> Tuple[Dict, Dict, float, int]:

    coverage_names = dict()
    entropy_names = dict()
    gini_impurity_coeff = None

    for i in range(len(dirs)):
        assert os.path.exists(
            os.path.join(folder, dirs[i])
        ), "File {} does not exist".format(os.path.join(folder, dirs[i]))
        assert os.path.isdir(
            os.path.join(folder, dirs[i])
        ), "File {} is not a folder".format(os.path.join(folder, dirs[i]))

    assert len(names) == len(
        dirs
    ), "Num failure search names {} != num directories {}".format(len(names), len(dirs))

    # colors for cluster representation == num of failure search methods. For more colors see:
    # https://matplotlib.org/stable/gallery/color/named_colors.html
    colors = [
        "black",
        "green",
        "blue",
        "purple",
        "magenta",
        "pink",
        "grey",
        "cyan",
        "lime",
        "orange",
        "peru",
    ]
    assert len(colors) >= len(names), "Choose {} more colors".format(
        len(names) - len(colors)
    )

    # markers for cluster representation == num of failure search methods. For more markers see:
    # https://matplotlib.org/stable/api/markers_api.html
    markers = ["o", "^", "2", "*", "s", "+", "x", "D", "h", "8", ">"]
    assert len(markers) >= len(names), "Choose {} more markers".format(
        len(names) - len(markers)
    )

    if visualize and visualize_name is not None:
        assert (
            visualize_name in names
        ), "Technique to plot {} is not in the list of techniques to analyze {}".format(
            visualize_name, names
        )

    folder_name = (
        os.path.join(folder, "input_diversity")
        if tp == "input"
        else os.path.join(folder, "output_diversity")
    )

    if output_dir_name is not None:
        folder_name += "_{}".format(output_dir_name)

    os.makedirs(folder_name, exist_ok=True)

    log_filename = (
        os.path.join(folder_name, "input-diversity.txt")
        if tp == "input"
        else os.path.join(folder_name, "output-diversity.txt")
    )

    if pca:
        log_filename = log_filename.split(".")[0] + "_pca.txt"

    logging.basicConfig(filename=log_filename, filemode="w", level=logging.DEBUG)

    # TODO: refactor
    if env_name == PARK_ENV_NAME:
        dataset = ParkingDataset(policy=avf_train_policy)
    elif env_name == HUMANOID_ENV_NAME:
        dataset = HumanoidDataset(policy=avf_train_policy)
    elif env_name == DONKEY_ENV_NAME:
        dataset = DonkeyDataset(policy=avf_train_policy)
    else:
        raise NotImplementedError("Unknown env name: {}".format(env_name))

    if tp == "input":
        to_cluster, env_configurations_names = input_diversity(
            folder=folder,
            names=names,
            directories=dirs,
            env_name=env_name,
            dataset=dataset,
            avf_train_policy=avf_train_policy,
        )

    else:
        to_cluster, env_configurations_names = output_diversity(
            env_name=env_name, folder=folder, names=names, directories=directories
        )

    if pca:
        # get number of components comprising 95% of the variance
        # https://www.mikulskibartosz.name/pca-how-to-choose-the-number-of-components/
        original_dimensions = to_cluster.shape[1:]
        pca = PCA(n_components=0.95, random_state=seed)
        pca.fit(to_cluster)
        to_cluster = pca.transform(to_cluster)
        dimensions_after_dim_reduction = to_cluster.shape[1:]
        logger.info(
            "Original dimensions: {}. Dimensions after dimensionality reduction: {}".format(
                original_dimensions, dimensions_after_dim_reduction
            )
        )

    max_num_of_clusters = (
        len(to_cluster) if max_num_clusters is None else max_num_clusters
    )

    try:
        clusterer, labels, centers, _ = cluster_data(
            data=to_cluster,
            n_clusters_interval=(2, min(to_cluster.shape[0], max_num_of_clusters)),
            seed=seed,
            silhouette_threshold=silhouette_threshold,
        )
        num_clusters = len(centers)
        coverage_set_names = dict()
        number_points_names = dict()
        name_idx = 0
        count_labels = 0
        for label in labels:
            if count_labels > len(env_configurations_names[names[name_idx]]) - 1:
                count_labels = 0
                name_idx += 1

            nm = names[name_idx]
            if nm not in coverage_set_names:
                coverage_set_names[nm] = set()
            coverage_set_names[nm].add(label)

            if nm not in number_points_names:
                number_points_names[nm] = []
            number_points_names[nm].append(label)

            count_labels += 1

        for nm, ls in number_points_names.items():
            assert len(ls) == len(
                env_configurations_names[nm]
            ), "Number of labels {} != Number of failures: {} for {}".format(
                len(ls), len(env_configurations_names[nm]), nm
            )

        logger.info("Number of clusters: {}".format(len(centers)))
        ideal_entropy = np.log2(len(centers))
        num_classes = len(env_configurations_names)
        logger.info(f"Num classes: {num_classes}")

        gini_dict = dict()
        counter_labels = Counter(labels)

        for nm, set_clusters in coverage_set_names.items():

            # TODO: computing gini coefficient only if with two classes for now
            if len(set_clusters) > 0 and num_classes == 2:
                number_points_name_array = np.asarray(number_points_names[nm])
                gini_dict[nm] = [
                    len(
                        number_points_name_array[
                            number_points_name_array == cluster_label
                        ]
                    )
                    / counter_labels[cluster_label]
                    for cluster_label in counter_labels
                ]

            coverage = 100 * len(set_clusters) / len(centers)
            coverage_names[nm] = coverage
            logger.info("Coverage for method {}: {}%".format(nm, coverage))
            counter = Counter(number_points_names[nm])
            distribution = [
                round(
                    100
                    * number_of_points_in_cluster
                    / len(env_configurations_names[nm]),
                    2,
                )
                for number_of_points_in_cluster in counter.values()
            ]
            assert math.isclose(
                a=sum(distribution), b=100.0, abs_tol=0.1
            ), "Sum of distribution values must be = 100. Found: {}".format(
                sum(distribution)
            )
            logger.info(
                "Distribution across clusters for method {}: {}".format(
                    nm, distribution
                )
            )
            entropy_names[nm] = 0.0
            if len(set_clusters) > 0:
                frequency_list = [
                    number_of_points_in_cluster / len(env_configurations_names[nm])
                    for number_of_points_in_cluster in counter.values()
                ]

                entropy = 0.0
                for freq in frequency_list:
                    entropy += freq * np.log2(freq)
                if not math.isclose(entropy, 0.0):
                    entropy *= -1
                logger.info("Entropy: {}, Ideal: {}".format(entropy, ideal_entropy))
                entropy = round(100 * entropy / ideal_entropy, 2)
                entropy_names[nm] = entropy
                logger.info("Entropy: {}%".format(entropy))

        # TODO: computing gini coefficient only if with two classes for now
        if num_classes == 2:
            same_length = all(
                len(probabilities) == len(list(gini_dict.values())[0])
                for probabilities in gini_dict.values()
            )
            assert (
                same_length
            ), f"All the probability scores are not of the same length: {gini_dict}"

            logger.info(f"Gini impurity coefficient dictionary: {gini_dict}")

            gini_impurity_coeff = 0
            for i in range(num_clusters):

                if sum([gini_dict[nm][i] for nm in gini_dict.keys()]) != 1.0:
                    raise RuntimeError(
                        f"Error in computing the gini coefficient: {gini_dict}"
                    )

                gini_impurity_coeff += 1 - sum(
                    [gini_dict[nm][i] ** 2 for nm in gini_dict.keys()]
                )

            gini_impurity_coeff /= num_clusters

            if gini_impurity_coeff > 1.0:
                raise RuntimeError(
                    f"The gini coefficient cannot be > 1.0: {gini_impurity_coeff}"
                )

            logger.info(f"Gini impurity coefficient: {gini_impurity_coeff}")
            logger.info(f"Gini purity coefficient: {1 - gini_impurity_coeff}")

        if show_ids:
            for nm, labels in number_points_names.items():
                logger.info("Name: {}".format(nm))
                for i, label in enumerate(labels):
                    logger.info("id: {}, cluster label: {}".format(i, label))

        if visualize:

            for c in centers:
                # add center of cluster as a new row
                to_cluster = np.append(to_cluster, c.reshape(1, -1), axis=0)

            for per in [2, 5, 10, 20, 30]:
                # assuming TSNE
                n_iterations = 5000
                tsne = TSNE(
                    n_components=2,
                    perplexity=per,
                    n_jobs=-1,
                    n_iter=n_iterations,
                    random_state=seed,
                )
                embeddings = tsne.fit_transform(to_cluster)

                _ = plt.figure()
                ax = plt.gca()
                ax.tick_params(left=False)
                ax.tick_params(bottom=False)
                ax.axes.yaxis.set_ticklabels([])
                ax.axes.xaxis.set_ticklabels([])

                count_configs = 0
                name_idx = 0

                embeddings_names = dict()
                for i in range(len(embeddings) - len(centers)):
                    if (
                        count_configs
                        > len(env_configurations_names[names[name_idx]]) - 1
                    ):
                        count_configs = 0
                        name_idx += 1

                    if names[name_idx] not in embeddings_names:
                        embeddings_names[names[name_idx]] = ([], [])
                    embeddings_names[names[name_idx]][0].append(embeddings[i][0])
                    embeddings_names[names[name_idx]][1].append(embeddings[i][1])

                    count_configs += 1

                for nm, ems in embeddings_names.items():
                    assert len(ems[0]) == len(
                        env_configurations_names[nm]
                    ), "Number of embeddings {} != Number of failures: {} for {}".format(
                        len(ems[0]), len(env_configurations_names[nm]), nm
                    )

                for i, name_embeddings in enumerate(embeddings_names.items()):
                    nm, embeddings_ = name_embeddings

                    if visualize_name is None or visualize_name == nm:
                        plt.scatter(
                            embeddings_[0],
                            embeddings_[1],
                            s=80,
                            color=colors[i],
                            marker=markers[i],
                            label=nm,
                        )

                    if annotate:
                        if visualize_name is None or visualize_name == nm:
                            embedding_id = 0
                            for embedding_x, embedding_y in zip(
                                embeddings_[0], embeddings_[1]
                            ):
                                plt.annotate(
                                    "{}_{}".format(nm, embedding_id),
                                    (embedding_x, embedding_y),
                                    fontsize=5,
                                )
                                embedding_id += 1

                embeddings_centroids = ([], [])
                for i in range(
                    len(embeddings) - 1, len(embeddings) - 1 - len(centers), -1
                ):
                    embeddings_centroids[0].append(embeddings[i][0])
                    embeddings_centroids[1].append(embeddings[i][1])

                plt.scatter(
                    embeddings_centroids[0],
                    embeddings_centroids[1],
                    s=50,
                    color="red",
                    marker="$c$",
                    label="centroid",
                )

                plt.legend(prop={"size": 6})

                filename = (
                    "input-diversity-clusters-tsne-per-{}".format(per)
                    if tp == "input"
                    else "output-diversity-clusters-tsne-per-{}".format(per)
                )

                if visualize_name is not None:
                    filename += "-{}".format(visualize_name)

                if annotate:
                    filename += "-annotated"

                if pca:
                    filename += "-pca"

                plt.savefig(os.path.join(folder_name, filename) + ".pdf", format="pdf")
                plt.clf()
                plt.close()
    except AssertionError as _:
        logger.warn(
            "Not possible to cluster {} configurations in more than cluster".format(
                to_cluster.shape[0]
            )
        )
        num_clusters = 1
        # in this case entropy is always zero (all configurations belong to the same cluster) but coverage
        # depends on the method, i.e. if it does not generate any failure then the coverage is zero, otherwise 100%.
        for nm in names:
            if len(env_configurations_names[nm]) > 0:
                coverage_names[nm] = 100.00
            else:
                coverage_names[nm] = 0.0
            entropy_names[nm] = 0.0

    gini_purity_coeff = (
        1 - gini_impurity_coeff if gini_impurity_coeff is not None else None
    )
    return coverage_names, entropy_names, gini_purity_coeff, num_clusters


def statistical_analysis(
    logger: Log,
    names_dict: Dict,
    nms: List[str],
    alpha: float,
    beta: float,
    adjust: bool,
    metric_name: str,
) -> None:
    if not adjust:
        for name_i in range(len(names_dict)):
            metrics_a = names_dict[nms[name_i]]
            name_a = nms[name_i]
            for name_j in range(name_i + 1, len(names_dict)):
                metrics_b = names_dict[nms[name_j]]
                name_b = nms[name_j]
                logger.info(
                    "{} {}: {}, {} {}: {}".format(
                        metric_name, name_a, metrics_a, metric_name, name_b, metrics_b
                    )
                )
                if (
                    np.array_equal(a1=np.asarray(metrics_a), a2=np.asarray(metrics_b))
                    or np.array_equal(
                        a1=np.asarray(metrics_a),
                        a2=np.asarray(metrics_b[: len(metrics_a)]),
                    )
                    or np.array_equal(
                        a1=np.asarray(metrics_a[: len(metrics_b)]),
                        a2=np.asarray(metrics_b),
                    )
                ):
                    logger.info(
                        "{} ({}) vs {} ({}), not significant".format(
                            name_a, np.mean(metrics_a), name_b, np.mean(metrics_b)
                        )
                    )
                else:
                    _, p_value = mannwhitney_test(a=list(metrics_a), b=list(metrics_b))
                    if p_value < args.alpha:
                        eff_size_magnitude = vargha_delaney_unpaired(
                            a=list(metrics_a), b=list(metrics_b)
                        )
                        logger.info(
                            "{} ({}) vs {} ({}), p-value: {}, effect size: {}, significant".format(
                                name_a,
                                np.mean(metrics_a),
                                name_b,
                                np.mean(metrics_b),
                                p_value,
                                eff_size_magnitude,
                            )
                        )
                    else:
                        if len(metrics_a) == len(metrics_b) and len(metrics_a) > 0:
                            effect_size, _ = cohend(
                                a=list(metrics_a), b=list(metrics_b)
                            )
                            if math.isclose(effect_size, 0.0):
                                logger.info(
                                    "{} ({}) vs {} ({}), not significant".format(
                                        name_a,
                                        np.mean(metrics_a),
                                        name_b,
                                        np.mean(metrics_b),
                                    )
                                )
                            else:
                                sample_size = parametric_power_analysis(
                                    effect=effect_size, alpha=alpha, power=beta
                                )
                                logger.info(
                                    "{} ({}) vs {} ({}), sample size: {}".format(
                                        name_a,
                                        np.mean(metrics_a),
                                        name_b,
                                        np.mean(metrics_b),
                                        (
                                            int(sample_size)
                                            if sample_size != math.inf
                                            else math.inf
                                        ),
                                    )
                                )
                        else:
                            logger.info(
                                "Not possible to compute parametric effect size {} "
                                "with {} for {}, since metrics lengths are different {} != {}".format(
                                    name_a,
                                    name_b,
                                    metric_name,
                                    len(metrics_a),
                                    len(metrics_b),
                                )
                            )
                            logger.info(
                                "{} ({}) vs {} ({}), not significant".format(
                                    name_a,
                                    np.mean(metrics_a),
                                    name_b,
                                    np.mean(metrics_b),
                                )
                            )
    else:
        raise NotImplementedError("P-value adjustment not supported")


if __name__ == "__main__":

    lgg = Log("input_diversity") if args.type == "input" else Log("output_diversity")
    lgg.info("Args: {}".format(args))

    if args.seed == -1:
        args.seed = np.random.randint(2**32 - 1)

    if args.sil_threshold > 0:
        assert (
            0 < args.sil_threshold <= 100
        ), "Silhouette threshold needs to be in (0, 100]. Found: {}".format(
            args.sil_threshold
        )

    set_random_seed(args.seed)

    directories_names = dict()
    if args.pattern:
        for idx in range(len(args.names)):
            name = args.names[idx]
            if name == "random" or "prioritized_replay" in name:
                directories = glob.glob(
                    os.path.join(
                        args.folder, "replay_test_failure_{}-*-trial".format(name)
                    )
                )
            else:
                directories = glob.glob(
                    os.path.join(
                        args.folder,
                        "replay_test_failure_{}-{}-*-trial".format(
                            args.avf_train_policy, name
                        ),
                    )
                )
            if name not in directories_names:
                directories_names[name] = []
            assert len(directories) > 0, "No directory match for {}".format(name)
            directories_names[name].extend(sorted(directories, key=sort_directory))

    if len(args.directories) > 0:
        _, _, _, _ = compute_coverage_entropy(
            logger=lgg,
            seed=args.seed,
            folder=args.folder,
            dirs=args.directories,
            names=args.names,
            visualize=args.visualize,
            visualize_name=args.visualize_name,
            tp=args.type,
            output_dir_name=args.output_dir_name,
            avf_train_policy=args.avf_train_policy,
            env_name=args.env_name,
            max_num_clusters=args.max_num_clusters,
            show_ids=args.show_ids,
            annotate=args.annotate,
            pca=args.pca,
            silhouette_threshold=args.sil_threshold,
        )
    else:
        assert len(directories_names) > 0, "Cannot compute coverage and entropy"
        max_num_trials = max([len(directories_names[key]) for key in directories_names])
        coverage_trials_names = dict()
        entropy_trials_names = dict()
        gini_coefficient_trials = []
        num_clusters_trials = []
        for num_trial in range(max_num_trials):
            directories = []
            names = []
            for idx in range(len(args.names)):
                dir_name = list(
                    filter(
                        lambda s: int(s.split("-")[-2]) == num_trial + 1,
                        directories_names[args.names[idx]],
                    )
                )
                assert (
                    len(dir_name) <= 1
                ), "Not possible to have more than one match for trial {} in {}: {}".format(
                    num_trial, args.names[idx], dir_name
                )
                if len(dir_name) == 1:
                    directories.append(dir_name[0].replace(args.folder + "/", ""))
                    names.append(args.names[idx])

            if len(directories) == 0:
                lgg.debug(
                    f"There are no directories (args.names) with executions to cluster. Num trial: {num_trial + 1}"
                )
                continue

            if args.num_runs_clustering > 1:
                (
                    all_coverage_names,
                    all_entropy_names,
                    all_gini_coefficients,
                    all_nums_clusters,
                ) = ([], [], [], [])
                for _ in range(args.num_runs_clustering):
                    seed = np.random.randint(2**32 - 1)
                    set_random_seed(seed)
                    lgg.debug("Seed for clustering: {}".format(seed))
                    (
                        coverage_names_run,
                        entropy_names_run,
                        gini_purity_coefficient_run,
                        num_clusters_run,
                    ) = compute_coverage_entropy(
                        logger=lgg,
                        seed=seed,
                        folder=args.folder,
                        dirs=directories,
                        names=names,
                        visualize=args.visualize,
                        visualize_name=args.visualize_name,
                        tp=args.type,
                        output_dir_name=(
                            "{}-{}-trial".format(args.output_dir_name, num_trial)
                            if args.output_dir_name is not None
                            else "{}-trial".format(num_trial)
                        ),
                        avf_train_policy=args.avf_train_policy,
                        env_name=args.env_name,
                        max_num_clusters=args.max_num_clusters,
                        show_ids=args.show_ids,
                        annotate=args.annotate,
                        pca=args.pca,
                        silhouette_threshold=args.sil_threshold,
                    )
                    all_coverage_names.append(coverage_names_run)
                    all_entropy_names.append(entropy_names_run)
                    if gini_purity_coefficient_run is not None:
                        all_gini_coefficients.append(gini_purity_coefficient_run)
                    all_nums_clusters.append(num_clusters_run)

                coverage_names, entropy_names = dict(), dict()
                for name in args.names:
                    coverage_names_name = [
                        cv_names_dict[name]
                        for cv_names_dict in all_coverage_names
                        if name in cv_names_dict
                    ]
                    if len(coverage_names_name) > 0:
                        coverage_names[name] = np.mean(coverage_names_name)
                    entropy_names_name = [
                        ent_names_dict[name]
                        for ent_names_dict in all_entropy_names
                        if name in ent_names_dict
                    ]
                    if len(entropy_names_name) > 0:
                        entropy_names[name] = np.mean(entropy_names_name)

                all_values_num_clusters = [
                    n_clusters for n_clusters in all_nums_clusters
                ]
                num_clusters_trials.append(np.mean(all_nums_clusters))
                if len(all_gini_coefficients) > 0:
                    gini_coefficient_trials.append(np.mean(all_gini_coefficients))
                    lgg.debug(
                        f"Gini purity coefficients: {all_gini_coefficients}, mean: {np.mean(all_gini_coefficients)}"
                    )

                lgg.debug(
                    "All values num clusters: {}. Mean: {}, Std: {}".format(
                        all_values_num_clusters,
                        np.mean(all_nums_clusters),
                        np.std(all_values_num_clusters),
                    )
                )
                lgg.debug("Coverage names: {}".format(coverage_names))
                lgg.debug("Entropy names: {}".format(entropy_names))
            else:
                (
                    coverage_names,
                    entropy_names,
                    gini_purity_coefficient,
                    num_clusters,
                ) = compute_coverage_entropy(
                    logger=lgg,
                    seed=args.seed,
                    folder=args.folder,
                    dirs=directories,
                    names=names,
                    visualize=args.visualize,
                    visualize_name=args.visualize_name,
                    tp=args.type,
                    output_dir_name=(
                        "{}-{}-trial".format(args.output_dir_name, num_trial)
                        if args.output_dir_name is not None
                        else "{}-trial".format(num_trial)
                    ),
                    avf_train_policy=args.avf_train_policy,
                    env_name=args.env_name,
                    max_num_clusters=args.max_num_clusters,
                    show_ids=args.show_ids,
                    annotate=args.annotate,
                    pca=args.pca,
                    silhouette_threshold=args.sil_threshold,
                )
                num_clusters_trials.append(num_clusters)
                if gini_purity_coefficient is not None:
                    gini_coefficient_trials.append(gini_purity_coefficient)

            # when a certain method does not produce any failure the dictionaries coverage_names and entropy_names
            # do not have a coverage and an entropy value. In such case the coverage of that method will be 0.0 and
            # as well as its entropy.
            if len(coverage_names) < len(args.names):
                for name in args.names:
                    if name not in coverage_names:
                        coverage_names[name] = 0.0
                        entropy_names[name] = 0.0

            coverage_trials_names[num_trial] = coverage_names
            entropy_trials_names[num_trial] = entropy_names

        coverages_names = dict()
        entropies_names = dict()
        for idx in range(len(args.names)):
            coverages = [
                coverage_trials_names[key][args.names[idx]]
                for key in coverage_trials_names.keys()
                if args.names[idx] in coverage_trials_names[key]
            ]
            coverages_names[args.names[idx]] = coverages
            lgg.info(
                "Mean coverage for name {} across {} trials: {}".format(
                    args.names[idx], max_num_trials, round(np.mean(coverages), 2)
                )
            )
            entropies = [
                entropy_trials_names[key][args.names[idx]]
                for key in entropy_trials_names.keys()
                if args.names[idx] in coverage_trials_names[key]
            ]
            entropies_names[args.names[idx]] = entropies
            lgg.info(
                "Mean entropy for name {} across {} trials: {}".format(
                    args.names[idx], max_num_trials, round(np.mean(entropies), 2)
                )
            )

        lgg.info(
            "Mean num clusters: {}, Max num clusters: {}, Min num clusters: {}".format(
                np.mean(num_clusters_trials),
                np.max(num_clusters_trials),
                np.min(num_clusters_trials),
            )
        )
        lgg.info("Num clusters array: {}".format(num_clusters_trials))

        if len(gini_coefficient_trials) > 0:
            lgg.info(
                "Mean Gini purity coefficients: {}, Max Gini purity coefficient: {}, Min Gini purity  coefficient: {}".format(
                    np.mean(gini_coefficient_trials) * 100,
                    np.max(gini_coefficient_trials) * 100,
                    np.min(gini_coefficient_trials) * 100,
                )
            )
        lgg.info("Gini purity coefficient trials: {}".format(gini_coefficient_trials))

        print()
        # statistical comparison coverage no adjust
        lgg.info("Coverage statistical analysis")
        statistical_analysis(
            logger=lgg,
            names_dict=coverages_names,
            nms=args.names,
            alpha=args.alpha,
            beta=args.beta,
            adjust=args.adjust,
            metric_name="Coverage",
        )
        lgg.info("Entropy statistical analysis")
        # statistical comparison entropy no adjust
        statistical_analysis(
            logger=lgg,
            names_dict=entropies_names,
            nms=args.names,
            alpha=args.alpha,
            beta=args.beta,
            adjust=args.adjust,
            metric_name="Entropy",
        )
