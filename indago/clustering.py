from typing import Tuple

import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from log import Log


def cluster_data(
    data: np.ndarray,
    n_clusters_interval: Tuple,
    seed: int,
    silhouette_threshold: int,
) -> Tuple:
    logger = Log("cluster_data")
    optimal_n_clusters = 1
    optimal_score = -1
    if n_clusters_interval[0] != 1:
        range_n_clusters = np.arange(n_clusters_interval[0], n_clusters_interval[1])
        optimal_score = -1
        optimal_n_clusters = -1
        for n_clusters in range_n_clusters:
            clusterer = KMeans(n_clusters=n_clusters, random_state=seed)
            cluster_labels = clusterer.fit_predict(data)
            try:
                silhouette_avg = silhouette_score(
                    data, cluster_labels, random_state=seed
                )
                logger.debug(
                    "For n_clusters = {}, the average silhouette score is: {}".format(
                        n_clusters, silhouette_avg
                    )
                )
                if silhouette_avg > optimal_score:
                    if silhouette_threshold < 0 or optimal_score == -1:
                        optimal_score = silhouette_avg
                        optimal_n_clusters = n_clusters
                        logger.debug(
                            "New optimal silhouette score: {}. Num clusters: {}".format(
                                silhouette_avg, n_clusters
                            )
                        )
                    else:
                        assert (
                            0 < silhouette_threshold <= 100
                        ), "Silhouette threshold needs to be in (0, 100]. Found {}".format(
                            silhouette_threshold
                        )
                        percentage_increase = round(
                            100 * (silhouette_avg - optimal_score) / optimal_score, 2
                        )
                        if percentage_increase >= silhouette_threshold:
                            optimal_score = silhouette_avg
                            optimal_n_clusters = n_clusters
                            logger.debug(
                                "New optimal silhouette score: {} > {}% of previous score {}. Num clusters: {}".format(
                                    silhouette_avg,
                                    silhouette_threshold,
                                    percentage_increase,
                                    n_clusters,
                                )
                            )
            except ValueError:
                logger.warn("Clustering ValueError exception")
                break

        assert optimal_n_clusters != -1, "Error in silhouette analysis"
        logger.debug(
            "Best score is {} for n_cluster = {}".format(
                optimal_score, optimal_n_clusters
            )
        )

    clusterer = KMeans(n_clusters=optimal_n_clusters).fit(data)
    labels = clusterer.labels_
    centers = clusterer.cluster_centers_
    return clusterer, labels, centers, optimal_score


def compute_pairwise_distances(data: np.ndarray) -> float:
    distances = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            distances.append(euclidean(u=data[i], v=data[j]))
    if len(distances) == 0:
        return 0.0
    return float(np.mean(distances))
