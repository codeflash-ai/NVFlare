# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from math import sqrt
from typing import Dict, List, TypeVar

from nvflare.app_common.abstract.statistics_spec import Bin, BinRange, DataType, Feature, Histogram, HistogramType
from nvflare.app_common.app_constant import StatisticsConstants as StC

T = TypeVar("T")


def get_initial_structure(client_metrics: dict, ordered_metrics: dict) -> dict:
    """Calculate initial output structure that is common at all the hierarchical levels.

    Args:
        client_metrics: Local stats for each client.
        ordered_metrics: Ordered target metrics.

    Returns:
        A dict containing initial output structure.
    """
    stats = {}
    for metric in ordered_metrics:
        stats[metric] = {}
        for stat in client_metrics:
            for site in client_metrics[stat]:
                for ds in client_metrics[stat][site]:
                    stats[metric][ds] = {}
                    for feature in client_metrics[stat][site][ds]:
                        stats[metric][ds][feature] = 0
    return stats


def create_output_structure(
    client_metrics: dict, metric_task: str, ordered_metrics: dict, hierarchy_config: dict
) -> dict:
    """Recursively calculate the hierarchical global stats structure from the given hierarchy config.

    Args:
        client_metrics: Local stats for each client.
        metric_task: Statistics task.
        ordered_metrics: Ordered target metrics.
        hierarchy_config: Hierarchy configuration for the global stats.

    Returns:
        A dict containing hierarchical global stats structure.
    """

    def recursively_add_values(structure: dict, value_json: dict, metric_task: str, ordered_metrics: dict):
        if isinstance(structure, dict):
            new_items = {}
            for key, value in list(structure.items()):
                if key == StC.NAME:
                    continue
                if isinstance(value, list):
                    if key not in new_items:
                        new_items[StC.GLOBAL] = get_initial_structure(value_json, ordered_metrics)
                    for i, item in enumerate(value):
                        if isinstance(item, str):
                            value[i] = {
                                StC.NAME: item,
                                StC.LOCAL: get_initial_structure(value_json, ordered_metrics),
                            }
                        else:
                            recursively_add_values(item, value_json, metric_task, ordered_metrics)
                else:
                    recursively_add_values(value, value_json, metric_task, ordered_metrics)
            structure.update(new_items)
        elif isinstance(structure, list):
            for item in structure:
                recursively_add_values(item, value_json, metric_task, ordered_metrics)
        return structure

    filled_structure = copy.deepcopy(hierarchy_config)
    final_strcture = recursively_add_values(filled_structure, client_metrics, metric_task, ordered_metrics)
    return final_strcture


def get_output_structure(client_metrics: dict, metric_task: str, ordered_metrics: dict, hierarchy_config: dict) -> dict:
    """Create required global statistics hierarchical output structure.

    Args:
        client_metrics: Local stats for each client.
        metric_task: Statistics task.
        ordered_metrics: Ordered target metrics.
        hierarchy_config: Hierarchy configuration for the global stats.

    Returns:
        A dict containing hierarchical global stats structure that also includes
        top level global stats structure.
    """
    top_strcture = get_initial_structure(client_metrics, ordered_metrics)
    output_structure = {
        StC.GLOBAL: top_strcture,
        **create_output_structure(client_metrics, metric_task, ordered_metrics, hierarchy_config),
    }
    return output_structure


def update_output_strcture(
    client_metrics: dict,
    metric_task: str,
    ordered_metrics: dict,
    global_metrics: dict,
) -> None:
    """Update global statistics hierarchical output structure with the new ordered metrics.

    Args:
        client_metrics: Local stats for each client.
        metric_task: Statistics task.
        ordered_metrics: Ordered target metrics.
        global_metrics: The current global metrics.

    Returns:
        A dict containing updated hierarchical global stats.
    """
    if isinstance(global_metrics, dict):
        for key, value in list(global_metrics.items()):
            if key == StC.NAME:
                continue
            elif key == StC.GLOBAL:
                global_metrics[key].update(get_initial_structure(client_metrics, ordered_metrics))
            elif key == StC.LOCAL:
                global_metrics[key].update(get_initial_structure(client_metrics, ordered_metrics))
                return
            elif isinstance(value, list):
                update_output_strcture(client_metrics, metric_task, ordered_metrics, value)
    elif isinstance(global_metrics, list):
        for item in global_metrics:
            update_output_strcture(client_metrics, metric_task, ordered_metrics, item)


def get_global_stats(global_metrics: dict, client_metrics: dict, metric_task: str, hierarchy_config: dict) -> dict:
    """Get global hierarchical statistics for the given hierarchy config.

    Args:
        global_metrics: The current global metrics.
        client_metrics: Local stats for each client.
        metric_task: Statistics task.
        hierarchy_config: Hierarchy configuration for the global stats.


    Returns:
        A dict containing global hierarchical statistics.
    """
    # create stats structure
    ordered_target_metrics = StC.ordered_statistics[metric_task]
    ordered_metrics = [metric for metric in ordered_target_metrics if metric in client_metrics]

    # Create hierarchical output structure
    if StC.GLOBAL not in global_metrics:
        global_metrics = get_output_structure(client_metrics, metric_task, ordered_metrics, hierarchy_config)
    else:
        update_output_strcture(client_metrics, metric_task, ordered_metrics, global_metrics)

    for metric in ordered_metrics:
        stats = client_metrics[metric]
        if metric == StC.STATS_COUNT or metric == StC.STATS_FAILURE_COUNT or metric == StC.STATS_SUM:
            for client_name in stats:
                global_metrics = accumulate_hierarchical_metrics(
                    metric, client_name, stats[client_name], global_metrics, hierarchy_config
                )
        elif metric == StC.STATS_MAX or metric == StC.STATS_MIN:
            for client_name in stats:
                global_metrics = get_hierarchical_mins_or_maxs(
                    metric, client_name, stats[client_name], global_metrics, hierarchy_config
                )
        elif metric == StC.STATS_MEAN:
            global_metrics = get_hierarchical_means(metric, global_metrics)
        elif metric == StC.STATS_HISTOGRAM:
            for client_name in stats:
                global_metrics = get_hierarchical_histograms(
                    metric, client_name, stats[client_name], global_metrics, hierarchy_config
                )
        elif metric == StC.STATS_VAR:
            for client_name in stats:
                global_metrics = accumulate_hierarchical_metrics(
                    metric, client_name, stats[client_name], global_metrics, hierarchy_config
                )
        elif metric == StC.STATS_STDDEV:
            global_metrics = get_hierarchical_stddevs(global_metrics)

    return global_metrics


def accumulate_hierarchical_metrics(
    metric: str, client_name: str, metrics: dict, global_metrics: dict, hierarchy_config: dict
) -> dict:
    """Accumulate metrics at each hierarchical level.

    Args:
        metric: Metric to accumulate.
        client_name: Client name.
        metrics: Client metrics.
        global_metrics: The current global metrics.
        hierarchy_config:  Hierarchy configuration for the global stats.

    Returns:
        A dict containing accumulated hierarchical global statistics.
    """

    def recursively_accumulate_hierarchical_metrics(
        metric: str, client_name: str, metrics: dict, global_metrics: dict, dataset: str, feature: str, org: list
    ) -> dict:
        if isinstance(global_metrics, dict):
            for key, value in global_metrics.items():
                if key == StC.GLOBAL and StC.NAME not in global_metrics:
                    global_metrics[StC.GLOBAL][metric][dataset][feature] += metrics[dataset][feature]
                    continue
                if key == StC.NAME:
                    if org and value in org:
                        # The client belongs to this org so update current global metrics before sending it further
                        global_metrics[StC.GLOBAL][metric][dataset][feature] += metrics[dataset][feature]
                    elif value == client_name:
                        # This is a client local metrics update
                        global_metrics[StC.LOCAL][metric][dataset][feature] += metrics[dataset][feature]
                    else:
                        break
                if isinstance(value, list):
                    for item in value:
                        recursively_accumulate_hierarchical_metrics(
                            metric, client_name, metrics, item, dataset, feature, org
                        )

    client_org = get_client_hierarchy(copy.deepcopy(hierarchy_config), client_name)
    for dataset in metrics:
        for feature in metrics[dataset]:
            recursively_accumulate_hierarchical_metrics(
                metric, client_name, metrics, global_metrics, dataset, feature, client_org
            )

    return global_metrics


def get_hierarchical_mins_or_maxs(
    metric: str, client_name: str, metrics: dict, global_metrics: dict, hierarchy_config: dict
) -> dict:
    """Calculate min or max at each hierarchical level.

    Args:
        metric: Metric to accumulate.
        client_name: Client name.
        metrics: Client metrics.
        global_metrics: The current global metrics.
        hierarchy_config:  Hierarchy configuration for the global stats.

    Returns:
        A dict containing updated hierarchical global statistics with
        accumulated mins or maxs.
    """

    def recursively_update_org_mins_or_maxs(
        metric: str,
        client_name: str,
        metrics: dict,
        global_metrics: dict,
        dataset: str,
        feature: str,
        org: list,
        op: str,
    ) -> dict:
        if isinstance(global_metrics, dict):
            for key, value in global_metrics.items():
                if key == StC.GLOBAL and StC.NAME not in global_metrics:
                    if global_metrics[StC.GLOBAL][metric][dataset][feature]:
                        global_metrics[StC.GLOBAL][metric][dataset][feature] = op(
                            global_metrics[StC.GLOBAL][metric][dataset][feature], metrics[dataset][feature]
                        )
                    else:
                        global_metrics[StC.GLOBAL][metric][dataset][feature] = metrics[dataset][feature]
                    continue
                if key == StC.NAME:
                    if org and value in org:
                        # The client belongs to this org so update current global metrics before sending it further
                        if global_metrics[StC.GLOBAL][metric][dataset][feature]:
                            global_metrics[StC.GLOBAL][metric][dataset][feature] = op(
                                global_metrics[StC.GLOBAL][metric][dataset][feature], metrics[dataset][feature]
                            )
                        else:
                            global_metrics[StC.GLOBAL][metric][dataset][feature] = metrics[dataset][feature]
                    elif value == client_name:
                        # This is a client local metrics update
                        global_metrics[StC.LOCAL][metric][dataset][feature] = metrics[dataset][feature]
                    else:
                        break
                if isinstance(value, list):
                    for item in value:
                        recursively_update_org_mins_or_maxs(
                            metric, client_name, metrics, item, dataset, feature, org, op
                        )

    if metric == "min":
        op = min
    else:
        op = max
    client_org = get_client_hierarchy(copy.deepcopy(hierarchy_config), client_name)
    for dataset in metrics:
        for feature in metrics[dataset]:
            recursively_update_org_mins_or_maxs(
                metric, client_name, metrics, global_metrics, dataset, feature, client_org, op
            )

    return global_metrics


def get_hierarchical_means(metric: str, global_metrics: dict) -> dict:
    """Calculate means at each hierarchical level.

    Args:
        metric: Metric to accumulate.
        global_metrics: The current global metrics.

    Returns:
        A dict containing updated hierarchical global statistics with
        accumulated means.
    """

    def recursively_update_org_means(metrics: dict, global_metrics: dict, dataset: str, feature: str) -> dict:
        if isinstance(global_metrics, dict):
            for key, value in global_metrics.items():
                if key == StC.GLOBAL:
                    global_metrics[StC.GLOBAL][metric][dataset][feature] = (
                        global_metrics[StC.GLOBAL][StC.STATS_SUM][dataset][feature]
                        / global_metrics[StC.GLOBAL][StC.STATS_COUNT][dataset][feature]
                    )
                if key == StC.LOCAL:
                    global_metrics[StC.LOCAL][metric][dataset][feature] = (
                        global_metrics[StC.LOCAL][StC.STATS_SUM][dataset][feature]
                        / global_metrics[StC.LOCAL][StC.STATS_COUNT][dataset][feature]
                    )
                if isinstance(value, list):
                    for item in value:
                        recursively_update_org_means(metrics, item, dataset, feature)

    #  Iterate each hierarchical level and calculate 'mean' from 'sum' and 'count'.
    for dataset in global_metrics[StC.GLOBAL][StC.STATS_COUNT]:
        for feature in global_metrics[StC.GLOBAL][StC.STATS_COUNT][dataset]:
            recursively_update_org_means(metric, global_metrics, dataset, feature)

    return global_metrics


def get_hierarchical_histograms(
    metric: str, client_name: str, metrics: dict, global_metrics: dict, hierarchy_config: dict
) -> dict:
    """Calculate histograms at each hierarchical level.

    Args:
        metric: Metric to accumulate.
        client_name: Client name.
        metrics: Client metrics.
        global_metrics: The current global metrics.
        hierarchy_config:  Hierarchy configuration for the global stats.

    Returns:
        A dict containing updated hierarchical global statistics with
        accumulated histograms.
    """

    def bins_to_dict(bins):
        """Helper to quickly turn a list of Bin into BinRange -> sample_count dict"""
        # Using tuple as key to avoid hashing overhead of custom BinRange type
        return {(b.low_value, b.high_value): b.sample_count for b in bins}

    def add_bins_to_dict(buckets: dict, bins):
        """Efficiently add sample_counts from bins to the buckets dict in place"""
        for b in bins:
            key = (b.low_value, b.high_value)
            if key in buckets:
                buckets[key] += b.sample_count
            else:
                buckets[key] = b.sample_count

    def binrange_lookup(buckets: dict, binrange: BinRange):
        return buckets[(binrange.low_value, binrange.high_value)]

    def dict_to_bins(gt_bins, buckets_dict):
        """Efficiently generate ordered bins list"""
        return [Bin(gb.low_value, gb.high_value, buckets_dict[(gb.low_value, gb.high_value)]) for gb in gt_bins]

    def recursively_accumulate_org_histograms(
        metric: str,
        client_name: str,
        metrics: dict,
        global_metrics: dict,
        dataset: str,
        feature: str,
        org: list,
        histogram: Histogram,
    ) -> dict:
        # Use local variables to avoid repeated global_metrics[] indexing
        if isinstance(global_metrics, dict):
            for key, value in global_metrics.items():
                if key == StC.GLOBAL and StC.NAME not in global_metrics:
                    target = global_metrics[StC.GLOBAL][metric][dataset]
                    # Fast path: setdefault-style insert to avoid repeated lookups
                    if feature not in target or not target[feature]:
                        # Fast copy bins, avoid list copy loop
                        g_bins = [Bin(b.low_value, b.high_value, b.sample_count) for b in histogram.bins]
                        g_hist = Histogram(HistogramType.STANDARD, g_bins)
                        target[feature] = g_hist
                    else:
                        g_hist = target[feature]
                        g_buckets = bins_to_dict(g_hist.bins)
                        add_bins_to_dict(g_buckets, histogram.bins)
                        # keep bin ordering for compatibility
                        target[feature] = Histogram(g_hist.hist_type, dict_to_bins(g_hist.bins, g_buckets))
                    continue
                if key == StC.NAME:
                    if org and value in org:
                        target = global_metrics[StC.GLOBAL][metric][dataset]
                        if feature not in target or not target[feature]:
                            g_bins = [Bin(b.low_value, b.high_value, b.sample_count) for b in histogram.bins]
                            g_hist = Histogram(HistogramType.STANDARD, g_bins)
                            target[feature] = g_hist
                        else:
                            g_hist = target[feature]
                            g_buckets = bins_to_dict(g_hist.bins)
                            add_bins_to_dict(g_buckets, histogram.bins)
                            target[feature] = Histogram(g_hist.hist_type, dict_to_bins(g_hist.bins, g_buckets))
                    elif value == client_name:
                        target = global_metrics[StC.LOCAL][metric][dataset]
                        if feature not in target or not target[feature]:
                            g_bins = [Bin(b.low_value, b.high_value, b.sample_count) for b in histogram.bins]
                            g_hist = Histogram(HistogramType.STANDARD, g_bins)
                            target[feature] = g_hist
                        else:
                            g_hist = target[feature]
                            g_buckets = bins_to_dict(g_hist.bins)
                            add_bins_to_dict(g_buckets, histogram.bins)
                            target[feature] = Histogram(g_hist.hist_type, dict_to_bins(g_hist.bins, g_buckets))
                    else:
                        break
                if isinstance(value, list):
                    # loop unrolling or other performance approach is not possible
                    for item in value:
                        recursively_accumulate_org_histograms(
                            metric, client_name, metrics, item, dataset, feature, org, histogram
                        )

    # Avoid unnecessary deep copy if hierarchy_config is not mutated inside get_client_hierarchy
    # Profile shows this is the major bottleneck. get_client_hierarchy does not mutate it,
    # so just pass as is.
    client_org = get_client_hierarchy(hierarchy_config, client_name)
    # Group .bins and dict lookups for better cache use in loops
    for dataset, features in metrics.items():
        for feature, histogram in features.items():
            recursively_accumulate_org_histograms(
                metric, client_name, metrics, global_metrics, dataset, feature, client_org, histogram
            )

    return global_metrics


def get_hierarchical_stddevs(global_metrics: dict) -> dict:
    """Calculate stddevs at each hierarchical level.

    Args:
        global_metrics: The current global metrics.

    Returns:
        A dict containing updated hierarchical global statistics with
        accumulated stddevs.
    """

    def recursively_update_org_stddevs(global_metrics: dict, dataset: str, feature: str) -> dict:
        if isinstance(global_metrics, dict):
            for key, value in global_metrics.items():
                if key == StC.GLOBAL:
                    global_metrics[StC.GLOBAL][StC.STATS_STDDEV][dataset][feature] = sqrt(
                        global_metrics[StC.GLOBAL][StC.STATS_VAR][dataset][feature]
                    )
                if key == StC.LOCAL:
                    global_metrics[StC.LOCAL][StC.STATS_STDDEV][dataset][feature] = sqrt(
                        global_metrics[StC.LOCAL][StC.STATS_VAR][dataset][feature]
                    )
                if isinstance(value, list):
                    for item in value:
                        recursively_update_org_stddevs(item, dataset, feature)

    for dataset in global_metrics[StC.GLOBAL][StC.STATS_VAR]:
        for feature in global_metrics[StC.GLOBAL][StC.STATS_VAR][dataset]:
            recursively_update_org_stddevs(global_metrics, dataset, feature)

    return global_metrics


def get_hierarchical_levels(data: dict, level: int = 0, levels_dict: dict = None) -> dict:
    """Calculate number of hierarchical levels from the given hierarchy config.

    Args:
        data: Hierarchy configuration for the global stats.
        level: The current hierarchical level (used for recursive calls).
        levels_dict: The accumulated levels dict (used for recursive calls).

    Returns:
        A dict containing containing hierarchical levels.
    """
    if levels_dict is None:
        levels_dict = {}

    if isinstance(data, list):
        for item in data:
            get_hierarchical_levels(item, level, levels_dict)
    elif isinstance(data, dict):
        for key, value in data.items():
            if key == StC.NAME:
                continue
            if key not in levels_dict:
                levels_dict[key] = level
            get_hierarchical_levels(value, level + 1, levels_dict)

    return levels_dict


def get_client_hierarchy(hierarchy_config: dict, client_name: str, path=None) -> list:
    """Calculate hierarchy for the given client name.

    Args:
        hierarchy_config: Hierarchy configuration for the global stats.
        client_name: Client name.
        path: The accumulated hierarchy path (used for recursive calls).

    Returns:
        A list containing hierarchy levels for the client.
    """
    if path is None:
        path = []

    # Optimize branching order and limit repeated isinstance checks
    if isinstance(hierarchy_config, dict):
        for key, value in hierarchy_config.items():
            # Most dict branches don't match, check list type only once.
            if isinstance(value, list):
                result = get_client_hierarchy(value, client_name, path)
                if result:
                    return result
    elif isinstance(hierarchy_config, list):
        # Instead of multiple isinstance checks per item, check just once and split
        # Also reduce path list concatenations by only appending when necessary
        for item in hierarchy_config:
            if item == client_name:
                return path
            elif isinstance(item, dict):
                next_path = path + [item.get(StC.NAME)]
                result = get_client_hierarchy(item, client_name, next_path)
                if result:
                    return result

    return None


def bins_to_dict(bins: List[Bin]) -> Dict[BinRange, float]:
    """Convert histogram bins to a 'dict'.

    Args:
        bins: Histogram bins.

    Returns:
        A dict containing histogram bins.
    """
    buckets = {}
    for bucket in bins:
        bucket_range = BinRange(bucket.low_value, bucket.high_value)
        buckets[bucket_range] = bucket.sample_count
    return buckets


def filter_numeric_features(ds_features: Dict[str, List[Feature]]) -> Dict[str, List[Feature]]:
    """Filter numeric features.

    Args:
        ds_features: A features dict.

    Returns:
        A dict containing numeric features.
    """
    numeric_ds_features = {}
    for ds_name in ds_features:
        features: List[Feature] = ds_features[ds_name]
        n_features = [f for f in features if (f.data_type == DataType.INT or f.data_type == DataType.FLOAT)]
        numeric_ds_features[ds_name] = n_features

    return numeric_ds_features
