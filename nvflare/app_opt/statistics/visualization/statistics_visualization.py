# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Dict

from nvflare.fuel.utils.import_utils import optional_import


def convert_data(feature_metrics) -> dict:
    converted = {}
    for statistic in feature_metrics:
        converted[statistic] = {}
        for site in feature_metrics[statistic]:
            for ds in feature_metrics[statistic][site]:
                site_dataset = f"{site}-{ds}"
                converted[statistic][site_dataset] = feature_metrics[statistic][site][ds]
    return converted


class Visualization:
    def import_modules(self):
        if self._imported:
            return self._display, self._pd

        display, import_flag = optional_import(module="IPython.display", name="display")
        if not import_flag:
            print(display.failure)
        pd, import_flag = optional_import(module="pandas")
        if not import_flag:
            print(pd.failure)
        self._display = display
        self._pd = pd
        self._imported = True
        return display, pd

    def show_stats(self, data, white_list_features=None):
        if white_list_features is None:
            white_list_features = []

        display, pd = self.import_modules()
        all_features = [k for k in data]
        target_features = self._get_target_features(all_features, white_list_features)
        for feature in target_features:
            print(f"\n{feature}\n")
            feature_metrics = data[feature]
            converted = convert_data(feature_metrics)
            df = pd.DataFrame.from_dict(converted)
            display(df)

    def show_histograms(self, data, display_format="sample_count", white_list_features=None, plot_type="both"):
        if white_list_features is None:
            white_list_features = []
        feature_dfs = self.get_histogram_dataframes(data, display_format, white_list_features)
        self.show_dataframe_plots(feature_dfs, plot_type)

    def show_dataframe_plots(self, feature_dfs, plot_type="both"):
        for feature in feature_dfs:
            df = feature_dfs[feature]
            if plot_type == "both":
                axes = df.plot.line(rot=40, title=feature)
                axes = df.plot.line(rot=40, subplots=True, title=feature)
            elif plot_type == "main":
                axes = df.plot.line(rot=40, title=feature)
            elif plot_type == "subplot":
                axes = df.plot.line(rot=40, subplots=True, title=feature)
            else:
                print(f"not supported plot type: '{plot_type}'")

    def get_histogram_dataframes(self, data, display_format="sample_count", white_list_features=None) -> Dict:
        if white_list_features is None:
            white_list_features = []
        display, pd = self.import_modules()

        # _prepare_histogram_data returns feature_hists, feature_edges
        hists, edges = self._prepare_histogram_data(data, display_format, white_list_features)
        # Fast lookup -- no unnecessary list-comp
        all_features = list(edges.keys())
        target_features = self._get_target_features(all_features, white_list_features)

        feature_dfs = {}
        # Localize pd.DataFrame for speed
        pd_DataFrame = pd.DataFrame
        for feature in target_features:
            hist_data = hists[feature]
            index = edges[feature]
            feature_dfs[feature] = pd_DataFrame(hist_data, index=index)

        return feature_dfs

    def _prepare_histogram_data(self, data, display_format="sample_count", white_list_features=None):
        if white_list_features is None:
            white_list_features = []
        # Use keys directly for performance
        all_features = list(data.keys())
        target_features = self._get_target_features(all_features, white_list_features)

        feature_hists = {}
        feature_edges = {}

        # Avoid repeated lookup in display_format
        is_percent = display_format == "percent"
        sum_counts_in_histogram = self.sum_counts_in_histogram

        for feature in target_features:
            converted = convert_data(data[feature])
            xs = converted["histogram"]
            hists = {}
            # Pre-allocate expected length for feature_edges if possible (not possible without knowing length)
            # Faster insert: bulk append only on first ds for edges
            feature_edges_list = []
            ds_items = list(xs.items())
            for i, (ds, ds_hist) in enumerate(ds_items):
                # Avoid repeated lookups, build list in single pass
                ds_bucket_counts = []
                # If first ds, collect edges in bulk for performance
                if i == 0:
                    feature_edges_list = [bucket[0] for bucket in ds_hist]
                if is_percent:
                    sum_value = sum_counts_in_histogram(ds_hist)
                    ds_bucket_counts = [bucket[2] / sum_value for bucket in ds_hist]
                else:
                    ds_bucket_counts = [bucket[2] for bucket in ds_hist]
                hists[ds] = ds_bucket_counts
            feature_edges[feature] = feature_edges_list
            feature_hists[feature] = hists

        return feature_hists, feature_edges

    def sum_counts_in_histogram(self, hist):
        sum_value = 0
        for bucket in hist:
            sum_value += bucket[2]
        return sum_value

    def _get_target_features(self, all_features, white_list_features=None):
        if white_list_features is None:
            white_list_features = []

        target_features = white_list_features
        if not white_list_features:
            target_features = all_features
        return target_features

    def __init__(self):
        self._display = None
        self._pd = None
        self._imported = False
