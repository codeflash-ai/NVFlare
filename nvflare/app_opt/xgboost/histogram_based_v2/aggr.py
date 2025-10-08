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


class Aggregator:
    def __init__(self, initial_value=0):
        self.initial_value = initial_value

    def add(self, a, b):
        return a + b

    def _update_aggregation(self, gh_values, sample_bin_assignment, sample_id, aggr):
        bin_id = sample_bin_assignment[sample_id]
        if bin_id < 0:
            return

        sample_value = gh_values[sample_id]
        current_value = aggr[bin_id]
        if current_value == 0:
            # avoid add since sample_value may be cypher-text!
            aggr[bin_id] = sample_value
        else:
            aggr[bin_id] = self.add(current_value, sample_value)

    def aggregate(self, gh_values: list, sample_bin_assignment, num_bins, sample_ids):
        aggr_result = [self.initial_value] * num_bins
        add = self.add if hasattr(self, "add") else None  # Capture method reference if exists
        # Fused loop: move _update_aggregation logic here for lower call overhead
        if not sample_ids:
            for sample_id, sample_value in enumerate(gh_values):
                bin_id = sample_bin_assignment[sample_id]
                if bin_id < 0:
                    continue
                current_value = aggr_result[bin_id]
                if current_value == 0:
                    # avoid add since sample_value may be cypher-text!
                    aggr_result[bin_id] = sample_value
                else:
                    aggr_result[bin_id] = add(current_value, sample_value)
        else:
            for sample_id in sample_ids:
                bin_id = sample_bin_assignment[sample_id]
                if bin_id < 0:
                    continue
                sample_value = gh_values[sample_id]
                current_value = aggr_result[bin_id]
                if current_value == 0:
                    # avoid add since sample_value may be cypher-text!
                    aggr_result[bin_id] = sample_value
                else:
                    aggr_result[bin_id] = add(current_value, sample_value)
        return aggr_result
