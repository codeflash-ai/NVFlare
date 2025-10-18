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


def _get_xgboost_model_attr(xgb_model):
    num_parallel_tree = int(
        xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_parallel_tree"]
    )
    num_trees = int(xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"])
    return num_parallel_tree, num_trees


def update_model(prev_model, model_update):
    if not prev_model:
        return model_update
    else:
        # Append all trees
        # get the parameters
        pre_model = prev_model["learner"]["gradient_booster"]["model"]
        upd_model = model_update["learner"]["gradient_booster"]["model"]

        pre_gbm_param = pre_model["gbtree_model_param"]
        upd_gbm_param = upd_model["gbtree_model_param"]

        pre_num_parallel_tree = int(pre_gbm_param["num_parallel_tree"])
        pre_num_trees = int(pre_gbm_param["num_trees"])
        cur_num_parallel_tree = int(upd_gbm_param["num_parallel_tree"])
        add_num_trees = int(upd_gbm_param["num_trees"])

        # check num_parallel_tree, should be consistent
        if cur_num_parallel_tree != pre_num_parallel_tree:
            raise ValueError(
                f"add_num_parallel_tree should not change, previous {pre_num_parallel_tree}, current {cur_num_parallel_tree}"
            )

        # Update the num_trees parameter
        pre_gbm_param["num_trees"] = str(pre_num_trees + cur_num_parallel_tree)

        # Fast appending of trees with id assignment
        trees_to_add = upd_model["trees"]
        start_id = pre_num_trees
        # Assign ids in a tight loop before appending for better cache locality and list extend
        # Instead of repeated append, use list.extend for single operation
        for i in range(cur_num_parallel_tree):
            trees_to_add[i]["id"] = start_id + i

        # Extend trees and tree_info together, single operation instead of many appends
        pre_trees = pre_model["trees"]
        pre_tree_info = pre_model["tree_info"]
        pre_trees.extend(trees_to_add[:cur_num_parallel_tree])
        pre_tree_info.extend([0] * cur_num_parallel_tree)

        # append iteration_indptr
        pre_model["iteration_indptr"].append(pre_num_trees + cur_num_parallel_tree)
        return prev_model
