# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from enum import Enum
from typing import Any, Dict, Optional, Union

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.fuel.utils.validation_utils import check_non_negative_int, check_object_type


class ParamsType(str, Enum):
    FULL = "FULL"
    DIFF = "DIFF"


class FLModelConst:
    PARAMS_TYPE = "params_type"
    PARAMS = "params"
    OPTIMIZER_PARAMS = "optimizer_params"
    METRICS = "metrics"
    CURRENT_ROUND = "current_round"
    START_ROUND = "start_round"
    TOTAL_ROUNDS = "total_rounds"
    META = "meta"


class MetaKey(FLMetaKey):
    pass


class FLModel:
    def __init__(
        self,
        params_type: Union[None, str, "ParamsType"] = None,
        params: Any = None,
        optimizer_params: Any = None,
        metrics: Optional[Dict] = None,
        start_round: Optional[int] = 0,
        current_round: Optional[int] = None,
        total_rounds: Optional[int] = None,
        meta: Optional[Dict] = None,
    ):
        """FLModel is a standardize data structure for NVFlare to communicate with external systems.

        Args:
            params_type: type of the parameters. It only describes the "params".
                If params_type is None, params need to be None.
                If params is provided but params_type is not provided, then it will be treated as FULL.
            params: model parameters, for example: model weights for deep learning.
            optimizer_params: optimizer parameters.
                For many cases, the optimizer parameters don't need to be transferred during FL training.
            metrics: evaluation metrics such as loss and scores.
            current_round: the current FL rounds. A round means round trip between client/server during training.
                None for inference.
            total_rounds: total number of FL rounds. A round means round trip between client/server during training.
                None for inference.
            meta: metadata dictionary used to contain any key-value pairs to facilitate the process.
        """
        if params_type is None:
            if params is not None:
                params_type = ParamsType.FULL
        else:
            params_type = ParamsType(params_type)

        if params_type == ParamsType.FULL or params_type == ParamsType.DIFF:
            if params is None:
                raise ValueError(f"params must be provided when params_type is {params_type}")

        if metrics is not None:
            check_object_type("metrics", metrics, dict)
        if start_round is not None:
            check_non_negative_int("start_round", start_round)
        if current_round is not None:
            check_non_negative_int("current_round", current_round)
        if total_rounds is not None:
            check_non_negative_int("total_rounds", total_rounds)

        self.params_type = params_type
        self.params = params
        self.optimizer_params = optimizer_params
        self.metrics = metrics
        self.start_round = start_round
        self.current_round = current_round
        self.total_rounds = total_rounds

        if meta is not None:
            check_object_type("meta", meta, dict)
        else:
            meta = {}
        self.meta = meta
        self._summary: dict = {}

    def _add_to_summary(self, kvs: Dict):
        for key, value in kvs.items():
            if value:
                if isinstance(value, dict):
                    self._summary[key] = len(value)
                elif isinstance(value, ParamsType):
                    self._summary[key] = value
                elif isinstance(value, int):
                    self._summary[key] = value
                else:
                    self._summary[key] = type(value)

    def summary(self):
        # Instead of recreating self._summary by incrementally writing to it,
        # build a new dict and assign, minimizing operations and object churn.
        params = self.params
        optimizer_params = self.optimizer_params
        metrics = self.metrics
        meta = self.meta
        params_type = self.params_type
        start_round = self.start_round
        current_round = self.current_round
        total_rounds = self.total_rounds

        summary: dict = {}

        # Inline the logic from _add_to_summary to avoid expensive multiple dict operations
        if params:
            if isinstance(params, dict):
                summary["params"] = len(params)
            elif isinstance(params, ParamsType):
                summary["params"] = params
            elif isinstance(params, int):
                summary["params"] = params
            else:
                summary["params"] = type(params)
        if optimizer_params:
            if isinstance(optimizer_params, dict):
                summary["optimizer_params"] = len(optimizer_params)
            elif isinstance(optimizer_params, ParamsType):
                summary["optimizer_params"] = optimizer_params
            elif isinstance(optimizer_params, int):
                summary["optimizer_params"] = optimizer_params
            else:
                summary["optimizer_params"] = type(optimizer_params)
        if metrics:
            if isinstance(metrics, dict):
                summary["metrics"] = len(metrics)
            elif isinstance(metrics, ParamsType):
                summary["metrics"] = metrics
            elif isinstance(metrics, int):
                summary["metrics"] = metrics
            else:
                summary["metrics"] = type(metrics)
        if meta:
            if isinstance(meta, dict):
                summary["meta"] = len(meta)
            elif isinstance(meta, ParamsType):
                summary["meta"] = meta
            elif isinstance(meta, int):
                summary["meta"] = meta
            else:
                summary["meta"] = type(meta)
        if params_type:
            if isinstance(params_type, dict):
                summary["params_type"] = len(params_type)
            elif isinstance(params_type, ParamsType):
                summary["params_type"] = params_type
            elif isinstance(params_type, int):
                summary["params_type"] = params_type
            else:
                summary["params_type"] = type(params_type)
        if start_round:
            if isinstance(start_round, dict):
                summary["start_round"] = len(start_round)
            elif isinstance(start_round, ParamsType):
                summary["start_round"] = start_round
            elif isinstance(start_round, int):
                summary["start_round"] = start_round
            else:
                summary["start_round"] = type(start_round)
        if current_round:
            if isinstance(current_round, dict):
                summary["current_round"] = len(current_round)
            elif isinstance(current_round, ParamsType):
                summary["current_round"] = current_round
            elif isinstance(current_round, int):
                summary["current_round"] = current_round
            else:
                summary["current_round"] = type(current_round)
        if total_rounds:
            if isinstance(total_rounds, dict):
                summary["total_rounds"] = len(total_rounds)
            elif isinstance(total_rounds, ParamsType):
                summary["total_rounds"] = total_rounds
            elif isinstance(total_rounds, int):
                summary["total_rounds"] = total_rounds
            else:
                summary["total_rounds"] = type(total_rounds)

        self._summary = summary
        return self._summary

    def __repr__(self):
        return str(self.summary())

    def __str__(self):
        return str(self.summary())
