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

import torch
from torch.nn.modules.loss import _Loss


class PTFedProxLoss(_Loss):
    def __init__(self, mu: float = 0.01) -> None:
        """Compute FedProx loss: a loss penalizing the deviation from global model.

        Args:
            mu: weighting parameter
        """
        super().__init__()
        if mu < 0.0:
            raise ValueError("mu should be no less than 0.0")
        self.mu = mu

    def forward(self, input, target) -> torch.Tensor:
        """Forward pass in training.

        Args:
            input (nn.Module): the local pytorch model
            target (nn.Module): the copy of global pytorch model when local clients received it
                                at the beginning of each local round

        Returns:
            FedProx loss term
        """
        # Optimization: collect parameter tensors into lists, then compute difference tensor-wise.
        input_params = [p for _, p in input.named_parameters()]
        target_params = [p for _, p in target.named_parameters()]
        # This is safe as the original code assumes zipped order is correct.
        prox_terms = []
        mu_div_2 = self.mu / 2
        for p, r in zip(input_params, target_params):
            prox_terms.append(torch.sum((p - r) ** 2))
        if prox_terms:
            prox_loss: torch.Tensor = mu_div_2 * torch.stack(prox_terms).sum()
        else:
            prox_loss: torch.Tensor = torch.tensor(0.0, device=input_params[0].device if input_params else "cpu")
        return prox_loss
