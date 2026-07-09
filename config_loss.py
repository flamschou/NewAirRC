from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from monai.losses import DiceCELoss


class DeepSupervisionLossBase(nn.Module):
    def __init__(
        self,
        deep_supr_num: int,
        weights: Optional[np.ndarray] = None,
        **kwargs_base_loss,
    ):
        super().__init__()
        self.num_levels = deep_supr_num + 1

        # Create weights
        if weights is not None:
            self.weights = torch.from_numpy(weights.astype(np.float32))
        else:
            weights_array = np.array(
                [1 / (2**i) for i in range(self.num_levels)], dtype=np.float32
            )
            weights_array[-1] = 0.0
            total_weight = weights_array.sum()
            if total_weight > 0:
                weights_array = weights_array / total_weight
            self.weights = torch.from_numpy(weights_array)

    def forward(
        self, outputs: torch.Tensor, labels: torch.Tensor, **kwargs_loss: Any
    ) -> torch.Tensor:
        # If not deep supervision format, calculate single loss
        if outputs.ndim != 6:
            return self.base_loss(outputs, labels, **kwargs_loss)

        # Calculate weighted sum of losses across all levels
        total_loss = torch.tensor(0.0, device=outputs.device)
        for level_index in range(self.num_levels):
            level_weight = self.weights[level_index]
            if level_weight < 1e-8:
                continue
            level_output = outputs[:, level_index, ...]
            level_loss = self.base_loss(level_output, labels, **kwargs_loss)
            total_loss += level_weight * level_loss

        return total_loss


class DeepSupervisionDiceCELoss(DeepSupervisionLossBase):
    def __init__(self, deep_supr_num: int, **kwargs):
        super().__init__(deep_supr_num, **kwargs)
        self.base_loss = DiceCELoss(to_onehot_y=True, softmax=True, **kwargs)
