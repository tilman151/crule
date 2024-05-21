from typing import Tuple, Any

import numpy as np
import torch
from rul_adapt.model import TwoStageExtractor
from torch import nn


class XjtuSyExtractor:
    def __init__(self, window_size: int) -> None:
        self.window_size = window_size

    def __call__(
        self, features: np.ndarray, targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        num_slices = features.shape[1] // self.window_size
        cutoff = num_slices * self.window_size
        features = features[:, :cutoff].reshape(-1, self.window_size, 2)
        targets = targets.repeat(num_slices)

        return features, targets


class XjtuSyWindowExtractor:
    def __init__(self, upper_window=10, lower_window=2560):
        self.upper_window = upper_window
        self.lower_window = lower_window

    def __call__(self, features, targets):
        num_channels = features.shape[-1]
        num_slices = features.shape[1] // self.lower_window
        last_idx = num_slices * self.lower_window
        reshaped = features[:, :last_idx].reshape(-1, self.lower_window, num_channels)
        reshaped = np.pad(reshaped, ((0, num_slices - 1), (0, 0), (0, 0)), mode="empty")

        window_shape = (self.upper_window * num_slices, self.lower_window, num_channels)
        features = np.lib.stride_tricks.sliding_window_view(reshaped, window_shape)
        features = features[:, 0, 0, ::num_slices]

        window_cutoff = (self.upper_window - 1) * num_slices
        targets = targets.repeat(num_slices)[window_cutoff:]

        return features, targets

    def __repr__(self):
        return (
            f"XjtuSyWindowExtractor(upper_window={self.upper_window}, "
            f"lower_window={self.lower_window})"
        )


class TwoStageWrapper(nn.Module):
    """
    Wrapper for TwoStageExtractor to ignore any additional kwargs.

    This is needed so that the hydra config can contain additional keys that are
    needed for reference elsewhere.
    """

    def __init__(
        self, lower_stage: nn.Module, upper_stage: nn.Module, **_kwargs: Any
    ) -> None:
        """
        Create a new TwoStageExtractor but ignore any additional kwargs.

        :param lower_stage: the lower stage extractor
        :param upper_stage: the upper stage extractor
        :param _kwargs: ignored kwargs
        """
        super().__init__()
        self.lower_stage = lower_stage
        self.upper_stage = upper_stage
        self._kwargs = _kwargs  # save as member to be compatible with checkpointing
        self._wrapped = TwoStageExtractor(lower_stage, upper_stage)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._wrapped(x)
