from typing import Tuple

import numpy as np


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
