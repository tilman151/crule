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
    def __init__(self, upper_window_size=10, lower_window_size=2560):
        self.upper_window_size = upper_window_size
        self.lower_window_size = lower_window_size

    def __call__(self, features, targets):
        features = features.astype(np.float16)
        num_slices = features.shape[1] // self.lower_window_size
        num_features = features.shape[-1]
        window_shape = (self.upper_window_size, self.lower_window_size, num_features)
        features = np.lib.stride_tricks.sliding_window_view(features, window_shape)
        features = features[:, :: self.lower_window_size]  # tumbling over lower windows
        window_cutoff = (self.upper_window_size - 1) * num_slices
        targets = targets.repeat(num_slices)[window_cutoff:]

        return features, targets

    def __repr__(self):
        return (
            f"XjtuSyWindowExtractor(upper_window_size={self.upper_window_size}, "
            f"lower_window_size={self.lower_window_size})"
        )
