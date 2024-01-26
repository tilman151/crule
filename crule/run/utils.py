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
        cutoff = num_slices * self.lower_window_size
        features = features[:, :cutoff].reshape(
            -1, self.lower_window_size, num_features
        )
        features = self.extract_windows(features, self.upper_window_size, num_slices)
        window_cutoff = self._cutoff(self.upper_window_size, num_slices)
        targets = targets.repeat(num_slices)[window_cutoff:]

        return features, targets

    def extract_windows(self, seq, window_size, dilation):
        if window_size * dilation > len(seq):
            raise ValueError(
                f"Cannot extract windows of size {window_size} with stride {dilation} "
                f"from a sequence of length {len(seq)}."
            )

        num_frames = seq.shape[0] - self._cutoff(window_size, dilation)
        window_idx = np.arange(window_size, dtype=np.uint16)[None, :] * dilation
        window_idx = window_idx + np.arange(num_frames, dtype=np.uint16)[:, None]
        windows = seq[window_idx]

        return windows

    def _cutoff(self, window_size, dilation):
        return (window_size - 1) * dilation
