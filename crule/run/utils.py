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


class NcmapssAverageExtractor:
    def __init__(self, num_sections: int, padding_value: float) -> None:
        self.num_sections = num_sections
        self.padding_value = padding_value

    def __call__(self, features: np.ndarray, targets: np.ndarray):
        num_samples, _, num_channels = features.shape
        reduced_features = np.empty([num_samples, self.num_sections * num_channels])
        for i, cycle in enumerate(features):
            first_padding_idx = np.argmax(np.all(cycle == self.padding_value, axis=1))
            sections = np.split(cycle[:first_padding_idx], 2)
            reduced_features[i] = np.concatenate([s.mean(axis=0) for s in sections])

        return reduced_features, targets
