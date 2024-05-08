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
    def __init__(
        self, num_sections: int, padding_value: float, window_size: Optional[int] = None
    ) -> None:
        self.num_sections = num_sections
        self.padding_value = padding_value
        self.window_size = window_size

    def __call__(self, features: np.ndarray, targets: np.ndarray):
        num_samples, _, num_channels = features.shape
        reduced_features = np.empty([num_samples, self.num_sections * num_channels])
        for i, cycle in enumerate(features):
            last_padding_idx = np.argmin(np.all(cycle == self.padding_value, axis=1))
            section_size = (len(cycle) - last_padding_idx) // self.num_sections
            split_idx = list(range(0, len(cycle) - last_padding_idx + 1, section_size))
            sections = np.split(cycle[last_padding_idx:], split_idx[1:-1])
            reduced_features[i] = np.concatenate([s.mean(axis=0) for s in sections])

        if self.window_size is not None and num_samples < self.window_size:
            padding = ((self.window_size - num_samples, 0), (0, 0))
            reduced_features = np.pad(
                reduced_features,
                padding,
                mode="constant",
                constant_values=self.padding_value,
            )
            targets = np.pad(
                targets,
                (self.window_size - num_samples, 0),
                mode="constant",
                constant_values=0,
            )

        return reduced_features, targets

    def __repr__(self):
        return (
            f"NcmapssAverageExtractor(num_sections={self.num_sections}, "
            f"padding_value={self.padding_value}, window_size={self.window_size})"
        )
