import numpy as np
import numpy.testing as npt

from crule.run.utils import NcmapssAverageExtractor


def test_ncmapss_average_extractor():
    features = np.random.randn(2, 100, 5)
    features[0, 50:] = -1.0  # add padding
    features[1, 60:] = -1.0
    targets = np.random.rand(2)

    extractor = NcmapssAverageExtractor(num_sections=2, padding_value=-1.0)
    reduced_features, reduced_targets = extractor(features, targets)

    assert reduced_features.shape == (2, 10)
    assert reduced_targets.shape == (2,)
    npt.assert_equal(reduced_targets, targets)

    first_sample = reduced_features[0]
    npt.assert_almost_equal(first_sample[:5], np.mean(features[0, :25], axis=0))
    npt.assert_almost_equal(first_sample[5:], np.mean(features[0, 25:50], axis=0))

    second_sample = reduced_features[1]
    npt.assert_almost_equal(second_sample[:5], np.mean(features[1, :30], axis=0))
    npt.assert_almost_equal(second_sample[5:], np.mean(features[1, 30:60], axis=0))
