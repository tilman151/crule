from typing import List

import pytest

from crule.run import common


@pytest.mark.parametrize("percent_fail_runs", [None, 1.0, 0.8, 0.6, 0.4, 0.2])
def test_get_cv_config(percent_fail_runs):
    config = {
        "target": {"reader": {"fd": 1, "percent_fail_runs": percent_fail_runs}},
        "replications": 5,
        "dataset": "ncmapss",
    }

    cv_configs = list(common.get_cv_configs(config))

    assert len(cv_configs) == 5
    entity_idx = [
        cv_config["target"]["reader"]["percent_fail_runs"] for cv_config in cv_configs
    ]
    if percent_fail_runs is None or percent_fail_runs == 1.0:
        assert all(idx == entity_idx[0] for idx in entity_idx)
    else:
        for i in range(5):
            for j in range(5):
                if not i == j:
                    assert not entity_idx[i] == entity_idx[j]
