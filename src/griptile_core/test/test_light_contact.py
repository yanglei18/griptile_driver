import numpy as np
import pytest

from griptile_core.grid import SHAPE
from griptile_core.light_contact import LightContactTracker


STEP = 33_000_000
SENSOR_TIME = 978_459_534_000_000_000


class Scan:
    def __init__(self):
        self.tracker = LightContactTracker()
        self.time = SENSOR_TIME
        self.rows = np.zeros(SHAPE[0], dtype=np.int64)
        self.delta = np.zeros(SHAPE)
        self.response = np.zeros(SHAPE)
        self.threshold = np.ones(SHAPE)

    def read(self, rows=slice(None), step=STEP):
        self.time += step
        self.rows[rows] = self.time - SENSOR_TIME
        return self.tracker.update(self.time, self.rows, self.delta, self.response, self.threshold)

    def warm(self):
        for _ in range(8):
            assert not self.read().any()


@pytest.mark.parametrize("sign", (1, -1))
def test_neighboring_light_contact_is_signed_held_and_released(sign):
    s = Scan()
    s.warm()
    s.delta[4, 5:7] = sign * 1.8
    s.response[4, 5:7] = 1.3
    assert not s.read().any()
    for _ in range(12):
        result = s.read()
        assert result.sum() == 2 and result[4, 5] and result[4, 6]
    with pytest.raises(ValueError):
        result[4, 5] = False
    s.delta.fill(0)
    s.response.fill(0)
    assert not s.read().any()
    assert result.sum() == 2  # The archived mask is independent of subsequent frames.


def test_scattered_weak_changes_do_not_vote_and_duplicate_rows_do_not_vote_twice():
    s = Scan()
    s.warm()
    s.delta[3, 4] = s.delta[12, 9] = 1.8
    s.response[3, 4] = s.response[12, 9] = 1.3
    for _ in range(3):
        assert not s.read().any()
    s.delta[11, 17], s.response[11, 17] = 4, 3.5
    assert not s.read().any()
    assert not s.read(rows=slice(0, 5)).any()
    assert not s.read(rows=slice(0, 5)).any()
    result = s.read(rows=slice(11, 12))
    assert result.sum() == 1 and result[11, 17]


def test_time_gap_or_missing_source_cannot_confirm_a_stale_observation():
    s = Scan()
    s.warm()
    s.delta[4, 4] = 4
    s.response[4, 4] = 3.5
    assert not s.read().any()
    assert not s.read(step=180_000_000).any()
    for _ in range(15):
        assert not s.read().any()
    s.threshold[4, 4] = np.nan
    assert not s.read().any()


def test_rebound_next_to_recent_strong_contact_does_not_start_a_weak_contact():
    s = Scan()
    s.delta[7, 7], s.response[7, 7] = 12, 11.5
    s.warm()
    s.delta[7, 7], s.response[7, 7] = 0, 0
    s.delta[8, 8:10] = -2.5
    s.response[8, 8:10] = 2.
    s.delta[16, 8:10] = -2.5
    s.response[16, 8:10] = 2.
    assert not s.read().any()
    mask = s.read()
    assert not mask[8].any() and mask[16, 8:10].all()


def test_adjacent_rows_may_corroborate_on_staggered_physical_updates():
    s = Scan()
    s.warm()
    s.delta[7:9, 5] = 2
    s.response[7:9, 5] = 1.5
    assert not s.read(rows=7, step=15_000_000).any()
    assert not s.read(rows=8, step=15_000_000).any()
    assert not s.read(rows=7, step=15_000_000).any()
    mask = s.read(rows=8, step=15_000_000)
    assert mask[8, 5] and mask.sum() == 1
    mask = s.read(rows=7, step=15_000_000)
    assert mask[7, 5] and mask[8, 5] and mask.sum() == 2


def test_missing_row_clears_held_point_and_profile_gate_remains_required():
    s = Scan()
    s.warm()
    s.delta[9, 9:11] = 4
    s.response[9, 10] = 3.5
    assert not s.read().any()
    mask = s.read()
    assert mask.sum() == 1 and mask[9, 10]
    s.response[9] = np.nan
    assert not s.read().any()
    s.tracker.reset()
    s.response[9] = 3.5
    assert not s.read().any()
