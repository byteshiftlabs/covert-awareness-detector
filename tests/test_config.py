"""Unit tests for config.py."""

import sys
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import (
    SUBJECTS, N_SUBJECTS, LOR_TIME, ROR_TIME,
    CONSCIOUS_CONDITIONS, UNCONSCIOUS_CONDITIONS, CONDITIONS,
    N_ROIS, FD_THRESHOLD, FD_COLUMN, TRANSITION_BUFFER,
    SPECIAL_SUBJECTS, RANDOM_STATE
)


def test_subject_count():
    """25 usable subjects (sub-01 excluded, sub-30 has no timing data)."""
    assert N_SUBJECTS == 25
    assert len(SUBJECTS) == N_SUBJECTS


def test_subjects_are_sorted():
    """Subject IDs should be in order."""
    assert SUBJECTS == sorted(SUBJECTS)


def test_all_subjects_have_timing():
    """Every subject must have both LOR and ROR timing entries."""
    for subject in SUBJECTS:
        assert subject in LOR_TIME, f"{subject} missing LOR time"
        assert subject in ROR_TIME, f"{subject} missing ROR time"


def test_timing_values_positive():
    """All LOR/ROR times must be positive TR indices."""
    for subject in SUBJECTS:
        assert LOR_TIME[subject] > 0, f"{subject} LOR <= 0"
        assert ROR_TIME[subject] > 0, f"{subject} ROR <= 0"


def test_conditions():
    """7 conditions total: 6 conscious + 1 unconscious (LOR)."""
    assert len(CONDITIONS) == 7
    assert set(CONDITIONS.keys()) == {0, 1, 2, 3, 4, 5, 6}
    assert UNCONSCIOUS_CONDITIONS == [3]
    assert sorted(CONSCIOUS_CONDITIONS + UNCONSCIOUS_CONDITIONS) == [0, 1, 2, 3, 4, 5, 6]


def test_atlas_parameters():
    """446 ROIs from 4S456Parcels atlas."""
    assert N_ROIS == 446
    assert FD_THRESHOLD == 0.8
    assert FD_COLUMN == 7
    assert TRANSITION_BUFFER == 375


def test_special_subjects():
    """sub-29 is the only special case (no post-ROR segment)."""
    assert SPECIAL_SUBJECTS == {"sub-29"}
    assert "sub-29" in SUBJECTS


def test_random_state():
    """Random state should be set for reproducibility."""
    assert isinstance(RANDOM_STATE, int)
