"""Provide BVI user profile loading and validation helpers.

This module is part of the BVI ACT-R navigation simulation workflow.
"""

try:
    from .utils import clamp
except ImportError:
    from utils import clamp

EXPERTISE_FIXED = 0.8


def normalize_familiarity_level(familiarity_level=1):
    """Handle normalize familiarity level behavior."""
    try:
        value = float(familiarity_level)
    except (TypeError, ValueError):
        value = 1.0
    return 1 if value >= 0.5 else 0


def get_user_profile_adjustments(familiarity_level=1, user_id="default"):
    """Handle get user profile adjustments behavior."""
    familiarity_level = normalize_familiarity_level(familiarity_level)
    expertise_proxy = clamp(EXPERTISE_FIXED)

    LANDMARK_EXPECTANCY_BONUS = 0.05 + 0.5 * familiarity_level
    SOUND_SOURCE_THRESHOLD = clamp(
        0.95 - 0.5 * expertise_proxy - 0.2 * familiarity_level,
        low=0.40,
        high=1.10,
    )
    profile_adjustments = {
        "D": 0.5,
        "MAS": 1.5,
        "RT": -2.0,
        "ANS": 0.2,
    }

    familiarity_level = round(familiarity_level, 2)
    expertise_proxy = round(expertise_proxy, 2)
    landmark_expectancy_bonus = round(LANDMARK_EXPECTANCY_BONUS, 3)
    sound_source_threshold = round(SOUND_SOURCE_THRESHOLD, 3)

    return {
        "USER_ID": user_id,
        "familiarity_level": familiarity_level,
        "FAMILIARITY_LEVEL": familiarity_level,
        "expertise_proxy": expertise_proxy,
        "EXPERTISE_PROXY": expertise_proxy,
        "landmark_expectancy_bonus": landmark_expectancy_bonus,
        "LANDMARK_EXPECTANCY_BONUS": landmark_expectancy_bonus,
        "sound_source_threshold": sound_source_threshold,
        "SOUND_SOURCE_THRESHOLD": sound_source_threshold,
        **profile_adjustments,
    }
