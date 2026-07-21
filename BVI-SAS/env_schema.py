"""Define environment schemas and validation helpers for navigation maps.

This module is part of the BVI ACT-R navigation simulation workflow.
"""

import math
import random

try:
    from .utils import clamp
except ImportError:
    from utils import clamp


ENVIRONMENT_TYPES = {
    "intersection": {
        "cn": "路口/十字路口",
        "landmark_presence": 0.72,
        "sound_salience": 0.68,
        "drift_risk": 0.35,
        "guidance_absent_risk": 0.45,
    },
    "tactile_guidance": {
        "cn": "盲道/导航路段",
        "landmark_presence": 0.85,
        "sound_salience": 0.38,
        "drift_risk": 0.08,
        "guidance_absent_risk": 0.05,
    },
    "flat_road": {
        "cn": "平整路面（人行道/广场）",
        "landmark_presence": 0.42,
        "sound_salience": 0.32,
        "drift_risk": 0.28,
        "guidance_absent_risk": 0.50,
    },
    "uneven_natural": {
        "cn": "不平整自然路面",
        "landmark_presence": 0.25,
        "sound_salience": 0.48,
        "drift_risk": 0.65,
        "guidance_absent_risk": 0.75,
    },
    "slope_surface": {
        "cn": "坡度路面",
        "landmark_presence": 0.38,
        "sound_salience": 0.55,
        "drift_risk": 0.72,
        "guidance_absent_risk": 0.68,
    },
    "height_drop": {
        "cn": "高度落差（楼梯/高路缘）",
        "landmark_presence": 0.88,
        "sound_salience": 0.75,
        "drift_risk": 0.15,
        "guidance_absent_risk": 0.10,
    },
}


def compute_chunk_base_level(
    chunk_type,
    environment_type,
    familiarity_level,
    frequency_scale=1.0,
):
    """Handle compute chunk base level behavior."""
    env_schema = ENVIRONMENT_TYPES.get(environment_type, {})
    familiarity = clamp(familiarity_level)

    base_relevance = {
        "landmark": env_schema.get("landmark_presence", 0.5),
        "sound_salience_high": env_schema.get("sound_salience", 0.5),
        "spatial_drift": env_schema.get("drift_risk", 0.5),
        "guidance_absent": env_schema.get("guidance_absent_risk", 0.5),
        "obstacle_alert": 0.5,
        "safe_progress": 1.0 - env_schema.get("drift_risk", 0.5),
    }.get(chunk_type, 0.5)

    base_level = base_relevance * (1.0 + 2.5 * familiarity * frequency_scale)

    return clamp(base_level, low=0.0, high=3.0)


def compute_associative_strength(
    source_chunk_type,
    target_chunk_type,
    environment_type,
    familiarity_level,
    co_occurrence_scale=1.0,
):
    """Handle compute associative strength behavior."""
    env_schema = ENVIRONMENT_TYPES.get(environment_type, {})
    familiarity = clamp(familiarity_level)

    co_occurrence_matrix = {
        ("landmark", "spatial_drift"): (1.0 - env_schema.get("drift_risk", 0.5)),
        ("guidance_absent", "spatial_drift"): env_schema.get(
            "guidance_absent_risk", 0.5
        ),
        ("guidance_absent", "obstacle_alert"): env_schema.get(
            "guidance_absent_risk", 0.5
        ),
        ("sound_salience_high", "attention_gate"): env_schema.get(
            "sound_salience", 0.5
        ),
        ("landmark", "attention_gate"): env_schema.get("landmark_presence", 0.5),
    }

    key = (source_chunk_type, target_chunk_type)
    co_occur_prob = co_occurrence_matrix.get(key, 0.3)

    S = (1.0 * (2.0 * co_occur_prob - 1.0) + (familiarity - 0.5)) * co_occurrence_scale

    return clamp(S, low=-2.5, high=1.5)


def build_environment_schema(familiarity_level, expertise_proxy=0.5):
    """Handle build environment schema behavior."""
    chunk_types = [
        "landmark",
        "sound_salience_high",
        "spatial_drift",
        "guidance_absent",
        "obstacle_alert",
        "safe_progress",
    ]

    schema = {}
    for env_type in ENVIRONMENT_TYPES.keys():
        schema[env_type] = {}
        for chunk_type in chunk_types:
            base_level = compute_chunk_base_level(
                chunk_type,
                env_type,
                familiarity_level,
                frequency_scale=1.0,
            )

            associations = {}
            for src_chunk in chunk_types:
                if src_chunk != chunk_type:
                    S = compute_associative_strength(
                        src_chunk,
                        chunk_type,
                        env_type,
                        familiarity_level,
                        co_occurrence_scale=1.0,
                    )
                    associations[src_chunk] = S

            schema[env_type][chunk_type] = {
                "base_level": base_level,
                "associative_strength": associations,
            }

    return schema


def _compute_context_weights(environment_type, context=None):
    """Handle compute context weights behavior."""
    env_schema = ENVIRONMENT_TYPES.get(environment_type, {})
    context = context or {}

    base_weights = {
        "landmark": 0.20,
        "sound_salience_high": env_schema.get("sound_salience", 0.5),
        "guidance_absent": env_schema.get("guidance_absent_risk", 0.5),
        "spatial_drift": env_schema.get("drift_risk", 0.5),
    }

    if context.get("crossing_active"):
        base_weights["sound_salience_high"] += 0.20
    if context.get("cane_guidance_present"):
        base_weights["guidance_absent"] *= 0.35
    if context.get("vehicle_approach"):
        base_weights["sound_salience_high"] += 0.15
    if context.get("no_reference"):
        base_weights["guidance_absent"] += 0.20
        base_weights["spatial_drift"] += 0.10

    total = sum(max(0.0, v) for v in base_weights.values())
    if total <= 0:
        return {k: 0.25 for k in base_weights}
    return {k: max(0.0, v) / total for k, v in base_weights.items()}


def get_chunk_activation_terms(
    environment_type,
    chunk_type,
    familiarity_level,
    experience_boost=1.0,
    context=None,
    noise_scale=0.03,
):
    """Handle get chunk activation terms behavior."""
    env_schema = ENVIRONMENT_TYPES.get(environment_type, {})
    familiarity = clamp(familiarity_level)

    if chunk_type == "landmark":
        base_prob = env_schema.get("landmark_presence", 0.5)
    elif chunk_type == "spatial_drift":
        base_prob = env_schema.get("drift_risk", 0.5)
    elif chunk_type == "sound_salience_high":
        base_prob = env_schema.get("sound_salience", 0.5)
    elif chunk_type == "guidance_absent":
        base_prob = env_schema.get("guidance_absent_risk", 0.5)
    else:
        base_prob = 0.5

    base_level = 0.15 + 0.55 * familiarity * base_prob

    context_weights = _compute_context_weights(environment_type, context=context)
    spreading = 0.0
    for source_chunk, weight in context_weights.items():
        strength = compute_associative_strength(
            source_chunk,
            chunk_type,
            environment_type,
            familiarity,
            co_occurrence_scale=experience_boost,
        )
        spreading += weight * strength

    noise = random.gauss(0.0, max(0.0, noise_scale))
    activation = base_level + spreading + noise

    return {
        "base_level": round(base_level, 4),
        "spreading": round(spreading, 4),
        "noise": round(noise, 4),
        "activation": round(activation, 4),
    }


def get_chunk_activation_modulation(
    environment_type,
    chunk_type,
    familiarity_level,
    experience_boost=1.0,
    context=None,
):
    """Handle get chunk activation modulation behavior."""
    terms = get_chunk_activation_terms(
        environment_type=environment_type,
        chunk_type=chunk_type,
        familiarity_level=familiarity_level,
        experience_boost=experience_boost,
        context=context,
    )

    sigmoid = 1.0 / (1.0 + math.exp(-terms["activation"]))
    modulation = 0.3 + 2.2 * sigmoid
    return clamp(modulation, low=0.3, high=2.5)
