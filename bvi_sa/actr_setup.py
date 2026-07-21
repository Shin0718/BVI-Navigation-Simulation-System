"""Configure ACT-R chunks, buffers, productions, and runtime adapters.

This module is part of the BVI ACT-R navigation simulation workflow.
"""

import pyactr as actr

try:
    from .env_schema import (
        compute_chunk_base_level,
    )
    from .utils import clamp
except ImportError:
    from env_schema import (
        compute_chunk_base_level,
    )
    from utils import clamp


ACTION_KEYS = {
    "move_direct": "SPACE",
    "stop_and_probe": "S",
    "wait_at_red": "F",
}
KEY_TO_ACTION = {v: k for k, v in ACTION_KEYS.items()}

UTILITY_BOOKKEEPING = 14.0
UTILITY_COMMIT = 11.5
UTILITY_SAFETY_CRITICAL = 12.0
UTILITY_DANGER_RESPONSE = 9.0
UTILITY_ROUTINE_NAV = 6.0
UTILITY_DEFAULT_FORWARD = 4.0
UTILITY_TRAFFIC_REACT_CROSSING_BASE = UTILITY_DANGER_RESPONSE
UTILITY_TRAFFIC_REACT_SIDEWALK_HORN = 4.9
UTILITY_TRAFFIC_REACT_SIDEWALK_REVERSE = 4.4
UTILITY_CUE_JUST_ENTERED_CROSSING_PROBE = 11.8


def define_chunk_types():
    """Handle define chunk types behavior."""
    chunk_specs = {
        "landmark": "name type reliability location",
        "path": "surface_type surface_cn from_node to_node steps texture wm_modifier drift_rate cpt_prob",
        "current_state": (
            "position actr_iw actr_wave "
            "overload_phase reference_phase safety_phase "
            "load_state risk attention_gated attention_source salience_band"
        ),
        "goal": "task next_action",
        "perception_input": (
            "sound_type sound_intensity sound_salience sound_identified "
            "dominant_cane_type surface_change "
            "cane_imu cane_variability cane_obstacle cane_guidance cane_guidance_type "
            "at_node at_intersection crossing_active light_state crossing_node"
        ),
        "spatial_state": (
            "anchored anchor_state unanchored_steps load risk_weight "
            "surface_type surface_cn wm_modifier drift_rate segment_remaining"
        ),
        "tick_signal": (
            "risk_band iw_high prev_action vehicle_or_obstacle reference_now "
            "crossing_active crossing_subphase light_state just_entered_crossing"
        ),
    }
    for typename, slots in chunk_specs.items():
        try:
            actr.chunktype(typename, slots)
        except Exception:
            continue


def build_model(profile):
    """Handle build model behavior."""
    define_chunk_types()
    ans = float(profile.get("ans", profile.get("ANS", 0.2)))
    rt = float(profile.get("rt", profile.get("RT", -2.0)))
    decay = float(profile.get("d", profile.get("D", 0.5)))
    mas = float(profile.get("mas", profile.get("MAS", 1.5)))

    model = actr.ACTRModel(
        subsymbolic=True,
        instantaneous_noise=ans,
        retrieval_threshold=rt,
        decay=decay,
        strength_of_association=mas,
        utility_learning=False,
        utility_noise=0.05,
        motor_prepared=True,
    )
    print(
        f"ACT-R model created with utility_learning=False (forward simulation), "
        f"motor_module enabled, mas={mas}, rt={rt}, ans={ans}"
    )
    return model


def seed_memory(
    model,
    initial_load,
    surface_profiles=None,
    surface_cpt_probs=None,
    familiarity_level=0.5,
):
    """Handle seed memory behavior."""
    dm = model.decmem

    if surface_profiles and surface_cpt_probs:
        for surface_type, meta in surface_profiles.items():
            dm.add(
                actr.makechunk(
                    typename="path",
                    surface_type=surface_type,
                    surface_cn=meta.get("cn", surface_type),
                    from_node="segment_start",
                    to_node="segment_end",
                    steps=6,
                    texture=meta.get("texture", "mixed"),
                    wm_modifier=meta.get("wm_modifier", 1.0),
                    drift_rate=meta.get("drift_rate", 0.02),
                    cpt_prob=surface_cpt_probs.get(surface_type, 0.01),
                )
            )
    else:
        dm.add(
            actr.makechunk(
                typename="path",
                surface_type="flat_road",
                surface_cn="平整路面",
                from_node="segment_start",
                to_node="segment_end",
                steps=6,
                texture="smooth",
                wm_modifier=1.0,
                drift_rate=0.01,
                cpt_prob=0.691,
            )
        )

    audio_base = compute_chunk_base_level(
        "landmark", "intersection", familiarity_level, frequency_scale=1.2
    )
    tactile_base = compute_chunk_base_level(
        "landmark", "tactile_guidance", familiarity_level, frequency_scale=1.1
    )
    spatial_base = compute_chunk_base_level(
        "landmark", "flat_road", familiarity_level, frequency_scale=0.9
    )

    dm.add(
        actr.makechunk(
            typename="landmark",
            name="audio_landmark",
            type="audio",
            reliability=0.8,
            location="nearby",
        )
    )
    dm.add(
        actr.makechunk(
            typename="landmark",
            name="tactile_landmark",
            type="tactile",
            reliability=0.7,
            location="underfoot",
        )
    )
    dm.add(
        actr.makechunk(
            typename="landmark",
            name="spatial_landmark",
            type="spatial",
            reliability=0.6,
            location="adjacent",
        )
    )

    dm.add(
        actr.makechunk(
            typename="landmark",
            name="drift_warning_unanchored",
            type="meta",
            reliability=compute_chunk_base_level(
                "spatial_drift", "flat_road", familiarity_level
            ),
            location="none",
        )
    )
    dm.add(
        actr.makechunk(
            typename="landmark",
            name="safe_progress_anchored",
            type="meta",
            reliability=compute_chunk_base_level(
                "safe_progress", "tactile_guidance", familiarity_level
            ),
            location="none",
        )
    )


def setup_buffers(model, initial_load, spatial_base_wm_load):
    """Handle setup buffers behavior."""
    goal = model.goal
    imaginal = model.set_goal("imaginal")
    retrieval = model.retrieval
    perception_buf = model.set_goal("perception_buf")
    spatial_loc = model.set_goal("spatial_loc")
    tick_signal = model.set_goal("tick_signal")

    imaginal.add(
        actr.makechunk(
            typename="current_state",
            position="init",
            actr_iw=initial_load,
            actr_wave=initial_load,
            overload_phase="none",
            reference_phase="present",
            safety_phase="none",
            load_state="normal",
            risk="low",
            attention_gated="no",
            attention_source="none",
            salience_band="low",
        )
    )
    perception_buf.add(
        actr.makechunk(
            typename="perception_input",
            sound_type="none",
            dominant_cane_type="none",
            surface_change="no",
            sound_intensity=0.0,
            sound_salience=0.0,
            sound_identified="no",
            cane_imu=0.2,
            cane_variability=0.0,
            cane_obstacle="no",
            cane_guidance="no",
            cane_guidance_type="none",
            at_node="yes",
            at_intersection="no",
            crossing_active="no",
            light_state="green",
            crossing_node="none",
        )
    )
    spatial_loc.add(
        actr.makechunk(
            typename="spatial_state",
            anchored="yes",
            anchor_state="anchored",
            unanchored_steps=0,
            load=round(spatial_base_wm_load, 3),
            risk_weight=0.0,
            surface_type="flat_road",
            surface_cn="平整路面",
            wm_modifier=1.0,
            drift_rate=0.01,
            segment_remaining=0,
        )
    )
    tick_signal.add(
        actr.makechunk(
            typename="tick_signal",
            risk_band="low",
            iw_high="no",
            prev_action="none",
            vehicle_or_obstacle="no",
            reference_now="yes",
            crossing_active="no",
            crossing_subphase="none",
            light_state="green",
            just_entered_crossing="no",
        )
    )
    goal.add(
        actr.makechunk(
            typename="goal", task="navigating", next_action="continue_forward"
        )
    )
    return goal, imaginal, retrieval, perception_buf, spatial_loc, tick_signal


def register_productions(model, expertise_proxy=0.5):
    """Handle register productions behavior."""
    expertise = clamp(float(expertise_proxy), low=0.0, high=1.0)

    model.productionstring(
        name="bk_overload_none_to_starting",
        string="""
    =g>
        isa goal
        task navigating
    =imaginal>
        isa current_state
        overload_phase none
    =tick_signal>
        isa tick_signal
        iw_high yes
    ==>
    =imaginal>
        isa current_state
        overload_phase starting
""",
    )
    model.productionstring(
        name="bk_overload_starting_to_sustained",
        string="""
    =g>
        isa goal
        task navigating
    =imaginal>
        isa current_state
        overload_phase starting
    =tick_signal>
        isa tick_signal
        iw_high yes
    ==>
    =imaginal>
        isa current_state
        overload_phase sustained
        load_state overloaded
""",
    )
    model.productionstring(
        name="bk_overload_starting_to_none",
        string="""
    =g>
        isa goal
        task navigating
    =imaginal>
        isa current_state
        overload_phase starting
    =tick_signal>
        isa tick_signal
        iw_high no
    ==>
    =imaginal>
        isa current_state
        overload_phase none
        load_state normal
""",
    )
    model.productionstring(
        name="bk_overload_sustained_to_none",
        string="""
    =g>
        isa goal
        task navigating
    =imaginal>
        isa current_state
        overload_phase sustained
    =tick_signal>
        isa tick_signal
        iw_high no
    ==>
    =imaginal>
        isa current_state
        overload_phase none
        load_state normal
""",
    )

    model.productionstring(
        name="bk_reference_short_to_present",
        string="""
    =g>
        isa goal
        task navigating
    =imaginal>
        isa current_state
        reference_phase absent_short
    =tick_signal>
        isa tick_signal
        reference_now yes
    ==>
    =imaginal>
        isa current_state
        reference_phase present
""",
    )
    model.productionstring(
        name="bk_reference_long_to_present",
        string="""
    =g>
        isa goal
        task navigating
    =imaginal>
        isa current_state
        reference_phase absent_long
    =tick_signal>
        isa tick_signal
        reference_now yes
    ==>
    =imaginal>
        isa current_state
        reference_phase present
""",
    )
    model.productionstring(
        name="bk_reference_present_to_absent_short",
        string="""
    =g>
        isa goal
        task navigating
    =imaginal>
        isa current_state
        reference_phase present
    =tick_signal>
        isa tick_signal
        reference_now no
    ==>
    =imaginal>
        isa current_state
        reference_phase absent_short
""",
    )
    model.productionstring(
        name="bk_reference_short_to_long",
        string="""
    =g>
        isa goal
        task navigating
    =imaginal>
        isa current_state
        reference_phase absent_short
    =tick_signal>
        isa tick_signal
        reference_now no
    ==>
    =imaginal>
        isa current_state
        reference_phase absent_long
""",
    )

    model.productionstring(
        name="bk_safety_none_to_probing",
        string="""
    =g>
        isa goal
        task navigating
    =imaginal>
        isa current_state
        safety_phase none
    =tick_signal>
        isa tick_signal
        prev_action probe
        vehicle_or_obstacle no
    ==>
    =imaginal>
        isa current_state
        safety_phase probing
""",
    )
    model.productionstring(
        name="bk_safety_probing_to_safe_long",
        string="""
    =g>
        isa goal
        task navigating
    =imaginal>
        isa current_state
        safety_phase probing
    =tick_signal>
        isa tick_signal
        prev_action probe
        vehicle_or_obstacle no
    ==>
    =imaginal>
        isa current_state
        safety_phase safe_long
""",
    )
    model.productionstring(
        name="bk_safety_probing_to_probing_under_threat",
        string="""
    =g>
        isa goal
        task navigating
    =imaginal>
        isa current_state
        safety_phase safe_long
    =tick_signal>
        isa tick_signal
        prev_action probe
        vehicle_or_obstacle yes
    ==>
    =imaginal>
        isa current_state
        safety_phase probing
""",
    )
    model.productionstring(
        name="bk_safety_probing_to_none_after_move",
        string="""
    =g>
        isa goal
        task navigating
    =imaginal>
        isa current_state
        safety_phase probing
    =tick_signal>
        isa tick_signal
        prev_action move
    ==>
    =imaginal>
        isa current_state
        safety_phase none
""",
    )
    model.productionstring(
        name="bk_safety_safe_long_to_none_after_move",
        string="""
    =g>
        isa goal
        task navigating
    =imaginal>
        isa current_state
        safety_phase safe_long
    =tick_signal>
        isa tick_signal
        prev_action move
    ==>
    =imaginal>
        isa current_state
        safety_phase none
""",
    )

    model.productionstring(
        name="bk_sync_risk_low",
        string="""
    =g>
        isa goal
        task navigating
    =imaginal>
        isa current_state
        risk medium
    =tick_signal>
        isa tick_signal
        risk_band low
    ==>
    =imaginal>
        isa current_state
        risk low
""",
    )
    model.productionstring(
        name="bk_sync_risk_medium",
        string="""
    =g>
        isa goal
        task navigating
    =imaginal>
        isa current_state
        risk low
    =tick_signal>
        isa tick_signal
        risk_band medium
    ==>
    =imaginal>
        isa current_state
        risk medium
""",
    )
    model.productionstring(
        name="bk_sync_risk_high_from_low",
        string="""
    =g>
        isa goal
        task navigating
    =imaginal>
        isa current_state
        risk low
    =tick_signal>
        isa tick_signal
        risk_band high
    ==>
    =imaginal>
        isa current_state
        risk high
""",
    )
    model.productionstring(
        name="bk_sync_risk_high_from_medium",
        string="""
    =g>
        isa goal
        task navigating
    =imaginal>
        isa current_state
        risk medium
    =tick_signal>
        isa tick_signal
        risk_band high
    ==>
    =imaginal>
        isa current_state
        risk high
""",
    )
    model.productionstring(
        name="bk_sync_risk_low_from_high",
        string="""
    =g>
        isa goal
        task navigating
    =imaginal>
        isa current_state
        risk high
    =tick_signal>
        isa tick_signal
        risk_band low
    ==>
    =imaginal>
        isa current_state
        risk low
""",
    )
    model.productionstring(
        name="bk_sync_risk_medium_from_high",
        string="""
    =g>
        isa goal
        task navigating
    =imaginal>
        isa current_state
        risk high
    =tick_signal>
        isa tick_signal
        risk_band medium
    ==>
    =imaginal>
        isa current_state
        risk medium
""",
    )

    model.productionstring(
        name="cue_overload_sustained_probe",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =imaginal>
        isa current_state
        overload_phase sustained
    =perception_buf>
        isa perception_input
        crossing_active no
    ==>
    =g>
        isa goal
        task navigating
        next_action stop_and_probe
""",
    )
    model.productionstring(
        name="cue_reference_long_absent_probe",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =imaginal>
        isa current_state
        reference_phase absent_long
        load_state normal
    =perception_buf>
        isa perception_input
        crossing_active no
    ==>
    =g>
        isa goal
        task navigating
        next_action stop_and_probe
""",
    )
    model.productionstring(
        name="cue_post_probe_safe_go",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =imaginal>
        isa current_state
        safety_phase safe_long
        load_state normal
        risk low
    =perception_buf>
        isa perception_input
        crossing_active no
    ==>
    =g>
        isa goal
        task navigating
        next_action move_direct
""",
    )
    model.productionstring(
        name="cue_just_entered_crossing_probe",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =tick_signal>
        isa tick_signal
        just_entered_crossing yes
    =perception_buf>
        isa perception_input
        crossing_active yes
    ==>
    =g>
        isa goal
        task navigating
        next_action stop_and_probe
""",
    )

    model.productionstring(
        name="predict_goal_high_load",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =perception_buf>
        isa perception_input
        crossing_active no
    =imaginal>
        isa current_state
        load_state overloaded
    ==>
    =g>
        isa goal
        task navigating
        next_action stop_and_probe
""",
    )
    model.productionstring(
        name="predict_goal_high_risk",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =perception_buf>
        isa perception_input
        crossing_active no
    =imaginal>
        isa current_state
        load_state normal
        risk high
    ==>
    =g>
        isa goal
        task navigating
        next_action stop_and_probe
""",
    )
    model.productionstring(
        name="predict_goal_medium_risk",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =perception_buf>
        isa perception_input
        crossing_active no
    =imaginal>
        isa current_state
        load_state normal
        risk medium
    ==>
    =g>
        isa goal
        task navigating
        next_action move_direct
""",
    )
    model.productionstring(
        name="predict_goal_low_risk",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =perception_buf>
        isa perception_input
        crossing_active no
    =imaginal>
        isa current_state
        load_state normal
        risk low
    ==>
    =g>
        isa goal
        task navigating
        next_action move_direct
""",
    )

    model.productionstring(
        name="probe_when_spatial_lost",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =perception_buf>
        isa perception_input
        crossing_active no
    =spatial_loc>
        isa spatial_state
        anchor_state lost
    =imaginal>
        isa current_state
        load_state normal
    ==>
    =g>
        isa goal
        task navigating
        next_action stop_and_probe
""",
    )
    model.productionstring(
        name="probe_when_spatial_drifting",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =perception_buf>
        isa perception_input
        crossing_active no
    =spatial_loc>
        isa spatial_state
        anchor_state drifting
    =imaginal>
        isa current_state
        risk high
    ==>
    =g>
        isa goal
        task navigating
        next_action stop_and_probe
""",
    )

    model.productionstring(
        name="crossing_red_wait",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =perception_buf>
        isa perception_input
        crossing_active yes
        light_state red
    ==>
    =g>
        isa goal
        task navigating
        next_action wait_at_red
""",
    )
    model.productionstring(
        name="crossing_green_probe_when_overloaded",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =perception_buf>
        isa perception_input
        at_node yes
        crossing_active yes
        light_state green
    =imaginal>
        isa current_state
        load_state overloaded
    ==>
    =g>
        isa goal
        task navigating
        next_action stop_and_probe
""",
    )
    model.productionstring(
        name="crossing_green_probe_when_high_risk",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =perception_buf>
        isa perception_input
        at_node yes
        crossing_active yes
        light_state green
    =imaginal>
        isa current_state
        risk high
    ==>
    =g>
        isa goal
        task navigating
        next_action stop_and_probe
""",
    )
    model.productionstring(
        name="crossing_green_probe_when_reference_lost",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =perception_buf>
        isa perception_input
        at_node yes
        crossing_active yes
        light_state green
    =imaginal>
        isa current_state
        reference_phase absent_long
    ==>
    =g>
        isa goal
        task navigating
        next_action stop_and_probe
""",
    )
    model.productionstring(
        name="crossing_green_go",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =perception_buf>
        isa perception_input
        crossing_active yes
        light_state green
    ==>
    =g>
        isa goal
        task navigating
        next_action move_direct
""",
    )

    model.productionstring(
        name="react_horn_at_crossing",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =perception_buf>
        isa perception_input
        sound_type horn
        crossing_active yes
    ==>
    =g>
        isa goal
        task navigating
        next_action stop_and_probe
""",
    )
    model.productionstring(
        name="react_horn_on_sidewalk",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =perception_buf>
        isa perception_input
        sound_type horn
        crossing_active no
    ==>
    =g>
        isa goal
        task navigating
        next_action stop_and_probe
""",
    )
    model.productionstring(
        name="react_reverse_beep_at_crossing",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =perception_buf>
        isa perception_input
        sound_type reverse_beep
        crossing_active yes
    ==>
    =g>
        isa goal
        task navigating
        next_action stop_and_probe
""",
    )
    model.productionstring(
        name="react_reverse_beep_on_sidewalk",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =perception_buf>
        isa perception_input
        sound_type reverse_beep
        crossing_active no
    ==>
    =g>
        isa goal
        task navigating
        next_action stop_and_probe
""",
    )
    model.productionstring(
        name="react_human_activity_bottom_up",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =perception_buf>
        isa perception_input
        sound_type human_activity
    ==>
    =g>
        isa goal
        task navigating
        next_action move_direct
""",
    )

    model.productionstring(
        name="react_cane_obstacle_bottom_up",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =perception_buf>
        isa perception_input
        dominant_cane_type obstacle
    ==>
    =g>
        isa goal
        task navigating
        next_action stop_and_probe
""",
    )
    model.productionstring(
        name="react_cane_tactile_bottom_up",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =perception_buf>
        isa perception_input
        dominant_cane_type tactile
    ==>
    =g>
        isa goal
        task navigating
        next_action move_direct
""",
    )
    model.productionstring(
        name="react_cane_curb_bottom_up",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =perception_buf>
        isa perception_input
        dominant_cane_type curb
    ==>
    =g>
        isa goal
        task navigating
        next_action move_direct
""",
    )
    model.productionstring(
        name="react_cane_wall_bottom_up",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =perception_buf>
        isa perception_input
        dominant_cane_type wall
    ==>
    =g>
        isa goal
        task navigating
        next_action move_direct
""",
    )
    model.productionstring(
        name="react_cane_railing_bottom_up",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =perception_buf>
        isa perception_input
        dominant_cane_type railing
    ==>
    =g>
        isa goal
        task navigating
        next_action move_direct
""",
    )
    model.productionstring(
        name="react_surface_change_bottom_up",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =perception_buf>
        isa perception_input
        dominant_cane_type surface_change
    ==>
    =g>
        isa goal
        task navigating
        next_action move_direct
""",
    )

    model.productionstring(
        name="crossing_guidance_lost",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =perception_buf>
        isa perception_input
        crossing_active yes
        dominant_cane_type none
    ==>
    =g>
        isa goal
        task navigating
        next_action stop_and_probe
""",
    )
    model.productionstring(
        name="crossing_obstacle_alert",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =perception_buf>
        isa perception_input
        crossing_active yes
        dominant_cane_type obstacle
    ==>
    =g>
        isa goal
        task navigating
        next_action stop_and_probe
""",
    )

    model.productionstring(
        name="request_landmark_audio_from_dm",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =imaginal>
        isa current_state
        attention_gated yes
        attention_source landmark
        load_state normal
    ==>
    +retrieval>
        isa landmark
        type audio
        location nearby
""",
    )
    model.productionstring(
        name="confirm_landmark_audio_retrieved",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =imaginal>
        isa current_state
        attention_gated yes
        attention_source landmark
        load_state normal
    =retrieval>
        isa landmark
        type audio
        location nearby
    ==>
    =g>
        isa goal
        task navigating
        next_action move_direct
""",
    )
    model.productionstring(
        name="attend_gated_sound_high",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =imaginal>
        isa current_state
        attention_gated yes
        attention_source sound
        salience_band high
        load_state normal
    ==>
    =g>
        isa goal
        task navigating
        next_action stop_and_probe
""",
    )
    model.productionstring(
        name="attend_gated_tactile_high",
        string="""
    =g>
        isa goal
        task navigating
        next_action continue_forward
    =imaginal>
        isa current_state
        attention_gated yes
        attention_source tactile
        salience_band high
        load_state normal
    ==>
    =g>
        isa goal
        task navigating
        next_action stop_and_probe
""",
    )

    model.productionstring(
        name="commit_move_direct",
        string="""
    =g>
        isa goal
        task navigating
        next_action move_direct
    ?manual>
        state free
    ==>
    =g>
        isa goal
        task navigating
        next_action committed
    +manual>
        isa _manual
        cmd press_key
        key SPACE
""",
    )
    model.productionstring(
        name="commit_stop_and_probe",
        string="""
    =g>
        isa goal
        task navigating
        next_action stop_and_probe
    ?manual>
        state free
    ==>
    =g>
        isa goal
        task navigating
        next_action committed
    +manual>
        isa _manual
        cmd press_key
        key S
""",
    )
    model.productionstring(
        name="commit_wait_at_red",
        string="""
    =g>
        isa goal
        task navigating
        next_action wait_at_red
    ?manual>
        state free
    ==>
    =g>
        isa goal
        task navigating
        next_action committed
    +manual>
        isa _manual
        cmd press_key
        key F
""",
    )

    model.productionstring(
        name="release_goal_after_commit",
        string="""
    =g>
        isa goal
        task navigating
        next_action committed
    ?manual>
        state free
    ==>
    =g>
        isa goal
        task navigating
        next_action continue_forward
""",
    )

    high_load_probe_utility = 7.0 - 2.5 * expertise
    overload_cue_probe_utility = 8.2 - 3.0 * expertise

    _initial_utilities = {
        "bk_overload_none_to_starting": UTILITY_BOOKKEEPING,
        "bk_overload_starting_to_sustained": UTILITY_BOOKKEEPING,
        "bk_overload_starting_to_none": UTILITY_BOOKKEEPING,
        "bk_overload_sustained_to_none": UTILITY_BOOKKEEPING,
        "bk_reference_short_to_present": UTILITY_BOOKKEEPING,
        "bk_reference_long_to_present": UTILITY_BOOKKEEPING,
        "bk_reference_present_to_absent_short": UTILITY_BOOKKEEPING,
        "bk_reference_short_to_long": UTILITY_BOOKKEEPING,
        "bk_safety_none_to_probing": UTILITY_BOOKKEEPING,
        "bk_safety_probing_to_safe_long": UTILITY_BOOKKEEPING,
        "bk_safety_probing_to_probing_under_threat": UTILITY_BOOKKEEPING,
        "bk_safety_probing_to_none_after_move": UTILITY_BOOKKEEPING,
        "bk_safety_safe_long_to_none_after_move": UTILITY_BOOKKEEPING,
        "bk_sync_risk_low": UTILITY_BOOKKEEPING,
        "bk_sync_risk_medium": UTILITY_BOOKKEEPING,
        "bk_sync_risk_high_from_low": UTILITY_BOOKKEEPING,
        "bk_sync_risk_high_from_medium": UTILITY_BOOKKEEPING,
        "bk_sync_risk_low_from_high": UTILITY_BOOKKEEPING,
        "bk_sync_risk_medium_from_high": UTILITY_BOOKKEEPING,
        "commit_move_direct": UTILITY_COMMIT,
        "commit_stop_and_probe": UTILITY_COMMIT,
        "commit_wait_at_red": UTILITY_COMMIT,
        "release_goal_after_commit": UTILITY_COMMIT - 0.5,
        "crossing_red_wait": UTILITY_SAFETY_CRITICAL,
        "crossing_green_probe_when_overloaded": UTILITY_SAFETY_CRITICAL - 0.4,
        "crossing_green_probe_when_high_risk": UTILITY_SAFETY_CRITICAL - 0.4,
        "crossing_green_probe_when_reference_lost": UTILITY_SAFETY_CRITICAL - 0.4,
        "crossing_green_go": UTILITY_SAFETY_CRITICAL - 2.0,
        "crossing_obstacle_alert": UTILITY_DANGER_RESPONSE + 0.5,
        "crossing_guidance_lost": UTILITY_DANGER_RESPONSE - 0.2,
        "react_horn_at_crossing": UTILITY_TRAFFIC_REACT_CROSSING_BASE - 1.0,
        "react_horn_on_sidewalk": UTILITY_TRAFFIC_REACT_SIDEWALK_HORN,
        "react_reverse_beep_at_crossing": UTILITY_TRAFFIC_REACT_CROSSING_BASE - 1.5,
        "react_reverse_beep_on_sidewalk": UTILITY_TRAFFIC_REACT_SIDEWALK_REVERSE,
        "react_cane_obstacle_bottom_up": UTILITY_DANGER_RESPONSE,
        "cue_just_entered_crossing_probe": UTILITY_CUE_JUST_ENTERED_CROSSING_PROBE,
        "cue_overload_sustained_probe": round(overload_cue_probe_utility, 3),
        "cue_reference_long_absent_probe": 5.2,
        "cue_post_probe_safe_go": 6.2,
        "predict_goal_high_load": round(high_load_probe_utility, 3),
        "predict_goal_high_risk": 6.5,
        "probe_when_spatial_lost": 5.0,
        "probe_when_spatial_drifting": 3.7,
        "predict_goal_medium_risk": UTILITY_ROUTINE_NAV,
        "predict_goal_low_risk": UTILITY_ROUTINE_NAV,
        "react_cane_tactile_bottom_up": UTILITY_DEFAULT_FORWARD + 2.0,
        "react_cane_curb_bottom_up": UTILITY_DEFAULT_FORWARD,
        "react_cane_wall_bottom_up": UTILITY_DEFAULT_FORWARD,
        "react_cane_railing_bottom_up": UTILITY_DEFAULT_FORWARD,
        "react_surface_change_bottom_up": UTILITY_DEFAULT_FORWARD - 0.5,
        "react_human_activity_bottom_up": UTILITY_DEFAULT_FORWARD - 0.5,
        "request_landmark_audio_from_dm": 5.0,
        "confirm_landmark_audio_retrieved": 5.6,
        "attend_gated_sound_high": 5.5,
        "attend_gated_tactile_high": 5.0,
    }
    for prod_name, utility_val in _initial_utilities.items():
        if prod_name in model.productions:
            production = model.productions[prod_name]
            production["utility"] = float(utility_val)
            production.utility = float(utility_val)
    print(
        f"[ACT-R] 已注册 {len(model.productions)} 条产生式，"
        f"配置初始 utility {len(_initial_utilities)} 条；"
        f"bookkeeping={UTILITY_BOOKKEEPING}，commit={UTILITY_COMMIT}，"
        f"safety_critical={UTILITY_SAFETY_CRITICAL}（不做 learning，固定先验）"
    )
