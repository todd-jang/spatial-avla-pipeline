"""Fixed, display-only MPC scenario for demonstrating numeric action outputs.

The inputs are virtual demo values, not measurements from camera/audio sensors.
The selected steering/brake/throttle values are never sent to vehicle hardware.
"""
from __future__ import annotations

import math

FIXED_DEMO_SCENARIO = {
    "lateral_error_m": 0.4,
    "heading_error_deg": 3.0,
    "speed_kmh": 36.0,
    "target_speed_kmh": 40.0,
    "obstacle_ttc_s": 2.2,
}

_STEERING_CANDIDATES_DEG = (
    -12.0, -8.0, -6.0, -4.0, -2.0, -1.0, 0.0,
    1.0, 2.0, 4.0, 6.0, 8.0, 12.0,
)
_ACCELERATION_CANDIDATES_MPS2 = (-5.0, -3.0, -1.0, 0.0, 1.5)
_HORIZON_STEPS = 15
_DT_SECONDS = 0.2
_WHEELBASE_M = 2.8
_MAX_BRAKE_MPS2 = 5.0


def _trajectory_cost(
    steering_deg: float, acceleration_mps2: float, scenario: dict
) -> tuple[float, float, float]:
    lateral_error = float(scenario["lateral_error_m"])
    heading_error = math.radians(float(scenario["heading_error_deg"]))
    speed = max(0.0, float(scenario["speed_kmh"]) / 3.6)
    target_speed = max(0.0, float(scenario["target_speed_kmh"]) / 3.6)
    ttc = float(scenario["obstacle_ttc_s"])
    obstacle_distance = speed * max(0.0, ttc) if ttc > 0 else math.inf
    distance_travelled = 0.0
    cost = 0.0

    steering_rad = math.radians(steering_deg)
    yaw_rate = speed / _WHEELBASE_M * math.tan(steering_rad)
    for _ in range(_HORIZON_STEPS):
        speed = max(0.0, speed + acceleration_mps2 * _DT_SECONDS)
        heading_error += yaw_rate * _DT_SECONDS
        lateral_error += speed * math.sin(heading_error) * _DT_SECONDS
        distance_travelled += speed * _DT_SECONDS
        remaining = obstacle_distance - distance_travelled

        cost += (
            8.0 * lateral_error**2
            + 2.0 * heading_error**2
            + 0.12 * (speed - target_speed) ** 2
            + 0.035 * steering_deg**2
            + 0.12 * acceleration_mps2**2
        )
        if math.isfinite(remaining):
            safe_gap = 4.0 + speed
            if remaining <= 0.0:
                cost += 100_000.0 + 1000.0 * abs(remaining)
            elif remaining < safe_gap:
                cost += 40.0 * (safe_gap - remaining) ** 2

    return cost, lateral_error, speed


def compute_fixed_demo_mpc() -> dict:
    """Choose the lowest-cost constant action over a short virtual horizon."""
    scenario = dict(FIXED_DEMO_SCENARIO)
    candidates = []
    for steering_deg in _STEERING_CANDIDATES_DEG:
        for acceleration_mps2 in _ACCELERATION_CANDIDATES_MPS2:
            cost, final_lateral_error, final_speed = _trajectory_cost(
                steering_deg, acceleration_mps2, scenario
            )
            candidates.append(
                (
                    cost,
                    steering_deg,
                    acceleration_mps2,
                    final_lateral_error,
                    final_speed,
                )
            )
    cost, steering_deg, acceleration_mps2, final_lateral, final_speed = min(
        candidates
    )
    brake_percent = round(
        min(100.0, max(0.0, -acceleration_mps2 / _MAX_BRAKE_MPS2 * 100.0)), 1
    )
    throttle_percent = round(
        min(100.0, max(0.0, acceleration_mps2 / 2.0 * 100.0)), 1
    )
    action = (
        f"BRAKE {brake_percent:.0f}%"
        if brake_percent > 0.0
        else f"THROTTLE {throttle_percent:.0f}%"
        if throttle_percent > 0.0
        else "COAST"
    )
    return {
        "status": "simulation_only",
        "actuators_connected": False,
        "actuation_status": "display_only_not_sent",
        "scenario": scenario,
        "horizon_seconds": _HORIZON_STEPS * _DT_SECONDS,
        "steering_angle_degrees": steering_deg,
        "longitudinal_acceleration_mps2": acceleration_mps2,
        "brake_percent": brake_percent,
        "brake_lamp_on": brake_percent > 0.0,
        "throttle_percent": throttle_percent,
        "action": action,
        "objective_cost": round(cost, 3),
        "predicted_final_lateral_error_m": round(final_lateral, 3),
        "predicted_final_speed_kmh": round(final_speed * 3.6, 2),
    }
