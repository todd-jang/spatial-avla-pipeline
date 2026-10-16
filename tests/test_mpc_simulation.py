from src.app.mpc_simulation import (
    FIXED_DEMO_SCENARIO,
    compute_fixed_demo_mpc,
)


def test_fixed_demo_mpc_returns_bounded_display_only_actions():
    result = compute_fixed_demo_mpc()

    assert result["status"] == "simulation_only"
    assert result["actuators_connected"] is False
    assert -18.0 <= result["steering_angle_degrees"] <= 18.0
    assert 0.0 <= result["brake_percent"] <= 100.0
    assert result["brake_lamp_on"] is True
    assert 0.0 <= result["throttle_percent"] <= 100.0
    assert result["actuation_status"] == "display_only_not_sent"
    assert result["horizon_seconds"] == 3.0
    assert result["scenario"] == FIXED_DEMO_SCENARIO


def test_fixed_demo_mpc_is_deterministic():
    assert compute_fixed_demo_mpc() == compute_fixed_demo_mpc()
