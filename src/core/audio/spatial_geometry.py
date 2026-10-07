"""Single source of truth for head geometry.

Every consumer (renderer, estimator, tests) MUST import the radius and the
Woodworth relation from here. A radius mismatch between renderer and
estimator scales the error by (r_est/r_syn - 1) * theta -- about 2.7 deg at
90 deg -- which is enough to break the todo-12 accuracy gate.
"""
import math

HEAD_RADIUS = 0.0875
SPEED_OF_SOUND = 343.0


def woodworth_tau(theta):
    """Interaural time difference in seconds. theta in radians."""
    return (HEAD_RADIUS / SPEED_OF_SOUND) * (theta + math.sin(theta))


MAX_ITD_SECONDS = woodworth_tau(math.pi / 2)


def woodworth_angle(tau):
    """Inverse of woodworth_tau. Returns degrees, clamped to +-90."""
    if tau == 0.0:
        return 0.0
    sign = 1.0 if tau > 0.0 else -1.0
    target = abs(tau) * SPEED_OF_SOUND / HEAD_RADIUS
    lo, hi = 0.0, math.pi / 2.0
    if target >= hi + math.sin(hi):
        return sign * 90.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if mid + math.sin(mid) < target:
            lo = mid
        else:
            hi = mid
    return math.degrees(0.5 * (lo + hi)) * sign
