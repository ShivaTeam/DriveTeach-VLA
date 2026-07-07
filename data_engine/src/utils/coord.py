"""Coordinate and trajectory formatting utilities."""

import json
import math
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ir.schema import EgoPoint


def format_number(n: float, decimal_places: int = 2) -> str:
    """Format a number with sign and fixed decimals. 0.00 returns '0.00'."""
    if abs(n) <= 1e-10:
        return "0.00"
    if abs(round(n, decimal_places)) <= 10 ** (-decimal_places - 1):
        return "0.00"
    return f"{n:+.{decimal_places}f}"


def format_point(point, decimals: int = 2) -> str:
    """Format an EgoPoint or (x,y,heading) tuple as (x, y, heading)."""
    if hasattr(point, 'x'):
        x, y, h = point.x, point.y, point.heading
    else:
        x, y, h = point[0], point[1], point[2]
    return f"({format_number(x, decimals)}, {format_number(y, decimals)}, {format_number(h, decimals)})"


def format_traj(trajectory: Optional[list], decimals: int = 2,
                t_min: Optional[float] = None, t_max: Optional[float] = None,
                separator: str = ", ") -> str:
    """Format trajectory points as [PT, (x,y,h), ...]."""
    if trajectory is None:
        return "[PT, ]"
    pts = []
    for p in trajectory:
        t = p.t if hasattr(p, 't') else 0
        if t_min is not None and t < t_min - 1e-9:
            continue
        if t_max is not None and t > t_max + 1e-9:
            continue
        pts.append(format_point(p, decimals))
    return f"[PT, {separator.join(pts)}]"


def format_past_timeline(trajectory: list) -> str:
    """Format past trajectory points including current frame: t-3: (x,y,h) ... t-0: (0,0,0)."""
    pts = [p for p in trajectory if hasattr(p, 't') and p.t <= 1e-9]
    if not pts:
        return "no history"
    sorted_pts = sorted(pts, key=lambda p: p.t)
    num_pts = len(sorted_pts)
    lines = []
    for i, p in enumerate(sorted_pts):
        idx = num_pts - 1 - i  # t-3, t-2, t-1, t-0
        lines.append(f"   - t-{idx}: {format_point(p)}")
    return " ".join(lines)


def describe_past(trajectory: Optional[list]) -> str:
    """Describe past trajectory: '2.0-second past trajectory (4 steps at 2 Hz)'."""
    if trajectory is None:
        return "no history"
    pts = [p for p in trajectory if hasattr(p, 't') and p.t < -1e-9]
    if not pts:
        return "no history"
    # Use the full window span (from earliest past to current frame t=0)
    earliest_t = min(p.t for p in pts)
    duration = abs(earliest_t)
    if duration == int(duration):
        duration = int(duration)
    return f"{duration}-second past trajectory({len(pts)} steps at 2 Hz)"


def describe_future(trajectory: Optional[list]) -> str:
    """Describe future trajectory: '4.0-second future trajectory (8 steps at 2 Hz)'."""
    if trajectory is None:
        return "no future"
    pts = [p for p in trajectory if hasattr(p, 't') and p.t > 1e-9]
    if not pts:
        return "no future"
    # Use the full window span (from t=0 to latest future)
    latest_t = max(p.t for p in pts)
    duration = latest_t
    if duration == int(duration):
        duration = int(duration)
    return f"{duration}-second future trajectory({len(pts)} steps at 2 Hz)"


def format_2d_tgp_keypoints(trajectory_2d: list[dict]) -> str:
    """Format 2D-TGP keypoints WITHOUT heading: [{"point_2d": [x, y]}, ...].

    Used in Planner prompt as spatial guidance.
    """
    items = [{"point_2d": p["point_2d"]} for p in trajectory_2d]
    return f"[{', '.join(json.dumps(item) for item in items)}]"
