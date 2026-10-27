"""Drawing strokes built from mission steps.

A stroke lowers to the first point, draws the path, then lifts.
The lift is along z, so the tool stays at the same pitch.
"""

from robot.mission import slide


def _pose(x, y, z, pitch):
    return (float(x), float(y), float(z), float(pitch))


def _hover(point, lift):
    return (point[0], point[1], point[2] + lift, point[3])


def _stroke(path, lift, duration_s, label):
    if len(path) < 2:
        raise ValueError(f"{label} needs at least two points")
    if lift <= 0:
        raise ValueError("lift must be positive")
    travel = max(0.5, duration_s * 0.5)
    steps = [slide(_hover(path[0], lift), path[0], travel, f"{label} lower")]
    for start, end in zip(path, path[1:]):
        steps.append(slide(start, end, duration_s, label))
    steps.append(slide(path[-1], _hover(path[-1], lift), travel, f"{label} lift"))
    return steps


def draw_square(corner_xy, side, z, pitch, lift, duration_s=2.0):
    """Four sides of a square in the horizontal plane, then a lift.

    corner_xy is the (x, y) of one corner. side is the length of each edge.
    z and pitch stay constant while the pen is down.
    """
    if side == 0:
        raise ValueError("side must be non-zero")
    x, y = corner_xy
    corners = [
        _pose(x, y, z, pitch),
        _pose(x + side, y, z, pitch),
        _pose(x + side, y + side, z, pitch),
        _pose(x, y + side, z, pitch),
    ]
    return _stroke(corners + [corners[0]], lift, duration_s, "square")


def draw_x(corner_xy, side, z, pitch, lift, duration_s=2.0):
    """The two diagonals of the same square, lifting between them."""
    if side == 0:
        raise ValueError("side must be non-zero")
    x, y = corner_xy
    start = _pose(x, y, z, pitch)
    across = _pose(x + side, y, z, pitch)
    opposite = _pose(x + side, y + side, z, pitch)
    other = _pose(x, y + side, z, pitch)
    travel = max(0.5, duration_s * 0.5)
    return [
        *_stroke([start, opposite], lift, duration_s, "x"),
        slide(_hover(opposite, lift), _hover(across, lift), travel, "x travel"),
        *_stroke([across, other], lift, duration_s, "x"),
    ]
