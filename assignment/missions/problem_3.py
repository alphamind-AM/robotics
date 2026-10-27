"""go to 37 points equidistant on a circle of radius 32 mm,
centered at (150, 0, 120) with pitch 0.
"""

from robot.mission import Mission, go
import numpy as np

RADIUS = 32.0
center = (150.0, 0.0, 120.0)
pitch = 0.0



def _pose(x, y, z, pitch):
    return (float(x), float(y), float(z), float(pitch))

def circle_pose(phi, circle_center=None, radius=None):
    """One pose on the Problem 3 circle. phi is in radians.

    The circle stands in the yz plane and the pitch stays 0.
    phi = 0 is the +y point, (x, y + radius, z).
    phi = pi/2 is the high point, (x, y, z + radius).
    """
    if circle_center is None:
        circle_center = center
    if radius is None:
        radius = RADIUS
    cx, cy, cz = circle_center
    return _pose(
        cx,
        cy + radius * np.cos(phi),
        cz + radius * np.sin(phi),
        pitch,
    )


def series_of_points_on_circle(circle_center, radius, num_points):
    """Generate a list of points on a circle in the yz plane."""
    points = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        points.append(go(circle_pose(angle, circle_center, radius), f"circle {i + 1}"))
    return points


problem_3 = Mission(
    "problem_3",
    [
        go(_pose(*center, pitch), "go to center"),
        *series_of_points_on_circle(center, RADIUS, 37)
    ],
)
