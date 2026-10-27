"""Draw a square, then the X inside it.

Change the numbers below to move or resize the drawing. The corner is the
(x, y) where the pen first touches. side, z, and pitch use the same units
and pitch convention as forward().
"""

from missions.shapes import draw_square, draw_x
from robot.mission import Mission

CORNER_XY = (-150.0, -25.0)
SIDE = 50.0
Z = 180.0
PITCH = 60.0
LIFT = 30.0

square_with_x = Mission(
    "square_with_x",
    [
        *draw_square(CORNER_XY, SIDE, Z, PITCH, LIFT),
        *draw_x(CORNER_XY, SIDE, Z, PITCH, LIFT),
    ],
)
