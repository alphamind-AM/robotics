from missions.dynamic import dynamic
from missions.known_poses import known_poses
from missions.point_ring import point_ring
from missions.problem_3 import problem_3
from missions.square_with_x import square_with_x

MISSIONS = {
    square_with_x.name: square_with_x,
    problem_3.name: problem_3,
    dynamic.name: dynamic,
    known_poses.name: known_poses,
    point_ring.name: point_ring,
}
