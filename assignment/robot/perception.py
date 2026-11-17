"""Find a thin metal ring in a camera frame.

The ring is a narrow shiny band. At the working distance its inner and
outer edges are a few pixels apart, so the camera sees one circle, not
two. The hole center is the center of that circle. r_inner and r_outer
are both that radius.

This matches the two poses in use: held up in front of the laptop camera,
and lying on the table once the arm has pointed the camera at it. Both
are nearly frontal. A side view, with the ring still flat on the table
and the camera not yet aimed, is an ellipse and this step will miss it.

The circle has to be about a tenth to a quarter of the image width. That
is the size of this one ring. Edges are shared across the frame so a hand
around the ring cannot push the band out of the vote.
"""

from collections import namedtuple

import numpy as np
from PIL import Image


# (u, v) is the hole center in pixels. For this thin band, r_inner and
# r_outer are the same circle. Strongest circle first.
Ring = namedtuple("Ring", "u v r_inner r_outer")

_WORK_LONG_SIDE = 320
# Hole diameter as a fraction of the image width. The real ring sits
# between about 1/10 and 1/4; the band on either side leaves margin.
_DIAMETER_FRACTION = (0.08, 0.32)
_RADIUS_STEP = 2
_MAX_EDGES = 4000
_EDGE_PERCENTILE = 70.0
# Share of 36 angular bins that must contain a radial edge. Fingers can
# hide part of a ring held up to the camera, so this stays below a half turn.
_MIN_ARC = 0.35


def detect_rings(frame):
    """Find rings in an RGB image, shape (height, width, 3).

    Return a list of Ring(u, v, r, r), strongest first. Both radii are the
    band. An empty list means no ring in this frame.

    The ring must face the camera and be roughly a tenth to a quarter of
    the image width. Try it on the laptop camera, with no robot connected:

        python3 MyRobot.py --vision
    """
    gray, scale_x, scale_y = _prepare(frame)
    if gray is None:
        return []

    scale_r = 0.5 * (scale_x + scale_y)
    full_width = gray.shape[1] * scale_x
    min_radius = max(6, int(round(0.5 * _DIAMETER_FRACTION[0] * full_width / scale_r)))
    max_radius = int(round(0.5 * _DIAMETER_FRACTION[1] * full_width / scale_r))
    circles = _circles(gray, min_radius, max_radius)
    found = []
    for _score, x, y, radius in circles:
        found.append(
            Ring(
                float(x * scale_x),
                float(y * scale_y),
                float(radius * scale_r),
                float(radius * scale_r),
            )
        )
    return found


def _prepare(frame):
    """Return (gray float image, scale_x, scale_y) in the work resolution."""
    if frame is None or getattr(frame, "ndim", 0) != 3 or frame.shape[2] < 3:
        return None, 1.0, 1.0
    height, width = int(frame.shape[0]), int(frame.shape[1])
    if height < 16 or width < 16:
        return None, 1.0, 1.0

    gray = _gray(frame[:, :, :3])
    longest = max(height, width)
    if longest > _WORK_LONG_SIDE:
        work_scale = _WORK_LONG_SIDE / float(longest)
        work_w = max(16, int(round(width * work_scale)))
        work_h = max(16, int(round(height * work_scale)))
        small = Image.fromarray(np.clip(np.rint(gray), 0, 255).astype(np.uint8), mode="L")
        gray = np.asarray(small.resize((work_w, work_h), Image.BILINEAR), dtype=np.float32)
    else:
        gray = gray.astype(np.float32)
        work_h, work_w = height, width
    return gray, width / float(work_w), height / float(work_h)


def _gray(frame):
    red = frame[:, :, 0].astype(np.float32)
    green = frame[:, :, 1].astype(np.float32)
    blue = frame[:, :, 2].astype(np.float32)
    return 0.299 * red + 0.587 * green + 0.114 * blue


def _smooth(gray):
    """Two 3x3 binomial passes. Enough to steady a webcam edge."""
    blurred = gray
    for _ in range(2):
        padded = np.pad(blurred, 1, mode="edge")
        blurred = (
            padded[:-2, :-2] + 2.0 * padded[:-2, 1:-1] + padded[:-2, 2:]
            + 2.0 * padded[1:-1, :-2] + 4.0 * padded[1:-1, 1:-1] + 2.0 * padded[1:-1, 2:]
            + padded[2:, :-2] + 2.0 * padded[2:, 1:-1] + padded[2:, 2:]
        ) * (1.0 / 16.0)
    return blurred


def _gradients(gray):
    padded = np.pad(gray, 1, mode="edge")
    gx = (
        -padded[:-2, :-2] - 2.0 * padded[1:-1, :-2] - padded[2:, :-2]
        + padded[:-2, 2:] + 2.0 * padded[1:-1, 2:] + padded[2:, 2:]
    )
    gy = (
        -padded[:-2, :-2] - 2.0 * padded[:-2, 1:-1] - padded[:-2, 2:]
        + padded[2:, :-2] + 2.0 * padded[2:, 1:-1] + padded[2:, 2:]
    )
    return gx, gy


def _edge_pixels(gray):
    """Thin, strong gradients. Each pixel votes along its own normal."""
    gx, gy = _gradients(_smooth(gray))
    magnitude = np.hypot(gx, gy)
    interior = magnitude[1:-1, 1:-1]
    gx_i = gx[1:-1, 1:-1]
    gy_i = gy[1:-1, 1:-1]
    horizontal = np.abs(gx_i) >= np.abs(gy_i)
    keep = np.zeros(magnitude.shape, dtype=bool)
    keep[1:-1, 1:-1] = (
        horizontal
        & (interior >= magnitude[1:-1, :-2])
        & (interior >= magnitude[1:-1, 2:])
    ) | (
        ~horizontal
        & (interior >= magnitude[:-2, 1:-1])
        & (interior >= magnitude[2:, 1:-1])
    )
    strength = magnitude[keep]
    if strength.size == 0:
        return None
    limit = float(np.percentile(strength, _EDGE_PERCENTILE))
    ys, xs = np.nonzero(keep & (magnitude >= max(limit, 6.0)))
    if xs.size == 0:
        return None
    xs, ys = _spread_edges(xs, ys, magnitude, gray.shape)
    mag = magnitude[ys, xs]
    return xs.astype(np.float32), ys.astype(np.float32), gx[ys, xs] / mag, gy[ys, xs] / mag


def _spread_edges(xs, ys, magnitude, shape):
    """Keep edges from every part of the frame.

    Taking only the strongest edges lets a hand crowd out a thin metal
    band. Each cell of a grid keeps its own strongest edges.
    """
    if xs.size <= _MAX_EDGES:
        return xs, ys
    cells = 8
    height, width = shape
    col = np.minimum((xs * cells) // width, cells - 1)
    row = np.minimum((ys * cells) // height, cells - 1)
    cell = row * cells + col
    per_cell = max(8, _MAX_EDGES // (cells * cells))
    chosen = []
    for index in range(cells * cells):
        members = np.nonzero(cell == index)[0]
        if members.size == 0:
            continue
        if members.size > per_cell:
            local = magnitude[ys[members], xs[members]]
            members = members[np.argpartition(local, -per_cell)[-per_cell:]]
        chosen.append(members)
    if not chosen:
        return xs[:0], ys[:0]
    picked = np.concatenate(chosen)
    return xs[picked], ys[picked]


def _circles(gray, min_radius, max_radius):
    edges = _edge_pixels(gray)
    if edges is None or max_radius <= min_radius:
        return []
    xs, ys, nx, ny = edges
    height, width = gray.shape

    found = []
    for radius in range(min_radius, max_radius + 1, _RADIUS_STEP):
        acc = np.zeros((height, width), dtype=np.int32)
        for sign in (1.0, -1.0):
            cols = np.rint(xs + sign * radius * nx).astype(np.int32)
            rows = np.rint(ys + sign * radius * ny).astype(np.int32)
            valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
            np.add.at(acc, (rows[valid], cols[valid]), 1)
        min_votes = max(8, int(round(0.06 * 2.0 * np.pi * radius)))
        found.extend(_peaks(acc, radius, min_votes))
    return _keep_circular(xs, ys, nx, ny, _merge_circles(found))


def _peaks(acc, radius, min_votes):
    """Centers that beat their neighbours. The full-image window was the slow part."""
    if int(acc.max()) < min_votes:
        return []
    rows, cols = np.nonzero(acc >= min_votes)
    if cols.size == 0:
        return []
    height, width = acc.shape
    votes = acc[rows, cols]
    above = np.clip(rows - 2, 0, height - 1)
    below = np.clip(rows + 2, 0, height - 1)
    left = np.clip(cols - 2, 0, width - 1)
    right = np.clip(cols + 2, 0, width - 1)
    neighbour = np.maximum.reduce(
        [
            acc[above, cols],
            acc[below, cols],
            acc[rows, left],
            acc[rows, right],
            acc[above, left],
            acc[above, right],
            acc[below, left],
            acc[below, right],
        ]
    )
    keep = votes >= neighbour
    rows, cols, votes = rows[keep], cols[keep], votes[keep]
    if cols.size == 0:
        return []
    order = np.argsort(votes)[::-1][:6]
    return [
        (float(cols[index]), float(rows[index]), float(radius), int(votes[index]))
        for index in order
    ]


def _merge_circles(circles):
    """Collapse the same edge found in two neighbouring radius bins."""
    circles = sorted(circles, key=lambda circle: circle[3], reverse=True)
    kept = []
    for x, y, radius, votes in circles:
        match = None
        for index, (ox, oy, other_r, other_votes) in enumerate(kept):
            if abs(radius - other_r) > _RADIUS_STEP:
                continue
            if (x - ox) ** 2 + (y - oy) ** 2 > 25.0:
                continue
            match = index
            total = votes + other_votes
            kept[index] = (
                (x * votes + ox * other_votes) / total,
                (y * votes + oy * other_votes) / total,
                (radius * votes + other_r * other_votes) / total,
                total,
            )
            break
        if match is None:
            kept.append((x, y, radius, votes))
    return kept


def _arc_fraction(xs, ys, nx, ny, center_x, center_y, radius):
    """Fraction of the circumference that has an edge pointing at the center."""
    dx = xs - center_x
    dy = ys - center_y
    distance = np.hypot(dx, dy)
    on_band = np.abs(distance - radius) <= 2.5
    if int(on_band.sum()) < 8:
        return 0.0
    distance = np.maximum(distance[on_band], 1e-3)
    radial_x = dx[on_band] / distance
    radial_y = dy[on_band] / distance
    aligned = np.abs(radial_x * nx[on_band] + radial_y * ny[on_band]) >= 0.70
    if int(aligned.sum()) < 8:
        return 0.0
    angle = np.arctan2(dy[on_band][aligned], dx[on_band][aligned])
    bins = np.floor((angle + np.pi) / (2.0 * np.pi) * 36.0).astype(np.int32)
    bins = np.clip(bins, 0, 35)
    occupied = np.zeros(36, dtype=bool)
    occupied[bins] = True
    return float(occupied.mean())


def _keep_circular(xs, ys, nx, ny, circles):
    """Drop Hough peaks that are not a ring arc, strongest remaining first."""
    ranked = []
    for x, y, radius, votes in circles:
        arc = _arc_fraction(xs, ys, nx, ny, x, y, radius)
        if arc < _MIN_ARC:
            continue
        ranked.append((votes * arc, x, y, radius))
    ranked.sort(key=lambda circle: circle[0], reverse=True)
    kept = []
    for score, x, y, radius in ranked:
        if any((x - ox) ** 2 + (y - oy) ** 2 < (0.6 * radius) ** 2 for _s, ox, oy, _r in kept):
            continue
        kept.append((score, x, y, radius))
        if len(kept) == 3:
            break
    return kept
