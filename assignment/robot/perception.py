import base64
import zlib
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

# OpenCV's frontal-face LBP cascade (Apache-2.0). Loaded once, then run with
# an integral image so the laptop preview stays near 10 frames per second.
_CASCADE_XML = """
eNrFnd2vHNdy3d8N+H+Y3HeK+/sDoGX4OkjyENhGcuE8CsfS0RURSRRI6l7nv8/6rT38mqGD091MUxCkw8PpmZ7d
e1etqlq16sXf//svP1/+8vj6zctXv/7dH+I34Q9//+3f/s2L//Ts2d/+za+///Jvj68vr368/Pbqzcu3L//yeHnz
8MtvPz++ueQQwscv+PXxzw+fvCBWXvDsGW/26rfHX7//y3dv3r56/fDnR37z/cOb7x9+eLy8/T+/PX738oe/+8N6
ybPr7599//PDmzcvf3z5+PoPevnl8uLNW135J7362z/+8z//zz+9eP7hF/77Hx8f3v7+ev3iv//xX148//gXfsVP
jy///NPbb1N58fz6o3/915c/vP3Jv10/ffi0f3l4/fDLG/9Cv/q3V6/evPW7/dd/+OOL5x/+eP37X17++t9evv0f
D28fvw3fzKlvr39KbyP32F48/+iv313w8O//5eHnN4//8PPD6190zbriwz+65pNXXC/7q+/9T69f/vLus+rtdTcv
+fB5//nxN33D6HdeP3/4u//1+PC///HV77++/TZeP/vDb66r/dGCvFvvT9dIF/3jw9t1TarN7/L+F+8fycdv4/f9
p99/+TaF64fwhw9/9f6ttR8v/s0lXLyp/Nvvrj/cfoV88wXev2rtmp9eP7756dXPP3z7LHzTawpjppBKKCGl9G5n
vX/N+2v/ytu935dv3v/F9e7evn785OZubtB/fPnr28fXvz78/E+vfnh888lfXbg0Xkq7PGs95tDDvDxLsTb9L5bU
YgohZv2ceo1J91z916N++ibPYsujXurosdQxeY8SQn3x/D/+5Bc/Pz78+K8PP/9+d0danVZLTjHEMkaMuYRL+Gas
f2atMbbUtWAfvcGL5999dl3iwXXxV29ZX703LVFvs87c69BvQ0g9eLVa0pe+PKuxlNpHLbdLM7MuylVvVVNJg5Xd
vTC955liY+uU0WrXwvTUR8s67rlrH6X5pIVJBxdG1+eWZ2596s9j5hRy86bRncwyQuRluc3Qx7xZkRjG6CP13HhJ
SbH3xOKWWEuZTc8s1hn0PvtXKbRRW466Jz2skrRKrbdY+bAS4uy93q0S9uvTg/Zh7T4yA/EJZqA83QyU0ZOWKMly
J47M1zUDo1y0QL2M3Eq/xFl0/rVju3ZubkO2QefBL6hh9PnpG/Q8Yk5F2yGGFnV2a+uXpGMc9C5zXVZaq/uNwghy
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
86v/C9Lob2Y=
"""

_MODEL = None
_WINDOW = 24


def _model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    root = ET.fromstring(zlib.decompress(base64.b64decode(_CASCADE_XML)))
    cascade = root.find("cascade")
    features = [
        tuple(int(value) for value in node.findtext("rect").split())
        for node in cascade.find("features")
    ]
    codes = np.arange(256, dtype=np.uint32)
    word = codes >> 5
    bit = codes & 31
    stages = []
    for stage in cascade.find("stages"):
        threshold = np.float32(float(stage.findtext("stageThreshold")) - 1e-5)
        weaks = []
        for weak in stage.find("weakClassifiers"):
            numbers = [int(value) for value in weak.findtext("internalNodes").split()]
            leaves = [float(value) for value in weak.findtext("leafValues").split()]
            subsets = np.asarray(numbers[3:11], dtype=np.int32).view(np.uint32)
            selected = ((subsets[word] >> bit) & np.uint32(1)) != 0
            table = np.where(selected, leaves[0], leaves[1]).astype(np.float32)
            weaks.append((numbers[2], table))
        stages.append((threshold, weaks))
    _MODEL = (features, stages)
    return _MODEL


def _rectangle_sum(integral, x, y, width, height, rows, cols):
    return (
        integral[y:y + rows, x:x + cols]
        - integral[y:y + rows, x + width:x + width + cols]
        - integral[y + height:y + height + rows, x:x + cols]
        + integral[y + height:y + height + rows, x + width:x + width + cols]
    )


def _lbp_codes(integral, rect, rows, cols):
    origin_x, origin_y, block_w, block_h = rect
    cells = [
        _rectangle_sum(
            integral,
            origin_x + ix * block_w,
            origin_y + iy * block_h,
            block_w,
            block_h,
            rows,
            cols,
        )
        for iy in range(3)
        for ix in range(3)
    ]
    center = cells[4]
    code = np.zeros((rows, cols), dtype=np.uint8)
    for shift, index in zip((7, 6, 5, 4, 3, 2, 1, 0), (0, 1, 2, 5, 8, 7, 6, 3)):
        code |= (cells[index] >= center).astype(np.uint8) << np.uint8(shift)
    return code


def _boxes_at_scales(gray, scale_factor, min_size):
    features, stages = _model()
    height, width = gray.shape
    boxes = []
    scale = 1.0
    while True:
        window = int(round(_WINDOW * scale))
        if window > width or window > height:
            break
        if window >= min_size:
            scaled_w = int(round(width / scale))
            scaled_h = int(round(height / scale))
            small = np.asarray(
                Image.fromarray(gray, mode="L").resize((scaled_w, scaled_h), Image.BILINEAR),
                dtype=np.uint8,
            )
            integral = np.zeros((scaled_h + 1, scaled_w + 1), dtype=np.int32)
            integral[1:, 1:] = np.cumsum(
                np.cumsum(small, axis=0, dtype=np.int32), axis=1, dtype=np.int32
            )
            rows = scaled_h - _WINDOW + 1
            cols = scaled_w - _WINDOW + 1
            if rows > 0 and cols > 0:
                active = np.ones((rows, cols), dtype=bool)
                for threshold, weaks in stages:
                    ys, xs = np.nonzero(active)
                    if ys.size == 0:
                        break
                    y0 = int(ys.min())
                    y1 = int(ys.max()) + 1
                    x0 = int(xs.min())
                    x1 = int(xs.max()) + 1
                    region = active[y0:y1, x0:x1]
                    total = np.zeros(region.shape, dtype=np.float32)
                    window_integral = integral[y0:, x0:]
                    for feature_index, table in weaks:
                        codes = _lbp_codes(
                            window_integral, features[feature_index], region.shape[0], region.shape[1]
                        )
                        total += table[codes]
                    active[y0:y1, x0:x1] = region & (total >= threshold)
                ys, xs = np.nonzero(active)
                for y, x in zip(ys.tolist(), xs.tolist()):
                    boxes.append(
                        (
                            int(round(x * scale)),
                            int(round(y * scale)),
                            window,
                            window,
                        )
                    )
        scale *= scale_factor
    return boxes


def _overlap_box(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[0] + first[2], second[0] + second[2])
    bottom = min(first[1] + first[3], second[1] + second[3])
    return max(0, right - left) * max(0, bottom - top)


def _same_face(first, second):
    overlap = _overlap_box(first, second)
    if overlap <= 0:
        return False
    area_a = first[2] * first[3]
    area_b = second[2] * second[3]
    union = area_a + area_b - overlap
    if union > 0 and overlap / union >= 0.3:
        return True
    smaller = min(area_a, area_b)
    return smaller > 0 and overlap / smaller >= 0.55


def _merge_overlaps(boxes):
    """Average boxes that cover the same face, largest cluster first."""
    remaining = sorted(boxes, key=lambda box: box[2] * box[3], reverse=True)
    merged = []
    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        kept = []
        for box in remaining:
            if any(_same_face(box, member) for member in group):
                group.append(box)
            else:
                kept.append(box)
        remaining = kept
        total = np.asarray(group, dtype=np.float64).mean(axis=0)
        merged.append(tuple(int(round(value)) for value in total))
    merged.sort(key=lambda box: box[2] * box[3], reverse=True)
    return merged


def _gray(frame):
    red = frame[:, :, 0].astype(np.uint16)
    green = frame[:, :, 1].astype(np.uint16)
    blue = frame[:, :, 2].astype(np.uint16)
    return ((red * 77 + green * 150 + blue * 29) >> 8).astype(np.uint8)


def detect_faces(frame):
    """Find faces in an RGB image, shape (height, width, 3).

    Return a list of (u, v, w, h) boxes in pixels, largest first.
    An empty list means no face in this frame.

    Try it on the laptop camera, with no robot connected:

        python3 MyRobot.py --vision
    """
    if frame is None or getattr(frame, "ndim", 0) != 3 or frame.shape[2] < 3:
        return []
    height, width = int(frame.shape[0]), int(frame.shape[1])
    if height < _WINDOW or width < _WINDOW:
        return []

    # The live camera is 480x640. Searching a 320px-wide copy is enough for a
    # face in front of the laptop and keeps the preview interactive.
    longest = max(height, width)
    if longest > 320:
        work_scale = 320.0 / longest
        work_w = max(_WINDOW, int(round(width * work_scale)))
        work_h = max(_WINDOW, int(round(height * work_scale)))
    else:
        work_w, work_h = width, height
    gray = _gray(frame[:, :, :3])
    if (work_w, work_h) != (width, height):
        gray = np.asarray(
            Image.fromarray(gray, mode="L").resize((work_w, work_h), Image.BILINEAR),
            dtype=np.uint8,
        )

    # Skip windows smaller than about 96px on a 640-wide frame. A face in
    # front of the laptop is larger than that, and the finer scales are slow.
    min_size = max(48, int(round(0.15 * min(work_h, work_w))))
    found = _boxes_at_scales(gray, scale_factor=1.25, min_size=min_size)
    if not found:
        return []

    scale_x = width / float(work_w)
    scale_y = height / float(work_h)
    boxes = []
    for u, v, w, h in _merge_overlaps(found):
        left = int(round(u * scale_x))
        top = int(round(v * scale_y))
        box_w = int(round(w * scale_x))
        box_h = int(round(h * scale_y))
        if box_w <= 0 or box_h <= 0:
            continue
        left = min(max(left, 0), width - 1)
        top = min(max(top, 0), height - 1)
        box_w = min(box_w, width - left)
        box_h = min(box_h, height - top)
        boxes.append((left, top, box_w, box_h))
    boxes.sort(key=lambda box: box[2] * box[3], reverse=True)
    return boxes
