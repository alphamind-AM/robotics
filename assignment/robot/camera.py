import os
import subprocess
import threading

import numpy as np

# consider() returns this from watch_until to keep the window open. The
# reader keeps only the newest frame, so a slow detect step does not
# replay the pictures that arrived while the arm was moving.
KEEP_LOOKING = object()


class _LatestFrames:
    """Read the camera pipe on its own thread and keep the newest frame.

    One RGB frame is about 900 kB and the pipe holds about 64 kB, so a
    reader on the UI thread never sees two whole frames at once and cannot
    drop the stale ones. This thread stays caught up with ffmpeg. take()
    hands the UI the newest frame and forgets it.
    """

    def __init__(self, proc, frame_bytes):
        self._proc = proc
        self._frame_bytes = int(frame_bytes)
        self._lock = threading.Lock()
        self._latest = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        buf = bytearray()
        need = self._frame_bytes
        try:
            fd = self._proc.stdout.fileno()
        except OSError:
            with self._lock:
                self._closed = True
            return
        while True:
            try:
                chunk = os.read(fd, 1 << 20)
            except OSError:
                break
            if not chunk:
                break
            buf.extend(chunk)
            count = len(buf) // need
            if count == 0:
                continue
            start = (count - 1) * need
            raw = bytes(buf[start:start + need])
            del buf[:start + need]
            with self._lock:
                self._latest = raw
        with self._lock:
            self._closed = True

    def take(self):
        """Return (raw, closed). raw is None when no new frame is waiting."""
        with self._lock:
            raw = self._latest
            self._latest = None
            return raw, self._closed


def _capture_nodes():
    root = "/sys/class/video4linux"
    if not os.path.isdir(root):
        return
    for node in sorted(os.listdir(root)):
        folder = os.path.join(root, node)
        try:
            label = open(os.path.join(folder, "name"), encoding="utf-8").read().strip()
            index = open(os.path.join(folder, "index"), encoding="utf-8").read().strip()
        except OSError:
            continue
        if index != "0":
            continue
        device = f"/dev/{node}"
        if os.path.exists(device):
            yield label, device


def find_laptop_camera():
    """The built-in color camera, skipping the infrared sensor."""
    integrated = [(label, device) for label, device in _capture_nodes() if "Integrated" in label]
    for label, device in integrated:
        if label.rstrip().endswith(" C"):
            return label, device
    if integrated:
        return integrated[0]
    return None


def find_camera_device():
    """Prefer a plugged-in webcam over the laptop's integrated camera.

    Returns (label, /dev/videoN) for the capture node, or None.
    """
    external = []
    integrated = []
    for label, device in _capture_nodes():
        if "Integrated" in label:
            integrated.append((label, device))
        else:
            external.append((label, device))
    if external:
        return external[-1]
    if integrated:
        return integrated[0]
    return None


class Camera:
    """Live view for the arm, and frames for vision development.

    start() opens an ffplay window and leaves the arm loop free.
    read_frame() and preview_detection() pull RGB frames into Python.
    Stop the ffplay window before reading frames. Both cannot share the device.
    """

    WIDTH = 640
    HEIGHT = 480

    def __init__(self):
        self._proc = None
        self._ffmpeg = None
        self._device = None
        self._label = None

    def start(self, device=None):
        if device:
            label = device
        else:
            found = find_camera_device()
            if not found:
                print("No camera found.")
                return False
            label, device = found
        if not os.path.exists(device):
            print(f"Camera device not found: {device}")
            return False
        self._device = device
        print(f"Live camera: {label} ({device})")
        try:
            self._proc = subprocess.Popen(
                [
                    "ffplay",
                    "-hide_banner",
                    "-loglevel", "error",
                    "-fflags", "nobuffer",
                    "-flags", "low_delay",
                    "-framedrop",
                    "-window_title", "MyRobot camera",
                    "-f", "v4l2",
                    "-input_format", "mjpeg",
                    "-video_size", "640x480",
                    "-framerate", "30",
                    device,
                ],
                stdin=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            print("ffplay is not installed, so the camera window could not be opened.")
            self._proc = None
            return False
        return True

    def stop(self):
        self._close_ffmpeg()
        proc = self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

    def read_frame(self, device=None):
        """Return one RGB image as a numpy array of shape (height, width, 3).

        Stop the ffplay preview before calling this. Both cannot use the device.
        device, when given, selects /dev/videoN for this read.
        """
        if self._proc is not None and self._proc.poll() is None:
            raise RuntimeError("Stop the camera preview before reading frames.")
        if device is not None and device != self._device:
            self._close_ffmpeg()
            self._label, self._device = device, device
        if self._ffmpeg is None:
            if self._device is None:
                found = find_camera_device()
                if not found:
                    raise RuntimeError("No camera found.")
                self._label, self._device = found
            self._open_ffmpeg()
        raw = self._read_raw_frame()
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.HEIGHT, self.WIDTH, 3))
        return frame.copy()

    def preview_detection(self, detect, device=None):
        """Show the webcam and the rings from detect(frame).

        Does not open the robot. Press q or close the window to stop.
        detect receives an RGB image and returns Ring(u, v, r_inner, r_outer).
        The yellow circle is the hole. The green circle is the outer rim.
        The cyan cross is the stylus pixel. The cyan line is the pixel error
        from that cross to the hole.
        """
        if device is None:
            found = find_laptop_camera() or find_camera_device()
            if not found:
                raise RuntimeError("No camera found.")
            self._label, self._device = found
        else:
            self._label, self._device = device, device
        print(f"Vision bench: {self._label} ({self._device})")
        print("Press q in the window to quit. The robot is not connected.")
        self._open_ffmpeg()

        import tkinter as tk
        from PIL import Image, ImageDraw, ImageTk

        root = tk.Tk()
        root.title(f"MyRobot vision — {self._label}")
        image_label = tk.Label(root)
        image_label.pack()
        status = tk.Label(root, text="")
        status.pack()
        state = {"run": True, "told": False}
        frames = _LatestFrames(self._ffmpeg, self.WIDTH * self.HEIGHT * 3)

        def finish():
            state["run"] = False
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", finish)
        root.bind("<q>", lambda _event: finish())
        root.bind("<Escape>", lambda _event: finish())

        def tick():
            if not state["run"]:
                return
            raw, closed = frames.take()
            if closed and raw is None:
                print("Camera closed.")
                finish()
                return
            if raw is None:
                root.after(10, tick)
                return
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.HEIGHT, self.WIDTH, 3)).copy()
            rings, note = _call_detect(detect, frame)
            if note and not state["told"]:
                print(note)
                state["told"] = True

            image = Image.fromarray(frame, mode="RGB")
            draw = ImageDraw.Draw(image)
            for u, v, r_inner, r_outer in rings:
                if abs(r_outer - r_inner) <= max(3.0, 0.08 * max(r_inner, 1.0)):
                    _draw_circle(draw, u, v, r_inner, "yellow")
                else:
                    _draw_circle(draw, u, v, r_outer, "lime")
                    _draw_circle(draw, u, v, r_inner, "yellow")
                draw.line([(u - 8, v), (u + 8, v)], fill="red", width=2)
                draw.line([(u, v - 8), (u, v + 8)], fill="red", width=2)
            _draw_stylus_travel(draw, rings, frame.shape)
            photo = ImageTk.PhotoImage(image)
            image_label.configure(image=photo)
            image_label.image = photo
            text = note or _ring_status(rings, frame.shape)
            if not rings and not note and float(frame.mean()) < 12.0:
                text = "image is very dark — check the lens cover"
            status.configure(text=text)
            root.after(1, tick)

        root.after(1, tick)
        root.mainloop()

    def watch_until(self, detect, device, consider):
        """Show device until consider(frame, rings) returns a value, or the user quits.

        consider returns None or KEEP_LOOKING to keep the window open. Any
        other return value closes the window and is returned. q or closing
        the window returns None.
        """
        self._label, self._device = device, device
        print(f"Arm camera: {device}")
        print("Press q in the window to keep the arm where it is.")
        self._open_ffmpeg()

        import tkinter as tk
        from PIL import Image, ImageDraw, ImageTk

        root = tk.Tk()
        root.title(f"MyRobot — {device}")
        image_label = tk.Label(root)
        image_label.pack()
        status = tk.Label(root, text="")
        status.pack()
        state = {"run": True, "result": None, "told": False}
        frames = _LatestFrames(self._ffmpeg, self.WIDTH * self.HEIGHT * 3)

        def finish(result=None):
            state["result"] = result
            state["run"] = False
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", lambda: finish(None))
        root.bind("<q>", lambda _event: finish(None))
        root.bind("<Escape>", lambda _event: finish(None))

        def tick():
            if not state["run"]:
                return
            raw, closed = frames.take()
            if closed and raw is None:
                print("Camera closed.")
                finish(None)
                return
            if raw is None:
                root.after(10, tick)
                return
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.HEIGHT, self.WIDTH, 3)).copy()
            rings, note = _call_detect(detect, frame)
            if note and not state["told"]:
                print(note)
                state["told"] = True
            result = consider(frame, rings)

            image = Image.fromarray(frame, mode="RGB")
            draw = ImageDraw.Draw(image)
            for u, v, r_inner, r_outer in rings:
                if abs(r_outer - r_inner) <= max(3.0, 0.08 * max(r_inner, 1.0)):
                    _draw_circle(draw, u, v, r_inner, "yellow")
                else:
                    _draw_circle(draw, u, v, r_outer, "lime")
                    _draw_circle(draw, u, v, r_inner, "yellow")
                draw.line([(u - 8, v), (u + 8, v)], fill="red", width=2)
                draw.line([(u, v - 8), (u, v + 8)], fill="red", width=2)
            _draw_stylus_travel(draw, rings, frame.shape)
            photo = ImageTk.PhotoImage(image)
            image_label.configure(image=photo)
            image_label.image = photo
            text = note or _ring_status(rings, frame.shape)
            if not rings and not note and float(frame.mean()) < 12.0:
                text = "image is very dark — check the lens cover"
            status.configure(text=text)
            if result is None or result is KEEP_LOOKING:
                root.after(1, tick)
                return
            finish(result)

        try:
            root.after(1, tick)
            root.mainloop()
        finally:
            self.stop()
        return state["result"]

    def _open_ffmpeg(self):
        self._close_ffmpeg()
        self._ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-fflags", "nobuffer",
                "-flags", "low_delay",
                "-thread_queue_size", "1",
                "-f", "v4l2",
                "-input_format", "mjpeg",
                "-video_size", f"{self.WIDTH}x{self.HEIGHT}",
                "-framerate", "30",
                "-i", self._device,
                "-fps_mode", "passthrough",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def _read_raw_frame(self):
        frame_bytes = self.WIDTH * self.HEIGHT * 3
        raw = self._ffmpeg.stdout.read(frame_bytes)
        if raw is None or len(raw) != frame_bytes:
            raise RuntimeError(f"Could not read a frame from {self._device}")
        return raw

    def _close_ffmpeg(self):
        proc = self._ffmpeg
        self._ffmpeg = None
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        if proc.stdout is not None:
            proc.stdout.close()


def _draw_circle(draw, u, v, radius, color):
    radius = float(radius)
    draw.ellipse([u - radius, v - radius, u + radius, v + radius], outline=color, width=2)


def _draw_cross(draw, u, v, color, arm=10):
    draw.line([(u - arm, v), (u + arm, v)], fill=color, width=2)
    draw.line([(u, v - arm), (u, v + arm)], fill=color, width=2)


def _draw_stylus_travel(draw, rings, image_shape):
    """Cyan cross is the stylus pixel. The line is the pixel error to the hole."""
    from robot.calibration import pixel_travel, stylus_pixel

    su, sv = stylus_pixel(image_shape)
    _draw_cross(draw, su, sv, "cyan")
    if not rings:
        return
    u, v = rings[0][0], rings[0][1]
    draw.line([(su, sv), (u, v)], fill="cyan", width=2)
    du, dv = pixel_travel((u, v), image_shape)
    draw.text((8, 8), f"du={du:+.0f}  dv={dv:+.0f} px", fill="cyan")


def _ring_status(rings, image_shape):
    from robot.calibration import pixel_travel, stylus_pixel

    su, sv = stylus_pixel(image_shape)
    if not rings:
        return f"no ring, stylus at ({su:.0f}, {sv:.0f})"
    u, v, r_inner, _r_outer = rings[0]
    du, dv = pixel_travel((u, v), image_shape)
    return (
        f"ring r={r_inner:.0f}px  travel du={du:+.0f} dv={dv:+.0f} px"
        f"  (stylus {su:.0f},{sv:.0f} -> hole {u:.0f},{v:.0f})"
    )


def _call_detect(detect, frame):
    try:
        rings = detect(frame) or []
    except NotImplementedError:
        return [], "detect_rings() is not written yet. The window is the webcam only."
    except Exception as exc:
        return [], f"detect_rings error: {exc}"
    parsed = []
    for ring in rings:
        try:
            u, v, r_inner, r_outer = ring
            parsed.append((float(u), float(v), float(r_inner), float(r_outer)))
        except (TypeError, ValueError):
            return [], "detect_rings must return (u, v, r_inner, r_outer)"
    return parsed, ""
