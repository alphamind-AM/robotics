import os
import select
import subprocess

import numpy as np


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

    def read_frame(self):
        """Return one RGB image as a numpy array of shape (height, width, 3).

        Stop the ffplay preview before calling this. Both cannot use the device.
        """
        if self._proc is not None and self._proc.poll() is None:
            raise RuntimeError("Stop the camera preview before reading frames.")
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
        """Show the laptop camera and boxes from detect(frame).

        Does not open the robot. Press q or close the window to stop.
        detect receives an RGB image and returns (u, v, w, h) boxes.
        """
        if device is None:
            found = find_laptop_camera()
            if not found:
                raise RuntimeError("No laptop camera found.")
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
        state = {"run": True, "buf": bytearray(), "told": False}

        def finish():
            state["run"] = False
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", finish)
        root.bind("<q>", lambda _event: finish())
        root.bind("<Escape>", lambda _event: finish())

        frame_bytes = self.WIDTH * self.HEIGHT * 3
        fd = self._ffmpeg.stdout.fileno()

        def tick():
            if not state["run"]:
                return
            while len(state["buf"]) < frame_bytes:
                ready, _, _ = select.select([fd], [], [], 0.02)
                if not ready:
                    root.after(10, tick)
                    return
                chunk = os.read(fd, frame_bytes - len(state["buf"]))
                if not chunk:
                    print("Camera closed.")
                    finish()
                    return
                state["buf"].extend(chunk)

            raw = bytes(state["buf"][:frame_bytes])
            del state["buf"][:frame_bytes]
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.HEIGHT, self.WIDTH, 3)).copy()
            faces, note = _call_detect(detect, frame)
            if note and not state["told"]:
                print(note)
                state["told"] = True

            image = Image.fromarray(frame, mode="RGB")
            draw = ImageDraw.Draw(image)
            for box in faces:
                u, v, w, h = [int(x) for x in box]
                draw.rectangle([u, v, u + w, v + h], outline="lime", width=3)
            photo = ImageTk.PhotoImage(image)
            image_label.configure(image=photo)
            image_label.image = photo
            status.configure(text=note or f"{len(faces)} face(s)")
            root.after(1, tick)

        root.after(1, tick)
        root.mainloop()

    def _open_ffmpeg(self):
        self._close_ffmpeg()
        self._ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-f", "v4l2",
                "-input_format", "mjpeg",
                "-video_size", f"{self.WIDTH}x{self.HEIGHT}",
                "-framerate", "30",
                "-i", self._device,
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


def _call_detect(detect, frame):
    try:
        faces = detect(frame) or []
    except NotImplementedError:
        return [], "detect_faces() is not written yet. The window is the laptop camera only."
    except Exception as exc:
        return [], f"detect_faces error: {exc}"
    boxes = []
    for box in faces:
        try:
            u, v, w, h = box
            boxes.append((float(u), float(v), float(w), float(h)))
        except (TypeError, ValueError):
            return [], "detect_faces must return (u, v, w, h) boxes"
    return boxes, ""
