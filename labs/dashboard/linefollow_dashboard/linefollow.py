"""
MIT BWSI Autonomous RACECAR
GNU General Public License v3.0
racecar-neo-outreach-labs

File Name: linefollow.py

Title: Line follow dashboard

Purpose: HSV threshold, crop band, and PD steering for a line follower, tuned
from trackbars in a browser. Adapted from MITRacecarNeo/linefollow_dashboard,
with the ROS node replaced by racecar_core calls, so the same program runs
against the simulator and the car. linefollow.html is the upstream page.

Pipeline (same as rc_utils.find_contours and get_largest_contour):

  frame -> resize to process_width -> HSV -> inRange(low, high) -> crop band
        -> largest contour above min_area -> its center column

The center's offset from the image center is the error, -1 (left edge) to +1
(right edge). Steering is angle_kp * error + angle_kd * (change in error);
throttle is speed reduced by speed_kp per unit of |error|. With no line in the
band the last command is held.

Usage:
    python linefollow.py -s         RacecarSim (enter user program mode to start)
    python linefollow.py            physical car
    then open http://localhost:8086 in a browser.

Options:
    --port N        HTTP port (default 8086)

Threading: update() is the only code that calls racecar_core. The HTTP server
thread reads the shared state and writes parameters under _lock.

The shipped yaml has speed at 0.0, so the car does not move until the speed
slider is raised. Angles use the student convention (positive = right).
"""

import argparse
import json
import re
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

PORT = 8086
BASE = Path(__file__).resolve().parent
YAML_PATH = BASE / "linefollow.yaml"
LOG_DIR = BASE / "logs"

MAX_SPEED = 1.0         # hard throttle cap before the max_speed scaling in rc.drive
PROCESS_RATE_HZ = 20.0  # frames thresholded per second; update() runs at ~60
MIN_BAND = 0.05         # crop_bottom - crop_top never below this fraction
# The live chart is 420 px wide; send an even sample of the history rather
# than every row. The log csv keeps all of them.
HIST_POINTS = 300
# Overlay colors, BGR, matching the dashboard's tokens.
GOLD = (0, 215, 255)
RED = (51, 0, 255)
GREY = (200, 200, 200)

# Tunable from the dashboard, in the order the trackbars show them.
HSV_KEYS = ("h_low", "h_high", "s_low", "s_high", "v_low", "v_high")
HSV_MAX = {"h_low": 179, "h_high": 179, "s_low": 255, "s_high": 255,
           "v_low": 255, "v_high": 255}
FLOAT_KEYS = ("speed", "angle_kp", "crop_top", "crop_bottom", "speed_kp",
              "angle_kd", "max_speed")
INT_KEYS = HSV_KEYS + ("min_area",)
TUNE_KEYS = HSV_KEYS + FLOAT_KEYS + ("min_area",)

_lock = threading.Lock()
_params: Dict[str, Any] = {}
_state: Dict[str, Any] = {"found": False, "error": 0.0, "steer": 0.0, "speed_cmd": 0.0,
                          "cx": None, "cy": None, "area": 0, "lost_for": None,
                          "res": [0, 0], "proc_res": [0, 0], "cam_fps": 0.0,
                          "proc_fps": 0.0, "enc_speed": 0.0, "hist": [], "marks": []}
_preview_color = b""
_preview_mask = b""
_hist: Deque[List[Any]] = deque(maxlen=1200)
# Full rows behind the live chart. POST /logs/save snapshots this to a csv,
# so a saved log is exactly what was on the screen.
_rows: Deque[str] = deque(maxlen=1200)
# Parameter-change markers [t, label] shown on the charts. Saved into logs as
# "# mark" lines.
_marks: Deque[List[Any]] = deque(maxlen=40)
_T0 = time.monotonic()


def _now() -> float:
    return round(time.monotonic() - _T0, 2)


def _add_mark(label: str) -> None:
    # Coalesce rapid updates: every trackbar fires while it is dragged.
    if _marks and _now() - _marks[-1][0] < 1.0 and _marks[-1][1].split(" ")[0] == label.split(" ")[0]:
        _marks[-1] = [_now(), label]
    else:
        _marks.append([_now(), label])


def set_max_speed(speed: float) -> float:
    """Clamp throttle to [0, MAX_SPEED]. Every speed sent to rc.drive passes
    through here."""
    return max(0.0, min(MAX_SPEED, float(speed)))


def _clean(p: Dict[str, Any]) -> None:
    """Clamp every tunable into its range. Caller holds _lock."""
    for k in HSV_KEYS:
        p[k] = max(0, min(HSV_MAX[k], int(p[k])))
    p["speed"] = set_max_speed(p["speed"])
    p["angle_kp"] = max(0.0, float(p["angle_kp"]))
    p["angle_kd"] = max(0.0, float(p["angle_kd"]))
    p["speed_kp"] = max(0.0, float(p["speed_kp"]))
    p["max_speed"] = max(0.0, min(1.0, float(p["max_speed"])))
    p["min_area"] = max(1, int(p["min_area"]))
    # The band keeps its order and a minimum height, so a slider dragged past
    # its partner pushes the partner along instead of inverting the crop.
    top = max(0.0, min(1.0 - MIN_BAND, float(p["crop_top"])))
    bottom = max(top + MIN_BAND, min(1.0, float(p["crop_bottom"])))
    p["crop_top"], p["crop_bottom"] = round(top, 3), round(bottom, 3)


def load_params() -> None:
    with _lock:
        before = json.dumps(_params, sort_keys=True)
        _params.update(yaml.safe_load(YAML_PATH.read_text()))
        _clean(_params)
        if before != "{}" and before != json.dumps(_params, sort_keys=True):
            _add_mark("yaml load")


def save_params() -> None:
    """Rewrite the numeric values in place so the yaml comments survive."""
    scalar = re.compile(r"^([A-Za-z_]\w*):([ \t]*)[-\d.eE+]+(.*)$")
    out = []
    with _lock:
        for raw in YAML_PATH.read_text().splitlines(keepends=True):
            m = scalar.match(raw.rstrip("\n"))
            if m:
                key, gap, tail = m.groups()
                v = _params.get(key)
                if isinstance(v, (int, float)):
                    out.append(f"{key}:{gap}{v}{tail}\n")
                    continue
            out.append(raw)
    YAML_PATH.write_text("".join(out))


def update_params(data: Dict[str, Any]) -> List[str]:
    """Apply a posted subset of TUNE_KEYS. Returns the change labels."""
    with _lock:
        before = dict(_params)
        for k in TUNE_KEYS:
            if k in data:
                _params[k] = int(data[k]) if k in INT_KEYS else float(data[k])
        _clean(_params)
        changed = [f"{k} {_params[k]:g}" for k in TUNE_KEYS if before.get(k) != _params[k]]
        if changed:
            _add_mark(", ".join(changed))
    return changed


def threshold(hsv: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    """inRange over an HSV image. A hue low above its high wraps through 0,
    so red (which straddles hue 0/179) is one range rather than two."""
    lo = np.array([p["h_low"], p["s_low"], p["v_low"]], np.uint8)
    hi = np.array([p["h_high"], p["s_high"], p["v_high"]], np.uint8)
    if p["h_low"] <= p["h_high"]:
        return cv2.inRange(hsv, lo, hi)
    lo2, hi2 = lo.copy(), hi.copy()
    lo2[0], hi[0] = 0, 179  # [h_low..179] or [0..h_high]
    return cv2.bitwise_or(cv2.inRange(hsv, lo, hi), cv2.inRange(hsv, lo2, hi2))


def find_line(mask: np.ndarray, top: int, bottom: int,
              min_area: int) -> Optional[Tuple[np.ndarray, Tuple[int, int], float]]:
    """Largest contour in mask[top:bottom] with area >= min_area.

    Returns (contour in full-mask coordinates, (cx, cy), area) or None.
    Same selection rc_utils.get_largest_contour makes."""
    band = mask[top:bottom]
    if band.size == 0:
        return None
    contours, _ = cv2.findContours(band, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0.0
    for c in contours:
        a = cv2.contourArea(c)
        if a > best_area:
            best, best_area = c, a
    if best is None or best_area < min_area:
        return None
    m = cv2.moments(best)
    if m["m00"] == 0:
        return None
    best = best + np.array([[[0, top]]], dtype=best.dtype)
    return best, (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]) + top), best_area


class LineFollower:
    """Controller state between frames. process() takes a BGR frame and
    updates the command; actuate() sends it through a racecar."""

    def __init__(self) -> None:
        self._enc = 0.0
        self._last_error = 0.0
        self._error = 0.0
        self._steer_cmd = 0.0
        self._speed_cmd = 0.0
        self._found = False
        self._last_seen: Optional[float] = None
        self._res = (0, 0)
        self._last_proc = 0.0
        self._frame_stamps: List[float] = []
        self._proc_stamps: List[float] = []
        self._max_speed: Optional[float] = None  # last value sent to rc.drive

    def tick(self, rc: Any) -> None:
        """One update(): read the camera at PROCESS_RATE_HZ, drive every call."""
        now = time.monotonic()
        self._frame_stamps.append(now)
        self._frame_stamps = [t for t in self._frame_stamps if now - t < 2.0]
        if now - self._last_proc >= 1.0 / PROCESS_RATE_HZ:
            self._last_proc = now
            frame = rc.camera.get_color_image()
            if frame is not None:
                self._res = (frame.shape[1], frame.shape[0])
                self.process(frame)
        self.actuate(rc)

    def process(self, frame: np.ndarray) -> None:
        """Threshold one BGR frame, update the command, and render the previews."""
        global _preview_color, _preview_mask
        with _lock:
            p = dict(_params)

        h, w = frame.shape[:2]
        pw = int(p["process_width"])
        img = cv2.resize(frame, (pw, max(1, round(pw * h / w))))
        H, W = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = threshold(hsv, p)
        top, bottom = int(p["crop_top"] * H), int(p["crop_bottom"] * H)
        hit = find_line(mask, top, bottom, p["min_area"])

        now = time.monotonic()
        if hit is not None:
            contour, (cx, cy), area = hit
            error = (cx - W / 2) / (W / 2)  # -1 left edge .. +1 right edge
            cmd = p["angle_kp"] * error + p["angle_kd"] * (error - self._last_error)
            self._last_error = error
            self._steer_cmd = max(-1.0, min(1.0, cmd))
            # Constant throttle by default; speed_kp bleeds it off as the
            # line drifts from center so a sharp turn is taken slower.
            self._speed_cmd = set_max_speed(p["speed"] * max(0.0, 1.0 - p["speed_kp"] * abs(error)))
            self._error = error
            self._found = True
            self._last_seen = now
        else:
            # No line in the band: hold the last command so a gap in a dashed
            # line does not straighten the wheels.
            contour, cx, cy, area = None, None, None, 0
            self._found = False

        self._proc_stamps.append(now)
        self._proc_stamps = [t for t in self._proc_stamps if now - t < 2.0]
        t = _now()
        lost = 0 if self._found else 1
        _rows.append(f"{t},{self._error:.4f},{self._steer_cmd:.4f},{self._speed_cmd:.3f},"
                     f'{self._enc:.3f},{lost},{"" if cx is None else cx},{area:.0f}\n')
        _hist.append([t, round(self._error, 4), lost])

        # Previews: the color frame carries the band, the contour, and its
        # center; the mask is the whole frame with the band outlined.
        color = img if int(p["preview_width"]) == W else cv2.resize(
            img, (int(p["preview_width"]), max(1, round(int(p["preview_width"]) * H / W))))
        sx = color.shape[1] / W
        shade = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if shade.shape[1] != color.shape[1]:
            shade = cv2.resize(shade, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
        for view in (color, shade):
            cv2.rectangle(view, (0, int(top * sx)), (view.shape[1] - 1, int(bottom * sx) - 1),
                          GOLD, 1)
        cv2.line(color, (color.shape[1] // 2, 0), (color.shape[1] // 2, color.shape[0]),
                 GREY, 1)
        if contour is not None:
            cv2.drawContours(color, [(contour * sx).astype(np.int32)], -1, RED, 2)
            cv2.circle(color, (int(cx * sx), int(cy * sx)), 5, RED, -1)
            cv2.line(color, (color.shape[1] // 2, int(cy * sx)), (int(cx * sx), int(cy * sx)),
                     RED, 1)
        quality = [cv2.IMWRITE_JPEG_QUALITY, int(p["preview_quality"])]
        with _lock:
            _preview_color = cv2.imencode(".jpg", color, quality)[1].tobytes()
            _preview_mask = cv2.imencode(".jpg", shade, quality)[1].tobytes()
            _state.update({
                "found": self._found, "error": round(self._error, 4),
                "steer": round(self._steer_cmd, 3), "speed_cmd": round(self._speed_cmd, 3),
                "cx": cx, "cy": cy, "area": int(area),
                "lost_for": None if self._found else (
                    round(now - self._last_seen, 1) if self._last_seen else None),
                "res": list(self._res), "proc_res": [W, H],
                "cam_fps": round(len(self._frame_stamps) / 2.0, 1),
                "proc_fps": round(len(self._proc_stamps) / 2.0, 1),
                "hist": _thin(_hist), "marks": list(_marks),
            })

    def actuate(self, rc: Any) -> None:
        with _lock:
            max_speed = _params["max_speed"]
        if max_speed != self._max_speed:
            rc.drive.set_max_speed(max_speed)
            self._max_speed = max_speed
        rc.drive.set_speed_angle(set_max_speed(self._speed_cmd), self._steer_cmd)
        self._enc = float(rc.physics.get_encoder_speed())
        with _lock:
            _state["enc_speed"] = round(self._enc, 3)


def _thin(hist: Deque[List[Any]]) -> List[List[Any]]:
    """Even sample of `hist` down to HIST_POINTS, newest row always kept."""
    rows = list(hist)
    stride = max(1, len(rows) // HIST_POINTS)
    if stride == 1:
        return rows
    sent = rows[::stride]
    if sent[-1] is not rows[-1]:
        sent.append(rows[-1])
    return sent


def _log_number(name: str) -> int:
    """Log7.csv -> 7, so lists sort numerically. Anything else sorts first."""
    m = re.match(r"Log(\d+)\.csv$", name)
    return int(m.group(1)) if m else 0


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path == "/":
            self._send((BASE / "linefollow.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path.startswith("/frame/color") or self.path.startswith("/frame/mask"):
            with _lock:
                data = _preview_mask if "/mask" in self.path else _preview_color
            if data:
                self._send(data, "image/jpeg", cache="no-store")
            else:
                self.send_error(503)
        elif self.path == "/state":
            with _lock:
                body = json.dumps({**_state, "params": _params, "marks": list(_marks)})
            self._send(body.encode(), "application/json")
        elif self.path == "/logs":
            names = sorted((f.name for f in LOG_DIR.glob("*.csv")), key=_log_number)
            self._send(json.dumps(names).encode(), "application/json")
        elif self.path.startswith("/logs/"):
            f = LOG_DIR / Path(self.path).name
            if f.is_file():
                self._send(f.read_bytes(), "text/csv")
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/params":
            update_params(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        elif self.path == "/logs/save":
            LOG_DIR.mkdir(exist_ok=True)
            latest = max((_log_number(f.name) for f in LOG_DIR.glob("Log*.csv")), default=0)
            name = f"Log{latest + 1}.csv"
            rows = list(_rows)
            tmin = float(rows[0].split(",")[0]) if rows else 0.0
            with open(LOG_DIR / name, "w") as f:
                f.write("t,error,steer,speed_cmd,speed_ms,lost,cx,area\n")
                f.writelines(f"# mark,{t},{label}\n" for t, label in list(_marks) if t >= tmin)
                f.writelines(rows)
            self._send(name.encode(), "text/plain")
            return
        elif self.path == "/save":
            save_params()
        elif self.path == "/load":
            load_params()
        else:
            self.send_error(404)
            return
        self._send(b"ok", "text/plain")

    def _send(self, body: bytes, ctype: str, cache: str = "") -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *log_args: Any) -> None:
        pass


def main() -> None:
    import racecar_core

    # add_help=False: racecar_core uses -h for headless mode
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int, default=PORT)
    args, _ = parser.parse_known_args()

    load_params()
    rc = racecar_core.create_racecar()
    follower = LineFollower()

    def start() -> None:
        rc.drive.stop()

    def update() -> None:
        follower.tick(rc)

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f">> Line follow dashboard on http://localhost:{args.port}; enter user program mode to start.")
    rc.set_start_update(start, update)
    rc.go()
    server.shutdown()


if __name__ == "__main__":
    main()
