"""
MIT BWSI Autonomous RACECAR
GNU General Public License v3.0
racecar-neo-outreach-labs

File Name: wallfollow.py

Title: Wall follow dashboard

Purpose: Lidar wall following with PD steering and optional closed-loop speed,
tuned from a browser. Adapted from MITRacecarNeo/wallfollow_dashboard, with
the ROS node replaced by racecar_core calls, so the same program runs against
the simulator and the car. wallfollow.html keeps the upstream page and its
shared RACECAR Neo styles, vendored in labs/web/.

Two controllers, selected by `mode` in wallfollow.yaml (file edit only):

  static:  mirrored side windows at look_angle. Error = right - left, meters.
           Setpoint 0 keeps the car centered between the walls.
  dynamic: steer toward the most open heading in a front arc, with obstacles
           inflated by the car's half-width. side_weight biases one side.
           Error = chosen heading / width, -1 to +1.

Steering is kp * error + kd * (change in error), clamped to -1..1. Throttle
is the speed slider; with speed_kp or speed_kd above 0 a trim term pulls the
encoder speed toward a target that shrinks as the open road ahead shortens.

Usage:
    python wallfollow.py -s         RacecarSim (enter user program mode to start)
    python wallfollow.py            physical car
    then open http://localhost:8083 in a browser.

Options:
    --port N        HTTP port (default 8083)

Threading: update() is the only code that calls racecar_core. The HTTP server
thread reads the shared state and writes parameters under _lock.

Angles use the student convention (0 = nose, positive = right), the order
rc.lidar.get_samples() documents: sample 0 in front, clockwise, cm, 0.0 for
no return. Unlike upstream there is no LIDAR_MOUNT_YAW_DEG; any mount offset
belongs in racecar_core. The shipped yaml has speed at 0.0, so the car does
not move until the speed slider is raised.
"""

import argparse
import json
import math
import re
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import yaml

PORT = 8083
BASE = Path(__file__).resolve().parent
YAML_PATH = BASE / "wallfollow.yaml"
LOG_DIR = BASE / "logs"
# Shared styles, fonts and logos, served under /ui/ (labs/web/webui.py).
sys.path.insert(0, str(BASE.parent / "web"))
import webui  # noqa: E402

MAX_SPEED = 1.0       # hard throttle cap before the max_speed scaling in rc.drive
SCAN_RATE_HZ = 10.0   # scans processed per second, about the RPLIDAR's own rate,
                      # so kd tuned here means the same thing on the car
SEARCH_STEP = 2       # degrees between candidate headings (dynamic)
HALF_CLEAR = 0.21     # meters, car half-width plus margin, inflates obstacles
CENTER_BIAS = 0.006   # score penalty per degree off-center, stops ping-pong
SCAN_STRIDE = 3       # every 3rd lidar point is sent to the browser
CHART_SECONDS = 30.0  # live chart span
# The live chart is 420 px wide; send an even sample of the last
# CHART_SECONDS rather than every row. The log csv keeps all of them.
HIST_POINTS = 300

TUNE_KEYS = ("speed", "kp", "kd", "speed_kp", "speed_kd", "window", "lookahead",
             "look_angle", "width", "side_weight", "max_speed")

_lock = threading.Lock()
_params: Dict[str, Any] = {}
_state: Dict[str, Any] = {"scan": [], "hist": [], "overlay": {}, "enc_speed": 0.0,
                          "scan_fps": 0.0, "points": 0, "marks": []}
# 600 scans at SCAN_RATE_HZ is the last 60 s.
_hist: Deque[List[float]] = deque(maxlen=600)
# Full rows behind the live chart. POST /logs/save snapshots this to a csv.
_rows: Deque[str] = deque(maxlen=600)
# Parameter-change markers [t, label] shown on the charts. Saved into logs as
# "# mark" lines.
_marks: Deque[List[Any]] = deque(maxlen=40)
_T0 = time.monotonic()


def _now() -> float:
    return round(time.monotonic() - _T0, 2)


def _add_mark(label: str) -> None:
    # Coalesce rapid updates: the speed slider fires while it is dragged.
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
    p["mode"] = 1 if str(p["mode"]).strip() in ("1", "dynamic") else 0
    p["speed"] = set_max_speed(p["speed"])
    p["max_speed"] = max(0.0, min(1.0, float(p["max_speed"])))
    for k in ("kp", "kd", "speed_kp", "speed_kd"):
        p[k] = max(0.0, float(p[k]))
    p["window"] = max(1.0, min(90.0, float(p["window"])))
    p["lookahead"] = max(0.1, float(p["lookahead"]))
    p["max_mps"] = max(0.1, float(p["max_mps"]))
    p["look_angle"] = max(0.0, min(180.0, float(p["look_angle"])))
    p["width"] = max(float(SEARCH_STEP), min(180.0, float(p["width"])))
    p["side_weight"] = float(p["side_weight"])


def load_params() -> None:
    with _lock:
        before = json.dumps(_params, sort_keys=True)
        _params.update(yaml.safe_load(YAML_PATH.read_text()))
        _clean(_params)
        if before != "{}" and before != json.dumps(_params, sort_keys=True):
            _add_mark("yaml load")


def save_params() -> None:
    """Rewrite the numeric values in place so the yaml comments survive.
    `mode` is left as written; it is a file-only setting."""
    scalar = re.compile(r"^([A-Za-z_]\w*):([ \t]*)[-\d.eE+]+(.*)$")
    out = []
    with _lock:
        for raw in YAML_PATH.read_text().splitlines(keepends=True):
            m = scalar.match(raw.rstrip("\n"))
            if m and m.group(1) != "mode":
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
                _params[k] = float(data[k])
        _clean(_params)
        changed = [f"{k} {_params[k]:g}" for k in TUNE_KEYS if before.get(k) != _params[k]]
        if changed:
            _add_mark(", ".join(changed))
    return changed


def index_at(n: int, deg: float) -> int:
    """Sample index for a student angle (0 = nose, positive = right)."""
    return round(deg * n / 360.0) % n


def student_deg(n: int, i: int) -> float:
    """Student angle of sample `i`, wrapped to [-180, 180)."""
    return (i * 360.0 / n + 180.0) % 360.0 - 180.0


def mean_at(scan_m: np.ndarray, deg: float, window: float, la: float) -> float:
    """Average range (m) in a `window`-deg cone at `deg`, capped at `la`.

    Samples of 0 are no return, which reads as open space (`la`) rather than
    as zero; zero would pull the car toward gaps."""
    n = len(scan_m)
    center = index_at(n, deg)
    half = max(1, round(window / 2 * n / 360.0))
    vals = scan_m[[(center + k) % n for k in range(-half, half + 1)]]
    vals = vals[vals > 0]
    return min(float(vals.mean()), la) if vals.size else la


def static_error(scan_m: np.ndarray, p: Dict[str, Any], la: float) -> Tuple[float, Dict[str, Any]]:
    left = mean_at(scan_m, -p["look_angle"], p["window"], la)
    right = mean_at(scan_m, p["look_angle"], p["window"], la)
    overlay = {"mode": "static", "look_angle": p["look_angle"], "window": p["window"],
               "left": round(left, 2), "right": round(right, 2), "la": la}
    return right - left, overlay  # >0: drifted left -> steer right


def dynamic_error(scan_m: np.ndarray, p: Dict[str, Any],
                  la: float) -> Tuple[float, float, Dict[str, Any]]:
    """Most open heading in +/- width. Returns (error, road ahead on it, overlay)."""
    width = int(p["width"])
    degs = list(range(-width, width + 1, SEARCH_STEP))
    dists = [mean_at(scan_m, d, p["window"], la) for d in degs]
    # An obstacle at d blocks every heading within atan(HALF_CLEAR/d) of it;
    # the car body would sideswipe it there.
    corridor = list(dists)
    for j, d in enumerate(dists):
        if d >= la:
            continue
        block = math.degrees(math.atan2(HALF_CLEAR, d))
        span = int(block // SEARCH_STEP) + 1
        for i in range(max(0, j - span), min(len(degs), j + span + 1)):
            if abs(degs[i] - degs[j]) <= block and d < corridor[i]:
                corridor[i] = d
    best = max(range(len(degs)),
               key=lambda i: corridor[i] - CENTER_BIAS * abs(degs[i]) + p["side_weight"] * degs[i])
    overlay = {"mode": "dynamic", "width": p["width"], "target_deg": degs[best], "la": la}
    return degs[best] / p["width"], corridor[best], overlay


class WallFollower:
    """Controller state between scans. process() takes a scan in cm and
    updates the command; actuate() sends it through a racecar."""

    def __init__(self) -> None:
        self._enc = 0.0
        self._last_error = 0.0
        self._error = 0.0
        self._steer_cmd = 0.0
        self._speed_cmd = 0.0
        self._trim = 0.0
        self._last_speed_error = 0.0
        self._last_proc = 0.0
        self._proc_stamps: List[float] = []
        self._max_speed: Optional[float] = None  # last value sent to rc.drive

    def tick(self, rc: Any) -> None:
        """One update(): read the lidar at SCAN_RATE_HZ, drive every call.
        The encoder is read first so the speed loop sees this frame's speed."""
        self._enc = float(rc.physics.get_encoder_speed())
        with _lock:
            _state["enc_speed"] = round(self._enc, 3)
        now = time.monotonic()
        if now - self._last_proc >= 1.0 / SCAN_RATE_HZ:
            self._last_proc = now
            scan = rc.lidar.get_samples()
            if scan is not None and len(scan) > 0:
                self.process(np.asarray(scan, dtype=np.float32))
        self.actuate(rc)

    def process(self, scan_cm: np.ndarray) -> None:
        """Run one scan through the active controller and publish the state."""
        with _lock:
            p = dict(_params)
        scan_m = scan_cm / 100.0
        la = p["lookahead"]

        if p["mode"] == 1:
            error, road, overlay = dynamic_error(scan_m, p, la)
        else:
            error, overlay = static_error(scan_m, p, la)
            road = mean_at(scan_m, 0.0, p["window"], la)

        cmd = p["kp"] * error + p["kd"] * (error - self._last_error)
        self._last_error = error
        self._error = error
        self._steer_cmd = max(-1.0, min(1.0, cmd))
        self._shape_speed(p, road, la)

        now = time.monotonic()
        self._proc_stamps.append(now)
        self._proc_stamps = [t for t in self._proc_stamps if now - t < 2.0]
        t = _now()
        _rows.append(f"{t},{error:.4f},{self._steer_cmd:.4f},{self._speed_cmd:.3f},{self._enc:.3f}\n")
        _hist.append([t, round(error, 4)])
        overlay.update({"error": round(error, 4), "steer": round(self._steer_cmd, 3),
                        "speed_cmd": round(self._speed_cmd, 3)})
        n = len(scan_m)
        points = [[round(student_deg(n, i), 1), round(float(scan_m[i]), 3)]
                  for i in range(0, n, SCAN_STRIDE) if scan_m[i] > 0]
        with _lock:
            _state.update({"scan": points, "overlay": overlay, "hist": _thin(_hist),
                           "marks": list(_marks), "points": n,
                           "scan_fps": round(len(self._proc_stamps) / 2.0, 1)})

    def _shape_speed(self, p: Dict[str, Any], road: float, la: float) -> None:
        """With speed_kp and speed_kd at 0 the slider is the throttle. Otherwise
        a trim term pulls the encoder speed toward a target that scales with
        the open road ahead: the corridor along the chosen heading (dynamic)
        or straight ahead (static). Corners shorten it and slow the car."""
        # Target in m/s: what the slider would reach on an open straight,
        # after rc.drive scales it by max_speed.
        target = p["speed"] * p["max_speed"] * p["max_mps"] * min(road / la, 1.0)
        serr = target - self._enc
        if p["speed"] <= 1e-3:
            self._trim = 0.0
            self._speed_cmd = 0.0
        elif p["speed_kp"] == 0 and p["speed_kd"] == 0:
            self._trim = 0.0
            self._speed_cmd = p["speed"]
        else:
            # Never integrate up while commanded but not moving, such as
            # pinned against a wall.
            blocked = self._enc < 0.05 and self._speed_cmd > 0.3
            if serr < 0 or not blocked:
                self._trim += p["speed_kp"] * serr + p["speed_kd"] * (serr - self._last_speed_error)
            self._trim = max(-1.0, min(0.3, self._trim))
            self._speed_cmd = set_max_speed(p["speed"] + self._trim)
        self._last_speed_error = serr

    def actuate(self, rc: Any) -> None:
        with _lock:
            max_speed = _params["max_speed"]
        if max_speed != self._max_speed:
            rc.drive.set_max_speed(max_speed)
            self._max_speed = max_speed
        rc.drive.set_speed_angle(set_max_speed(self._speed_cmd), self._steer_cmd)


def _thin(hist: Deque[List[float]]) -> List[List[float]]:
    """Rows from the last CHART_SECONDS, evenly sampled down to HIST_POINTS,
    newest row always kept."""
    rows = list(hist)
    if rows:
        start = rows[-1][0] - CHART_SECONDS
        rows = [r for r in rows if r[0] >= start]
    stride = max(1, -(-len(rows) // HIST_POINTS))
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
        asset = webui.static_file(self.path)
        if asset is not None:
            self._send(asset[0], asset[1], cache=webui.CACHE_CONTROL)
            return
        if self.path == "/":
            self._send((BASE / "wallfollow.html").read_bytes(), "text/html; charset=utf-8")
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
                f.write("t,error,steer,speed_cmd,speed_ms\n")
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
    follower = WallFollower()

    def start() -> None:
        rc.drive.stop()

    def update() -> None:
        follower.tick(rc)

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    mode = "dynamic" if _params["mode"] == 1 else "static"
    print(f">> Wall follow dashboard ({mode}) on http://localhost:{args.port}; "
          "enter user program mode to start.")
    rc.set_start_update(start, update)
    rc.go()
    server.shutdown()


if __name__ == "__main__":
    main()
