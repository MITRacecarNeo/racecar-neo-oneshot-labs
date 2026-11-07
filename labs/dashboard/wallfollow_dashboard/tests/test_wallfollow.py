"""Controller and server tests for wallfollow.py.

Runs without RacecarSim or a lidar: scans are synthetic corridors traced ray
by ray, and FakeRacecar records what reaches rc.drive.

    pytest -q labs/wallfollow_dashboard/tests
"""

import importlib.util
import json
import math
import re
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pytest
import yaml

BASE = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("wallfollow", BASE / "wallfollow.py")
wf = importlib.util.module_from_spec(_spec)
sys.modules["wallfollow"] = wf
_spec.loader.exec_module(wf)

SHIPPED: Optional[str] = None  # yaml as committed, restored after any test that writes it
N = 1080       # racecar_core samples per scan
MAX_RANGE = 10.0  # meters; beyond this a ray reads 0, no return


def corridor(left: Optional[float], right: Optional[float],
             front: Optional[float] = None) -> np.ndarray:
    """Scan in cm from walls parallel to the car at `left` and `right` meters
    and one across its path at `front`. None leaves that side open."""
    out = np.zeros(N, np.float32)
    for i in range(N):
        a = math.radians(i * 360.0 / N)  # clockwise from the nose
        sx, cy = math.sin(a), math.cos(a)
        hits = []
        if right is not None and sx > 1e-9:
            hits.append(right / sx)
        if left is not None and sx < -1e-9:
            hits.append(left / -sx)
        if front is not None and cy > 1e-9:
            hits.append(front / cy)
        r = min(hits, default=0.0)
        out[i] = r * 100.0 if 0 < r <= MAX_RANGE else 0.0
    return out


class FakeRacecar:
    """Stands in for racecar_core: a fixed lidar scan and a drive recorder."""

    def __init__(self, scan: Optional[np.ndarray] = None, enc: float = 0.5) -> None:
        self.scan = scan
        self.enc = enc
        self.drives: List[Tuple[float, float]] = []
        self.max_speeds: List[float] = []
        self.lidar = self
        self.drive = self
        self.physics = self

    def get_samples(self) -> np.ndarray:
        return np.empty(0, np.float32) if self.scan is None else self.scan.copy()

    def set_speed_angle(self, speed: float, angle: float) -> None:
        assert -1.0 <= speed <= 1.0 and -1.0 <= angle <= 1.0
        self.drives.append((speed, angle))

    def set_max_speed(self, max_speed: float = 0.25) -> None:
        assert 0.0 <= max_speed <= 1.0
        self.max_speeds.append(max_speed)

    def get_encoder_speed(self) -> float:
        return self.enc


@pytest.fixture(autouse=True)
def params() -> Iterator[Dict[str, Any]]:
    """Fresh parameters per test; restore the file if a test saved over it."""
    global SHIPPED
    if SHIPPED is None:
        SHIPPED = (BASE / "wallfollow.yaml").read_text()
    wf._params.clear()
    wf._marks.clear()
    wf._rows.clear()
    wf._hist.clear()
    wf.load_params()
    yield wf._params
    if (BASE / "wallfollow.yaml").read_text() != SHIPPED:
        (BASE / "wallfollow.yaml").write_text(SHIPPED)


@pytest.fixture
def node() -> Any:
    return wf.WallFollower()


def feed(node: Any, scan: np.ndarray, enc: float = 0.5) -> FakeRacecar:
    """Push a scan through tick(), ignoring the rate gate."""
    node._last_proc = 0.0
    rc = FakeRacecar(scan, enc)
    node.tick(rc)
    return rc


def yaml_value(key: str) -> Any:
    return yaml.safe_load(SHIPPED)[key]


# ---- parameters ----

def test_speed_is_capped() -> None:
    assert wf.set_max_speed(5.0) == wf.MAX_SPEED
    assert wf.set_max_speed(-1.0) == 0.0


def test_shipped_yaml_cannot_drive() -> None:
    assert yaml_value("speed") == 0.0
    assert yaml_value("mode") in (0, 1)
    assert yaml_value("max_speed") == 0.25


def test_mode_accepts_the_spelled_out_form(params: Dict[str, Any]) -> None:
    params["mode"] = "dynamic"
    wf._clean(params)
    assert params["mode"] == 1
    params["mode"] = "static"
    wf._clean(params)
    assert params["mode"] == 0


def test_save_keeps_comments_and_mode(params: Dict[str, Any]) -> None:
    params.update(kp=0.8, look_angle=60.0, mode=0)
    wf.save_params()
    text = (BASE / "wallfollow.yaml").read_text()
    assert "Raise this if the car wobbles." in text
    assert f"mode: {yaml_value('mode')}" in text   # file-only, never rewritten
    assert "kp: 0.8 " in text
    wf._params.clear()
    wf.load_params()
    assert wf._params["kp"] == 0.8
    assert wf._params["look_angle"] == 60.0


def test_update_clamps(params: Dict[str, Any]) -> None:
    wf.update_params({"speed": 4.0, "max_speed": 3.0, "width": 999, "window": 0,
                      "lookahead": -1, "kp": -2})
    assert params["speed"] == wf.MAX_SPEED
    assert params["max_speed"] == 1.0
    assert params["width"] == 180.0
    assert params["window"] == 1.0
    assert params["lookahead"] == 0.1
    assert params["kp"] == 0.0


def test_update_ignores_file_only_keys(params: Dict[str, Any]) -> None:
    mode, max_mps = params["mode"], params["max_mps"]
    assert wf.update_params({"mode": 1 - mode, "max_mps": 99}) == []
    assert (params["mode"], params["max_mps"]) == (mode, max_mps)


def test_update_marks_only_real_changes() -> None:
    assert wf.update_params({"speed": 0.0}) == []
    assert wf.update_params({"speed": 0.2}) == ["speed 0.2"]
    assert wf._marks[-1][1] == "speed 0.2"


# ---- scan geometry ----

def test_angles_follow_the_student_convention() -> None:
    assert wf.index_at(N, 0) == 0
    assert wf.index_at(N, 90) == 270     # right is clockwise
    assert wf.index_at(N, -90) == 810
    assert wf.student_deg(N, 270) == pytest.approx(90)
    assert wf.student_deg(N, 810) == pytest.approx(-90)
    assert wf.student_deg(N, 540) == pytest.approx(-180)


def test_synthetic_corridor_puts_the_right_wall_on_the_right() -> None:
    scan = corridor(left=2.0, right=0.5) / 100.0
    assert scan[wf.index_at(N, 90)] == pytest.approx(0.5, abs=0.01)
    assert scan[wf.index_at(N, -90)] == pytest.approx(2.0, abs=0.01)


def test_mean_at_skips_no_return_and_caps_at_lookahead() -> None:
    scan = np.full(N, 2.0, np.float32)
    scan[:5] = 0.0
    scan[-5:] = 0.0
    assert wf.mean_at(scan, 0, 10, 3.0) == pytest.approx(2.0)   # zeros dropped
    assert wf.mean_at(np.zeros(N), 0, 10, 3.0) == 3.0          # nothing = open
    assert wf.mean_at(np.full(N, 8.0), 0, 10, 3.0) == 3.0      # capped


# ---- static controller ----

def test_static_centered_has_no_error(node: Any, params: Dict[str, Any]) -> None:
    params["mode"] = 0
    feed(node, corridor(1.0, 1.0))
    assert node._error == pytest.approx(0.0, abs=0.01)
    assert node._steer_cmd == pytest.approx(0.0, abs=0.02)


def test_static_near_the_left_wall_steers_right(node: Any, params: Dict[str, Any]) -> None:
    params.update(mode=0, kd=0.0)
    feed(node, corridor(0.6, 1.4))
    assert node._error > 0.5
    assert node._steer_cmd > 0
    assert wf._state["overlay"]["left"] < wf._state["overlay"]["right"]


def test_static_near_the_right_wall_steers_left(node: Any, params: Dict[str, Any]) -> None:
    params.update(mode=0, kd=0.0)
    feed(node, corridor(1.4, 0.6))
    assert node._steer_cmd < 0


def test_static_open_side_reads_as_lookahead(node: Any, params: Dict[str, Any]) -> None:
    params.update(mode=0, lookahead=3.0)
    feed(node, corridor(1.0, None))
    assert wf._state["overlay"]["right"] == 3.0


# ---- dynamic controller ----

def test_dynamic_open_corridor_goes_straight(node: Any, params: Dict[str, Any]) -> None:
    params.update(mode=1, side_weight=0.0)
    feed(node, corridor(1.0, 1.0))
    assert abs(wf._state["overlay"]["target_deg"]) <= 10


def test_dynamic_turns_into_the_opening(node: Any, params: Dict[str, Any]) -> None:
    params.update(mode=1, kd=0.0)
    feed(node, corridor(0.5, None, front=1.0))   # wall ahead, right is open
    assert wf._state["overlay"]["target_deg"] > 30
    assert node._steer_cmd > 0
    feed(node, corridor(None, 0.5, front=1.0))   # mirror: left is open
    assert wf._state["overlay"]["target_deg"] < -30
    assert node._steer_cmd < 0


def test_dynamic_side_weight_picks_a_side(node: Any, params: Dict[str, Any]) -> None:
    params.update(mode=1, side_weight=-0.05)
    feed(node, corridor(None, None))
    assert wf._state["overlay"]["target_deg"] == -int(params["width"])
    params["side_weight"] = 0.05
    feed(node, corridor(None, None))
    assert wf._state["overlay"]["target_deg"] == int(params["width"])


# ---- steering and speed ----

def test_derivative_term_damps_a_jump(node: Any, params: Dict[str, Any]) -> None:
    params.update(mode=0, kp=1.0, kd=0.0)
    feed(node, corridor(1.0, 1.0))
    feed(node, corridor(0.8, 1.2))
    p_only = node._steer_cmd
    params["kd"] = 0.5
    feed(node, corridor(1.0, 1.0))
    feed(node, corridor(0.8, 1.2))
    assert node._steer_cmd > p_only


def test_steer_is_clamped(node: Any, params: Dict[str, Any]) -> None:
    params.update(mode=0, kp=10.0)
    feed(node, corridor(0.3, 2.5))
    assert node._steer_cmd == 1.0


def test_slider_is_the_throttle_with_speed_gains_off(node: Any, params: Dict[str, Any]) -> None:
    params.update(speed=0.4, speed_kp=0.0, speed_kd=0.0)
    feed(node, corridor(1.0, 1.0, front=0.5))
    assert node._speed_cmd == 0.4


def test_zero_speed_never_drives(node: Any, params: Dict[str, Any]) -> None:
    params.update(speed=0.0, speed_kp=0.5)
    rc = feed(node, corridor(1.0, 1.0), enc=0.0)
    assert node._speed_cmd == 0.0
    assert rc.drives[-1][0] == 0.0


def test_speed_trim_tracks_the_target(node: Any, params: Dict[str, Any]) -> None:
    params.update(mode=0, speed=0.5, speed_kp=0.1, max_speed=1.0, max_mps=4.0)
    feed(node, corridor(1.0, 1.0), enc=0.5)    # target 2 m/s, car slow
    assert node._speed_cmd > 0.5
    node._trim = 0.0
    feed(node, corridor(1.0, 1.0), enc=3.5)    # car fast
    assert node._speed_cmd < 0.5


def test_wall_ahead_lowers_the_target(node: Any, params: Dict[str, Any]) -> None:
    params.update(mode=0, speed=0.5, speed_kp=0.1, max_speed=1.0, max_mps=4.0, lookahead=3.0)
    feed(node, corridor(1.0, 1.0, front=0.6), enc=1.0)   # target 0.4 m/s
    assert node._speed_cmd < 0.5


def test_blocked_car_does_not_wind_up(node: Any, params: Dict[str, Any]) -> None:
    params.update(mode=0, speed=0.5, speed_kp=0.1, max_speed=1.0)
    for _ in range(20):
        feed(node, corridor(1.0, 1.0), enc=0.0)
    assert node._speed_cmd <= 0.5 + 0.3 + 1e-9


# ---- actuation and rate ----

def test_actuate_keeps_student_sign_and_caps_speed(node: Any) -> None:
    rc = FakeRacecar()
    node._steer_cmd, node._speed_cmd = 0.4, 99.0
    node.actuate(rc)
    assert rc.drives[-1] == (wf.MAX_SPEED, pytest.approx(0.4))  # + is right, no negation


def test_encoder_is_read_before_the_scan(node: Any, params: Dict[str, Any]) -> None:
    params.update(mode=0, speed=0.5, speed_kp=0.1, max_speed=1.0, max_mps=4.0)
    feed(node, corridor(1.0, 1.0), enc=3.5)   # first frame already sees 3.5 m/s
    assert node._speed_cmd < 0.5
    assert wf._state["enc_speed"] == 3.5


def test_max_speed_is_sent_only_when_it_changes(node: Any) -> None:
    rc = FakeRacecar()
    node.actuate(rc)
    node.actuate(rc)
    assert rc.max_speeds == [0.25]
    wf.update_params({"max_speed": 0.6})
    node.actuate(rc)
    node.actuate(rc)
    assert rc.max_speeds == [0.25, 0.6]


def test_tick_drives_every_call_but_reads_the_lidar_at_the_scan_rate(node: Any) -> None:
    rc = FakeRacecar(corridor(1.0, 1.0))
    for _ in range(5):
        node.tick(rc)
    assert len(rc.drives) == 5
    assert len(wf._rows) == 1


def test_tick_survives_an_empty_scan(node: Any) -> None:
    rc = FakeRacecar(None)
    node.tick(rc)
    assert rc.drives == [(0.0, 0.0)]
    assert not wf._rows


def test_state_carries_the_scan_in_meters(node: Any) -> None:
    feed(node, corridor(1.0, 1.0))
    s = wf._state
    assert s["points"] == N
    assert 0 < len(s["scan"]) <= N // wf.SCAN_STRIDE
    deg, r = min(s["scan"], key=lambda p: abs(p[0] - 90))
    assert r == pytest.approx(1.0, abs=0.02)
    row = wf._rows[-1].strip().split(",")
    assert len(row) == 5


def test_thin_keeps_the_newest_row_and_the_chart_span() -> None:
    rows = [[i * 0.1, 0.0] for i in range(600)]   # 60 s
    sent = wf._thin(rows)
    assert sent[-1] is rows[-1]
    assert sent[0][0] >= rows[-1][0] - wf.CHART_SECONDS - 1e-9
    many = [[i * 0.01, 0.0] for i in range(2000)]
    assert len(wf._thin(many)) <= wf.HIST_POINTS + 1


# ---- http ----

@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.setattr(wf, "LOG_DIR", tmp_path / "logs")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), wf.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _post(url: str, body: Optional[Dict[str, Any]] = None) -> bytes:
    data = json.dumps(body).encode() if body is not None else b""
    with urllib.request.urlopen(urllib.request.Request(url, data=data, method="POST")) as r:
        return r.read()


def test_http_serves_the_page_and_state(server: str) -> None:
    with urllib.request.urlopen(server + "/") as r:
        assert b"<title>Wall follow - RACECAR Neo</title>" in r.read()
    with urllib.request.urlopen(server + "/state") as r:
        d = json.loads(r.read())
    assert d["params"]["speed"] == 0.0


def test_http_params_and_log_round_trip(server: str, node: Any) -> None:
    feed(node, corridor(1.0, 1.0))
    _post(server + "/params", {"speed": 0.3})
    assert wf._params["speed"] == 0.3
    feed(node, corridor(1.0, 1.0))
    name = _post(server + "/logs/save").decode()
    assert name == "Log1.csv"
    with urllib.request.urlopen(server + "/logs/" + name) as r:
        lines = r.read().decode().splitlines()
    assert lines[0] == "t,error,steer,speed_cmd,speed_ms"
    assert any(l.startswith("# mark") and "speed 0.3" in l for l in lines)
    with urllib.request.urlopen(server + "/logs") as r:
        assert json.loads(r.read()) == ["Log1.csv"]


def test_http_serves_every_shared_asset_the_page_links(server: str) -> None:
    page = (BASE / "wallfollow.html").read_text()
    refs = set(re.findall(r'(/ui/[^"?#]+)', page))
    assert {"/ui/rcn.css", "/ui/charts.js", "/ui/icons.svg", "/ui/brand/mark.png"} <= refs
    for ref in refs:
        with urllib.request.urlopen(server + ref + "?v=13") as r:
            assert r.status == 200 and r.read()
            assert r.headers["Cache-Control"] == "max-age=300"


def test_page_loads_no_car_only_scripts() -> None:
    page = (BASE / "wallfollow.html").read_text()
    for name in ("battery.js", "lost.js", "override.js"):
        assert f"/ui/{name}" not in page
