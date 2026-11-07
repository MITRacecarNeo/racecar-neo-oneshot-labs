"""Pipeline and controller tests for linefollow.py.

Runs without RacecarSim or a camera: frames are synthetic BGR images carrying
a painted stripe, and FakeRacecar records what reaches rc.drive.

    pytest -q labs/linefollow_dashboard/tests
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import pytest
import yaml

BASE = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("linefollow", BASE / "linefollow.py")
lf = importlib.util.module_from_spec(_spec)
sys.modules["linefollow"] = lf
_spec.loader.exec_module(lf)

SHIPPED: Optional[str] = None  # yaml as committed, restored after any test that writes it

W, H = 640, 480
BLUE = (255, 80, 0)      # BGR; hue about 105 in OpenCV's 0..179
BLUE_HSV = dict(h_low=95, h_high=125, s_low=100, s_high=255, v_low=100, v_high=255)
RED = (0, 0, 255)        # hue 0, on the wrap

Stripe = Tuple[float, int, Tuple[int, int, int], float, float]


class FakeRacecar:
    """Stands in for racecar_core: a fixed camera frame and a drive recorder."""

    def __init__(self, image: Optional[np.ndarray] = None) -> None:
        self.image = image
        self.drives: List[Tuple[float, float]] = []
        self.max_speeds: List[float] = []
        self.camera = self
        self.drive = self
        self.physics = self

    def get_color_image(self) -> Optional[np.ndarray]:
        return None if self.image is None else self.image.copy()

    def set_speed_angle(self, speed: float, angle: float) -> None:
        assert -1.0 <= speed <= 1.0 and -1.0 <= angle <= 1.0
        self.drives.append((speed, angle))

    def set_max_speed(self, max_speed: float = 0.25) -> None:
        assert 0.0 <= max_speed <= 1.0
        self.max_speeds.append(max_speed)

    def get_encoder_speed(self) -> float:
        return 0.5


@pytest.fixture(autouse=True)
def params() -> Iterator[Dict[str, Any]]:
    """Fresh parameters per test; restore the file if a test saved over it."""
    global SHIPPED
    if SHIPPED is None:
        SHIPPED = (BASE / "linefollow.yaml").read_text()
    lf._params.clear()
    lf._marks.clear()
    lf._rows.clear()
    lf._hist.clear()
    lf.load_params()
    lf._params.update(BLUE_HSV)
    yield lf._params
    if (BASE / "linefollow.yaml").read_text() != SHIPPED:
        (BASE / "linefollow.yaml").write_text(SHIPPED)


@pytest.fixture
def node() -> Any:
    return lf.LineFollower()


def frame(stripes: Tuple[Stripe, ...] = ()) -> np.ndarray:
    """Grey floor with vertical stripes. stripes: (x_frac, width_px, bgr, y0_frac, y1_frac)."""
    img = np.full((H, W, 3), 110, np.uint8)
    for x_frac, width, color, y0, y1 in stripes:
        x = int(x_frac * W)
        cv2.rectangle(img, (x - width // 2, int(y0 * H)), (x + width // 2, int(y1 * H)), color, -1)
    return img


def feed(node: Any, img: np.ndarray) -> None:
    """Push a frame through tick(), ignoring the rate gate."""
    node._last_proc = 0.0
    node.tick(FakeRacecar(img))


def yaml_value(key: str) -> Any:
    return yaml.safe_load(SHIPPED)[key]


# ---- parameters ----

def test_speed_is_capped() -> None:
    assert lf.set_max_speed(5.0) == lf.MAX_SPEED
    assert lf.set_max_speed(-1.0) == 0.0


def test_shipped_yaml_cannot_drive() -> None:
    assert yaml_value("speed") == 0.0


def test_save_keeps_comments_and_types(params: Dict[str, Any]) -> None:
    params.update(h_low=12, speed=0.35, crop_top=0.5, min_area=40)
    lf.save_params()
    text = (BASE / "linefollow.yaml").read_text()
    assert "Raise this if the car weaves." in text  # comments survive
    assert "h_low: 12\n" in text                     # ints stay ints
    lf._params.clear()
    lf.load_params()
    assert lf._params["h_low"] == 12
    assert lf._params["speed"] == 0.35
    assert lf._params["crop_top"] == 0.5
    assert lf._params["min_area"] == 40


def test_update_clamps_hsv_and_speed(params: Dict[str, Any]) -> None:
    lf.update_params({"h_high": 500, "v_low": -3, "speed": 4.0})
    assert params["h_high"] == 179
    assert params["v_low"] == 0
    assert params["speed"] == lf.MAX_SPEED


def test_update_clamps_max_speed(params: Dict[str, Any]) -> None:
    assert yaml_value("max_speed") == 0.25
    lf.update_params({"max_speed": 3.0})
    assert params["max_speed"] == 1.0
    lf.update_params({"max_speed": -0.5})
    assert params["max_speed"] == 0.0


def test_update_keeps_the_band_ordered(params: Dict[str, Any]) -> None:
    lf.update_params({"crop_top": 0.9, "crop_bottom": 0.4})
    assert params["crop_top"] < params["crop_bottom"]
    assert params["crop_bottom"] - params["crop_top"] >= lf.MIN_BAND - 1e-9
    assert params["crop_bottom"] <= 1.0


def test_update_marks_only_real_changes() -> None:
    assert lf.update_params({"speed": 0.0}) == []
    assert lf.update_params({"speed": 0.2}) == ["speed 0.2"]
    assert lf._marks[-1][1] == "speed 0.2"


# ---- threshold and contour ----

def test_threshold_wraps_hue_for_red(params: Dict[str, Any]) -> None:
    hsv = cv2.cvtColor(np.full((4, 4, 3), RED, np.uint8), cv2.COLOR_BGR2HSV)
    params.update(h_low=170, h_high=10, s_low=100, s_high=255, v_low=100, v_high=255)
    assert lf.threshold(hsv, params).all()
    params.update(h_low=20, h_high=160)
    assert not lf.threshold(hsv, params).any()


def test_find_line_takes_the_largest_contour_in_the_band() -> None:
    mask = np.zeros((240, 320), np.uint8)
    cv2.rectangle(mask, (40, 150), (60, 230), 255, -1)    # small, in band
    cv2.rectangle(mask, (200, 150), (260, 230), 255, -1)  # large, in band
    cv2.rectangle(mask, (150, 0), (170, 100), 255, -1)    # above the band
    contour, (cx, cy), area = lf.find_line(mask, 144, 240, 30)
    assert 225 <= cx <= 235
    assert 185 <= cy <= 195          # in full-mask rows, not band rows
    assert contour[:, 0, 1].min() >= 144


def test_find_line_ignores_specks() -> None:
    mask = np.zeros((240, 320), np.uint8)
    cv2.rectangle(mask, (100, 200), (103, 203), 255, -1)
    assert lf.find_line(mask, 144, 240, 30) is None


# ---- controller ----

def test_line_right_of_center_steers_right(node: Any, params: Dict[str, Any]) -> None:
    params["angle_kp"] = 1.0
    feed(node, frame(((0.75, 40, BLUE, 0.6, 1.0),)))
    assert node._found
    assert node._error == pytest.approx(0.5, abs=0.05)   # student convention: + is right
    assert node._steer_cmd == pytest.approx(0.5, abs=0.05)


def test_line_left_of_center_steers_left(node: Any) -> None:
    feed(node, frame(((0.25, 40, BLUE, 0.6, 1.0),)))
    assert node._error < -0.4
    assert node._steer_cmd < 0


def test_stripe_above_the_band_is_not_the_line(node: Any, params: Dict[str, Any]) -> None:
    params.update(crop_top=0.6, crop_bottom=1.0)
    feed(node, frame(((0.5, 40, BLUE, 0.0, 0.5),)))
    assert not node._found
    params.update(crop_top=0.1, crop_bottom=0.5)
    feed(node, frame(((0.5, 40, BLUE, 0.0, 0.5),)))
    assert node._found


def test_lost_line_holds_the_last_command(node: Any, params: Dict[str, Any]) -> None:
    params["speed"] = 0.3
    feed(node, frame(((0.75, 40, BLUE, 0.6, 1.0),)))
    steer, speed = node._steer_cmd, node._speed_cmd
    assert speed == 0.3
    feed(node, frame())
    assert not node._found
    assert (node._steer_cmd, node._speed_cmd) == (steer, speed)
    assert lf._state["lost_for"] is not None


def test_derivative_term_damps_a_jump(node: Any, params: Dict[str, Any]) -> None:
    params.update(angle_kp=1.0, angle_kd=0.0)
    feed(node, frame(((0.5, 40, BLUE, 0.6, 1.0),)))
    feed(node, frame(((0.75, 40, BLUE, 0.6, 1.0),)))
    p_only = node._steer_cmd
    params["angle_kd"] = 0.5
    feed(node, frame(((0.5, 40, BLUE, 0.6, 1.0),)))
    feed(node, frame(((0.75, 40, BLUE, 0.6, 1.0),)))
    assert node._steer_cmd > p_only  # the jump adds to the P command


def test_speed_kp_slows_off_center(node: Any, params: Dict[str, Any]) -> None:
    params.update(speed=1.0, speed_kp=0.0)
    feed(node, frame(((0.75, 40, BLUE, 0.6, 1.0),)))
    assert node._speed_cmd == 1.0
    params["speed_kp"] = 1.0
    feed(node, frame(((0.75, 40, BLUE, 0.6, 1.0),)))
    assert node._speed_cmd == pytest.approx(0.5, abs=0.05)
    feed(node, frame(((0.5, 40, BLUE, 0.6, 1.0),)))
    assert node._speed_cmd == pytest.approx(1.0, abs=0.05)


def test_steer_is_clamped(node: Any, params: Dict[str, Any]) -> None:
    params["angle_kp"] = 4.0
    feed(node, frame(((0.9, 40, BLUE, 0.6, 1.0),)))
    assert node._steer_cmd == 1.0


def test_previews_and_state_are_produced(node: Any) -> None:
    feed(node, frame(((0.5, 40, BLUE, 0.6, 1.0),)))
    assert lf._preview_color[:2] == b"\xff\xd8"
    assert lf._preview_mask[:2] == b"\xff\xd8"
    assert lf._state["res"] == [W, H]
    assert lf._state["proc_res"] == [320, 240]
    assert lf._state["cx"] == pytest.approx(160, abs=3)
    assert lf._hist[-1][2] == 0
    row = lf._rows[-1].strip().split(",")
    assert row[5] == "0" and row[6] == str(lf._state["cx"])


def test_lost_rows_are_flagged(node: Any) -> None:
    feed(node, frame())
    assert lf._hist[-1][2] == 1
    assert lf._rows[-1].strip().split(",")[5:7] == ["1", ""]


# ---- actuation ----

def test_actuate_keeps_student_sign_and_caps_speed(node: Any) -> None:
    rc = FakeRacecar()
    node._steer_cmd, node._speed_cmd = 0.4, 99.0
    node.actuate(rc)
    assert rc.drives[-1] == (lf.MAX_SPEED, pytest.approx(0.4))  # + is right, no negation
    assert lf._state["enc_speed"] == 0.5


def test_max_speed_is_sent_only_when_it_changes(node: Any) -> None:
    rc = FakeRacecar()
    node.actuate(rc)
    node.actuate(rc)
    assert rc.max_speeds == [0.25]
    lf.update_params({"max_speed": 0.6})
    node.actuate(rc)
    node.actuate(rc)
    assert rc.max_speeds == [0.25, 0.6]


def test_tick_drives_every_call_but_reads_the_camera_at_the_process_rate(node: Any) -> None:
    rc = FakeRacecar(frame(((0.5, 40, BLUE, 0.6, 1.0),)))
    for _ in range(5):
        node.tick(rc)
    assert len(rc.drives) == 5
    assert len(lf._rows) == 1


def test_tick_survives_a_missing_frame(node: Any) -> None:
    rc = FakeRacecar(None)
    node.tick(rc)
    assert rc.drives == [(0.0, 0.0)]
    assert not lf._rows


def test_thin_keeps_the_newest_row() -> None:
    rows = [[i, 0.0, 0] for i in range(1000)]
    sent = lf._thin(rows)
    assert len(sent) < len(rows)
    assert sent[-1] is rows[-1]
    assert lf._thin(rows[:50]) == rows[:50]  # short histories go through whole
