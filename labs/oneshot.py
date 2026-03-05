"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-oneshot-labs

File Name: oneshot.py

Title: RACECAR Neo OneShot - Unified Autonomy Lab

Purpose: Combines Line Following, Wall Following, and Car Tracking into a single
immersive application with a modern UI. Users can toggle between autonomy modes,
tune parameters in real-time, and see a quantitative performance score.

Controls:
    Right Trigger   = Drive (dead man's switch)
    A / B / Y       = Switch mode: Line Follow / Wall Follow / Car Track
    GUI sliders     = Tune parameters in real-time

Notes:
    - If tkinter is available, a full dark-themed GUI is launched.
    - Otherwise, an OpenCV-based dashboard + trackbar tuner is used.
"""

########################################################################################
# Imports
########################################################################################

import sys
import os
import socket as _socket
import json as _json
import yaml as _yaml
import cv2 as cv
import numpy as np
import math
import time
import threading

# ---------------------------------------------------------------------------
# Display-availability detection (MUST run before create_racecar so we can
# inject the headless flag when there is no X server).
# ---------------------------------------------------------------------------

def _x11_display_available() -> bool:
    """Return True only if an X11 display server is reachable (safe, no crash)."""
    disp = os.environ.get("DISPLAY", "")
    if not disp or ":" not in disp:
        return False
    try:
        host, rest = disp.rsplit(":", 1)
        display_num = int(rest.split(".")[0])
        port = 6000 + display_num
        host = host if host and host != "localhost" else "127.0.0.1"
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        s.settimeout(0.5)
        s.connect((host, port))
        s.close()
        return True
    except Exception:
        return False

HAS_DISPLAY = _x11_display_available()

if not HAS_DISPLAY and "-h" not in sys.argv:
    sys.argv.append("-h")          # tell racecar_core to create headless Display
    print(">> No display server detected — running in headless mode (terminal + controller)")

# Optional GUI imports — tkinter may not be available in all environments
HAS_TKINTER = False
if HAS_DISPLAY:
    try:
        import tkinter as tk
        from tkinter import ttk
        from tkinter import font as tkfont
        from PIL import Image, ImageTk, ImageDraw, ImageFont
        HAS_TKINTER = True
    except ImportError:
        print(">> tkinter not available — using OpenCV dashboard instead")

# RACECAR library
sys.path.insert(0, "../../../library")
import racecar_core
import racecar_utils as rc_utils

# PyCoral for car tracking (optional - ML object detection)
try:
    from pycoral.adapters.common import input_size
    from pycoral.adapters.detect import get_objects
    from pycoral.utils.dataset import read_label_file
    from pycoral.utils.edgetpu import make_interpreter, run_inference
    PYCORAL_AVAILABLE = True
except ImportError:
    PYCORAL_AVAILABLE = False

########################################################################################
# Create RACECAR
########################################################################################

rc = racecar_core.create_racecar()

########################################################################################
# Neobotics Color Palette
########################################################################################

class Colors:
    """Dark theme color palette inspired by neobotics branding."""
    # Backgrounds
    BG_DARK      = "#0A0A0F"
    BG_PANEL     = "#111118"
    BG_SURFACE   = "#191922"
    BG_HOVER     = "#252535"
    BG_ACTIVE    = "#303045"

    # Neobotics brand colors
    NEO_RED      = "#FF2D20"
    NEO_ORANGE   = "#FF6B35"
    NEO_AMBER    = "#FFA726"

    # Functional colors
    CYAN         = "#00D4FF"
    GREEN        = "#00E676"
    YELLOW       = "#FFD740"
    MAGENTA      = "#E040FB"

    # Neutrals
    WHITE        = "#EEEEF2"
    LIGHT_GRAY   = "#9999AA"
    MID_GRAY     = "#4A4A5A"
    DARK_GRAY    = "#2A2A3A"

    # Score gradient
    SCORE_HIGH   = "#00E676"
    SCORE_MID    = "#FFD740"
    SCORE_LOW    = "#FF2D20"


########################################################################################
# Global Application State
########################################################################################

class AppState:
    """
    Central state shared between the racecar update loop and the GUI.
    """

    # Current autonomy mode: "line_follow" | "wall_follow" | "car_track"
    mode = "line_follow"

    # Driving output
    speed = 0.0
    angle = 0.0

    # Performance
    score = 0.0
    error_history: list = []
    HIST_LEN = 300

    # -- Line Following -------------------------------------------------------
    lf_speed          = 50       # percent
    lf_angle_sens     = 50       # percent
    # Line-follow sub-mode: "standard" (hue+SV picker) | "advanced" (full HSV ranges)
    lf_sub_mode       = "standard"
    # Color priority: ordered list of color keys — tries top-to-bottom,
    # follows the first color that has a valid contour.
    lf_color_priority = ["red", "blue", "green", "orange", "yellow", "purple"]
    # Per-color enable/disable: disabled colors are skipped entirely.
    lf_color_enabled  = None    # populated in start(): {color_key: True/False}
    lf_active_color   = None    # which color was matched this frame
    contour_center    = None
    contour_area      = 0

    # Per-color tunable HSV ranges (mutable copy from LINE_COLORS defaults).
    # Key → {"H_low", "H_high", "S_low", "S_high", "V_low", "V_high"}
    lf_color_hsv      = None    # populated in start() from LINE_COLORS defaults

    # Standard-mode picker values per color:
    #   hue      = center hue (0-179)
    #   sv_pos   = combined S/V slider position (0-100)
    #              0 → (S=0,V=255) desaturated  |  50 → (S=255,V=255) pure  |  100 → (S=255,V=0) dark
    lf_basic_hue      = None    # populated in start(): {color_key: int}
    lf_basic_sv       = None    # populated in start(): {color_key: int 0-100}

    # -- Wall Following -------------------------------------------------------
    wf_speed         = 80       # percent  (old lab default: 80)
    wf_kp            = 50       # percent sensitivity (old lab default: 50)
    wf_scan_dir      = 90       # degrees from forward: 90=side, 30=front-facing (range 0-135)
    wf_window        = 30       # half-width of scan window in degrees (range 1-60)
    wf_use_avg       = False    # True = average distance, False = closest point
    wf_left_dist     = 0.0
    wf_right_dist    = 0.0

    # -- Car Tracking ---------------------------------------------------------
    ct_speed         = 70       # percent
    ct_kp            = 8.0
    ct_ki            = 0.0
    ct_kd            = 10.0
    ct_prev_error    = 0
    ct_integral      = 0
    ct_last_time     = 0
    ct_score_thresh  = 0.5

    # ML model handles
    interpreter      = None
    labels           = None
    inference_size   = None

    # Image buffers (written by update thread, read by GUI thread)
    raw_image        = None
    processed_image  = None

    # Pre-encoded JPEG bytes for web dashboard (written in main update(),
    # read by MJPEG stream threads — GIL makes reference swap atomic)
    _jpeg_raw        = b""
    _jpeg_proc       = b""

    # Cached telemetry (written by update(), read by web dashboard — avoids
    # cross-thread calls that could corrupt the racecar protocol)
    imu_accel        = (0.0, 0.0, 0.0)
    imu_gyro         = (0.0, 0.0, 0.0)
    ctrl_lt          = 0.0
    ctrl_rt          = 0.0
    ctrl_lj          = (0.0, 0.0)
    ctrl_rj          = (0.0, 0.0)
    _physics_ok      = None   # None = untested, True = works, False = disabled


# Contour detection constants
CROP_FLOOR = ((300, 0), (rc.camera.get_height(), rc.camera.get_width()))
MIN_CONTOUR_AREA = 30

state = AppState()

MODE_NAMES = {
    "line_follow": "LINE FOLLOWING",
    "wall_follow": "WALL FOLLOWING",
    "car_track":   "CAR TRACKING",
}

# ---- Line-following color definitions (HSV ranges) ----
# The car tries each color in state.lf_color_priority order, top → bottom.
# First color that yields a valid contour wins.
LINE_COLORS = {
    "red":    {"label": "Red",    "hsv_lo": (0,   120, 80),  "hsv_hi": (15,  255, 255), "dot": "#FF4444"},
    "blue":   {"label": "Blue",   "hsv_lo": (100, 100, 60),  "hsv_hi": (130, 255, 255), "dot": "#4488FF"},
    "green":  {"label": "Green",  "hsv_lo": (40,  50,  50),  "hsv_hi": (80,  255, 255), "dot": "#44DD66"},
    "orange": {"label": "Orange", "hsv_lo": (10,  150, 100), "hsv_hi": (25,  255, 255), "dot": "#FF8833"},
    "yellow": {"label": "Yellow", "hsv_lo": (20,  100, 100), "hsv_hi": (35,  255, 255), "dot": "#DDCC22"},
    "purple": {"label": "Purple", "hsv_lo": (130, 50,  50),  "hsv_hi": (165, 255, 255), "dot": "#BB55FF"},
}


_COLOR_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oneshot_colors.yaml")


def _sv_pos_to_sv(pos):
    """Convert a 0-100 SV slider position to (S, V) tuple.
    0   → (S=0,  V=255)  desaturated / pale
    50  → (S=255,V=255)  pure color
    100 → (S=255,V=0)    dark
    """
    pos = max(0, min(100, pos))
    if pos <= 50:
        t = pos / 50.0                     # 0..1
        s = int(round(t * 255))
        v = 255
    else:
        t = (pos - 50) / 50.0              # 0..1
        s = 255
        v = int(round(255 * (1.0 - t)))
    return (s, v)


def _apply_basic_to_hsv(color_key):
    """Translate Standard-mode hue + SV position into the full HSV range
    stored in state.lf_color_hsv so the same contour detection code works."""
    if not (state.lf_basic_hue and state.lf_basic_sv and state.lf_color_hsv):
        return
    hue = state.lf_basic_hue.get(color_key, 90)
    sv_pos = state.lf_basic_sv.get(color_key, 50)
    s, v = _sv_pos_to_sv(sv_pos)

    # Build a narrow HSV window around the chosen hue/SV
    hue_margin = 12  # ±12 on each side of the center hue
    s_margin   = 60
    v_margin   = 60

    h_lo = max(0,   hue - hue_margin)
    h_hi = min(179, hue + hue_margin)
    s_lo = max(0,   s - s_margin)
    s_hi = min(255, s + s_margin)
    v_lo = max(0,   v - v_margin)
    v_hi = min(255, v + v_margin)

    state.lf_color_hsv[color_key] = {
        "H_low": h_lo, "H_high": h_hi,
        "S_low": s_lo, "S_high": s_hi,
        "V_low": v_lo, "V_high": v_hi,
    }


def _save_color_config():
    """Save all color HSV ranges, priority, enabled states, and basic-mode
    picker values to a ROS2-compatible YAML parameter file next to oneshot.py.

    Structure:
        /**:
          ros__parameters:
            lf_sub_mode: ...
            color_priority: [...]
            colors:
              <key>:
                enabled: true/false
                hsv: {H_low, H_high, S_low, S_high, V_low, V_high}
                basic_hue: <int>
                basic_sv:  <int>
    """
    colors_block = {}
    color_hsv     = state.lf_color_hsv     or {}
    color_enabled = state.lf_color_enabled or {}
    basic_hue     = state.lf_basic_hue     or {}
    basic_sv      = state.lf_basic_sv      or {}

    for key in color_hsv:
        colors_block[key] = {
            "enabled":   color_enabled.get(key, True),
            "hsv":       color_hsv[key],
            "basic_hue": basic_hue.get(key, 90),
            "basic_sv":  basic_sv.get(key, 50),
        }

    data = {
        "/**": {
            "ros__parameters": {
                "lf_sub_mode":    state.lf_sub_mode,
                "color_priority": list(state.lf_color_priority),
                "colors":         colors_block,
            }
        }
    }
    try:
        with open(_COLOR_CONFIG_PATH, "w") as f:
            _yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        print(f">> Color config saved to {_COLOR_CONFIG_PATH}")
    except Exception as e:
        print(f">> Error saving color config: {e}")


def _load_color_config():
    """Load color config from a ROS2-compatible YAML parameter file.
    Returns True if loaded successfully."""
    if not os.path.exists(_COLOR_CONFIG_PATH):
        return False
    try:
        with open(_COLOR_CONFIG_PATH, "r") as f:
            raw = _yaml.safe_load(f)
        if not raw:
            return False

        # Navigate into the ROS2 parameter structure:  /**  →  ros__parameters
        params = raw
        if "/**" in params:
            params = params["/**"]
        if "ros__parameters" in params:
            params = params["ros__parameters"]

        if "lf_sub_mode" in params:
            state.lf_sub_mode = params["lf_sub_mode"]
        if "color_priority" in params:
            state.lf_color_priority = params["color_priority"]

        colors = params.get("colors", {})
        for key, cdata in colors.items():
            if state.lf_color_enabled is not None and key in state.lf_color_enabled:
                state.lf_color_enabled[key] = cdata.get("enabled", True)
            if state.lf_color_hsv is not None and key in state.lf_color_hsv:
                hsv = cdata.get("hsv")
                if hsv:
                    state.lf_color_hsv[key] = hsv
            if state.lf_basic_hue is not None and key in state.lf_basic_hue:
                state.lf_basic_hue[key] = cdata.get("basic_hue", 90)
            if state.lf_basic_sv is not None and key in state.lf_basic_sv:
                state.lf_basic_sv[key] = cdata.get("basic_sv", 50)

        print(f">> Color config loaded from {_COLOR_CONFIG_PATH}")
        return True
    except Exception as e:
        print(f">> Error loading color config: {e}")
        return False


def _move_color_priority(color_key, direction):
    """Move a color up (-1) or down (+1) in the priority list."""
    lst = state.lf_color_priority
    if color_key not in lst:
        return
    idx = lst.index(color_key)
    new_idx = idx + direction
    if 0 <= new_idx < len(lst):
        lst[idx], lst[new_idx] = lst[new_idx], lst[idx]
        print(f">> Color priority: {' > '.join(c.upper() for c in lst)}")


########################################################################################
# Helper Functions
########################################################################################

def generate_lidar_image(samples, radius=128, max_range=500,
                         highlighted=None, left_window=None, right_window=None):
    """
    Generate a LIDAR visualization image (numpy BGR) for embedding in the GUI.
    """
    image = np.zeros((2 * radius, 2 * radius, 3), np.uint8)
    num_samples = len(samples)

    for i in range(num_samples):
        if 0 < samples[i] < max_range:
            ang = 2 * math.pi * i / num_samples
            length = radius * samples[i] / max_range
            r = int(radius - length * math.cos(ang))
            c = int(radius + length * math.sin(ang))
            if 0 <= r < 2 * radius and 0 <= c < 2 * radius:
                image[r][c] = (80, 80, 255)

    cv.circle(image, (radius, radius), 3, (0, 255, 0), -1)

    if left_window is not None:
        _draw_lidar_arc(image, radius, left_window, (255, 200, 0), max_range)
    if right_window is not None:
        _draw_lidar_arc(image, radius, right_window, (0, 200, 255), max_range)

    if highlighted:
        for (angle_deg, distance) in highlighted:
            if 0 < distance < max_range:
                ang = angle_deg * math.pi / 180
                length = radius * distance / max_range
                r = int(radius - length * math.cos(ang))
                c = int(radius + length * math.sin(ang))
                if 0 <= r < 2 * radius and 0 <= c < 2 * radius:
                    cv.circle(image, (c, r), 4, (255, 255, 0), -1)

    return image


def _draw_lidar_arc(image, radius, window, color, max_range):
    """Draw faint arc lines to show the LIDAR scan window."""
    start_deg, end_deg = window
    for deg in [start_deg, end_deg]:
        ang = deg * math.pi / 180
        x1 = int(radius + 10 * math.sin(ang))
        y1 = int(radius - 10 * math.cos(ang))
        x2 = int(radius + (radius - 5) * math.sin(ang))
        y2 = int(radius - (radius - 5) * math.cos(ang))
        cv.line(image, (x1, y1), (x2, y2), color, 1)


def calculate_score():
    """
    Return a 0-100 score.  Higher = better tuning / less error.
    Uses the trailing 60 samples (~1 second at 60 FPS).
    """
    if len(state.error_history) < 5:
        return 0.0

    recent = state.error_history[-60:]
    avg_error = float(np.mean(recent))

    if state.mode == "line_follow":
        max_err = rc.camera.get_width() / 2
    elif state.mode == "wall_follow":
        max_err = 200.0
    else:
        max_err = rc.camera.get_width() / 2

    normalised = min(avg_error / max(max_err, 1), 1.0)
    return round(max(0, (1.0 - normalised) * 100), 1)


########################################################################################
# Mode Switching (shared between tkinter and OpenCV GUI)
########################################################################################

_pending_mode_switch = None  # set by web handler, applied by main thread

def _switch_mode(mode, from_main_thread=True):
    """Switch autonomy mode and reset state."""
    global _pending_mode_switch
    if not from_main_thread:
        # Defer to the main update thread to avoid cross-thread UDP calls
        _pending_mode_switch = mode
        return
    if state.mode == mode:
        return
    state.mode = mode
    state.error_history.clear()
    state.speed = 0
    state.angle = 0
    state.contour_center = None
    state.contour_area = 0
    state.ct_prev_error = 0
    state.ct_integral = 0
    state.ct_last_time = time.time()

    name = MODE_NAMES.get(mode, "???")
    print(f">> Mode: {name}")
    try:
        rc.display.show_text(f"Mode: {name}")
    except Exception:
        pass


########################################################################################
# Autonomy Mode: LINE FOLLOWING
########################################################################################

def _get_color_hsv(color_key):
    """Return (hsv_lo_tuple, hsv_hi_tuple) for a color, using tuned values."""
    hsv = state.lf_color_hsv
    if hsv and color_key in hsv:
        h = hsv[color_key]
        return ((h["H_low"], h["S_low"], h["V_low"]),
                (h["H_high"], h["S_high"], h["V_high"]))
    # Fallback to LINE_COLORS defaults
    cdef = LINE_COLORS.get(color_key, {})
    return (cdef.get("hsv_lo", (0, 0, 0)), cdef.get("hsv_hi", (179, 255, 255)))


def update_line_follow():
    """
    Multi-color priority line following with a proportional controller.

    Iterates through state.lf_color_priority top → bottom.  The first
    color whose HSV mask yields a large-enough contour is the one the
    car follows.  Lower-priority colors are only used when the higher-
    priority ones are not visible.

    Processing view: clean HSV mask — black background with only the
    pixels matching the active color visible (exactly like hsv_tuner).
    A thin green contour outline and yellow center dot are overlaid
    for visual feedback without polluting the mask.
    """
    image = rc.camera.get_color_image()
    if image is None:
        return

    state.raw_image = image.copy()

    # Prepare a small copy for the mask visualization
    small = cv.resize(image, (320, 240))
    hsv_small = cv.cvtColor(small, cv.COLOR_BGR2HSV)

    # Crop for contour detection (full-res)
    cropped = rc_utils.crop(image, CROP_FLOOR[0], CROP_FLOOR[1])

    best_contour = None
    best_color_key = None

    # Only consider enabled colors
    enabled = state.lf_color_enabled or {}
    active_priority = [k for k in state.lf_color_priority if enabled.get(k, True)]

    for color_key in active_priority:
        hsv_lo, hsv_hi = _get_color_hsv(color_key)

        # Contour detection on the cropped (full-res) image
        if best_contour is None and cropped is not None:
            contours = rc_utils.find_contours(cropped, hsv_lo, hsv_hi)
            contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)
            if contour is not None:
                best_contour = contour
                best_color_key = color_key

    # ---- Build the processing image (mask view) ----
    # Like the old hsv_tuner: show the mask for the active color so the user
    # can see exactly what the camera is picking up and fine-tune HSV.
    if best_color_key is not None:
        hsv_lo, hsv_hi = _get_color_hsv(best_color_key)
    else:
        # No match — show the mask for the first enabled color so the
        # user can still tune it while seeing the camera feed.
        top_key = active_priority[0] if active_priority else "red"
        hsv_lo, hsv_hi = _get_color_hsv(top_key)

    lo_arr = np.array(hsv_lo, np.uint8)
    hi_arr = np.array(hsv_hi, np.uint8)
    mask = cv.inRange(hsv_small, lo_arr, hi_arr)
    # Classic technique: bitwise_and reveals only the masked color region
    proc = cv.bitwise_and(small, small, mask=mask)

    # ---- Act on the highest-priority match ----
    if best_contour is not None:
        state.contour_center = rc_utils.get_contour_center(best_contour)
        state.contour_area = rc_utils.get_contour_area(best_contour)
        state.lf_active_color = best_color_key

        setpoint = rc.camera.get_width() // 2
        error = setpoint - state.contour_center[1]
        kp = -(2 / setpoint) * (state.lf_angle_sens / 100) * 2
        state.angle = rc_utils.clamp(kp * error, -1, 1)

        state.error_history.append(abs(error))
        if len(state.error_history) > state.HIST_LEN:
            state.error_history.pop(0)

        # Draw a thin contour outline + center dot on the mask (not the full image).
        # The contour lives in the *cropped* coordinate space, but proc is a
        # resize of the *full* image → scale by (proc / full), not (proc / crop),
        # then shift Y by the crop-top offset.
        full_h, full_w = image.shape[:2]
        proc_h, proc_w = proc.shape[:2]
        sx = proc_w / max(full_w, 1)
        sy = proc_h / max(full_h, 1)
        y_offset = CROP_FLOOR[0][0] * sy   # crop starts at this row in proc
        scaled_contour = best_contour.copy().astype(np.float64)
        scaled_contour[:, :, 0] = scaled_contour[:, :, 0] * sx               # x
        scaled_contour[:, :, 1] = scaled_contour[:, :, 1] * sy + y_offset    # y
        scaled_contour = scaled_contour.astype(np.int32)
        cv.drawContours(proc, [scaled_contour], -1, (0, 255, 0), 1)
        # Center dot (contour_center is (row, col) in cropped space)
        cx = int(state.contour_center[1] * sx)
        cy = int(state.contour_center[0] * sy + y_offset)
        cv.circle(proc, (cx, cy), 4, (0, 255, 255), -1)
    else:
        state.contour_center = None
        state.contour_area = 0
        state.lf_active_color = None

    # Overlay a small label showing which color is being tracked
    shown_key = best_color_key or (state.lf_color_priority[0] if state.lf_color_priority else "none")
    label_text = f"Mask: {shown_key.upper()}"
    if best_color_key:
        label_text += "  [TRACKING]"
    cv.putText(proc, label_text, (8, 18), cv.FONT_HERSHEY_SIMPLEX, 0.45,
               (0, 255, 255), 1, cv.LINE_AA)

    state.processed_image = proc
    state.speed = state.lf_speed / 100

    if rc.controller.get_trigger(rc.controller.Trigger.RIGHT) > 0.1:
        rc.drive.set_speed_angle(state.speed, state.angle)
    else:
        rc.drive.set_speed_angle(0, 0)


########################################################################################
# Autonomy Mode: WALL FOLLOWING
########################################################################################

def update_wall_follow():
    """
    LIDAR-based wall following with tuneable scan direction, window width,
    and a toggle between closest-point and average-distance measurement.

    Parameters:
      Speed %        → driving speed (0-100 maps to 0.0-1.0)
      Sensitivity %  → P-controller gain (kp/10000 * 2)
      Scan Dir °     → where the windows point: 90 = pure sides, 0 = front (0-135)
      Window °       → half-width of the scan window (1-60)
      Use Average    → False = closest point, True = average distance

    Right window centres at `scan_dir` degrees from forward.
    Left  window centres at `360 - scan_dir` degrees from forward.
    Each spans ± window degrees.
    """
    scan = rc.lidar.get_samples()

    image = rc.camera.get_color_image()
    if image is not None:
        state.raw_image = image.copy()

    # ---- Build LIDAR scan windows ----
    d  = state.wf_scan_dir    # centre angle from forward (0-135)
    hw = state.wf_window      # half-width

    right_center = d                # e.g. 90 = directly right
    left_center  = (360 - d) % 360  # e.g. 270 = directly left

    right_window = (right_center - hw, right_center + hw)
    left_window  = (left_center - hw,  left_center + hw)

    # ---- Measure wall distances ----
    if state.wf_use_avg:
        # Average distance across the entire window (smoother, less noise)
        left_dist  = rc_utils.get_lidar_average_distance(scan, left_center,  hw * 2)
        right_dist = rc_utils.get_lidar_average_distance(scan, right_center, hw * 2)
        # For visualization we still need closest-point angles
        left_angle, _  = rc_utils.get_lidar_closest_point(scan, left_window)
        right_angle, _ = rc_utils.get_lidar_closest_point(scan, right_window)
    else:
        # Closest point in window (more reactive, classic behaviour)
        left_angle, left_dist   = rc_utils.get_lidar_closest_point(scan, left_window)
        right_angle, right_dist = rc_utils.get_lidar_closest_point(scan, right_window)

    state.wf_left_dist  = left_dist
    state.wf_right_dist = right_dist

    # P-controller: error = difference between wall distances
    error = right_dist - left_dist
    kp_now = (state.wf_kp / 10000) * 2   # matches old wall-follow_tuner
    state.angle = rc_utils.clamp(kp_now * error, -1, 1)
    state.speed = state.wf_speed / 100

    state.error_history.append(abs(error))
    if len(state.error_history) > state.HIST_LEN:
        state.error_history.pop(0)

    # Send speed and angle to the car if trigger is pressed
    if rc.controller.get_trigger(rc.controller.Trigger.RIGHT) > 0:
        rc.drive.set_speed_angle(state.speed, state.angle)
    else:
        rc.drive.set_speed_angle(0, 0)

    # Display LIDAR to screen
    rc.display.show_lidar(scan, max_range=500)

    # ---- LIDAR visualization for GUI ----
    highlighted = [(left_angle, left_dist), (right_angle, right_dist)]
    lidar_img = generate_lidar_image(
        scan, radius=128, max_range=500,
        highlighted=highlighted,
        left_window=left_window, right_window=right_window,
    )

    mode_tag = "AVG" if state.wf_use_avg else "CLOSEST"
    dir_label = "Side" if d >= 80 else ("Diag" if d >= 45 else "Fwd")
    cv.putText(lidar_img, f"L:{left_dist:.0f}cm", (10, 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    cv.putText(lidar_img, f"R:{right_dist:.0f}cm", (170, 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    cv.putText(lidar_img, f"Err:{error:.0f} [{dir_label}] {mode_tag}", (30, 245),
               cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    state.processed_image = lidar_img


########################################################################################
# Autonomy Mode: CAR TRACKING
########################################################################################

def update_car_track():
    """
    ML object detection (PyCoral) + PID controller.
    Falls back gracefully if PyCoral / model is unavailable.
    """
    image = rc.camera.get_color_image()
    if image is None:
        return

    state.raw_image = image.copy()

    if not PYCORAL_AVAILABLE or state.interpreter is None:
        proc = image.copy()
        cv.putText(proc, "ML Model Unavailable", (40, 200),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv.putText(proc, "Install PyCoral + model to enable", (20, 240),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        state.processed_image = proc
        rc.drive.stop()
        return

    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    rgb_resized = cv.resize(rgb, state.inference_size)
    run_inference(state.interpreter, rgb_resized.tobytes())
    objs = get_objects(state.interpreter, state.ct_score_thresh)[:1]

    proc = image.copy()

    if objs:
        obj = objs[0]
        h, w, _ = image.shape
        sx = w / state.inference_size[0]
        sy = h / state.inference_size[1]
        bbox = obj.bbox.scale(sx, sy)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2

        cv.rectangle(proc, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv.circle(proc, (cx, cy), 6, (0, 0, 255), -1)
        cv.putText(proc, f"{obj.score:.0%}", (x0, y0 - 8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        img_center = rc.camera.get_width() // 2
        error = cx - img_center

        now = time.time()
        dt = max(now - state.ct_last_time, 0.001)
        state.ct_last_time = now

        p = state.ct_kp * error
        state.ct_integral += error * dt
        i = state.ct_ki * state.ct_integral
        d = state.ct_kd * (error - state.ct_prev_error) / dt
        state.ct_prev_error = error

        state.angle = float(np.clip((p + i + d) / 100, -1.0, 1.0))
        state.speed = state.ct_speed / 100

        state.error_history.append(abs(error))
        if len(state.error_history) > state.HIST_LEN:
            state.error_history.pop(0)

        rc.drive.set_speed_angle(state.speed, state.angle)
    else:
        cv.putText(proc, "No target detected", (50, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
        rc.drive.stop()

    state.processed_image = proc


########################################################################################
# Dot Matrix Display
########################################################################################

def update_dot_matrix():
    """Push mode-specific info to the physical dot-matrix LED display."""
    score = calculate_score()
    state.score = score

    labels = {"line_follow": "LF", "wall_follow": "WF", "car_track": "CT"}
    tag = labels.get(state.mode, "??")

    try:
        rc.display.show_text(f"{tag} {int(score)}%")
    except Exception:
        pass


########################################################################################
# OpenCV Dashboard (Fallback when tkinter is unavailable)
########################################################################################

# BGR colors for OpenCV rendering
_BGR = {
    "bg":         (15, 10, 10),
    "panel":      (24, 17, 17),
    "surface":    (34, 25, 25),
    "neo_red":    (32, 45, 255),
    "neo_orange": (53, 107, 255),
    "neo_amber":  (38, 167, 255),
    "cyan":       (255, 212, 0),
    "green":      (118, 230, 0),
    "yellow":     (64, 215, 255),
    "white":      (242, 238, 238),
    "light_gray": (170, 153, 153),
    "mid_gray":   (90, 74, 74),
    "dark_gray":  (58, 42, 42),
}

_BGR_MODE = {
    "line_follow": _BGR["green"],
    "wall_follow": _BGR["cyan"],
    "car_track":   _BGR["neo_orange"],
}

_TUNER_WIN = "RACECAR Neo Tuner"
_DASH_WIN  = "RACECAR Neo Dashboard"
_cv_current_tuner_mode = None    # tracks which mode the trackbar window is set to
_cv_current_lf_sub    = None    # tracks standard/advanced for line_follow


def _cv_setup_tuner(mode):
    """(Re)create the OpenCV trackbar window for the given mode."""
    if not HAS_DISPLAY:
        return
    global _cv_current_tuner_mode, _cv_current_lf_sub
    lf_sub = state.lf_sub_mode if mode == "line_follow" else None
    if _cv_current_tuner_mode == mode and _cv_current_lf_sub == lf_sub:
        return
    _cv_current_tuner_mode = mode
    _cv_current_lf_sub = lf_sub

    # Destroy and recreate to clear old trackbars
    try:
        cv.destroyWindow(_TUNER_WIN)
        cv.waitKey(1)
    except Exception:
        pass

    cv.namedWindow(_TUNER_WIN, cv.WINDOW_NORMAL)
    cv.resizeWindow(_TUNER_WIN, 480, 30)

    if mode == "line_follow":
        # Speed + Steer + (basic: Hue + SV for first enabled color)
        cv.createTrackbar("Speed %",  _TUNER_WIN, state.lf_speed,    100,
                          lambda v: setattr(state, "lf_speed", v))
        cv.createTrackbar("Steer %",  _TUNER_WIN, state.lf_angle_sens, 100,
                          lambda v: setattr(state, "lf_angle_sens", v))
        # In standard mode, add Hue + SV trackbars for the first enabled color
        if state.lf_sub_mode == "standard":
            enabled = state.lf_color_enabled or {}
            first_key = next((k for k in state.lf_color_priority if enabled.get(k, True)), None)
            if first_key and state.lf_basic_hue and state.lf_basic_sv:
                hue = state.lf_basic_hue.get(first_key, 90)
                sv = state.lf_basic_sv.get(first_key, 50)
                def _set_hue(v, k=first_key):
                    if state.lf_basic_hue is not None:
                        state.lf_basic_hue[k] = v
                        _apply_basic_to_hsv(k)
                def _set_sv(v, k=first_key):
                    if state.lf_basic_sv is not None:
                        state.lf_basic_sv[k] = v
                        _apply_basic_to_hsv(k)
                cv.createTrackbar(f"Hue ({first_key})", _TUNER_WIN, hue, 179, _set_hue)
                cv.createTrackbar(f"SV  ({first_key})", _TUNER_WIN, sv,  100, _set_sv)

    elif mode == "wall_follow":
        cv.createTrackbar("Speed %",           _TUNER_WIN, state.wf_speed,       100,
                          lambda v: setattr(state, "wf_speed", max(v, 1)))
        cv.createTrackbar("Sensitivity %",     _TUNER_WIN, state.wf_kp,          100,
                          lambda v: setattr(state, "wf_kp", max(v, 1)))
        cv.createTrackbar("Scan Dir (90=Side)", _TUNER_WIN, state.wf_scan_dir,   135,
                          lambda v: setattr(state, "wf_scan_dir", v))
        cv.createTrackbar("Window +/-",        _TUNER_WIN, state.wf_window,      60,
                          lambda v: setattr(state, "wf_window", max(v, 1)))
        cv.createTrackbar("Avg(1)/Closest(0)", _TUNER_WIN, int(state.wf_use_avg), 1,
                          lambda v: setattr(state, "wf_use_avg", bool(v)))

    elif mode == "car_track":
        cv.createTrackbar("Speed %",  _TUNER_WIN, state.ct_speed,              100,
                          lambda v: setattr(state, "ct_speed", v))
        cv.createTrackbar("Kp x10",  _TUNER_WIN, int(state.ct_kp * 10),       500,
                          lambda v: setattr(state, "ct_kp", v / 10.0))
        cv.createTrackbar("Ki x10",  _TUNER_WIN, int(state.ct_ki * 10),       100,
                          lambda v: setattr(state, "ct_ki", v / 10.0))
        cv.createTrackbar("Kd x10",  _TUNER_WIN, int(state.ct_kd * 10),       500,
                          lambda v: setattr(state, "ct_kd", v / 10.0))
        cv.createTrackbar("Conf %",  _TUNER_WIN, int(state.ct_score_thresh * 100), 100,
                          lambda v: setattr(state, "ct_score_thresh", max(v, 10) / 100.0))

    # Tiny placeholder so the window is visible
    cv.imshow(_TUNER_WIN, np.zeros((1, 480, 3), np.uint8))


def _cv_compose_dashboard():
    """Compose a 4-panel dashboard as a single OpenCV image."""
    W, H   = 880, 530
    IMG_W  = 425
    IMG_H  = 290
    HDR_H  = 44
    FTR_H  = H - HDR_H - IMG_H - 16

    dash = np.full((H, W, 3), _BGR["bg"], dtype=np.uint8)
    F = cv.FONT_HERSHEY_SIMPLEX
    AA = cv.LINE_AA

    # ---- Header bar ----
    cv.rectangle(dash, (0, 0), (W, HDR_H), _BGR["panel"], -1)
    cv.putText(dash, "RACECAR Neo", (12, 30), F, 0.85, _BGR["neo_red"], 2, AA)
    cv.putText(dash, "OneShot Lab", (220, 30), F, 0.6, _BGR["neo_orange"], 1, AA)

    mode_color = _BGR_MODE.get(state.mode, _BGR["white"])
    mode_name  = MODE_NAMES.get(state.mode, "???")
    cv.putText(dash, mode_name, (420, 30), F, 0.7, mode_color, 2, AA)

    score = state.score
    sc = _BGR["green"] if score >= 70 else (_BGR["yellow"] if score >= 35 else _BGR["neo_red"])
    cv.putText(dash, f"Score: {score:.0f}/100", (710, 30), F, 0.6, sc, 2, AA)

    # ---- Image panels ----
    y0 = HDR_H + 4
    x_l, x_r = 5, W // 2 + 3

    cv.putText(dash, "Camera / LIDAR", (x_l + 2, y0 + 12), F, 0.35, _BGR["light_gray"], 1, AA)
    cv.putText(dash, "Processing",     (x_r + 2, y0 + 12), F, 0.35, _BGR["light_gray"], 1, AA)

    iy = y0 + 18
    ih = IMG_H - 18

    if state.raw_image is not None:
        try:
            raw = cv.resize(state.raw_image, (IMG_W, ih))
            dash[iy:iy + ih, x_l:x_l + IMG_W] = raw
        except Exception:
            pass
    else:
        cv.putText(dash, "No Camera Feed", (x_l + 110, iy + ih // 2), F, 0.7, _BGR["mid_gray"], 2, AA)

    if state.processed_image is not None:
        try:
            proc = state.processed_image
            # Handle both color and grayscale
            if len(proc.shape) == 2:
                proc = cv.cvtColor(proc, cv.COLOR_GRAY2BGR)
            proc = cv.resize(proc, (IMG_W, ih))
            dash[iy:iy + ih, x_r:x_r + IMG_W] = proc
        except Exception:
            pass
    else:
        cv.putText(dash, "No Processing", (x_r + 120, iy + ih // 2), F, 0.7, _BGR["mid_gray"], 2, AA)

    # ---- Footer: Telemetry + Score bar ----
    ftr_y = y0 + IMG_H + 4
    cv.rectangle(dash, (0, ftr_y), (W, H), _BGR["panel"], -1)
    cv.line(dash, (0, ftr_y), (W, ftr_y), _BGR["mid_gray"], 1)

    # Read telemetry from cached state (never call rc.physics / rc.controller
    # from here — cross-thread calls can corrupt the racecar protocol).
    a = state.imu_accel
    w = state.imu_gyro
    lt = state.ctrl_lt
    rt = state.ctrl_rt
    lj = state.ctrl_lj

    ty = ftr_y + 18
    cv.putText(dash,
        f"IMU  Accel X:{a[0]:+6.2f} Y:{a[1]:+6.2f} Z:{a[2]:+6.2f}   "
        f"Gyro X:{w[0]:+6.2f} Y:{w[1]:+6.2f} Z:{w[2]:+6.2f}",
        (10, ty), F, 0.37, _BGR["cyan"], 1, AA)

    ty += 18
    cv.putText(dash,
        f"CTRL  LT:{lt:.2f}  RT:{rt:.2f}  Stick:({lj[0]:+.2f},{lj[1]:+.2f})  "
        f"Speed:{state.speed:+.3f}  Angle:{state.angle:+.3f}",
        (10, ty), F, 0.37, _BGR["cyan"], 1, AA)

    # Mode-specific data line
    ty += 18
    if state.mode == "line_follow":
        ctr = state.contour_center
        active = state.lf_active_color or "none"
        prio = " > ".join(c.capitalize() for c in state.lf_color_priority)
        info = (f"Priority:[{prio}]  Tracking:{active.capitalize()}  "
                f"Contour:{f'({ctr[0]},{ctr[1]})' if ctr else 'None'}  "
                f"Area:{state.contour_area}")
    elif state.mode == "wall_follow":
        dirl = "Side" if state.wf_scan_dir >= 80 else ("Diag" if state.wf_scan_dir >= 45 else "Fwd")
        avg_tag = "AVG" if state.wf_use_avg else "CLOSEST"
        info = (f"L:{state.wf_left_dist:.0f}cm  R:{state.wf_right_dist:.0f}cm  "
                f"\u0394:{state.wf_right_dist - state.wf_left_dist:.0f}cm  "
                f"Dir:{state.wf_scan_dir}\u00b0[{dirl}] \u00b1{state.wf_window}\u00b0 {avg_tag}")
    else:
        info = (f"Kp:{state.ct_kp:.1f} Ki:{state.ct_ki:.1f} Kd:{state.ct_kd:.1f}  "
                f"ML:{'OK' if PYCORAL_AVAILABLE and state.interpreter else 'N/A'}")
    cv.putText(dash, info, (10, ty), F, 0.37, _BGR["yellow"], 1, AA)

    # Score bar
    ty += 22
    bar_x0, bar_x1 = 10, 660
    bar_h = 12
    cv.rectangle(dash, (bar_x0, ty), (bar_x1, ty + bar_h), _BGR["dark_gray"], -1)
    fill = int((bar_x1 - bar_x0) * score / 100)
    if fill > 0:
        cv.rectangle(dash, (bar_x0, ty), (bar_x0 + fill, ty + bar_h), sc, -1)
    cv.putText(dash, f"{score:.0f}%", (bar_x1 + 8, ty + 10), F, 0.4, sc, 1, AA)

    # Controls hint
    ty += 22
    cv.putText(dash,
        "A = Save/Line   B = Wall Follow   Y = Car Track   |   Hold RT to Drive",
        (10, ty), F, 0.37, _BGR["light_gray"], 1, AA)

    return dash


def _cv_update_display():
    """Show the OpenCV dashboard. Called each frame when tkinter is unavailable."""
    if not HAS_DISPLAY:
        return
    dashboard = _cv_compose_dashboard()
    cv.imshow(_DASH_WIN, dashboard)
    cv.waitKey(1)


########################################################################################
# Web Dashboard (headless mode — browser UI, no X server needed)
########################################################################################

from http.server import HTTPServer as _HTTPServer, BaseHTTPRequestHandler as _BaseHandler
import socketserver as _socketserver
from functools import partial as _partial


class _ThreadedHTTPServer(_socketserver.ThreadingMixIn, _HTTPServer):
    """HTTP server that handles each request in a new thread.
    Required for MJPEG streaming (long-lived connections) alongside API calls."""
    daemon_threads = True

_WEB_PORT = 8080

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RACECAR Neo · OneShot Lab</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#060609;--panel:#0D0D14;--surface:#14141E;--hover:#1E1E30;--active:#282842;
  --border:#1A1A2E;--border-light:#252540;
  --red:#FF2D20;--orange:#FF6B35;--amber:#FFA726;
  --cyan:#00D4FF;--green:#00E676;--yellow:#FFD740;--magenta:#E040FB;
  --white:#F0F0F5;--lgray:#8888A0;--mgray:#4A4A60;--dgray:#252538;
  --glow-red:0 0 20px rgba(255,45,32,.15);
  --glow-green:0 0 20px rgba(0,230,118,.15);
  --glow-cyan:0 0 20px rgba(0,212,255,.15);
  --glow-orange:0 0 20px rgba(255,107,53,.15);
  --radius:10px;--radius-lg:14px;
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--white);font-family:'Inter',system-ui,-apple-system,sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden;-webkit-font-smoothing:antialiased}

/* ---- Scrollbar ---- */
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--mgray);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:var(--lgray)}

/* ---- Top Bar ---- */
.topbar{display:flex;align-items:center;background:var(--panel);height:64px;padding:0 24px;gap:16px;border-bottom:1px solid var(--border);flex-shrink:0;position:relative;z-index:10}
.topbar::after{content:'';position:absolute;bottom:-1px;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(255,45,32,.3),rgba(255,107,53,.2),transparent)}
.logo{display:flex;align-items:baseline;gap:8px;flex-shrink:0}
.logo-main{font-size:22px;font-weight:900;background:linear-gradient(135deg,var(--red),var(--orange));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;letter-spacing:-.5px}
.logo-sub{font-size:13px;font-weight:600;color:var(--lgray);letter-spacing:.5px;text-transform:uppercase}
.mode-nav{display:flex;gap:4px;margin-left:auto;background:var(--surface);border-radius:var(--radius);padding:4px;border:1px solid var(--border)}
.mode-btn{padding:10px 22px;border:none;border-radius:8px;background:transparent;color:var(--lgray);font-family:inherit;font-size:13px;font-weight:700;cursor:pointer;transition:all .2s ease;letter-spacing:.3px;position:relative}
.mode-btn:hover{background:var(--hover);color:var(--white)}
.mode-btn.active{color:var(--white);box-shadow:var(--glow-green)}
.mode-btn.active[data-mode="line_follow"]{background:rgba(0,230,118,.12);color:var(--green);box-shadow:var(--glow-green)}
.mode-btn.active[data-mode="wall_follow"]{background:rgba(0,212,255,.12);color:var(--cyan);box-shadow:var(--glow-cyan)}
.mode-btn.active[data-mode="car_track"]{background:rgba(255,107,53,.12);color:var(--orange);box-shadow:var(--glow-orange)}
.mode-btn.active::after{content:'';position:absolute;bottom:-1px;left:20%;right:20%;height:2px;border-radius:1px}
.mode-btn.active[data-mode="line_follow"]::after{background:var(--green)}
.mode-btn.active[data-mode="wall_follow"]::after{background:var(--cyan)}
.mode-btn.active[data-mode="car_track"]::after{background:var(--orange)}

/* ---- Status Pill ---- */
.status-area{display:flex;align-items:center;gap:16px;margin-left:16px}
.conn-status{display:flex;align-items:center;gap:6px;font-size:11px;font-weight:600;color:var(--lgray);text-transform:uppercase;letter-spacing:.8px}
.conn-dot{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 6px rgba(0,230,118,.5);animation:pulse 2s ease-in-out infinite}
.conn-dot.err{background:var(--red);box-shadow:0 0 6px rgba(255,45,32,.5)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}

/* ---- Score Display ---- */
.score-box{display:flex;flex-direction:column;align-items:flex-end;gap:0;min-width:100px}
.score-label{font-size:10px;color:var(--lgray);text-transform:uppercase;letter-spacing:1.5px;font-weight:700}
.score-val{font-size:40px;font-weight:900;line-height:1;letter-spacing:-2px;transition:color .3s}
.score-bar-wrap{height:3px;background:var(--surface);flex-shrink:0;position:relative;overflow:hidden}
.score-bar{height:100%;transition:width .4s ease,background .4s ease;position:relative}
.score-bar::after{content:'';position:absolute;top:0;right:0;width:40px;height:100%;background:linear-gradient(90deg,transparent,rgba(255,255,255,.3))}

/* ---- Main Grid ---- */
.main{flex:1;display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;gap:6px;padding:6px;min-height:0}

/* ---- Panel Card ---- */
.panel{background:var(--panel);border-radius:var(--radius-lg);overflow:hidden;display:flex;flex-direction:column;min-height:0;border:1px solid var(--border);transition:border-color .2s}
.panel:hover{border-color:var(--border-light)}
.panel-hdr{display:flex;align-items:center;gap:8px;padding:12px 16px 8px;flex-shrink:0}
.panel-hdr-icon{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.panel-hdr-text{font-size:11px;font-weight:700;color:var(--lgray);text-transform:uppercase;letter-spacing:1.8px}

/* ---- Image Panels ---- */
.img-wrap{flex:1;display:flex;align-items:center;justify-content:center;overflow:hidden;background:var(--surface);margin:0 8px 8px;border-radius:var(--radius);min-height:0;position:relative}
.img-wrap img{max-width:100%;max-height:100%;object-fit:contain;display:block}
.img-placeholder{color:var(--mgray);font-size:12px;font-weight:500;letter-spacing:.5px}
.img-wrap::before{content:'';position:absolute;inset:0;border-radius:var(--radius);border:1px solid var(--border);pointer-events:none;z-index:1}

/* ---- Telemetry Panel ---- */
.telem{padding:10px 16px;font-family:'JetBrains Mono','Consolas',monospace;font-size:12px;line-height:1.7;overflow-y:auto;flex:1}
.telem .tbl{display:grid;grid-template-columns:auto 1fr;gap:2px 12px;margin:2px 0}
.telem .tbl .lbl{color:var(--lgray);white-space:nowrap}
.telem .tbl .dat{color:var(--cyan);font-weight:600;white-space:nowrap}
.telem .section{color:var(--amber);font-weight:700;margin-top:10px;font-size:11px;text-transform:uppercase;letter-spacing:1.5px;display:flex;align-items:center;gap:8px}
.telem .section::after{content:'';flex:1;height:1px;background:var(--border)}
.telem .row{color:var(--lgray);font-size:12px;padding:2px 0}
.telem .val{color:var(--cyan);font-weight:600}
.telem .section:first-child{margin-top:0}

/* ---- Tuning Panel ---- */
.tuning{padding:10px 16px;overflow-y:auto;flex:1}
.slider-group{margin-bottom:14px}
.slider-row{display:flex;align-items:center;justify-content:space-between;margin-bottom:4px}
.slider-label{font-size:12px;color:var(--lgray);font-weight:600;letter-spacing:.2px}
.slider-val{font-size:13px;color:var(--amber);font-weight:700;min-width:44px;text-align:right;font-family:'JetBrains Mono',monospace}

/* ---- Range Slider ---- */
input[type=range]{width:100%;height:6px;-webkit-appearance:none;appearance:none;background:var(--dgray);border-radius:3px;outline:none;margin:2px 0;transition:background .2s}
input[type=range]:hover{background:var(--mgray)}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:var(--amber);cursor:pointer;border:2px solid var(--panel);box-shadow:0 0 0 1px rgba(255,167,38,.3);transition:box-shadow .2s,transform .1s}
input[type=range]::-webkit-slider-thumb:hover{box-shadow:0 0 0 4px rgba(255,167,38,.2);transform:scale(1.1)}
input[type=range]::-webkit-slider-thumb:active{transform:scale(.95)}
input[type=range]::-moz-range-thumb{width:16px;height:16px;border-radius:50%;background:var(--amber);cursor:pointer;border:2px solid var(--panel);box-shadow:0 0 0 1px rgba(255,167,38,.3)}

/* ---- Color Priority List ---- */
.priority-header{font-size:11px;font-weight:700;color:var(--amber);margin-bottom:8px;text-transform:uppercase;letter-spacing:1.2px;display:flex;align-items:center;gap:8px}
.priority-header::after{content:'';flex:1;height:1px;background:var(--border)}
.priority-list{list-style:none;margin:6px 0}
.priority-item{display:flex;align-items:center;gap:8px;padding:6px 10px;border-radius:8px;margin-bottom:2px;background:var(--surface);transition:all .15s;border:1px solid transparent}
.priority-item:hover{background:var(--hover);border-color:var(--border)}
.priority-item.active{background:var(--hover);border-color:var(--item-color,var(--border-light));box-shadow:inset 3px 0 0 var(--item-color)}
.priority-rank{font-size:11px;font-weight:800;color:var(--mgray);min-width:16px;text-align:center;font-family:'JetBrains Mono',monospace}
.priority-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;box-shadow:0 0 4px currentColor}
.priority-name{font-size:13px;font-weight:600;color:var(--lgray);flex:1}
.priority-item.active .priority-name{color:var(--white)}
.priority-arrows{display:flex;gap:2px}
.priority-arrows button{width:24px;height:24px;border:none;border-radius:6px;background:var(--dgray);color:var(--lgray);font-size:11px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all .12s;padding:0;line-height:1;font-family:inherit}
.priority-arrows button:hover{background:var(--active);color:var(--white)}
.priority-expand{height:24px;border:none;border-radius:6px;background:var(--dgray);color:var(--mgray);font-size:10px;font-weight:700;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all .12s;padding:0 8px;margin-right:3px;letter-spacing:.5px;text-transform:uppercase;font-family:inherit}
.priority-expand:hover{background:var(--active);color:var(--white)}
.priority-expand.open{color:var(--amber);background:var(--active)}

/* ---- Color Toggle Switch ---- */
.color-toggle{position:relative;width:34px;height:18px;flex-shrink:0}
.color-toggle input{opacity:0;width:0;height:0}
.color-toggle .slider{position:absolute;cursor:pointer;top:0;left:0;right:0;bottom:0;background:var(--dgray);border-radius:9px;transition:.25s}
.color-toggle .slider:before{content:"";position:absolute;height:12px;width:12px;left:3px;bottom:3px;background:var(--mgray);border-radius:50%;transition:.25s}
.color-toggle input:checked+.slider{background:var(--green)}
.color-toggle input:checked+.slider:before{transform:translateX(16px);background:var(--white)}
.priority-item.disabled{opacity:.35}
.priority-item.disabled .priority-name{text-decoration:line-through;color:var(--mgray)}

/* ---- HSV Tuning Expand ---- */
.color-hsv-panel{display:none;padding:6px 10px 10px 32px;background:var(--bg);border-radius:0 0 8px 8px;margin-top:-2px;margin-bottom:4px;border:1px solid var(--border);border-top:none}
.color-hsv-panel.open{display:block}
.color-hsv-panel .slider-group{margin-bottom:4px}
.color-hsv-panel .slider-label{font-size:11px;color:var(--mgray)}
.color-hsv-panel .slider-val{font-size:11px}
.color-hsv-panel input[type=range]{height:3px}

/* ---- Section Divider ---- */
.tune-divider{height:1px;background:linear-gradient(90deg,transparent,var(--border),transparent);margin:10px 0}

/* ---- Toggle Button Pair ---- */
.toggle-btn-row{display:flex;gap:6px;margin-top:6px}
.toggle-btn{flex:1;padding:8px 0;border-radius:8px;font-family:inherit;font-size:12px;font-weight:700;cursor:pointer;transition:all .15s;letter-spacing:.3px}

/* ---- Footer ---- */
.footer{height:32px;display:flex;align-items:center;justify-content:center;gap:24px;font-size:10px;color:var(--mgray);background:var(--panel);flex-shrink:0;border-top:1px solid var(--border);letter-spacing:.5px;font-weight:500}
.footer-sep{color:var(--border-light)}
.footer kbd{background:var(--surface);border:1px solid var(--border);border-radius:3px;padding:1px 5px;font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--lgray)}

/* ---- Animations ---- */
@keyframes fadeIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
.panel{animation:fadeIn .3s ease}

/* ---- Mobile / Narrow Screens ---- */
@media(max-width:900px){
  body{overflow:auto;height:auto}
  .topbar{height:auto;padding:12px 16px;gap:10px;flex-wrap:wrap}
  .logo{width:100%;justify-content:center}
  .mode-nav{margin:0 auto}
  .status-area{margin:0 auto;flex-wrap:wrap;justify-content:center}
  .score-box{align-items:center}
  .score-bar-wrap{margin:0}
  .main{grid-template-columns:1fr;grid-template-rows:auto auto auto auto;gap:4px;padding:4px;overflow-y:auto}
  .panel{min-height:200px}
  .img-wrap{min-height:180px}
  .mode-btn{padding:8px 14px;font-size:12px}
  .telem{font-size:11px}
  .slider-label{font-size:12px}
  .footer{height:auto;padding:8px 12px;flex-wrap:wrap;gap:12px}
}
@media(max-width:480px){
  .mode-btn{padding:6px 10px;font-size:11px}
  .logo-main{font-size:18px}
  .logo-sub{font-size:10px}
  .score-val{font-size:28px}
  .img-wrap{min-height:140px}
  .priority-item{padding:8px 10px}
  .priority-arrows button{width:28px;height:28px;font-size:13px}
}
</style>
</head>
<body>

<!-- ===== Top Bar ===== -->
<div class="topbar">
  <div class="logo">
    <span class="logo-main">RACECAR Neo</span>
    <span class="logo-sub">OneShot Lab</span>
  </div>
  <div class="mode-nav">
    <button class="mode-btn active" data-mode="line_follow" onclick="setMode('line_follow')">Line Following</button>
    <button class="mode-btn" data-mode="wall_follow" onclick="setMode('wall_follow')">Wall Following</button>
    <button class="mode-btn" data-mode="car_track" onclick="setMode('car_track')">Car Tracking</button>
  </div>
  <div class="status-area">
    <div class="conn-status"><span class="conn-dot" id="connDot"></span><span id="connText">Connected</span></div>
    <div class="score-box">
      <span class="score-label">Score</span>
      <span class="score-val" id="scoreVal">0</span>
    </div>
  </div>
</div>
<div class="score-bar-wrap"><div class="score-bar" id="scoreBar" style="width:0%;background:var(--red)"></div></div>

<!-- ===== Main 4-Panel Grid ===== -->
<div class="main">
  <div class="panel">
    <div class="panel-hdr"><span class="panel-hdr-icon" style="background:var(--green)"></span><span class="panel-hdr-text">Camera / LIDAR</span></div>
    <div class="img-wrap"><img id="imgRaw" src="/api/mjpeg/raw" alt=""><div class="img-placeholder" id="phRaw">Awaiting camera feed&hellip;</div></div>
  </div>
  <div class="panel">
    <div class="panel-hdr"><span class="panel-hdr-icon" style="background:var(--cyan)"></span><span class="panel-hdr-text">Processing View</span></div>
    <div class="img-wrap"><img id="imgProc" src="/api/mjpeg/proc" alt=""><div class="img-placeholder" id="phProc">Awaiting processing&hellip;</div></div>
  </div>
  <div class="panel">
    <div class="panel-hdr"><span class="panel-hdr-icon" style="background:var(--amber)"></span><span class="panel-hdr-text">Telemetry</span></div>
    <div class="telem" id="telemBox"></div>
  </div>
  <div class="panel">
    <div class="panel-hdr"><span class="panel-hdr-icon" style="background:var(--magenta)"></span><span class="panel-hdr-text" id="tuneHdr">Tuning &mdash; Line Following</span></div>
    <div class="tuning" id="tuneBox"></div>
  </div>
</div>

<!-- ===== Footer ===== -->
<div class="footer">
  <span>RACECAR Neo &middot; Neobotics</span>
  <span class="footer-sep">|</span>
  <span>Hold <kbd>RT</kbd> to Drive</span>
  <span class="footer-sep">|</span>
  <span><kbd>A</kbd> Save Colors / Line &nbsp; <kbd>B</kbd> Wall &nbsp; <kbd>Y</kbd> Car</span>
</div>

<script>
const API='/api';
let currentMode='line_follow';
let paramsDirty=false;
let lastParams=[];
let pollOk=true;
let wfUseAvg=false;
let lfSubMode='standard';
let basicHue={};
let basicSV={};

// ---- Mode switching ----
function setMode(m){
  fetch(API+'/mode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:m})});
  currentMode=m;
  document.querySelectorAll('.mode-btn').forEach(b=>b.classList.toggle('active',b.dataset.mode===m));
  paramsDirty=true;
}

// ---- Color priority ----
let colorPriority=[];
let activeColor=null;
const colorDots={red:'#FF4444',blue:'#4488FF',green:'#44DD66',orange:'#FF8833',yellow:'#DDCC22',purple:'#BB55FF'};
const colorLabels={red:'Red',blue:'Blue',green:'Green',orange:'Orange',yellow:'Yellow',purple:'Purple'};

function moveColor(key,dir){
  fetch(API+'/color_priority',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:dir>0?'move_down':'move_up',color:key})});
}

let colorHSV={};
let colorEnabled={};
let expandedColor=null;

function toggleColorEnabled(key){
  const on=!(colorEnabled[key]!==false);
  colorEnabled[key]=!on;
  fetch(API+'/color_enabled',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({color:key,enabled:!on})});
  buildPriorityList(colorPriority,activeColor);
}

function buildPriorityList(priority,active){
  const box=document.getElementById('priorityBox');
  if(!box)return;
  box.innerHTML='';
  let rank=0;
  priority.forEach((key)=>{
    const on=colorEnabled[key]!==false;
    if(on)rank++;
    const item=document.createElement('div');
    item.className='priority-item'+(key===active?' active':'')+(on?'':' disabled');
    item.style.setProperty('--item-color',colorDots[key]||'#999');
    const checked=on?'checked':'';
    item.innerHTML=
      '<label class="color-toggle" title="'+(on?'Disable':'Enable')+' '+key+'"><input type="checkbox" '+checked+' onchange="toggleColorEnabled(\''+key+'\')"><span class="slider"></span></label>'+
      '<span class="priority-rank">'+(on?rank:'\u2014')+'</span>'+
      '<span class="priority-dot" style="background:'+(colorDots[key]||'#999')+';color:'+(colorDots[key]||'#999')+'"></span>'+
      '<span class="priority-name">'+(colorLabels[key]||key)+'</span>'+
      (on?'<button class="priority-expand'+(expandedColor===key?' open':'')+'" onclick="toggleHSV(\''+key+'\')" title="Tune HSV">HSV</button>':'')+
      '<span class="priority-arrows">'+
        '<button onclick="moveColor(\''+key+'\',-1)" title="Move up">\u25B2</button>'+
        '<button onclick="moveColor(\''+key+'\',1)" title="Move down">\u25BC</button>'+
      '</span>';
    box.appendChild(item);
    // Expandable HSV tuning
    const panel=document.createElement('div');
    panel.className='color-hsv-panel'+(expandedColor===key?' open':'');
    panel.id='hsvPanel_'+key;
    const hsv=colorHSV[key]||{H_low:0,H_high:179,S_low:0,S_high:255,V_low:0,V_high:255};
    [{label:'H Low',attr:'H_low',min:0,max:179,val:hsv.H_low},
     {label:'H High',attr:'H_high',min:0,max:179,val:hsv.H_high},
     {label:'S Low',attr:'S_low',min:0,max:255,val:hsv.S_low},
     {label:'S High',attr:'S_high',min:0,max:255,val:hsv.S_high},
     {label:'V Low',attr:'V_low',min:0,max:255,val:hsv.V_low},
     {label:'V High',attr:'V_high',min:0,max:255,val:hsv.V_high}
    ].forEach(f=>{
      const g=document.createElement('div');g.className='slider-group';
      const r=document.createElement('div');r.className='slider-row';
      const l=document.createElement('span');l.className='slider-label';l.textContent=f.label;
      const v=document.createElement('span');v.className='slider-val';v.id='chsv_'+key+'_'+f.attr;
      v.textContent=Math.round(f.val);
      r.appendChild(l);r.appendChild(v);g.appendChild(r);
      const s=document.createElement('input');s.type='range';s.min=f.min;s.max=f.max;s.step=1;s.value=f.val;
      s.oninput=function(){const nv=parseInt(this.value);v.textContent=nv;setColorHSV(key,f.attr,nv);};
      g.appendChild(s);panel.appendChild(g);
    });
    box.appendChild(panel);
  });
}

function toggleHSV(key){expandedColor=(expandedColor===key)?null:key;buildPriorityList(colorPriority,activeColor);}

function setColorHSV(color,attr,value){
  if(!colorHSV[color])colorHSV[color]={};
  colorHSV[color][attr]=value;
  fetch(API+'/color_hsv',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({color,attr,value:parseInt(value)})});
}

// ---- Param update ----
function setParam(attr,value){
  fetch(API+'/param',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({attr,value:parseFloat(value)})});
}

// ---- LF sub-mode helpers ----
function setLfSubMode(sm){
  lfSubMode=sm;
  fetch(API+'/lf_sub_mode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({lf_sub_mode:sm})});
  paramsDirty=true;
}
function setBasicColor(color,field,value){
  const body={color};body[field]=parseInt(value);
  fetch(API+'/basic_color',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
}
function saveColors(){
  fetch(API+'/save_colors',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({})});
}
function loadColors(){
  fetch(API+'/load_colors',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({})})
  .then(()=>{paramsDirty=true;});
}

function buildBasicColorPicker(box){
  // For each enabled color in priority order, show Hue + SV sliders
  const enabledColors=(colorPriority||[]).filter(k=>colorEnabled[k]!==false);
  enabledColors.forEach(key=>{
    const dot=colorDots[key]||'#999';
    const lbl=colorLabels[key]||key;
    const hdr=document.createElement('div');
    hdr.style.cssText='display:flex;align-items:center;gap:6px;padding:4px 0 2px;margin-top:6px';
    hdr.innerHTML='<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:'+dot+'"></span>'
      +'<span style="font-size:12px;font-weight:700;color:var(--white);letter-spacing:.3px">'+lbl+'</span>'
      +(key===activeColor?'<span style="font-size:10px;color:var(--green);margin-left:auto;font-weight:700">TRACKING</span>':'');
    box.appendChild(hdr);
    // Hue slider
    const hueVal=basicHue[key]||0;
    const g1=document.createElement('div');g1.className='slider-group';
    const r1=document.createElement('div');r1.className='slider-row';
    const l1=document.createElement('span');l1.className='slider-label';l1.textContent='Hue';
    const v1=document.createElement('span');v1.className='slider-val';v1.id='bhue_'+key;v1.textContent=hueVal;
    r1.appendChild(l1);r1.appendChild(v1);g1.appendChild(r1);
    const s1=document.createElement('input');s1.type='range';s1.min=0;s1.max=179;s1.step=1;s1.value=hueVal;
    s1.style.cssText='background:linear-gradient(to right,#f00,#ff0,#0f0,#0ff,#00f,#f0f,#f00);height:6px';
    s1.oninput=function(){v1.textContent=this.value;basicHue[key]=parseInt(this.value);setBasicColor(key,'hue',this.value);};
    g1.appendChild(s1);box.appendChild(g1);
    // SV slider (dark ← pure → light)
    const svVal=basicSV[key]||50;
    const g2=document.createElement('div');g2.className='slider-group';
    const r2=document.createElement('div');r2.className='slider-row';
    const l2=document.createElement('span');l2.className='slider-label';l2.textContent='Shade';
    const svLabels=['Pale','Pure','Dark'];
    const svIdx=svVal<=33?0:(svVal<=66?1:2);
    const v2=document.createElement('span');v2.className='slider-val';v2.id='bsv_'+key;v2.textContent=svLabels[svIdx]+' ('+svVal+')';
    r2.appendChild(l2);r2.appendChild(v2);g2.appendChild(r2);
    const s2=document.createElement('input');s2.type='range';s2.min=0;s2.max=100;s2.step=1;s2.value=svVal;
    s2.style.cssText='background:linear-gradient(to right,#fff,hsl('+Math.round(hueVal*2)+',100%,50%),#000);height:6px';
    s2.oninput=function(){
      const nv=parseInt(this.value);basicSV[key]=nv;
      const si=nv<=33?0:(nv<=66?1:2);
      v2.textContent=svLabels[si]+' ('+nv+')';
      setBasicColor(key,'sv',this.value);
    };
    g2.appendChild(s2);box.appendChild(g2);
  });
}

// ---- Build sliders ----
function buildSliders(params){
  const box=document.getElementById('tuneBox');
  box.innerHTML='';
  const modeNames={line_follow:'Line Following',wall_follow:'Wall Following',car_track:'Car Tracking'};
  document.getElementById('tuneHdr').textContent='Tuning \u2014 '+(modeNames[currentMode]||'???');
  if(currentMode==='line_follow'){
    // Standard / Advanced toggle
    const toggleRow=document.createElement('div');toggleRow.style.cssText='display:flex;gap:4px;margin-bottom:8px;background:var(--surface);border-radius:8px;padding:3px;border:1px solid var(--border)';
    const mkSub=(label,sm)=>{
      const b=document.createElement('button');b.textContent=label;
      const active=(sm===lfSubMode);
      b.style.cssText='flex:1;padding:7px 0;border:none;border-radius:6px;background:'+(active?'rgba(0,230,118,.12)':'transparent')+';color:'+(active?'var(--green)':'var(--lgray)')+';font-family:inherit;font-size:12px;font-weight:700;cursor:pointer;transition:all .15s;letter-spacing:.3px';
      b.onclick=function(){setLfSubMode(sm);};
      return b;
    };
    toggleRow.appendChild(mkSub('Standard','standard'));
    toggleRow.appendChild(mkSub('Advanced','advanced'));
    box.appendChild(toggleRow);

    // Save / Load button row
    const slRow=document.createElement('div');slRow.style.cssText='display:flex;gap:6px;margin-bottom:8px';
    const saveBtn=document.createElement('button');
    saveBtn.textContent='\uD83D\uDCBE Save (or press A)';
    saveBtn.style.cssText='flex:1;padding:8px 0;border:1px solid var(--amber);border-radius:8px;background:rgba(255,167,38,.08);color:var(--amber);font-family:inherit;font-size:12px;font-weight:700;cursor:pointer;transition:all .15s;letter-spacing:.3px';
    saveBtn.onmouseenter=function(){this.style.background='rgba(255,167,38,.18)';};
    saveBtn.onmouseleave=function(){this.style.background='rgba(255,167,38,.08)';};
    saveBtn.onclick=saveColors;
    slRow.appendChild(saveBtn);
    const loadBtn=document.createElement('button');
    loadBtn.textContent='\uD83D\uDCC2 Load';
    loadBtn.style.cssText='flex:1;padding:8px 0;border:1px solid var(--cyan);border-radius:8px;background:rgba(0,229,255,.08);color:var(--cyan);font-family:inherit;font-size:12px;font-weight:700;cursor:pointer;transition:all .15s;letter-spacing:.3px';
    loadBtn.onmouseenter=function(){this.style.background='rgba(0,229,255,.18)';};
    loadBtn.onmouseleave=function(){this.style.background='rgba(0,229,255,.08)';};
    loadBtn.onclick=loadColors;
    slRow.appendChild(loadBtn);
    box.appendChild(slRow);

    if(lfSubMode==='standard'){
      // Standard mode: Hue + SV per color
      const hdr=document.createElement('div');hdr.className='priority-header';
      hdr.textContent='Color Picker \u2014 Standard';box.appendChild(hdr);
      buildBasicColorPicker(box);
    }else{
      // Advanced mode: full color priority + HSV ranges
      const hdr=document.createElement('div');hdr.className='priority-header';
      hdr.textContent='Color Priority (top = first)';box.appendChild(hdr);
      const pbox=document.createElement('div');pbox.id='priorityBox';pbox.className='priority-list';
      box.appendChild(pbox);
      buildPriorityList(colorPriority,activeColor);
    }
    const sep=document.createElement('div');sep.className='tune-divider';box.appendChild(sep);
  }
  params.forEach(p=>{
    const g=document.createElement('div');g.className='slider-group';
    const r=document.createElement('div');r.className='slider-row';
    const l=document.createElement('span');l.className='slider-label';l.textContent=p.label;
    const v=document.createElement('span');v.className='slider-val';v.id='sv_'+p.attr;
    v.textContent=p.fmt==='float'?p.value.toFixed(2):p.value;
    r.appendChild(l);r.appendChild(v);g.appendChild(r);
    const s=document.createElement('input');s.type='range';s.min=p.min;s.max=p.max;s.step=p.step;s.value=p.value;
    s.oninput=function(){
      const nv=parseFloat(this.value);
      v.textContent=p.fmt==='float'?nv.toFixed(2):Math.round(nv);
      setParam(p.attr,nv);
    };
    g.appendChild(s);box.appendChild(g);
  });
  // Wall follow: add measurement mode toggle (Closest vs Average)
  if(currentMode==='wall_follow'){
    const sep=document.createElement('div');sep.className='tune-divider';box.appendChild(sep);
    const tg=document.createElement('div');tg.className='slider-group';tg.id='wfToggleGroup';
    const hdr=document.createElement('div');hdr.className='slider-row';
    const lbl=document.createElement('span');lbl.className='slider-label';lbl.textContent='Measurement Mode';
    const valSpan=document.createElement('span');valSpan.className='slider-val';valSpan.id='wfModeVal';
    valSpan.textContent=wfUseAvg?'Average':'Closest';
    hdr.appendChild(lbl);hdr.appendChild(valSpan);tg.appendChild(hdr);
    const btnRow=document.createElement('div');btnRow.style.cssText='display:flex;gap:6px;margin-top:6px';
    const mkBtn=(label,isAvg)=>{
      const b=document.createElement('button');
      b.textContent=label;
      const active=(isAvg===wfUseAvg);
      b.style.cssText='flex:1;padding:8px 0;border:1px solid '+(active?'var(--cyan)':'var(--border)')+';border-radius:8px;background:'+(active?'rgba(0,212,255,.12)':'var(--surface)')+';color:'+(active?'var(--cyan)':'var(--lgray)')+';font-family:inherit;font-size:12px;font-weight:700;cursor:pointer;transition:all .15s;letter-spacing:.3px';
      b.onmouseenter=function(){if(isAvg!==wfUseAvg)this.style.background='var(--hover)';};
      b.onmouseleave=function(){if(isAvg!==wfUseAvg)this.style.background='var(--surface)';};
      b.onclick=function(){
        wfUseAvg=isAvg;
        fetch(API+'/wf_toggle',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({wf_use_avg:isAvg})});
        buildSliders(lastParams);
      };
      return b;
    };
    btnRow.appendChild(mkBtn('Closest Point',false));
    btnRow.appendChild(mkBtn('Average Distance',true));
    tg.appendChild(btnRow);box.appendChild(tg);
  }
  lastParams=params;
}

// ---- Score color ----
function scoreColor(s){return s>=70?'var(--green)':s>=35?'var(--yellow)':'var(--red)';}

// ---- MJPEG stream ----
(function(){
  const raw=document.getElementById('imgRaw'),proc=document.getElementById('imgProc');
  raw.onload=()=>{document.getElementById('phRaw').style.display='none';raw.style.display='block';};
  raw.onerror=()=>{raw.style.display='none';document.getElementById('phRaw').style.display='block';};
  proc.onload=()=>{document.getElementById('phProc').style.display='none';proc.style.display='block';};
  proc.onerror=()=>{proc.style.display='none';document.getElementById('phProc').style.display='block';};
})();

// ---- Telemetry render ----
function renderTelem(d){
  const imu=d.imu||{accel:[0,0,0],gyro:[0,0,0]};
  const c=d.ctrl||{lt:0,rt:0,lj:[0,0],rj:[0,0]};
  let h='<div class="section">IMU</div>';
  h+='<div class="tbl">';
  h+='<span class="lbl">Accel</span><span class="dat">X: '+imu.accel[0].toFixed(2)+' &nbsp; Y: '+imu.accel[1].toFixed(2)+' &nbsp; Z: '+imu.accel[2].toFixed(2)+' m/s\u00b2</span>';
  h+='<span class="lbl">Gyro</span><span class="dat">X: '+imu.gyro[0].toFixed(2)+' &nbsp; Y: '+imu.gyro[1].toFixed(2)+' &nbsp; Z: '+imu.gyro[2].toFixed(2)+' rad/s</span>';
  h+='</div>';
  h+='<div class="section">Controller</div>';
  h+='<div class="tbl">';
  h+='<span class="lbl">Triggers</span><span class="dat">L: '+c.lt.toFixed(2)+' &nbsp; R: '+c.rt.toFixed(2)+'</span>';
  h+='<span class="lbl">L Stick</span><span class="dat">('+c.lj[0].toFixed(2)+', '+c.lj[1].toFixed(2)+')</span>';
  h+='<span class="lbl">R Stick</span><span class="dat">('+c.rj[0].toFixed(2)+', '+c.rj[1].toFixed(2)+')</span>';
  h+='</div>';
  h+='<div class="section">Drive Output</div>';
  h+='<div class="tbl">';
  h+='<span class="lbl">Speed</span><span class="dat">'+d.speed.toFixed(3)+'</span>';
  h+='<span class="lbl">Angle</span><span class="dat">'+d.angle.toFixed(3)+'</span>';
  h+='</div>';
  if(d.mode==='line_follow'){
    const sub=d.lf_sub_mode||'standard';
    h+='<div class="section">Line Following ('+sub.charAt(0).toUpperCase()+sub.slice(1)+')</div>';
    h+='<div class="tbl">';
    const ac=d.active_color;
    const acDot=ac?'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:'+(colorDots[ac]||'#999')+';margin:0 4px;vertical-align:middle;box-shadow:0 0 4px '+(colorDots[ac]||'#999')+'"></span>':'';
    h+='<span class="lbl">Tracking</span><span class="dat">'+acDot+(ac?ac.charAt(0).toUpperCase()+ac.slice(1):'None')+'</span>';
    const cc=d.contour_center;
    h+='<span class="lbl">Contour</span><span class="dat">'+(cc?'('+cc[0]+', '+cc[1]+')':'None')+'</span>';
    h+='<span class="lbl">Area</span><span class="dat">'+d.contour_area+'</span>';
    h+='</div>';
  }else if(d.mode==='wall_follow'){
    h+='<div class="section">Wall Following</div>';
    h+='<div class="tbl">';
    const sd=d.wf_scan_dir||90;
    const w=d.wf_window||30;
    const dirL=sd>=80?'Side':(sd>=45?'Diag':'Fwd');
    const modeTag=d.wf_use_avg?'Average':'Closest';
    h+='<span class="lbl">Left</span><span class="dat">'+d.wf_left_dist.toFixed(0)+' cm</span>';
    h+='<span class="lbl">Right</span><span class="dat">'+d.wf_right_dist.toFixed(0)+' cm</span>';
    h+='<span class="lbl">\u0394</span><span class="dat">'+(d.wf_right_dist-d.wf_left_dist).toFixed(0)+' cm</span>';
    h+='<span class="lbl">Scan</span><span class="dat">'+sd+'\u00b0 ['+dirL+'] \u00b1'+w+'\u00b0</span>';
    h+='<span class="lbl">Measurement</span><span class="dat">'+modeTag+' Point</span>';
    h+='</div>';
  }
  document.getElementById('telemBox').innerHTML=h;
}

// ---- Connection indicator ----
function setConn(ok){
  const dot=document.getElementById('connDot'),txt=document.getElementById('connText');
  if(ok){dot.classList.remove('err');txt.textContent='Connected';pollOk=true;}
  else{dot.classList.add('err');txt.textContent='Reconnecting\u2026';pollOk=false;}
}

// ---- Main poll loop ----
async function poll(){
  try{
    const r=await fetch(API+'/state');
    if(!r.ok)throw new Error();
    const d=await r.json();
    setConn(true);
    const sv=document.getElementById('scoreVal');
    sv.textContent=Math.round(d.score);
    sv.style.color=scoreColor(d.score);
    const bar=document.getElementById('scoreBar');
    bar.style.width=d.score+'%';
    bar.style.background=scoreColor(d.score);
    if(d.mode!==currentMode){currentMode=d.mode;paramsDirty=true;
      document.querySelectorAll('.mode-btn').forEach(b=>b.classList.toggle('active',b.dataset.mode===d.mode));
    }
    if(d.color_priority){
      const orderChanged=JSON.stringify(d.color_priority)!==JSON.stringify(colorPriority)||d.active_color!==activeColor;
      colorPriority=d.color_priority;activeColor=d.active_color||null;
      let needRebuild=orderChanged;
      if(d.color_hsv){const c=JSON.stringify(d.color_hsv)!==JSON.stringify(colorHSV);colorHSV=d.color_hsv;if(c)needRebuild=true;}
      if(d.color_enabled){const c=JSON.stringify(d.color_enabled)!==JSON.stringify(colorEnabled);colorEnabled=d.color_enabled;if(c)needRebuild=true;}
      if(d.basic_hue)basicHue=d.basic_hue;
      if(d.basic_sv)basicSV=d.basic_sv;
      if(needRebuild){if(lfSubMode==='advanced')buildPriorityList(colorPriority,activeColor);}
    }
    // Sync LF sub-mode
    if(d.lf_sub_mode&&d.lf_sub_mode!==lfSubMode){lfSubMode=d.lf_sub_mode;paramsDirty=true;}
    // Sync wall-follow toggle
    if(d.wf_use_avg!==undefined&&d.wf_use_avg!==wfUseAvg){
      wfUseAvg=d.wf_use_avg;
      const mv=document.getElementById('wfModeVal');
      if(mv)mv.textContent=wfUseAvg?'Average':'Closest';
    }
    renderTelem(d);
    if(paramsDirty&&d.params){buildSliders(d.params);paramsDirty=false;}
    else if(d.params){d.params.forEach(p=>{const el=document.getElementById('sv_'+p.attr);if(el)el.textContent=p.fmt==='float'?p.value.toFixed(2):p.value;});}
  }catch(e){setConn(false);}
}

paramsDirty=true;
setInterval(poll,100);
</script>
</body>
</html>"""


class _DashboardHandler(_BaseHandler):
    """HTTP handler for the web dashboard."""

    def log_message(self, format, *args):
        pass  # suppress noisy access logs

    def do_GET(self):
        if self.path == "/":
            self._serve_html()
        elif self.path == "/api/state":
            self._serve_state()
        elif self.path.startswith("/api/mjpeg/"):
            self._serve_mjpeg()
        elif self.path.startswith("/api/img/"):
            self._serve_image()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/mode":
            self._handle_mode()
        elif self.path == "/api/param":
            self._handle_param()
        elif self.path == "/api/wf_toggle":
            self._handle_wf_toggle()
        elif self.path == "/api/color_priority":
            self._handle_color_priority()
        elif self.path == "/api/color_hsv":
            self._handle_color_hsv()
        elif self.path == "/api/color_enabled":
            self._handle_color_enabled()
        elif self.path == "/api/lf_sub_mode":
            self._handle_lf_sub_mode()
        elif self.path == "/api/basic_color":
            self._handle_basic_color()
        elif self.path == "/api/save_colors":
            self._handle_save_colors()
        elif self.path == "/api/load_colors":
            self._handle_load_colors()
        else:
            self.send_error(404)

    # ---- Endpoints ----

    def _serve_html(self):
        body = _DASHBOARD_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_state(self):
        score = calculate_score()
        state.score = score
        data = {
            "mode": state.mode,
            "mode_name": MODE_NAMES.get(state.mode, "???"),
            "score": round(score, 1),
            "speed": round(state.speed, 3),
            "angle": round(state.angle, 3),
            "contour_center": list(state.contour_center) if state.contour_center else None,
            "contour_area": state.contour_area,
            "wf_left_dist": round(state.wf_left_dist, 1),
            "wf_right_dist": round(state.wf_right_dist, 1),
            "wf_scan_dir": state.wf_scan_dir,
            "wf_window": state.wf_window,
            "wf_use_avg": state.wf_use_avg,
            "params": self._get_params(),
            "color_priority": list(state.lf_color_priority),
            "active_color": state.lf_active_color,
            "color_hsv": state.lf_color_hsv if state.lf_color_hsv else {},
            "color_enabled": state.lf_color_enabled if state.lf_color_enabled else {},
            "lf_sub_mode": state.lf_sub_mode,
            "basic_hue": state.lf_basic_hue if state.lf_basic_hue else {},
            "basic_sv": state.lf_basic_sv if state.lf_basic_sv else {},
        }
        # IMU (read from cache — NEVER call rc.* from this thread!)
        a = state.imu_accel
        w = state.imu_gyro
        data["imu"] = {
            "accel": [round(a[0], 2), round(a[1], 2), round(a[2], 2)],
            "gyro":  [round(w[0], 2), round(w[1], 2), round(w[2], 2)],
        }
        # Controller (read from cache)
        data["ctrl"] = {
            "lt": round(state.ctrl_lt, 2),
            "rt": round(state.ctrl_rt, 2),
            "lj": [round(state.ctrl_lj[0], 2), round(state.ctrl_lj[1], 2)],
            "rj": [round(state.ctrl_rj[0], 2), round(state.ctrl_rj[1], 2)],
        }

        body = _json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_image(self):
        """Serve a single JPEG frame from the pre-encoded cache."""
        img_type = self.path.split("/")[-1].split("?")[0]
        data = state._jpeg_raw if img_type == "raw" else state._jpeg_proc
        if not data:
            self.send_response(204)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_mjpeg(self):
        """MJPEG stream — persistent HTTP connection pushing ~20 fps.

        Much faster than the old polling approach (which was ~4 fps with
        per-request JPEG encoding overhead).
        """
        img_type = self.path.split("/")[-1].split("?")[0]
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        while True:
            data = state._jpeg_raw if img_type == "raw" else state._jpeg_proc
            if not data:
                time.sleep(0.1)
                continue
            try:
                hdr = (b"--frame\r\nContent-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n")
                self.wfile.write(hdr)
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                break
            time.sleep(0.05)  # ~20 fps push rate

    def _handle_mode(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = _json.loads(body)
            mode = data.get("mode")
            if mode in MODE_NAMES:
                _switch_mode(mode, from_main_thread=False)
        except Exception:
            pass
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def _handle_param(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = _json.loads(body)
            attr = data.get("attr")
            value = data.get("value")
            for _label, a, lo, hi, _step, fmt in _TUNE_PARAMS.get(state.mode, []):
                if a == attr:
                    if fmt == "int":
                        value = int(max(lo, min(hi, value)))
                    else:
                        value = float(max(lo, min(hi, value)))
                    setattr(state, attr, value)
                    break
        except Exception:
            pass
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def _handle_wf_toggle(self):
        """Toggle wall-follow measurement mode (closest vs average).
        Body: {"wf_use_avg": true/false}"""
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = _json.loads(body)
            state.wf_use_avg = bool(data.get("wf_use_avg", False))
        except Exception:
            pass
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def _handle_color_priority(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = _json.loads(body)
            action = data.get("action")  # "move_up" or "move_down"
            color = data.get("color")
            if action == "move_up":
                _move_color_priority(color, -1)
            elif action == "move_down":
                _move_color_priority(color, +1)
        except Exception:
            pass
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def _handle_color_hsv(self):
        """Set a single HSV field for a specific color.
        Body: {"color": "red", "attr": "H_low", "value": 10}"""
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = _json.loads(body)
            color = data.get("color")
            attr = data.get("attr")
            value = int(data.get("value", 0))
            valid_attrs = {"H_low", "H_high", "S_low", "S_high", "V_low", "V_high"}
            if (color and attr in valid_attrs and state.lf_color_hsv
                    and color in state.lf_color_hsv):
                # Clamp to valid range
                mx = 179 if attr.startswith("H") else 255
                value = max(0, min(mx, value))
                state.lf_color_hsv[color][attr] = value
        except Exception:
            pass
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def _handle_color_enabled(self):
        """Toggle a color on or off.
        Body: {"color": "red", "enabled": true/false}"""
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = _json.loads(body)
            color = data.get("color")
            enabled = bool(data.get("enabled", True))
            if color and state.lf_color_enabled is not None and color in state.lf_color_enabled:
                state.lf_color_enabled[color] = enabled
        except Exception:
            pass
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def _handle_lf_sub_mode(self):
        """Switch line-follow sub-mode.
        Body: {"lf_sub_mode": "standard"|"advanced"}"""
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = _json.loads(body)
            sm = data.get("lf_sub_mode")
            if sm in ("standard", "advanced"):
                state.lf_sub_mode = sm
                print(f">> Line-follow sub-mode: {sm}")
        except Exception:
            pass
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def _handle_basic_color(self):
        """Set basic-mode hue or SV for a color.
        Body: {"color": "red", "hue": 10}  or  {"color": "red", "sv": 50}"""
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = _json.loads(body)
            color = data.get("color")
            if color:
                if "hue" in data and state.lf_basic_hue is not None:
                    state.lf_basic_hue[color] = max(0, min(179, int(data["hue"])))
                    _apply_basic_to_hsv(color)
                if "sv" in data and state.lf_basic_sv is not None:
                    state.lf_basic_sv[color] = max(0, min(100, int(data["sv"])))
                    _apply_basic_to_hsv(color)
        except Exception:
            pass
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def _handle_save_colors(self):
        """Save color config to disk."""
        _save_color_config()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def _handle_load_colors(self):
        """Load color config from disk."""
        ok = _load_color_config()
        if ok and state.lf_sub_mode == "standard":
            for color_key in (state.lf_basic_hue or {}):
                _apply_basic_to_hsv(color_key)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(_json.dumps({"ok": ok}).encode())

    # ---- Helpers ----

    def _get_params(self):
        result = []
        for label, attr, lo, hi, step, fmt in _TUNE_PARAMS.get(state.mode, []):
            val = getattr(state, attr)
            result.append({
                "label": label, "attr": attr,
                "value": round(val, 2) if fmt == "float" else int(val),
                "min": lo, "max": hi, "step": step, "fmt": fmt,
            })
        return result


def _start_web_dashboard():
    """Launch the web dashboard HTTP server in a daemon thread."""
    port = _WEB_PORT
    server = None
    for attempt in range(5):
        try:
            server = _ThreadedHTTPServer(("0.0.0.0", port), _DashboardHandler)
            break
        except OSError:
            port += 1
    if server is None:
        print(">> Web dashboard: could not find an open port")
        return

    # Detect LAN IP so other devices on the same WiFi can connect
    lan_ip = None
    try:
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # doesn't actually send data
        lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    local_url = f"http://localhost:{port}"
    print(f"\n>> \033[1;96mWeb Dashboard running at {local_url}\033[0m")
    if lan_ip and lan_ip != "127.0.0.1":
        net_url = f"http://{lan_ip}:{port}"
        print(f">> \033[1;93mNetwork access: {net_url}\033[0m")
        print(f">> Anyone on the same WiFi can open the URL above!")
    else:
        print(f">> Open this URL in your browser!")
    print()

    # Start HTTP server in daemon thread
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    # Auto-open browser on Windows from WSL (in its own thread to avoid blocking)
    def _open():
        try:
            import subprocess
            subprocess.Popen(
                ["cmd.exe", "/c", "start", "", url],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
    threading.Thread(target=_open, daemon=True).start()


########################################################################################
# Headless Controller-Based Tuning (WSL 1 / no X server)
########################################################################################

# Tunable parameters per mode: (label, attr_name, min, max, step, fmt)
# "fmt" is "int" or "float"
_TUNE_PARAMS = {
    "line_follow": [
        # label              attr               min   max   step  fmt
        ("Speed %",          "lf_speed",         0,   100,  5,    "int"),
        ("Steer Sens %",     "lf_angle_sens",    0,   100,  5,    "int"),
    ],
    "wall_follow": [
        # label              attr               min   max   step  fmt
        ("Speed %",          "wf_speed",         1,   100,  5,    "int"),
        ("Sensitivity %",    "wf_kp",            1,   100,  5,    "int"),
        ("Scan Dir\u00b0",   "wf_scan_dir",      0,   135,  5,    "int"),
        ("Window \u00b1\u00b0",  "wf_window",    1,    60,  5,    "int"),
    ],
    "car_track": [
        # label              attr               min   max   step  fmt
        ("Speed %",          "ct_speed",         0,   100,  5,    "int"),
        ("Kp",               "ct_kp",            0.0, 50.0, 0.5,  "float"),
        ("Ki",               "ct_ki",            0.0, 10.0, 0.1,  "float"),
        ("Kd",               "ct_kd",            0.0, 50.0, 0.5,  "float"),
        ("Confidence %",     "ct_score_thresh",  0.1,  1.0, 0.05, "float"),
    ],
}

_tune_index = 0  # currently selected parameter index


def _headless_tune_update():
    """
    Controller-based parameter tuning for headless mode.
      X       = cycle to next parameter
      LB      = decrease current parameter by one step
      RB      = increase current parameter by one step
      L-Joy Y = fine-adjust (held)
    """
    global _tune_index

    params = _TUNE_PARAMS.get(state.mode, [])
    if not params:
        return

    _tune_index = _tune_index % len(params)

    # X: cycle parameter
    if rc.controller.was_pressed(rc.controller.Button.X):
        _tune_index = (_tune_index + 1) % len(params)
        label, attr, lo, hi, step, fmt = params[_tune_index]
        val = getattr(state, attr)
        vstr = f"{val:.2f}" if fmt == "float" else str(int(val))
        print(f"  >> Tuning: [{label}] = {vstr}")

    label, attr, lo, hi, step, fmt = params[_tune_index]
    changed = False

    # LB: decrease
    if rc.controller.was_pressed(rc.controller.Button.LB):
        val = getattr(state, attr)
        new_val = max(lo, val - step)
        if fmt == "int":
            new_val = int(new_val)
        setattr(state, attr, new_val)
        changed = True

    # RB: increase
    if rc.controller.was_pressed(rc.controller.Button.RB):
        val = getattr(state, attr)
        new_val = min(hi, val + step)
        if fmt == "int":
            new_val = int(new_val)
        setattr(state, attr, new_val)
        changed = True

    # Left joystick Y: fine continuous adjust (held, not pressed)
    try:
        lj_x, lj_y = rc.controller.get_joystick(rc.controller.Joystick.LEFT)
        if abs(lj_y) > 0.3:
            val = getattr(state, attr)
            fine_step = step * 0.2 * lj_y  # proportional to stick tilt
            new_val = max(lo, min(hi, val + fine_step))
            if fmt == "int":
                new_val = int(round(new_val))
            setattr(state, attr, new_val)
            changed = True
    except Exception:
        pass

    if changed:
        val = getattr(state, attr)
        vstr = f"{val:.2f}" if fmt == "float" else str(int(val))
        print(f"  [{label}] = {vstr}")


def _headless_print_status():
    """
    Print a compact status block to terminal. Called from update_slow (~1/sec).
    """
    mode_name = MODE_NAMES.get(state.mode, "???")
    score = calculate_score()
    state.score = score

    # Score color via ANSI
    if score >= 70:
        sc = "\033[92m"   # bright green
    elif score >= 35:
        sc = "\033[93m"   # yellow
    else:
        sc = "\033[91m"   # red
    rst = "\033[0m"

    params = _TUNE_PARAMS.get(state.mode, [])
    idx = _tune_index % len(params) if params else 0

    # Build parameter summary line
    param_strs = []
    for i, (label, attr, lo, hi, step, fmt) in enumerate(params):
        val = getattr(state, attr)
        vstr = f"{val:.2f}" if fmt == "float" else str(int(val))
        marker = ">" if i == idx else " "
        param_strs.append(f"{marker}{label}:{vstr}")
    param_line = "  ".join(param_strs)

    print(
        f"\r\033[K"
        f"[{mode_name}]  "
        f"Score: {sc}{score:.0f}/100{rst}  "
        f"Spd:{state.speed:+.2f}  Ang:{state.angle:+.2f}  "
    )
    if state.mode == "line_follow":
        sub = state.lf_sub_mode.capitalize()
        prio = " > ".join(c.capitalize() for c in state.lf_color_priority)
        active = (state.lf_active_color or "none").capitalize()
        print(f"  [{sub}]  Colors: {prio}  |  Tracking: \033[1m{active}\033[0m")
    if param_line:
        print(f"  Params: {param_line}")
    print(
        f"  Controls: A=Save/Line B=Wall Y=Car | X=NextParam LB=- RB=+ | RT=Drive",
        end="", flush=True,
    )


########################################################################################
# Tkinter GUI (only used when tkinter is available)
########################################################################################

class OneshotApp:
    """
    Modern dark-themed tkinter application that provides:
      - 4-panel sensor display (camera, processing, telemetry, score)
      - Top-bar mode switching
      - Dynamic toolbar with per-mode tuning sliders
    Only instantiated when HAS_TKINTER is True.
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RACECAR Neo \u00b7 OneShot Lab")
        self.root.geometry("1300x820")
        self.root.configure(bg=Colors.BG_DARK)
        self.root.minsize(1000, 650)

        self.font_title    = tkfont.Font(family="Helvetica", size=18, weight="bold")
        self.font_subtitle = tkfont.Font(family="Helvetica", size=12, weight="bold")
        self.font_label    = tkfont.Font(family="Helvetica", size=10, weight="bold")
        self.font_data     = tkfont.Font(family="Consolas",  size=10)
        self.font_score    = tkfont.Font(family="Helvetica", size=42, weight="bold")
        self.font_btn      = tkfont.Font(family="Helvetica", size=11, weight="bold")
        self.font_small    = tkfont.Font(family="Helvetica", size=9)

        self._tk_raw  = None
        self._tk_proc = None
        self._slider_vars = {}
        self._mode_btns = {}

        self._configure_styles()
        self._build_topbar()
        self._build_main()
        self._set_mode("line_follow")
        self._tick()

    # ---- ttk Style ----
    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.Horizontal.TScale",
            background=Colors.BG_PANEL, troughcolor=Colors.DARK_GRAY,
            bordercolor=Colors.BG_PANEL, lightcolor=Colors.BG_PANEL,
            darkcolor=Colors.BG_PANEL, sliderlength=18)
        style.configure("Dark.TFrame", background=Colors.BG_PANEL)

    # ---- Top Bar ----
    def _build_topbar(self):
        bar = tk.Frame(self.root, bg=Colors.BG_PANEL, height=56)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)

        tk.Label(bar, text="RACECAR Neo", bg=Colors.BG_PANEL,
                 fg=Colors.NEO_RED, font=self.font_title).pack(side="left", padx=(16, 4))
        tk.Label(bar, text="OneShot Lab", bg=Colors.BG_PANEL,
                 fg=Colors.NEO_ORANGE, font=self.font_subtitle).pack(side="left", padx=(0, 20))

        modes = [
            ("Line Following", "line_follow", Colors.GREEN),
            ("Wall Following", "wall_follow", Colors.CYAN),
            ("Car Tracking",   "car_track",   Colors.NEO_ORANGE),
        ]
        btn_box = tk.Frame(bar, bg=Colors.BG_PANEL)
        btn_box.pack(side="right", padx=16)

        for label, mode_id, accent in modes:
            btn = tk.Button(btn_box, text=f"  {label}  ", font=self.font_btn,
                bg=Colors.BG_SURFACE, fg=accent,
                activebackground=Colors.BG_ACTIVE, activeforeground=accent,
                bd=0, relief="flat", cursor="hand2",
                command=lambda m=mode_id: self._set_mode(m))
            btn.pack(side="left", padx=3)
            self._mode_btns[mode_id] = btn

    # ---- Main Content ----
    def _build_main(self):
        main = tk.Frame(self.root, bg=Colors.BG_DARK)
        main.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        left = tk.Frame(main, bg=Colors.BG_DARK)
        left.pack(side="left", fill="both", expand=True)

        # Row 1: camera + processing
        row1 = tk.Frame(left, bg=Colors.BG_DARK)
        row1.pack(fill="both", expand=True)

        raw_panel = self._panel(row1, "Camera / LIDAR View")
        raw_panel.pack(side="left", fill="both", expand=True, padx=(0, 3), pady=(0, 3))
        self.cvs_raw = tk.Canvas(raw_panel, bg=Colors.BG_SURFACE, highlightthickness=0)
        self.cvs_raw.pack(fill="both", expand=True, padx=3, pady=3)

        proc_panel = self._panel(row1, "Processing View")
        proc_panel.pack(side="left", fill="both", expand=True, padx=(3, 0), pady=(0, 3))
        self.cvs_proc = tk.Canvas(proc_panel, bg=Colors.BG_SURFACE, highlightthickness=0)
        self.cvs_proc.pack(fill="both", expand=True, padx=3, pady=3)

        # Row 2: telemetry + score
        row2 = tk.Frame(left, bg=Colors.BG_DARK)
        row2.pack(fill="both", expand=True)

        telem_panel = self._panel(row2, "Telemetry")
        telem_panel.pack(side="left", fill="both", expand=True, padx=(0, 3), pady=(3, 0))
        self.txt_telem = tk.Text(telem_panel, bg=Colors.BG_SURFACE, fg=Colors.CYAN,
            font=self.font_data, relief="flat", bd=0, state="disabled",
            wrap="word", insertbackground=Colors.CYAN)
        self.txt_telem.pack(fill="both", expand=True, padx=4, pady=4)

        score_panel = self._panel(row2, "Performance Score")
        score_panel.pack(side="left", fill="both", expand=True, padx=(3, 0), pady=(3, 0))

        self.lbl_score = tk.Label(score_panel, text="--", bg=Colors.BG_SURFACE,
            fg=Colors.SCORE_HIGH, font=self.font_score)
        self.lbl_score.pack(expand=True, pady=(10, 0))

        self.cvs_bar = tk.Canvas(score_panel, bg=Colors.BG_SURFACE,
            highlightthickness=0, height=14)
        self.cvs_bar.pack(fill="x", padx=12, pady=(0, 2))

        self.lbl_score_hint = tk.Label(score_panel,
            text="Higher number = better tuning!",
            bg=Colors.BG_SURFACE, fg=Colors.LIGHT_GRAY, font=self.font_small)
        self.lbl_score_hint.pack(pady=(0, 10))

        # Right: toolbar
        right = tk.Frame(main, bg=Colors.BG_PANEL, width=290)
        right.pack(side="right", fill="y", padx=(8, 0))
        right.pack_propagate(False)

        tk.Label(right, text="Tuning Parameters", bg=Colors.BG_PANEL,
            fg=Colors.WHITE, font=self.font_subtitle).pack(pady=(12, 2))

        self.lbl_mode = tk.Label(right, text="LINE FOLLOWING", bg=Colors.BG_PANEL,
            fg=Colors.GREEN, font=self.font_label)
        self.lbl_mode.pack(pady=(0, 8))

        tk.Frame(right, bg=Colors.MID_GRAY, height=1).pack(fill="x", padx=12)

        self.toolbar_container = tk.Frame(right, bg=Colors.BG_PANEL)
        self.toolbar_container.pack(fill="both", expand=True, pady=6)

        tk.Frame(right, bg=Colors.MID_GRAY, height=1).pack(fill="x", padx=12)

        self.lbl_instructions = tk.Label(right,
            text="Hold RIGHT TRIGGER to drive\nUse sliders to tune parameters",
            bg=Colors.BG_PANEL, fg=Colors.LIGHT_GRAY,
            font=self.font_small, justify="center")
        self.lbl_instructions.pack(pady=10)

    # ---- UI Helpers ----
    def _panel(self, parent, title):
        frame = tk.Frame(parent, bg=Colors.BG_PANEL)
        hdr = tk.Frame(frame, bg=Colors.BG_PANEL, height=24)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text=f"  {title}", bg=Colors.BG_PANEL,
            fg=Colors.LIGHT_GRAY, font=self.font_small, anchor="w").pack(side="left")
        return frame

    def _slider(self, parent, label, lo, hi, default, callback,
                accent=Colors.CYAN, fmt="int"):
        frame = tk.Frame(parent, bg=Colors.BG_PANEL)
        frame.pack(fill="x", padx=12, pady=5)

        row = tk.Frame(frame, bg=Colors.BG_PANEL)
        row.pack(fill="x")
        tk.Label(row, text=label, bg=Colors.BG_PANEL,
            fg=Colors.WHITE, font=self.font_label).pack(side="left")
        val_lbl = tk.Label(row, text=str(int(default)), bg=Colors.BG_PANEL,
            fg=accent, font=self.font_label)
        val_lbl.pack(side="right")

        var = tk.DoubleVar(value=default)
        scale = ttk.Scale(frame, from_=lo, to=hi, orient="horizontal",
            variable=var, style="Dark.Horizontal.TScale",
            command=lambda v, vl=val_lbl, cb=callback, f=fmt: (
                vl.config(text=(f"{float(v):.2f}" if f == "float" else str(int(float(v))))),
                cb(float(v))))
        scale.pack(fill="x", pady=(2, 0))
        return var

    def _section_label(self, parent, text):
        tk.Label(parent, text=text, bg=Colors.BG_PANEL,
            fg=Colors.NEO_AMBER, font=self.font_label).pack(pady=(8, 2))

    # ---- Mode Switching ----
    def _set_mode(self, mode):
        _switch_mode(mode)

        mode_meta = {
            "line_follow": ("LINE FOLLOWING", Colors.GREEN),
            "wall_follow": ("WALL FOLLOWING", Colors.CYAN),
            "car_track":   ("CAR TRACKING",   Colors.NEO_ORANGE),
        }
        name, accent = mode_meta[mode]
        self.lbl_mode.config(text=name, fg=accent)

        for mid, btn in self._mode_btns.items():
            btn.config(bg=Colors.BG_ACTIVE if mid == mode else Colors.BG_SURFACE)

        for w in self.toolbar_container.winfo_children():
            w.destroy()
        self._slider_vars.clear()

        if mode == "line_follow":
            self._build_lf_toolbar()
        elif mode == "wall_follow":
            self._build_wf_toolbar()
        elif mode == "car_track":
            self._build_ct_toolbar()

    # ---- Toolbar Builders ----
    def _build_lf_toolbar(self):
        p = self.toolbar_container

        # ---- Standard / Advanced toggle ----
        toggle_frame = tk.Frame(p, bg=Colors.BG_PANEL)
        toggle_frame.pack(fill="x", padx=12, pady=(4, 4))
        self._lf_sub_btns = {}
        for label, sm, col in [("Standard", "standard", Colors.GREEN),
                                ("Advanced", "advanced", Colors.CYAN)]:
            active = (state.lf_sub_mode == sm)
            btn = tk.Button(toggle_frame, text=label, font=self.font_btn,
                bg=Colors.BG_ACTIVE if active else Colors.BG_SURFACE,
                fg=col if active else Colors.LIGHT_GRAY,
                activebackground=Colors.BG_ACTIVE, activeforeground=col,
                bd=0, relief="flat", cursor="hand2",
                command=lambda m=sm: self._set_lf_sub_mode(m))
            btn.pack(side="left", fill="x", expand=True, padx=2)
            self._lf_sub_btns[sm] = btn

        # ---- Save / Load buttons ----
        sl_frame = tk.Frame(p, bg=Colors.BG_PANEL)
        sl_frame.pack(fill="x", padx=12, pady=(2, 6))
        save_btn = tk.Button(sl_frame, text="\U0001F4BE  Save (or press A)",
            font=self.font_small, bg=Colors.BG_SURFACE, fg=Colors.NEO_AMBER,
            activebackground=Colors.BG_ACTIVE, activeforeground=Colors.NEO_AMBER,
            bd=0, relief="flat", cursor="hand2",
            command=_save_color_config)
        save_btn.pack(side="left", fill="x", expand=True, padx=(0, 3))
        load_btn = tk.Button(sl_frame, text="\U0001F4C2  Load",
            font=self.font_small, bg=Colors.BG_SURFACE, fg=Colors.CYAN,
            activebackground=Colors.BG_ACTIVE, activeforeground=Colors.CYAN,
            bd=0, relief="flat", cursor="hand2",
            command=self._load_and_refresh_colors)
        load_btn.pack(side="left", fill="x", expand=True, padx=(3, 0))

        tk.Frame(p, bg=Colors.MID_GRAY, height=1).pack(fill="x", padx=12)

        # ---- Content area (changes based on sub-mode) ----
        self._lf_content_frame = tk.Frame(p, bg=Colors.BG_PANEL)
        self._lf_content_frame.pack(fill="both", expand=True)
        self._build_lf_content()

    def _load_and_refresh_colors(self):
        """Load color config from disk and rebuild the LF toolbar."""
        ok = _load_color_config()
        if ok:
            if state.lf_sub_mode == "standard":
                for color_key in (state.lf_basic_hue or {}):
                    _apply_basic_to_hsv(color_key)
            # Full rebuild of toolbar to reflect loaded sub-mode, priority, etc.
            if state.mode == "line_follow":
                self._set_mode("line_follow")

    def _set_lf_sub_mode(self, sm):
        """Switch between standard and advanced LF sub-mode."""
        state.lf_sub_mode = sm
        for mid, btn in self._lf_sub_btns.items():
            col = Colors.GREEN if mid == "standard" else Colors.CYAN
            btn.config(bg=Colors.BG_ACTIVE if mid == sm else Colors.BG_SURFACE,
                       fg=col if mid == sm else Colors.LIGHT_GRAY)
        # Rebuild the content area
        for w in self._lf_content_frame.winfo_children():
            w.destroy()
        self._build_lf_content()

    def _build_lf_content(self):
        """Build the Standard or Advanced line-follow content."""
        p = self._lf_content_frame

        if state.lf_sub_mode == "standard":
            self._build_lf_standard(p)
        else:
            self._build_lf_advanced(p)

        tk.Frame(p, bg=Colors.MID_GRAY, height=1).pack(fill="x", padx=12)

        # ---- Drive Sliders (always shown) ----
        self._section_label(p, "Drive Control")
        self._slider(p, "Speed %", 0, 100, state.lf_speed,
                     lambda v: setattr(state, "lf_speed", int(v)))
        self._slider(p, "Steer Sensitivity %", 0, 100, state.lf_angle_sens,
                     lambda v: setattr(state, "lf_angle_sens", int(v)))

    def _build_lf_standard(self, p):
        """Standard mode: Hue + SV picker per enabled color."""
        self._section_label(p, "Color Picker \u2014 Standard")
        enabled = state.lf_color_enabled or {}
        for key in state.lf_color_priority:
            if not enabled.get(key, True):
                continue
            cdef = LINE_COLORS.get(key, {})
            accent = self._dot_colors.get(key, Colors.LIGHT_GRAY)

            # Color header
            hdr = tk.Frame(p, bg=Colors.BG_PANEL)
            hdr.pack(fill="x", padx=12, pady=(6, 0))
            dot = tk.Canvas(hdr, width=10, height=10, bg=Colors.BG_PANEL, highlightthickness=0)
            dot.create_oval(1, 1, 9, 9, fill=accent, outline="")
            dot.pack(side="left", padx=(0, 4))
            tk.Label(hdr, text=cdef.get("label", key), bg=Colors.BG_PANEL,
                     fg=Colors.WHITE, font=self.font_label).pack(side="left")
            if key == state.lf_active_color:
                tk.Label(hdr, text="TRACKING", bg=Colors.BG_PANEL,
                         fg=Colors.GREEN, font=("Helvetica", 8, "bold")).pack(side="right")

            # Hue slider
            hue_val = (state.lf_basic_hue or {}).get(key, 90)
            self._slider(p, "Hue", 0, 179, hue_val,
                         lambda v, k=key: self._set_basic_hue(k, int(v)),
                         accent=Colors.NEO_RED)

            # SV slider
            sv_val = (state.lf_basic_sv or {}).get(key, 50)
            self._slider(p, "Shade (Pale \u2190 Pure \u2192 Dark)", 0, 100, sv_val,
                         lambda v, k=key: self._set_basic_sv(k, int(v)),
                         accent=Colors.CYAN)

    def _set_basic_hue(self, color_key, value):
        """Update basic-mode hue and recalculate HSV."""
        if state.lf_basic_hue is not None:
            state.lf_basic_hue[color_key] = value
            _apply_basic_to_hsv(color_key)

    def _set_basic_sv(self, color_key, value):
        """Update basic-mode SV position and recalculate HSV."""
        if state.lf_basic_sv is not None:
            state.lf_basic_sv[color_key] = value
            _apply_basic_to_hsv(color_key)

    def _build_lf_advanced(self, p):
        """Advanced mode: full color priority list + HSV ranges."""
        self._section_label(p, "Color Priority  (top = first)")
        self._priority_frame = tk.Frame(p, bg=Colors.BG_PANEL)
        self._priority_frame.pack(fill="x", padx=12, pady=(2, 8))
        self._rebuild_priority_list()

    _dot_colors = {
        "red": "#FF4444", "blue": "#4488FF", "green": "#44DD66",
        "orange": "#FF8833", "yellow": "#DDCC22", "purple": "#BB55FF",
    }

    _tk_expanded_color = None  # which color's HSV panel is expanded

    def _rebuild_priority_list(self):
        """Populate the priority list with toggle, up/down buttons, and expandable HSV tuning."""
        for w in self._priority_frame.winfo_children():
            w.destroy()
        enabled = state.lf_color_enabled or {}
        rank = 0
        for _i, key in enumerate(state.lf_color_priority):
            cdef = LINE_COLORS.get(key, {})
            accent = self._dot_colors.get(key, Colors.LIGHT_GRAY)
            is_active = (key == state.lf_active_color)
            is_on = enabled.get(key, True)
            if is_on:
                rank += 1

            row_bg = Colors.BG_ACTIVE if is_active else Colors.BG_SURFACE
            row = tk.Frame(self._priority_frame, bg=row_bg)
            row.pack(fill="x", pady=1)

            # Enable/disable toggle checkbox
            var = tk.BooleanVar(value=is_on)
            cb = tk.Checkbutton(row, variable=var, bg=row_bg, activebackground=row_bg,
                selectcolor=Colors.BG_DARK, bd=0, highlightthickness=0,
                command=lambda k=key, v=var: self._toggle_color_enabled(k, v.get()))
            cb.pack(side="left", padx=(4, 0))

            # Rank label (show — when disabled)
            rank_text = f"{rank}." if is_on else "\u2014"
            name_fg = Colors.WHITE if is_active else (Colors.LIGHT_GRAY if is_on else Colors.MID_GRAY)
            tk.Label(row, text=rank_text, bg=row_bg, fg=Colors.MID_GRAY,
                     font=self.font_small, width=2).pack(side="left")
            # Color dot
            dot = tk.Canvas(row, width=10, height=10, bg=row_bg, highlightthickness=0)
            dot.create_oval(1, 1, 9, 9, fill=accent if is_on else Colors.DARK_GRAY, outline="")
            dot.pack(side="left", padx=(2, 4))
            tk.Label(row, text=cdef.get("label", key), bg=row_bg,
                     fg=name_fg,
                     font=self.font_small).pack(side="left", expand=True, anchor="w")
            # Tune button (only shown when enabled)
            if is_on:
                tune_bg = Colors.BG_ACTIVE if self._tk_expanded_color == key else Colors.DARK_GRAY
                tk.Button(row, text="Tune", font=("Helvetica", 8, "bold"), width=4,
                    bg=tune_bg, fg=Colors.NEO_AMBER if self._tk_expanded_color == key else Colors.LIGHT_GRAY,
                    bd=0, relief="flat",
                    command=lambda k=key: self._toggle_hsv_expand(k)).pack(side="right", padx=1)
            # Arrow buttons
            btn_frame = tk.Frame(row, bg=row_bg)
            btn_frame.pack(side="right", padx=2)
            tk.Button(btn_frame, text="\u25B2", font=("Helvetica", 7), width=2,
                bg=Colors.DARK_GRAY, fg=Colors.LIGHT_GRAY, bd=0, relief="flat",
                command=lambda k=key: self._move_color(k, -1)).pack(side="left", padx=1)
            tk.Button(btn_frame, text="\u25BC", font=("Helvetica", 7), width=2,
                bg=Colors.DARK_GRAY, fg=Colors.LIGHT_GRAY, bd=0, relief="flat",
                command=lambda k=key: self._move_color(k, +1)).pack(side="left", padx=1)

            # Expandable HSV tuning panel (only when enabled)
            if is_on and self._tk_expanded_color == key and state.lf_color_hsv and key in state.lf_color_hsv:
                hsv = state.lf_color_hsv[key]
                panel = tk.Frame(self._priority_frame, bg=Colors.BG_PANEL)
                panel.pack(fill="x", padx=(20, 4), pady=(0, 2))
                for attr, label, lo, hi, color in [
                    ("H_low",  "H Low",  0, 179, Colors.NEO_RED),
                    ("H_high", "H High", 0, 179, Colors.NEO_RED),
                    ("S_low",  "S Low",  0, 255, Colors.GREEN),
                    ("S_high", "S High", 0, 255, Colors.GREEN),
                    ("V_low",  "V Low",  0, 255, Colors.CYAN),
                    ("V_high", "V High", 0, 255, Colors.CYAN),
                ]:
                    self._hsv_slider(panel, key, attr, label, lo, hi,
                                     hsv.get(attr, 0), color)

    def _hsv_slider(self, parent, color_key, attr, label, lo, hi, default, accent):
        """Create a compact HSV slider for a specific color."""
        frame = tk.Frame(parent, bg=Colors.BG_PANEL)
        frame.pack(fill="x", pady=1)
        row = tk.Frame(frame, bg=Colors.BG_PANEL)
        row.pack(fill="x")
        tk.Label(row, text=label, bg=Colors.BG_PANEL, fg=Colors.MID_GRAY,
                 font=("Helvetica", 9)).pack(side="left")
        val_lbl = tk.Label(row, text=str(int(default)), bg=Colors.BG_PANEL,
                 fg=accent, font=("Helvetica", 9))
        val_lbl.pack(side="right")
        var = tk.DoubleVar(value=default)
        scale = ttk.Scale(frame, from_=lo, to=hi, orient="horizontal", variable=var,
            style="Dark.Horizontal.TScale",
            command=lambda v, vl=val_lbl, ck=color_key, a=attr: (
                vl.config(text=str(int(float(v)))),
                self._set_color_hsv(ck, a, int(float(v)))))
        scale.pack(fill="x", pady=(1, 0))

    def _set_color_hsv(self, color_key, attr, value):
        """Update a per-color HSV value in state."""
        if state.lf_color_hsv and color_key in state.lf_color_hsv:
            state.lf_color_hsv[color_key][attr] = value

    def _toggle_color_enabled(self, key, enabled):
        """Toggle a color on or off in the priority list."""
        if state.lf_color_enabled is not None:
            state.lf_color_enabled[key] = enabled
            # Collapse HSV panel if disabling
            if not enabled and self._tk_expanded_color == key:
                self._tk_expanded_color = None
        self._rebuild_priority_list()

    def _toggle_hsv_expand(self, key):
        """Toggle expand/collapse of a color's HSV panel."""
        self._tk_expanded_color = None if self._tk_expanded_color == key else key
        self._rebuild_priority_list()

    def _move_color(self, key, direction):
        """Move a color up/down in the priority list and refresh the UI."""
        _move_color_priority(key, direction)
        self._rebuild_priority_list()

    def _build_wf_toolbar(self):
        p = self.toolbar_container
        self._section_label(p, "Wall Following")
        self._slider(p, "Speed %",              1, 100, state.wf_speed,
                     lambda v: setattr(state, "wf_speed", int(v)))
        self._slider(p, "Sensitivity %",        1, 100, state.wf_kp,
                     lambda v: setattr(state, "wf_kp", int(v)))
        self._section_label(p, "LIDAR Scan")
        self._slider(p, "Scan Dir \u00b0 (90=Side, 0=Fwd)", 0, 135, state.wf_scan_dir,
                     lambda v: setattr(state, "wf_scan_dir", int(v)))
        self._slider(p, "Window \u00b1\u00b0",           1, 60,  state.wf_window,
                     lambda v: setattr(state, "wf_window", int(v)))

        # Toggle: Closest Point vs Average Distance
        toggle_frame = tk.Frame(p, bg=Colors.BG_PANEL)
        toggle_frame.pack(fill="x", padx=12, pady=6)
        tk.Label(toggle_frame, text="Measurement Mode", bg=Colors.BG_PANEL,
                 fg=Colors.WHITE, font=self.font_label).pack(anchor="w")
        self._wf_avg_var = tk.BooleanVar(value=state.wf_use_avg)
        btn_row = tk.Frame(toggle_frame, bg=Colors.BG_PANEL)
        btn_row.pack(fill="x", pady=4)
        closest_btn = tk.Radiobutton(btn_row, text="Closest Point",
            variable=self._wf_avg_var, value=False,
            bg=Colors.BG_PANEL, fg=Colors.CYAN, selectcolor=Colors.BG_DARK,
            activebackground=Colors.BG_PANEL, activeforeground=Colors.CYAN,
            font=self.font_small, indicatoron=True,
            command=lambda: setattr(state, "wf_use_avg", False))
        closest_btn.pack(side="left", padx=(0, 12))
        avg_btn = tk.Radiobutton(btn_row, text="Average Distance",
            variable=self._wf_avg_var, value=True,
            bg=Colors.BG_PANEL, fg=Colors.CYAN, selectcolor=Colors.BG_DARK,
            activebackground=Colors.BG_PANEL, activeforeground=Colors.CYAN,
            font=self.font_small, indicatoron=True,
            command=lambda: setattr(state, "wf_use_avg", True))
        avg_btn.pack(side="left")

    def _build_ct_toolbar(self):
        p = self.toolbar_container
        if not PYCORAL_AVAILABLE:
            tk.Label(p, text="PyCoral not installed\nML model unavailable",
                bg=Colors.BG_PANEL, fg=Colors.NEO_RED,
                font=self.font_label, justify="center").pack(pady=12)
        self._section_label(p, "PID Controller")
        self._slider(p, "Speed %", 0, 100, state.ct_speed,
                     lambda v: setattr(state, "ct_speed", int(v)))
        self._slider(p, "Kp", 0, 50, state.ct_kp,
                     lambda v: setattr(state, "ct_kp", v),
                     accent=Colors.NEO_RED, fmt="float")
        self._slider(p, "Ki", 0, 10, state.ct_ki,
                     lambda v: setattr(state, "ct_ki", v),
                     accent=Colors.NEO_ORANGE, fmt="float")
        self._slider(p, "Kd", 0, 50, state.ct_kd,
                     lambda v: setattr(state, "ct_kd", v),
                     accent=Colors.YELLOW, fmt="float")
        self._section_label(p, "Detection")
        self._slider(p, "Confidence %", 10, 100,
                     int(state.ct_score_thresh * 100),
                     lambda v: setattr(state, "ct_score_thresh", v / 100))

    # ---- Periodic Refresh (~30 FPS) ----
    def _tick(self):
        try:
            self._render_canvas(self.cvs_raw,  state.raw_image,       "_tk_raw")
            self._render_canvas(self.cvs_proc, state.processed_image, "_tk_proc")
            self._refresh_telemetry()
            self._refresh_score()
        except Exception:
            pass
        self.root.after(33, self._tick)

    def _render_canvas(self, canvas, cv_img, attr):
        if cv_img is None:
            return
        try:
            cw = max(canvas.winfo_width(), 64)
            ch = max(canvas.winfo_height(), 48)
            img = cv_img
            if len(img.shape) == 2:
                img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            resized = cv.resize(img, (cw, ch))
            rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tk_img = ImageTk.PhotoImage(pil)
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=tk_img)
            setattr(self, attr, tk_img)
        except Exception:
            pass

    def _refresh_telemetry(self):
        # Read from cached state (never call rc.physics / rc.controller from
        # the GUI thread — cross-thread calls can corrupt the racecar protocol).
        a = state.imu_accel
        w = state.imu_gyro
        lt = state.ctrl_lt
        rt = state.ctrl_rt
        lj = state.ctrl_lj
        rj = state.ctrl_rj

        lines = [
            f"  \u2554\u2550\u2550 IMU \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557",
            f"  \u2551  Accel   X:{a[0]:+7.2f}  Y:{a[1]:+7.2f}  Z:{a[2]:+7.2f}  m/s\u00b2",
            f"  \u2551  Gyro    X:{w[0]:+7.2f}  Y:{w[1]:+7.2f}  Z:{w[2]:+7.2f}  rad/s",
            f"  \u2560\u2550\u2550 Controller Inputs \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563",
            f"  \u2551  L Trigger: {lt:.2f}     R Trigger: {rt:.2f}",
            f"  \u2551  L Stick: ({lj[0]:+.2f}, {lj[1]:+.2f})",
            f"  \u2551  R Stick: ({rj[0]:+.2f}, {rj[1]:+.2f})",
            f"  \u2560\u2550\u2550 Drive Output \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563",
            f"  \u2551  Speed: {state.speed:+.3f}    Angle: {state.angle:+.3f}",
            f"  \u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d",
        ]

        self.txt_telem.config(state="normal")
        self.txt_telem.delete("1.0", "end")
        self.txt_telem.insert("1.0", "\n".join(lines))
        self.txt_telem.config(state="disabled")

    def _refresh_score(self):
        score = calculate_score()
        state.score = score
        self.lbl_score.config(text=f"{score:.0f}")
        if score >= 70:
            color = Colors.SCORE_HIGH
        elif score >= 35:
            color = Colors.SCORE_MID
        else:
            color = Colors.SCORE_LOW
        self.lbl_score.config(fg=color)

        self.cvs_bar.delete("all")
        w = max(self.cvs_bar.winfo_width(), 1)
        h = max(self.cvs_bar.winfo_height(), 1)
        self.cvs_bar.create_rectangle(0, 0, w, h, fill=Colors.DARK_GRAY, outline="")
        bar_w = int(w * score / 100)
        if bar_w > 0:
            self.cvs_bar.create_rectangle(0, 0, bar_w, h, fill=color, outline="")

    def run(self):
        self.root.mainloop()


########################################################################################
# RACECAR Start / Update / Update Slow
########################################################################################

app = None


def start():
    """Called once when the RACECAR starts."""
    global app

    rc.drive.set_speed_angle(0, 0)
    state.ct_last_time = time.time()

    # Initialize per-color tunable HSV ranges from LINE_COLORS defaults
    state.lf_color_hsv = {}
    state.lf_color_enabled = {}
    state.lf_basic_hue = {}
    state.lf_basic_sv = {}
    for key, cdef in LINE_COLORS.items():
        lo = cdef["hsv_lo"]
        hi = cdef["hsv_hi"]
        state.lf_color_hsv[key] = {
            "H_low": lo[0], "H_high": hi[0],
            "S_low": lo[1], "S_high": hi[1],
            "V_low": lo[2], "V_high": hi[2],
        }
        state.lf_color_enabled[key] = True  # all enabled by default
        # Default standard-mode values: center of the default HSV range
        state.lf_basic_hue[key] = (lo[0] + hi[0]) // 2
        state.lf_basic_sv[key] = 50  # pure color (middle)

    # Try loading saved color config (overrides defaults above)
    _load_color_config()

    # If standard mode, apply basic picker values → HSV ranges
    if state.lf_sub_mode == "standard" and state.lf_basic_hue and state.lf_basic_sv:
        for key in state.lf_color_priority:
            _apply_basic_to_hsv(key)

    # Physics/IMU support: auto-detected on first update() frame.
    # If the runtime supports physics calls, IMU data will be live.
    # If not (e.g. older sim build), it gracefully falls back to zeros.

    # -- Try loading ML model for car tracking --
    if PYCORAL_AVAILABLE:
        model_search_paths = [
            os.path.join(os.path.dirname(__file__), "model"),
            os.path.expanduser("~/jupyter_ws/TPS/labs/model"),
        ]
        for base in model_search_paths:
            model_path = os.path.join(base, "machineVision.tflite")
            label_path = os.path.join(base, "labels.txt")
            if os.path.exists(model_path):
                try:
                    state.interpreter = make_interpreter(model_path)
                    state.interpreter.allocate_tensors()
                    state.labels = read_label_file(label_path)
                    state.inference_size = input_size(state.interpreter)
                    print(f">> ML model loaded from {base}")
                except Exception as e:
                    print(f">> ML model load failed: {e}")
                break
        else:
            print(">> ML model not found (car tracking will be disabled)")

    # -- Launch GUI --
    if HAS_TKINTER:
        gui_thread = threading.Thread(target=_launch_gui, daemon=True)
        gui_thread.start()
    elif HAS_DISPLAY:
        # OpenCV fallback: set up the trackbar tuner window
        _cv_setup_tuner(state.mode)
    else:
        # Headless: launch web dashboard in browser
        _start_web_dashboard()

    # Welcome banner
    if HAS_TKINTER:
        gui_type = "tkinter"
    elif HAS_DISPLAY:
        gui_type = "OpenCV"
    else:
        gui_type = "headless (terminal + controller)"
    print(
        "\n"
        "\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557\n"
        "\u2551        RACECAR Neo \u00b7 OneShot Autonomy Lab            \u2551\n"
        "\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563\n"
        "\u2551  Modes: Line Following \u00b7 Wall Following \u00b7 Car Track  \u2551\n"
        f"\u2551  GUI:   {gui_type:<47}\u2551\n"
        "\u2551  Drive: Hold RIGHT TRIGGER                             \u2551\n"
        "\u2551  Switch: A=Save/Line  B=Wall  Y=CarTrack               \u2551\n"
    )
    if not HAS_DISPLAY:
        print(
            "\u2551  Tune:  X=NextParam  LB=Decrease  RB=Increase         \u2551\n"
            "\u2551         Left Stick Y = Fine Adjust                     \u2551\n"
        )
    print(
        "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d\n"
    )


def _launch_gui():
    """Entry point for the tkinter GUI thread."""
    global app
    app = OneshotApp()
    app.run()


def update():
    """Called every frame (~60 FPS) by the racecar framework."""
    global _tune_index, _pending_mode_switch

    # -- Apply deferred mode switch from web dashboard (thread-safe) --
    if _pending_mode_switch is not None:
        _switch_mode(_pending_mode_switch, from_main_thread=True)
        _tune_index = 0
        _pending_mode_switch = None

    # -- Controller-based mode switching --
    # A button: save color config when in line_follow, otherwise switch to line_follow
    try:
        if rc.controller.was_pressed(rc.controller.Button.A):
            if state.mode == "line_follow":
                _save_color_config()
            else:
                _switch_mode("line_follow")
                _tune_index = 0
                if HAS_TKINTER and app is not None:
                    app.root.after(0, lambda: app._set_mode("line_follow"))
                elif HAS_DISPLAY:
                    _cv_setup_tuner("line_follow")
        elif rc.controller.was_pressed(rc.controller.Button.B):
            _switch_mode("wall_follow")
            _tune_index = 0
            if HAS_TKINTER and app is not None:
                app.root.after(0, lambda: app._set_mode("wall_follow"))
            elif HAS_DISPLAY:
                _cv_setup_tuner("wall_follow")
        elif rc.controller.was_pressed(rc.controller.Button.Y):
            _switch_mode("car_track")
            _tune_index = 0
            if HAS_TKINTER and app is not None:
                app.root.after(0, lambda: app._set_mode("car_track"))
            elif HAS_DISPLAY:
                _cv_setup_tuner("car_track")
    except Exception:
        pass

    # -- Cache controller state for web dashboard (main thread only!) --
    try:
        state.ctrl_lt = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
        state.ctrl_rt = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
        state.ctrl_lj = rc.controller.get_joystick(rc.controller.Joystick.LEFT)
        state.ctrl_rj = rc.controller.get_joystick(rc.controller.Joystick.RIGHT)
    except Exception:
        pass

    # -- Cache physics / IMU --
    # On first frame, probe whether rc.physics works (real car always has it
    # via ROS; sim may or may not support the physics protocol headers).
    # If the call succeeds, keep reading every frame.
    # If it fails, disable permanently to avoid protocol issues.
    if state._physics_ok is not False:          # None (untested) or True
        try:
            accel = rc.physics.get_linear_acceleration()
            gyro  = rc.physics.get_angular_velocity()
            state.imu_accel = (float(accel[0]), float(accel[1]), float(accel[2]))
            state.imu_gyro  = (float(gyro[0]),  float(gyro[1]),  float(gyro[2]))
            if state._physics_ok is None:
                state._physics_ok = True
                print(">> Physics/IMU: supported ✓")
        except Exception as e:
            if state._physics_ok is None:
                state._physics_ok = False
                print(f">> Physics/IMU: not available ({e})")
                print(">>   IMU telemetry will show zeros.")

    # -- Run current autonomy mode --
    try:
        if state.mode == "line_follow":
            update_line_follow()
        elif state.mode == "wall_follow":
            update_wall_follow()
        elif state.mode == "car_track":
            update_car_track()
    except Exception as e:
        print(f">> Mode update error: {e}")

    # -- Pre-encode images as JPEG for web MJPEG streams --
    # Done in the main thread so the HTTP threads never touch raw numpy arrays
    # or call cv.imencode (which is the main bottleneck in the old polling model).
    if not HAS_DISPLAY:
        try:
            _WEB_W, _WEB_H = 320, 240
            if state.raw_image is not None:
                small = cv.resize(state.raw_image, (_WEB_W, _WEB_H))
                _, buf = cv.imencode(".jpg", small, [cv.IMWRITE_JPEG_QUALITY, 55])
                state._jpeg_raw = buf.tobytes()
            if state.processed_image is not None:
                img = state.processed_image
                if len(img.shape) == 2:
                    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
                small = cv.resize(img, (_WEB_W, _WEB_H))
                _, buf = cv.imencode(".jpg", small, [cv.IMWRITE_JPEG_QUALITY, 55])
                state._jpeg_proc = buf.tobytes()
        except Exception:
            pass

    # -- Headless controller tuning (no display) --
    if not HAS_DISPLAY:
        _headless_tune_update()

    # -- OpenCV dashboard (when tkinter unavailable but display exists) --
    if not HAS_TKINTER and HAS_DISPLAY:
        _cv_update_display()


def update_slow():
    """Called once per second - updates dot matrix and score."""
    update_dot_matrix()
    if not HAS_DISPLAY:
        _headless_print_status()


########################################################################################
# Entry Point
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
