#!/usr/bin/env python3
"""
OAK-D Lite Focus & Camera Control Tool  (DepthAI v3)
──────────────────────────────────────────────────────
Elegant UI for testing focus modes, lens positions,
exposure, ISO, and white balance on the OAK-D Lite.

Ported to the DepthAI v3 API:
  • ColorCamera/XLinkIn/XLinkOut -> unified Camera node
  • device.getOutputQueue()/getInputQueue() -> node.createOutputQueue()/createInputQueue()
  • with dai.Device(pipeline) -> with dai.Pipeline() as pipeline: ... pipeline.start()
  • main loop keyed off pipeline.isRunning() instead of an infinite while True

FPS / latency fixes applied:
  • ISP output requested as NV12 (native, half the USB bandwidth of BGR888i);
    BGR conversion happens cheaply on the host via getCvFrame()
  • Output queue maxSize=1, blocking=False -> always the newest frame, no backlog
  • USB speed is printed at startup with a loud warning if running at USB2
    (the most common cause of ~5-6 fps on the OAK-D Lite)
  • Host polling loop no longer busy-spins the CPU while waiting on a frame

Controls:
  Focus Modes ─────────────────────────
    1  AUTO          (focus-once & lock)
    2  CONTINUOUS    (smooth video AF)
    3  MACRO         (close-up AF)
    4  MANUAL        (use slider / +/-)
    0  AF OFF        (freeze current)

  Manual Focus (when in MANUAL mode) ──
    +/-  or  =/−     Step lens ±5
    [ / ]            Step lens ±1

  Exposure ────────────────────────────
    e    Toggle auto/manual exposure
    w/s  Adjust exposure time ±500µs
    a/d  Adjust ISO ±50

  White Balance ───────────────────────
    b    Toggle auto/manual WB
    z/x  Adjust color temp ±200K

  General ─────────────────────────────
    f    Trigger single AF (in AUTO)
    r    Reset all to defaults
    p    Screenshot
    h    Toggle help overlay
    q    Quit

Usage:
    python oakd_focus_testing_v3.py
    python oakd_focus_testing_v3.py --resolution 4k
    python oakd_focus_testing_v3.py --resolution 1080p --fps 30
"""

import argparse
import time
import sys
from dataclasses import dataclass

try:
    import depthai as dai
except ImportError:
    print("❌ depthai not installed. Run: pip install depthai --pre")
    sys.exit(1)

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

FOCUS_MODES = {
    "OFF":        dai.CameraControl.AutoFocusMode.OFF,
    "AUTO":       dai.CameraControl.AutoFocusMode.AUTO,
    "MACRO":      dai.CameraControl.AutoFocusMode.MACRO,
    "CONT_VIDEO": dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO,
    "CONT_PIC":   dai.CameraControl.AutoFocusMode.CONTINUOUS_PICTURE,
    "EDOF":       dai.CameraControl.AutoFocusMode.EDOF,
}

# In v3, Camera.requestOutput() takes a plain (width, height) pixel size
# rather than a SensorResolution enum, so we map straight to resolutions.
RESOLUTIONS = {
    "1080p": (1920, 1080),
    "4k":    (3840, 2160),
    "12mp":  (4032, 3040),   # OAK-D Lite IMX214 max still resolution
}


@dataclass
class CameraState:
    """Tracks all current camera settings for display."""
    focus_mode: str = "CONT_VIDEO"
    lens_pos: int = 130             # 0-255, ~130 = infinity
    auto_exposure: bool = True
    exposure_us: int = 20000        # microseconds
    iso: int = 400                  # 100-1600
    auto_wb: bool = True
    wb_temp: int = 4500             # Kelvin
    show_help: bool = True
    screenshot_idx: int = 0
    # Read-back from frame metadata
    meta_exposure: int = 0
    meta_iso: int = 0
    meta_lens_pos: int = 0
    meta_wb_temp: int = 0
    fps: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# UI Drawing  (unchanged from v2 — pure OpenCV, no DepthAI API involved)
# ─────────────────────────────────────────────────────────────────────────────

# Color palette (BGR)
C_BG      = (20, 20, 20)
C_PANEL   = (30, 30, 35)
C_ACCENT  = (235, 160, 50)    # warm amber
C_TEXT    = (220, 220, 220)
C_DIM     = (120, 120, 130)
C_GREEN   = (80, 220, 120)
C_RED     = (80, 80, 240)
C_CYAN    = (220, 200, 60)
C_WHITE   = (255, 255, 255)


def draw_rounded_rect(img, pt1, pt2, color, radius=8, thickness=-1, alpha=0.75):
    """Draw a rounded rectangle with optional transparency."""
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_pill_badge(img, text, x, y, bg_color, text_color=C_WHITE, font_scale=0.5):
    """Draw a small pill-shaped badge."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    pad_x, pad_y = 10, 5
    cv2.rectangle(img, (x, y - th - pad_y), (x + tw + pad_x * 2, y + pad_y),
                  bg_color, -1, cv2.LINE_AA)
    r = (th + pad_y * 2) // 2
    cv2.circle(img, (x + r, y - th // 2), r, bg_color, -1, cv2.LINE_AA)
    cv2.circle(img, (x + tw + pad_x * 2 - r, y - th // 2), r, bg_color, -1, cv2.LINE_AA)
    cv2.putText(img, text, (x + pad_x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, text_color, 1, cv2.LINE_AA)


def draw_bar(img, x, y, w, h, value, max_val, bar_color, bg_color=C_PANEL):
    """Draw a horizontal progress bar."""
    cv2.rectangle(img, (x, y), (x + w, y + h), bg_color, -1)
    fill_w = int(w * min(value / max_val, 1.0))
    if fill_w > 0:
        cv2.rectangle(img, (x, y), (x + fill_w, y + h), bar_color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), C_DIM, 1)


def draw_status_panel(img, state: CameraState):
    """Draw the elegant status panel on the left side."""
    h, w = img.shape[:2]
    panel_w = 310
    panel_h = 320
    px, py = 12, 12

    draw_rounded_rect(img, (px, py), (px + panel_w, py + panel_h), C_BG, alpha=0.82)

    y = py + 28
    cv2.putText(img, "OAK-D Lite", (px + 14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_ACCENT, 2, cv2.LINE_AA)
    cv2.putText(img, "Focus Lab", (px + 145, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_TEXT, 1, cv2.LINE_AA)

    y += 12
    cv2.line(img, (px + 14, y), (px + panel_w - 14, y), C_DIM, 1)

    y += 24
    cv2.putText(img, "FOCUS", (px + 14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_DIM, 1, cv2.LINE_AA)

    mode_color = C_GREEN if state.focus_mode != "OFF" else C_RED
    draw_pill_badge(img, state.focus_mode, px + 80, y, mode_color, font_scale=0.45)

    y += 24
    cv2.putText(img, "Lens", (px + 14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_DIM, 1, cv2.LINE_AA)
    draw_bar(img, px + 65, y - 10, 180, 14, state.meta_lens_pos, 255, C_ACCENT)
    cv2.putText(img, f"{state.meta_lens_pos}", (px + 252, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_TEXT, 1, cv2.LINE_AA)

    if state.focus_mode == "OFF":
        y += 18
        cv2.putText(img, f"Target: {state.lens_pos}", (px + 65, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_CYAN, 1, cv2.LINE_AA)

    y += 28
    cv2.line(img, (px + 14, y - 10), (px + panel_w - 14, y - 10), (40, 40, 45), 1)
    cv2.putText(img, "EXPOSURE", (px + 14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_DIM, 1, cv2.LINE_AA)
    exp_badge = "AUTO" if state.auto_exposure else "MANUAL"
    badge_col = C_GREEN if state.auto_exposure else C_CYAN
    draw_pill_badge(img, exp_badge, px + 115, y, badge_col, font_scale=0.40)

    y += 22
    exp_val = state.meta_exposure if state.meta_exposure else state.exposure_us
    cv2.putText(img, f"Time: {exp_val}us", (px + 14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_TEXT, 1, cv2.LINE_AA)
    iso_val = state.meta_iso if state.meta_iso else state.iso
    cv2.putText(img, f"ISO: {iso_val}", (px + 170, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_TEXT, 1, cv2.LINE_AA)

    y += 28
    cv2.line(img, (px + 14, y - 10), (px + panel_w - 14, y - 10), (40, 40, 45), 1)
    cv2.putText(img, "WHITE BAL", (px + 14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_DIM, 1, cv2.LINE_AA)
    wb_badge = "AUTO" if state.auto_wb else "MANUAL"
    wb_col = C_GREEN if state.auto_wb else C_CYAN
    draw_pill_badge(img, wb_badge, px + 125, y, wb_col, font_scale=0.40)

    y += 22
    wb_val = state.meta_wb_temp if state.meta_wb_temp else state.wb_temp
    cv2.putText(img, f"Temp: {wb_val}K", (px + 14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_TEXT, 1, cv2.LINE_AA)

    y += 28
    cv2.line(img, (px + 14, y - 10), (px + panel_w - 14, y - 10), (40, 40, 45), 1)
    cv2.putText(img, f"FPS: {state.fps:.1f}", (px + 14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GREEN, 1, cv2.LINE_AA)


def draw_help_overlay(img, state: CameraState):
    """Draw condensed keyboard shortcuts on the right side."""
    if not state.show_help:
        return

    h, w = img.shape[:2]
    panel_w = 255
    panel_h = 295
    px = w - panel_w - 12
    py = 12

    draw_rounded_rect(img, (px, py), (px + panel_w, py + panel_h), C_BG, alpha=0.82)

    y = py + 24
    cv2.putText(img, "Keyboard Shortcuts", (px + 14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, C_ACCENT, 1, cv2.LINE_AA)

    shortcuts = [
        ("Focus Modes", None),
        ("1", "Auto (once)"),
        ("2", "Continuous Video"),
        ("3", "Macro (close-up)"),
        ("4", "Manual"),
        ("0", "AF Off"),
        ("", ""),
        ("Manual Focus", None),
        ("+/-", "Lens ±5"),
        ("[ / ]", "Lens ±1"),
        ("", ""),
        ("Exposure", None),
        ("e", "Toggle auto/manual"),
        ("w/s", "Time ±500µs"),
        ("a/d", "ISO ±50"),
        ("", ""),
        ("b", "Toggle auto WB"),
        ("f", "Trigger AF"),
        ("r", "Reset all"),
        ("p", "Screenshot"),
        ("h", "Toggle help"),
        ("q", "Quit"),
    ]

    y += 8
    for key, desc in shortcuts:
        y += 16
        if y > py + panel_h - 8:
            break
        if desc is None:
            cv2.putText(img, key, (px + 14, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_DIM, 1, cv2.LINE_AA)
            continue
        if not key:
            y -= 8
            continue
        cv2.putText(img, key, (px + 14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(img, desc, (px + 70, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_TEXT, 1, cv2.LINE_AA)


def draw_focus_ring(img, state: CameraState):
    """Draw a subtle focus ring indicator at bottom center."""
    h, w = img.shape[:2]
    cx, cy = w // 2, h - 40
    radius = 18

    if state.focus_mode in ("CONT_VIDEO", "CONT_PIC", "AUTO"):
        color = C_GREEN
        label = "AF"
    elif state.focus_mode == "OFF":
        color = C_CYAN
        label = "MF"
    elif state.focus_mode == "MACRO":
        color = (0, 180, 255)
        label = "MC"
    else:
        color = C_DIM
        label = state.focus_mode[:2]

    cv2.circle(img, (cx, cy), radius + 2, color, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), radius - 4, color, 1, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(img, label, (cx - tw // 2, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Camera Control Helpers
#
# CameraControl message API itself is unchanged between v2 and v3 — what
# changes is *how* you get the queue you send() it on (see main()).
# ─────────────────────────────────────────────────────────────────────────────

def send_focus_mode(q, state: CameraState):
    ctrl = dai.CameraControl()
    mode = FOCUS_MODES[state.focus_mode]
    ctrl.setAutoFocusMode(mode)
    if state.focus_mode == "OFF":
        ctrl.setManualFocus(state.lens_pos)
    q.send(ctrl)


def send_manual_focus(q, state: CameraState):
    ctrl = dai.CameraControl()
    ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
    ctrl.setManualFocus(state.lens_pos)
    q.send(ctrl)


def send_exposure(q, state: CameraState):
    ctrl = dai.CameraControl()
    if state.auto_exposure:
        ctrl.setAutoExposureEnable()
    else:
        ctrl.setManualExposure(state.exposure_us, state.iso)
    q.send(ctrl)


def send_white_balance(q, state: CameraState):
    ctrl = dai.CameraControl()
    if state.auto_wb:
        ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
    else:
        ctrl.setManualWhiteBalance(state.wb_temp)
    q.send(ctrl)


def send_af_trigger(q):
    ctrl = dai.CameraControl()
    ctrl.setAutoFocusTrigger()
    q.send(ctrl)


def reset_all(q, state: CameraState):
    state.focus_mode = "CONT_VIDEO"
    state.lens_pos = 130
    state.auto_exposure = True
    state.auto_wb = True
    state.exposure_us = 20000
    state.iso = 400
    state.wb_temp = 4500

    ctrl = dai.CameraControl()
    ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
    ctrl.setAutoExposureEnable()
    ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
    q.send(ctrl)


# ─────────────────────────────────────────────────────────────────────────────
# Trackbar callback (for manual focus slider)
# ─────────────────────────────────────────────────────────────────────────────

_trackbar_state = {"q": None, "state": None}

def on_focus_trackbar(val):
    s = _trackbar_state["state"]
    q = _trackbar_state["q"]
    if s and q:
        s.lens_pos = val
        s.focus_mode = "OFF"
        send_manual_focus(q, s)


# ─────────────────────────────────────────────────────────────────────────────
# FPS
# ─────────────────────────────────────────────────────────────────────────────

class FPSCounter:
    def __init__(self, n=30):
        self.times = []
        self.n = n
        self.last = time.time()

    def tick(self) -> float:
        now = time.time()
        self.times.append(now - self.last)
        self.last = now
        if len(self.times) > self.n:
            self.times.pop(0)
        return len(self.times) / sum(self.times) if self.times else 0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OAK-D Lite Focus Testing Tool (DepthAI v3)")
    parser.add_argument("--resolution", choices=["1080p", "4k", "12mp"],
                        default="1080p", help="Camera resolution (default: 1080p)")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS (default: 30)")
    args = parser.parse_args()

    state = CameraState()
    fps_counter = FPSCounter()

    print("─" * 50)
    print("  OAK-D Lite Focus Lab  (DepthAI v3)")
    print(f"  Resolution: {args.resolution}  |  FPS: {args.fps}")
    print("─" * 50)
    print("  Press 'h' to toggle help overlay")
    print("  Press 'q' to quit")
    print("─" * 50)

    size = RESOLUTIONS.get(args.resolution, RESOLUTIONS["1080p"])

    WIN_NAME = "OAK-D Lite Focus Lab"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 1280, 720)
    cv2.createTrackbar("Focus Lens", WIN_NAME, state.lens_pos, 255, on_focus_trackbar)

    # ── v3: pipeline owns/creates the device via the context manager ──────────
    with dai.Pipeline() as pipeline:
        # Unified Camera node replaces ColorCamera. .build() connects it to a
        # physical sensor on the given socket and auto-detects the device.
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

        # Set an initial AF mode before the pipeline starts (equivalent of
        # v2's cam.initialControl.setAutoFocusMode(...)).
        cam.initialControl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)

        # Control input: replaces the old XLinkIn("control") node + inputControl link.
        q_ctrl = cam.inputControl.createInputQueue()

        # Video/ISP output: replaces XLinkOut("isp") + cam.isp.link(...).
        # NV12 is the ISP's native output format (1.5 bytes/pixel) — it's cheaper
        # to produce on-device AND half the USB bandwidth of BGR888i (3 bytes/pixel).
        # getCvFrame() converts NV12 -> BGR on the HOST using OpenCV, which is fast
        # and keeps the device-to-host USB link from being the bottleneck.
        # maxSize=1 + blocking=False means we always grab the newest frame and
        # drop stale ones instead of letting a backlog build up (the usual cause
        # of latency creeping up the longer the app runs).
        q_isp = cam.requestOutput(
            size,
            type=dai.ImgFrame.Type.NV12,
            fps=args.fps,
        ).createOutputQueue(maxSize=1, blocking=False)

        # Wire up trackbar state
        _trackbar_state["q"] = q_ctrl
        _trackbar_state["state"] = state

        pipeline.start()

        # Optional device info (best-effort; API surface for this varies by
        # v3 point release, so this is wrapped defensively).
        try:
            device = pipeline.getDefaultDevice()
            print(f"✅ Connected to: {device.getDeviceInfo().getMxId()}")
            for cam_info in device.getConnectedCameraFeatures():
                has_af = getattr(cam_info, "hasAutofocus", "unknown")
                print(f"   Camera {cam_info.socket.name}: AF={has_af}")

            # ── USB speed diagnostic ────────────────────────────────────
            # This is the #1 cause of ~5-6 fps on OAK-D Lite: the device has
            # silently negotiated USB2 ("HIGH") instead of USB3 ("SUPER" /
            # "SUPER_PLUS"). USB2 bandwidth is roughly 1/10th of USB3, which
            # will bottleneck any resolution above a small preview.
            usb_speed = device.getUsbSpeed()
            print(f"   USB speed: {usb_speed.name}")
            if usb_speed.name in ("LOW", "FULL", "HIGH"):
                print("   ⚠️  Running at USB2 speed! This alone explains ~5-6 fps.")
                print("       Try: a short, good-quality USB3 cable, a direct")
                print("       port (no hub), and a different USB3 port on the host.")
        except Exception:
            print("✅ Pipeline started.")

        while pipeline.isRunning():
            isp_msg = q_isp.tryGet()
            if isp_msg is None:
                # Keep the UI responsive even with no new frame yet, without
                # busy-spinning the host CPU (which competes with the frame
                # decode/draw work and can itself shave a few fps off).
                key = cv2.waitKey(5) & 0xFF
                if key == ord("q"):
                    pipeline.stop()
                    break
                continue

            frame = isp_msg.getCvFrame()

            # ── Read metadata ────────────────────────────────────────────
            try:
                state.meta_exposure = isp_msg.getExposureTime().total_seconds() * 1e6
                state.meta_iso = isp_msg.getSensitivity()
                state.meta_lens_pos = isp_msg.getLensPosition()
                state.meta_wb_temp = isp_msg.getColorTemperature()
            except Exception:
                pass

            state.fps = fps_counter.tick()

            # ── Draw UI ──────────────────────────────────────────────────
            display = frame.copy()
            draw_status_panel(display, state)
            draw_help_overlay(display, state)
            draw_focus_ring(display, state)

            cv2.imshow(WIN_NAME, display)

            # ── Sync trackbar ────────────────────────────────────────────
            tb_val = cv2.getTrackbarPos("Focus Lens", WIN_NAME)
            if tb_val != state.lens_pos and state.focus_mode == "OFF":
                state.lens_pos = tb_val
                send_manual_focus(q_ctrl, state)

            # ── Handle keys ──────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                pipeline.stop()
                break

            # ── Focus mode keys ──────────────────────────────────────────
            elif key == ord("1"):
                state.focus_mode = "AUTO"
                send_focus_mode(q_ctrl, state)
                print("  → Focus: AUTO")

            elif key == ord("2"):
                state.focus_mode = "CONT_VIDEO"
                send_focus_mode(q_ctrl, state)
                print("  → Focus: CONTINUOUS VIDEO")

            elif key == ord("3"):
                state.focus_mode = "MACRO"
                send_focus_mode(q_ctrl, state)
                print("  → Focus: MACRO")

            elif key == ord("4"):
                state.focus_mode = "OFF"
                send_focus_mode(q_ctrl, state)
                cv2.setTrackbarPos("Focus Lens", WIN_NAME, state.lens_pos)
                print(f"  → Focus: MANUAL (lens={state.lens_pos})")

            elif key == ord("0"):
                state.focus_mode = "OFF"
                ctrl = dai.CameraControl()
                ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
                q_ctrl.send(ctrl)
                print("  → Focus: OFF (frozen)")

            # ── Manual focus adjustment ──────────────────────────────────
            elif key in (ord("+"), ord("=")):
                state.lens_pos = min(255, state.lens_pos + 5)
                state.focus_mode = "OFF"
                send_manual_focus(q_ctrl, state)
                cv2.setTrackbarPos("Focus Lens", WIN_NAME, state.lens_pos)
                print(f"  → Lens: {state.lens_pos}")

            elif key == ord("-"):
                state.lens_pos = max(0, state.lens_pos - 5)
                state.focus_mode = "OFF"
                send_manual_focus(q_ctrl, state)
                cv2.setTrackbarPos("Focus Lens", WIN_NAME, state.lens_pos)
                print(f"  → Lens: {state.lens_pos}")

            elif key == ord("]"):
                state.lens_pos = min(255, state.lens_pos + 1)
                state.focus_mode = "OFF"
                send_manual_focus(q_ctrl, state)
                cv2.setTrackbarPos("Focus Lens", WIN_NAME, state.lens_pos)

            elif key == ord("["):
                state.lens_pos = max(0, state.lens_pos - 1)
                state.focus_mode = "OFF"
                send_manual_focus(q_ctrl, state)
                cv2.setTrackbarPos("Focus Lens", WIN_NAME, state.lens_pos)

            # ── Exposure ─────────────────────────────────────────────────
            elif key == ord("e"):
                state.auto_exposure = not state.auto_exposure
                send_exposure(q_ctrl, state)
                mode_str = "AUTO" if state.auto_exposure else "MANUAL"
                print(f"  → Exposure: {mode_str}")

            elif key == ord("w"):
                if not state.auto_exposure:
                    state.exposure_us = min(33000, state.exposure_us + 500)
                    send_exposure(q_ctrl, state)
                    print(f"  → Exposure time: {state.exposure_us}µs")

            elif key == ord("s"):
                if not state.auto_exposure:
                    state.exposure_us = max(100, state.exposure_us - 500)
                    send_exposure(q_ctrl, state)
                    print(f"  → Exposure time: {state.exposure_us}µs")

            elif key == ord("d"):
                if not state.auto_exposure:
                    state.iso = min(1600, state.iso + 50)
                    send_exposure(q_ctrl, state)
                    print(f"  → ISO: {state.iso}")

            elif key == ord("a"):
                if not state.auto_exposure:
                    state.iso = max(100, state.iso - 50)
                    send_exposure(q_ctrl, state)
                    print(f"  → ISO: {state.iso}")

            # ── White balance ────────────────────────────────────────────
            elif key == ord("b"):
                state.auto_wb = not state.auto_wb
                send_white_balance(q_ctrl, state)
                mode_str = "AUTO" if state.auto_wb else "MANUAL"
                print(f"  → White Balance: {mode_str}")

            elif key == ord("x"):
                if not state.auto_wb:
                    state.wb_temp = min(12000, state.wb_temp + 200)
                    send_white_balance(q_ctrl, state)
                    print(f"  → WB Temp: {state.wb_temp}K")

            elif key == ord("z"):
                if not state.auto_wb:
                    state.wb_temp = max(1000, state.wb_temp - 200)
                    send_white_balance(q_ctrl, state)
                    print(f"  → WB Temp: {state.wb_temp}K")

            # ── General ──────────────────────────────────────────────────
            elif key == ord("f"):
                send_af_trigger(q_ctrl)
                print("  → AF trigger sent")

            elif key == ord("r"):
                reset_all(q_ctrl, state)
                cv2.setTrackbarPos("Focus Lens", WIN_NAME, state.lens_pos)
                print("  → Reset to defaults")

            elif key == ord("p"):
                fname = f"oakd_screenshot_{state.screenshot_idx:04d}.png"
                cv2.imwrite(fname, frame)  # Save raw frame without UI
                state.screenshot_idx += 1
                print(f"  📸 Saved: {fname}")

            elif key == ord("h"):
                state.show_help = not state.show_help

    cv2.destroyAllWindows()
    print("👋 Done.")


if __name__ == "__main__":
    main()