#!/usr/bin/env python3
"""
EKF Localization Node with automatic initialization and kidnapping recovery
=============================================================================
States (1:1 mit ekf_extended):
  GLOBAL_LOC  - initial or repeated localization by scoring pose hypotheses
  WAITING     - after LOST: odometry only, scan updates disabled
  TRACKING    - normal EKF operation

Inputs:
  /initialpose  - optional manual override from RViz or AMCL
  /odom         - odometry
  /scan         - LiDAR

Output:
  map -> odom TF, /ekf/beams, /map

Usage:
  python3 robot/ekf_live.py <path/to/map.yaml> --bag-id lab/fast
"""

import argparse
from bisect import bisect_right
import json
import time
import threading
import yaml
from dataclasses import asdict, dataclass
from datetime import datetime
import numpy as np
from pathlib import Path
from PIL import Image

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import Odometry, OccupancyGrid, Path as RosPath
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import (
    TransformStamped, Point, PoseStamped, PoseWithCovarianceStamped,
)
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster

@dataclass(frozen=True)
class LiveConfig:
    n_beams: int = 30
    max_range: float = 3.5
    sigma_qd: float = 0.05
    sigma_qth: float = 0.0893
    sigma_r: float = 0.06
    eps_h: float = 0.035
    gate: float = 0.157
    sigma_p0: float = 0.05
    lost_valid_ratio: float = 0.25
    lost_consec_steps: int = 5
    p_trace_max: float = 5.0
    recovery_wait_s: float = 1.0
    grid_spacing: float = 0.40
    n_theta: int = 16
    free_thresh: int = 220
    recovery_p_init: float = 0.3
    recovery_gate: float = 0.5
    score_n_eval: int = 30
    score_mu: float = 0.5


CFG = LiveConfig()


# Map loading
def load_map(yaml_path: Path):
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    pgm_path = yaml_path.parent / meta['image']
    img = np.array(Image.open(pgm_path))
    occ_thresh = float(meta.get('occupied_thresh', 0.65))
    occ_mask = img < int((1.0 - occ_thresh) * 255)
    return img, occ_mask, float(meta['resolution']), meta['origin'][:2]


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 Hilfsfunktionen \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def quat_to_yaw(qx, qy, qz, qw):
    return np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))


def yaw_to_quat(yaw):
    return 0.0, 0.0, np.sin(yaw/2.0), np.cos(yaw/2.0)


def world_to_arr(x, y, origin, res, map_h):
    col = int((x - origin[0]) / res)
    row = map_h - 1 - int((y - origin[1]) / res)
    return col, row


def raycast(x, y, ang, occ_mask, origin, res, max_range=3.5):
    map_h, map_w = occ_mask.shape
    step = res * 0.5
    cx, cy = np.cos(ang), np.sin(ang)
    n = int(max_range / step)
    for k in range(1, n + 1):
        d = k * step
        col, row = world_to_arr(x + cx*d, y + cy*d, origin, res, map_h)
        if col < 0 or col >= map_w or row < 0 or row >= map_h:
            return d
        if occ_mask[row, col]:
            return d
    return max_range


def expected_scan(pose, beam_angles, occ_mask, origin, res, max_range=3.5):
    x, y, th = pose
    return np.array([raycast(x, y, th+a, occ_mask, origin, res, max_range)
                     for a in beam_angles])


def compute_H_num(pose, beam_angles, occ_mask, origin, res, max_range, eps):
    H = np.zeros((len(beam_angles), 3))
    for col, dp in enumerate([
        np.array([eps, 0.0, 0.0]),
        np.array([0.0, eps, 0.0]),
        np.array([0.0, 0.0, eps]),
    ]):
        H[:, col] = (expected_scan(pose+dp, beam_angles, occ_mask, origin, res, max_range) -
                     expected_scan(pose-dp, beam_angles, occ_mask, origin, res, max_range)) / (2*eps)
    return H


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 EKF Node \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
class EKFLocalizationNode(Node):

    # \u2500\u2500 EKF Parameter \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    N_BEAMS   = CFG.n_beams
    MAX_RANGE = CFG.max_range
    SIGMA_QD  = CFG.sigma_qd
    SIGMA_QTH = CFG.sigma_qth
    SIGMA_R   = CFG.sigma_r
    EPS_H     = CFG.eps_h
    GATE      = CFG.gate
    SIGMA_P0  = CFG.sigma_p0

    # \u2500\u2500 Kidnapping/Lost-Detection (1:1 mit ekf_extended) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    LOST_VALID_RATIO  = CFG.lost_valid_ratio
    LOST_CONSEC_STEPS = CFG.lost_consec_steps
    P_TRACE_MAX       = CFG.p_trace_max
    RECOVERY_WAIT_S   = CFG.recovery_wait_s

    # \u2500\u2500 Globale Lokalisierung (1:1 mit ekf_extended) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    GRID_SPACING    = CFG.grid_spacing
    N_THETA         = CFG.n_theta
    FREE_THRESH     = CFG.free_thresh
    RECOVERY_P_INIT = CFG.recovery_p_init
    RECOVERY_GATE   = CFG.recovery_gate
    SCORE_N_EVAL    = CFG.score_n_eval
    SCORE_MU        = CFG.score_mu

    # \u2500\u2500 Zust\u00e4nde \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    TRACKING   = 'tracking'
    WAITING    = 'waiting'
    GLOBAL_LOC = 'global_loc'

    def __init__(self, map_yaml: str, bag_id: str, run_root: Path, init_pose=None):
        super().__init__('ekf_localization')

        yaml_path = Path(map_yaml)
        self.map_yaml = yaml_path
        self.bag_id = bag_id
        safe_bag_id = bag_id.replace("/", "_").replace(" ", "_")
        self.run_dir = run_root / f"{safe_bag_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._records = []
        self._events = []
        self._record_lock = threading.Lock()
        self._finalized = False
        self._last_recorded_time = None
        self._loc_thread = None
        self.map_img, self.occ_mask, self.res, self.origin = load_map(yaml_path)
        self.get_logger().info(f"Map loaded: {yaml_path.name} "
                               f"({self.occ_mask.shape[1]}x{self.occ_mask.shape[0]} px)")

        # Vorberechnete Raycast-Abst\u00e4nde f\u00fcr vektorisierten expected_scan
        step = self.res * 0.5
        n_steps = int(self.MAX_RANGE / step)
        self._ray_dists = np.arange(1, n_steps + 1, dtype=float) * step

        self.tf_broadcaster = TransformBroadcaster(self)

        map_qos = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', map_qos)
        self.map_msg = self._build_map_msg(self.map_img)
        self.create_timer(1.0, self._publish_map)

        self.beam_pub = self.create_publisher(MarkerArray, '/ekf/beams', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/ekf/pose', 10)
        self.path_pub = self.create_publisher(RosPath, '/ekf/path', 10)
        self.mode_pub = self.create_publisher(String, '/ekf/mode', 10)
        self.path_msg = RosPath()
        self.path_msg.header.frame_id = 'map'

        # \u2500\u2500 EKF Zustand \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        self.x_ekf       = None
        self.P           = None
        self.Q           = np.diag([self.SIGMA_QD**2, self.SIGMA_QD**2, self.SIGMA_QTH**2])
        self.I3          = np.eye(3)
        self.last_odom   = None
        self.latest_odom = None
        self.beam_angles = None
        self._odom_stamps = []
        self._odom_poses = []
        self._pending_scans = []

        # \u2500\u2500 Modus & Zustandsvariablen \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        self._mode        = self.GLOBAL_LOC
        self._loc_buffer  = []
        self._loc_running = False
        self._consec_bad  = 0
        self._wait_start  = None
        self._lost_time   = None
        self._node_start  = time.time()
        self._lock        = threading.Lock()

        log_path = self.run_dir / "ekf.log"
        self._logfile = open(log_path, 'w', buffering=1)
        self._logfile.write(f"# EKF kidnapping log started {datetime.now().isoformat()}\n")
        self._logfile.write(f"# Map: {yaml_path.name}\n")
        self._logfile.write(f"# Format: ISO-Zeit | t_seit_start | Typ | Details\n\n")
        self.get_logger().info(f"Kidnapping-Log: {log_path}")

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_cb, qos_profile_sensor_data)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self.initialpose_cb, 10)

        self.odom_count    = 0
        self.scan_count    = 0
        self.last_log_time = time.time()
        self.scan_durations = []
        self.create_timer(2.0, self._log_status)

        if init_pose is not None:
            self.x_ekf = np.asarray(init_pose, dtype=float)
            self.P = np.diag([self.SIGMA_P0**2] * 3)
            self._mode = self.TRACKING
            self._events.append({
                "time": None,
                "type": "initialpose",
                "pose": self.x_ekf.tolist(),
                "source": "start_poses.json",
            })
            self.get_logger().info(
                f"Start pose from start_poses.json: {self.x_ekf.tolist()}")

        self.get_logger().info(
            "EKF node started; global localization begins when data arrives ...")

    # \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 Map \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _build_map_msg(self, img: np.ndarray) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.info.resolution = self.res
        msg.info.width  = img.shape[1]
        msg.info.height = img.shape[0]
        msg.info.origin.position.x = float(self.origin[0])
        msg.info.origin.position.y = float(self.origin[1])
        occ = np.full(img.shape, -1, dtype=np.int8)
        occ[img > 200] = 0
        occ[img < 50]  = 100
        msg.data = np.flipud(occ).flatten().tolist()
        return msg

    def _publish_map(self):
        self.map_msg.header.stamp = self.get_clock().now().to_msg()
        self.map_pub.publish(self.map_msg)

    # \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 Vektorisierter Raycast \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _expected_scan_fast(self, pose, beam_angles: np.ndarray) -> np.ndarray:
        """Raycast f\u00fcr alle Beams gleichzeitig (numpy-vektorisiert)."""
        x, y, th = pose
        aa   = th + beam_angles
        cx   = np.cos(aa)[:, None]
        cy   = np.sin(aa)[:, None]
        px   = x + cx * self._ray_dists[None, :]
        py   = y + cy * self._ray_dists[None, :]
        map_h, map_w = self.occ_mask.shape
        cols = ((px - self.origin[0]) / self.res).astype(np.int32)
        rows = (map_h - 1 - (py - self.origin[1]) / self.res).astype(np.int32)
        oob  = (cols < 0) | (cols >= map_w) | (rows < 0) | (rows >= map_h)
        cols = np.clip(cols, 0, map_w - 1)
        rows = np.clip(rows, 0, map_h - 1)
        hit  = self.occ_mask[rows, cols] | oob
        first = np.argmax(hit, axis=1)
        return np.where(hit.any(axis=1), self._ray_dists[first], self.MAX_RANGE)

    # \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 Globale Lokalisierung \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _build_hypotheses(self) -> np.ndarray:
        """Pose-Hypothesen nur auf klar freier Fl\u00e4che. 1:1 mit ekf_extended.build_hypotheses."""
        map_h, map_w = self.occ_mask.shape
        step_px = max(1, int(self.GRID_SPACING / self.res))
        free_mask = (self.map_img > self.FREE_THRESH) & (~self.occ_mask)
        rows, cols = np.where(free_mask)
        sel = (rows % step_px == 0) & (cols % step_px == 0)
        rows, cols = rows[sel], cols[sel]
        xs = self.origin[0] + cols * self.res
        ys = self.origin[1] + (map_h - 1 - rows) * self.res
        thetas = np.linspace(-np.pi, np.pi, self.N_THETA, endpoint=False)
        poses = np.array([(x, y, th) for x, y in zip(xs, ys) for th in thetas])
        return poses

    def _score_hypothesis(self, x_start: np.ndarray, buffer: list) -> float:
        """Odom-Dead-Reckoning \u00fcber Buffer. Score-Normalisierung 1:1 mit ekf_extended._score_odom."""
        x_h = x_start.copy()
        total_valid = 0
        total_innov = 0.0
        count = 0
        n_eval_actual = max(len(buffer) - 1, 1)
        for i in range(1, len(buffer)):
            ox0, oy0, oth0, _     = buffer[i-1]
            ox1, oy1, oth1, z_obs = buffer[i]
            dx   = ox1 - ox0
            dy   = oy1 - oy0
            dist = dx * np.cos(oth0) + dy * np.sin(oth0)
            dth  = (oth1 - oth0 + np.pi) % (2*np.pi) - np.pi
            if abs(dist) > 0.3 or abs(dth) > 1.0:
                continue
            th = x_h[2]
            x_h[0] += dist * np.cos(th)
            x_h[1] += dist * np.sin(th)
            x_h[2]  = (th + dth + np.pi) % (2*np.pi) - np.pi
            z_hat = self._expected_scan_fast(x_h, self.beam_angles)
            innov = np.abs(z_obs - z_hat)
            valid = ((z_obs < self.MAX_RANGE - 1e-3) &
                     (z_hat < self.MAX_RANGE - 1e-3) &
                     (innov < self.RECOVERY_GATE))
            if valid.any():
                total_innov += float(innov[valid].mean())
                count += 1
            total_valid += int(valid.sum())
        mean_innov = total_innov / max(count, 1)
        raw = float(total_valid) - self.SCORE_MU * mean_innov * n_eval_actual
        return raw / max(n_eval_actual * self.N_BEAMS, 1)

    def _run_global_localization(self, buffer: list):
        """L\u00e4uft in eigenem Thread. Bewertet alle Hypothesen, setzt Winner."""
        hyps = self._build_hypotheses()
        self.get_logger().info(
            f'[GLOBAL LOC] {len(hyps)} Hypothesen, {len(buffer)} Scans ...')
        t0 = time.time()

        best_score = -np.inf
        best_pose  = None
        for x_h in hyps:
            sc = self._score_hypothesis(x_h, buffer)
            if sc > best_score:
                best_score = sc
                best_pose  = x_h.copy()

        dt = time.time() - t0
        t_since_start = time.time() - self._node_start
        is_initial = self._lost_time is None
        if is_initial:
            log_msg = (f'[INIT LOC] t={t_since_start:.1f}s since node start | '
                       f'Suchdauer={dt:.1f}s | '
                       f'winner=({best_pose[0]:.3f}, {best_pose[1]:.3f}, '
                       f'{np.degrees(best_pose[2]):.0f}\u00b0) | score={best_score:.3f}')
        else:
            total_s = time.time() - self._lost_time
            log_msg = (f'[RECOVERY] Suchdauer={dt:.1f}s | '
                       f'total time since kidnapping={total_s:.1f}s | '
                       f'winner=({best_pose[0]:.3f}, {best_pose[1]:.3f}, '
                       f'{np.degrees(best_pose[2]):.0f}\u00b0) | score={best_score:.3f}')
        self.get_logger().info(log_msg)
        self._logfile.write(f"{datetime.now().isoformat()} | t={t_since_start:.1f}s | {log_msg}\n")

        # Winner durch Buffer propagieren \u2192 aktuelle Pose statt Startpose setzen
        x_final = best_pose.copy()
        for i in range(1, len(buffer)):
            ox0, oy0, oth0, _ = buffer[i-1]
            ox1, oy1, oth1, _ = buffer[i]
            dx   = ox1 - ox0
            dy   = oy1 - oy0
            dist = dx * np.cos(oth0) + dy * np.sin(oth0)
            dth  = (oth1 - oth0 + np.pi) % (2*np.pi) - np.pi
            if abs(dist) > 0.3 or abs(dth) > 1.0:
                continue
            th = x_final[2]
            x_final[0] += dist * np.cos(th)
            x_final[1] += dist * np.sin(th)
            x_final[2]  = (th + dth + np.pi) % (2*np.pi) - np.pi

        prop_msg = (f'[PROPAGIERT] start=({best_pose[0]:.2f}, {best_pose[1]:.2f}, '
                    f'{np.degrees(best_pose[2]):.0f}\u00b0) \u2192 '
                    f'aktuell=({x_final[0]:.2f}, {x_final[1]:.2f}, {np.degrees(x_final[2]):.0f}\u00b0) | '
                    f'delta_yaw={np.degrees(x_final[2] - best_pose[2]):.0f}\u00b0')
        self.get_logger().info(prop_msg)
        self._logfile.write(f"{datetime.now().isoformat()} | t={t_since_start:.1f}s | {prop_msg}\n")

        with self._lock:
            self.x_ekf    = x_final
            self.P        = np.diag([self.RECOVERY_P_INIT**2] * 3)
            # last_odom auf Ende des Buffers setzen \u2192 n\u00e4chster Predict-Delta = latest_odom - buffer[-1]
            ox_end, oy_end, oth_end = buffer[-1][0], buffer[-1][1], buffer[-1][2]
            self.last_odom    = (ox_end, oy_end, oth_end)
            self._mode        = self.TRACKING
            self._consec_bad  = 0
            self._wait_start  = None
            self._loc_running = False
        self._event(
            "initialized" if is_initial else "recovered",
            pose=x_final.tolist(),
            search_pose=best_pose.tolist(),
            score=float(best_score),
            search_duration_s=float(dt),
        )

    def _trigger_global_loc(self, buf: list):
        with self._lock:
            self._loc_running = True
            self._loc_buffer  = []
        self._loc_thread = threading.Thread(
            target=self._run_global_localization, args=(buf,), daemon=True)
        self._loc_thread.start()

    def _event(self, event_type: str, **details):
        with self._record_lock:
            self._events.append({
                "time": self._last_recorded_time,
                "type": event_type,
                **details,
            })

    @staticmethod
    def _stamp_seconds(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _record_scan(self, stamp, z_obs, z_hat=None, valid_ratio=np.nan,
                     callback_duration=0.0, odom_pose=None):
        with self._lock:
            pose = np.full(3, np.nan) if self.x_ekf is None else self.x_ekf.copy()
            covariance = (np.full((3, 3), np.nan) if self.P is None
                          else self.P.copy())
            mode = self._mode
        record = {
            "time": self._stamp_seconds(stamp),
            "pose": pose,
            "covariance": covariance,
            "mode": {"global_loc": 0, "waiting": 1, "tracking": 2}[mode],
            "valid_ratio": float(valid_ratio),
            "z_obs": np.asarray(z_obs, dtype=float).copy(),
            "z_hat": (np.full(self.N_BEAMS, self.MAX_RANGE) if z_hat is None
                      else np.asarray(z_hat, dtype=float).copy()),
            "odom": (np.full(3, np.nan) if odom_pose is None
                     else np.asarray(odom_pose, dtype=float)),
            "callback_duration": float(callback_duration),
        }
        with self._record_lock:
            self._records.append(record)
            self._last_recorded_time = record["time"]
        self._publish_state(stamp, pose, mode)

    def _publish_state(self, stamp, pose, mode):
        self.mode_pub.publish(String(data=mode))
        if not np.all(np.isfinite(pose)):
            return
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        msg.pose.position.x = float(pose[0])
        msg.pose.position.y = float(pose[1])
        qx, qy, qz, qw = yaw_to_quat(pose[2])
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self.pose_pub.publish(msg)
        self.path_msg.header.stamp = stamp
        self.path_msg.poses.append(msg)
        self.path_pub.publish(self.path_msg)

    def finalize_recording(self):
        if self._finalized:
            return self.run_dir
        self._finalized = True
        if self._loc_thread is not None and self._loc_thread.is_alive():
            self.get_logger().info(
                "Warte vor dem Speichern auf laufende globale Lokalisierung ...")
            self._loc_thread.join()
        with self._record_lock:
            records = list(self._records)
            events = list(self._events)
        if records:
            np.savez_compressed(
                self.run_dir / "result.npz",
                scan_t=np.asarray([r["time"] for r in records]),
                poses=np.asarray([r["pose"] for r in records]),
                covariances=np.asarray([r["covariance"] for r in records]),
                modes=np.asarray([r["mode"] for r in records], dtype=np.int8),
                valid_ratios=np.asarray([r["valid_ratio"] for r in records]),
                z_obs=np.asarray([r["z_obs"] for r in records]),
                z_hat=np.asarray([r["z_hat"] for r in records]),
                odom=np.asarray([r["odom"] for r in records]),
                callback_durations=np.asarray(
                    [r["callback_duration"] for r in records]),
                beam_angles=(np.asarray(self.beam_angles)
                             if self.beam_angles is not None else np.empty(0)),
                map_img=self.map_img,
                map_origin=np.asarray(self.origin),
                map_res=np.asarray(self.res),
            )
        with open(self.run_dir / "events.json", "w") as f:
            json.dump(events, f, indent=2)
        with open(self.run_dir / "session.json", "w") as f:
            json.dump({
                "bag_id": self.bag_id,
                "map_yaml": str(self.map_yaml),
                "created_at": datetime.now().isoformat(),
                "scan_count": len(records),
                "config": asdict(CFG),
                "scan_odom_policy": "latest_odom_at_or_before_scan_stamp",
                "mode_names": ["GLOBAL_LOC", "WAITING", "TRACKING"],
            }, f, indent=2)
        return self.run_dir

    # \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 Callbacks \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _log_status(self):
        now = time.time()
        dt  = now - self.last_log_time
        odom_hz = self.odom_count / dt
        scan_hz = self.scan_count / dt
        with self._lock:
            mode    = self._mode
            buf_len = len(self._loc_buffer)
        if self.x_ekf is not None:
            pose = f"x={self.x_ekf[0]:+.3f} y={self.x_ekf[1]:+.3f} yaw={self.x_ekf[2]:+.3f}"
        else:
            pose = "<nicht initialisiert>"
        if mode == self.GLOBAL_LOC:
            extra = f" [GLOBAL LOC buf={buf_len}/{self.SCORE_N_EVAL}]"
        elif mode == self.WAITING:
            elapsed = time.time() - self._wait_start if self._wait_start else 0.0
            extra = f" [WAITING {elapsed:.1f}/{self.RECOVERY_WAIT_S}s buf={buf_len}]"
        elif self._consec_bad > 0:
            extra = f" [consec_bad={self._consec_bad}/{self.LOST_CONSEC_STEPS}]"
        else:
            extra = ""
        if self.scan_durations:
            avg_ms = 1000 * np.mean(self.scan_durations)
            extra += f" scan={avg_ms:.0f}ms"
        self.get_logger().info(
            f"odom={odom_hz:.1f}Hz scan={scan_hz:.1f}Hz pose=[{pose}]{extra}")
        self.odom_count     = 0
        self.scan_count     = 0
        self.scan_durations = []
        self.last_log_time  = now

    def initialpose_cb(self, msg: PoseWithCovarianceStamped):
        """Manueller Override \u2014 setzt Pose sofort, bricht laufende Lokalisierung ab."""
        p   = msg.pose.pose.position
        q   = msg.pose.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        with self._lock:
            self.x_ekf       = np.array([p.x, p.y, yaw])
            self.P           = np.diag([self.SIGMA_P0**2] * 3)
            self._mode       = self.TRACKING
            self._consec_bad = 0
            self._wait_start = None
            if self.latest_odom is not None:
                self.last_odom = self.latest_odom
        self.get_logger().info(
            f"Pose gesetzt (override): x={p.x:.3f} y={p.y:.3f} yaw={yaw:.3f}")
        self._event("initialpose", pose=[p.x, p.y, yaw], source="/initialpose")

    def odom_cb(self, msg: Odometry):
        self.odom_count += 1
        p   = msg.pose.pose.position
        q   = msg.pose.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        odom_pose = (p.x, p.y, yaw)
        odom_t = self._stamp_seconds(msg.header.stamp)
        self.latest_odom = odom_pose

        if not self._odom_stamps or odom_t >= self._odom_stamps[-1]:
            self._odom_stamps.append(odom_t)
            self._odom_poses.append(odom_pose)
        else:
            index = bisect_right(self._odom_stamps, odom_t)
            self._odom_stamps.insert(index, odom_t)
            self._odom_poses.insert(index, odom_pose)

        with self._lock:
            x_ekf = self.x_ekf
        if x_ekf is not None:
            if self.last_odom is None:
                self.last_odom = odom_pose
            self._publish_tf(msg.header.stamp)
        self._drain_pending_scans()

    def scan_cb(self, msg: LaserScan):
        scan_t = self._stamp_seconds(msg.header.stamp)
        index = bisect_right(
            [queued_t for queued_t, _ in self._pending_scans], scan_t)
        self._pending_scans.insert(index, (scan_t, msg))
        self._drain_pending_scans()

    def _drain_pending_scans(self):
        """Process scans with the newest odometry pose at or before their stamp."""
        while (
            self._pending_scans
            and self._odom_stamps
            and self._pending_scans[0][0] <= self._odom_stamps[-1]
        ):
            scan_t, msg = self._pending_scans.pop(0)
            odom_index = bisect_right(self._odom_stamps, scan_t) - 1
            if odom_index < 0:
                continue
            odom_pose = self._odom_poses[odom_index]
            self._process_scan(msg, odom_pose)

            # Keep the selected predecessor plus all newer odometry samples.
            if odom_index > 0:
                del self._odom_stamps[:odom_index]
                del self._odom_poses[:odom_index]

    def _process_scan(self, msg: LaserScan, odom_pose):
        callback_started = time.perf_counter()

        if self.beam_angles is None:
            angles = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment
            idx = np.linspace(0, len(angles), self.N_BEAMS, endpoint=False).astype(int)
            self.beam_angles = angles[idx]

        ranges = np.array(msg.ranges, dtype=float)
        ranges[~np.isfinite(ranges)] = self.MAX_RANGE
        ranges[ranges < msg.range_min] = self.MAX_RANGE
        ranges[ranges > self.MAX_RANGE] = self.MAX_RANGE
        z_obs = ranges[np.linspace(0, len(ranges), self.N_BEAMS, endpoint=False).astype(int)]

        with self._lock:
            mode        = self._mode
            loc_running = self._loc_running

        # \u2500\u2500 Modus: Globale Lokalisierung \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if mode == self.GLOBAL_LOC:
            if not loc_running:
                ox, oy, oyaw = odom_pose
                buf_ready = False
                buf = None
                with self._lock:
                    self._loc_buffer.append((ox, oy, oyaw, z_obs.copy()))
                    if len(self._loc_buffer) >= self.SCORE_N_EVAL:
                        buf_ready = True
                        buf = list(self._loc_buffer)
                if buf_ready:
                    self._trigger_global_loc(buf)
            self._record_scan(msg.header.stamp, z_obs, odom_pose=odom_pose)
            return

        # \u2500\u2500 Modus: WAITING \u2014 nur Odom-Predict, kein Scan-Update \u2500\u2500\u2500
        # Entspricht dem wait_left-Block in ekf_extended.run_ekf_extended
        if mode == self.WAITING:
            ox, oy, oyaw = odom_pose
            with self._lock:
                self._loc_buffer.append((ox, oy, oyaw, z_obs.copy()))
                wait_start = self._wait_start

            # Odom-Predict (kein Scan-Update, wie in ekf_extended)
            with self._lock:
                x_ekf = self.x_ekf
                P     = self.P
            if x_ekf is not None and self.last_odom is not None:
                dx_o = ox - self.last_odom[0]
                dy_o = oy - self.last_odom[1]
                prev = self.last_odom[2]
                dist = dx_o * np.cos(prev) + dy_o * np.sin(prev)
                dth  = (oyaw - prev + np.pi) % (2*np.pi) - np.pi
                if abs(dist) <= 0.3 and abs(dth) <= 1.0:
                    th = x_ekf[2]
                    x_ekf = np.array([
                        x_ekf[0] + dist * np.cos(th),
                        x_ekf[1] + dist * np.sin(th),
                        (x_ekf[2] + dth + np.pi) % (2*np.pi) - np.pi,
                    ])
                    F = np.array([
                        [1.0, 0.0, -dist * np.sin(th)],
                        [0.0, 1.0,  dist * np.cos(th)],
                        [0.0, 0.0,  1.0],
                    ])
                    P = F @ P @ F.T + self.Q
                with self._lock:
                    self.x_ekf = x_ekf
                    self.P     = P
            self.last_odom = (ox, oy, oyaw)
            self._publish_tf(msg.header.stamp, odom_pose)

            # Waiting period elapsed; start global search.
            if wait_start is not None and (time.time() - wait_start) >= self.RECOVERY_WAIT_S:
                with self._lock:
                    buf              = list(self._loc_buffer)
                    self._mode       = self.GLOBAL_LOC
                    self._loc_buffer = []
                self.get_logger().info(
                    f'[WAITING->GLOBAL_LOC] {self.RECOVERY_WAIT_S:.0f}s elapsed, '
                    f'{len(buf)} scans collected; starting global search ...')
                self._trigger_global_loc(buf)
            self._record_scan(msg.header.stamp, z_obs, odom_pose=odom_pose)
            return

        # \u2500\u2500 Modus: Tracking (normaler EKF) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        with self._lock:
            x_ekf = self.x_ekf
            P     = self.P
        if x_ekf is None:
            return

        t0 = time.time()

        # EKF Pr\u00e4diktion
        dx_o  = odom_pose[0] - self.last_odom[0]
        dy_o  = odom_pose[1] - self.last_odom[1]
        prev  = self.last_odom[2]
        dist  = dx_o * np.cos(prev) + dy_o * np.sin(prev)
        dth   = (odom_pose[2] - prev + np.pi) % (2*np.pi) - np.pi

        th = x_ekf[2]
        x_ekf = np.array([
            x_ekf[0] + dist * np.cos(th),
            x_ekf[1] + dist * np.sin(th),
            (x_ekf[2] + dth + np.pi) % (2*np.pi) - np.pi,
        ])
        F = np.array([
            [1.0, 0.0, -dist * np.sin(th)],
            [0.0, 1.0,  dist * np.cos(th)],
            [0.0, 0.0,  1.0],
        ])
        P = F @ P @ F.T + self.Q
        self.last_odom = odom_pose

        # EKF Update
        z_hat = expected_scan(x_ekf, self.beam_angles,
                              self.occ_mask, self.origin, self.res, self.MAX_RANGE)
        innov = z_obs - z_hat
        valid = ((z_obs < self.MAX_RANGE - 1e-3) &
                 (z_hat < self.MAX_RANGE - 1e-3) &
                 (np.abs(innov) < self.GATE))
        n_valid = int(valid.sum())

        if n_valid >= 3:
            H = compute_H_num(x_ekf, self.beam_angles[valid],
                              self.occ_mask, self.origin, self.res,
                              self.MAX_RANGE, self.EPS_H)
            innov_v = innov[valid]
            S = H @ P @ H.T + np.eye(n_valid) * self.SIGMA_R**2
            try:
                K = P @ H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                pass
            else:
                corr = K @ innov_v
                raw_corr = corr.copy()
                corr[:2] = np.clip(corr[:2], -0.3, 0.3)
                corr[2]  = np.clip(corr[2],  -0.3, 0.3)
                x_ekf    = x_ekf + corr
                x_ekf[2] = (x_ekf[2] + np.pi) % (2*np.pi) - np.pi
                P        = (self.I3 - K @ H) @ P
                self.get_logger().info(
                    f"UPDATE valid={n_valid}/{self.N_BEAMS} "
                    f"raw_corr=[{raw_corr[0]:+.3f} {raw_corr[1]:+.3f} {raw_corr[2]:+.3f}] "
                    f"pose=[{x_ekf[0]:+.3f} {x_ekf[1]:+.3f} {x_ekf[2]:+.3f}]")
        else:
            self.get_logger().warn(f"UPDATE SKIP: nur {n_valid}/{self.N_BEAMS} valid beams")

        with self._lock:
            self.x_ekf = x_ekf
            self.P     = P

        # \u2500\u2500 Lost-Detection (1:1 mit ekf_extended) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        # bad = ratio < LOST_VALID_RATIO ODER trace(P) > P_TRACE_MAX
        ratio = n_valid / self.N_BEAMS
        bad   = (ratio < self.LOST_VALID_RATIO) or (np.trace(P) > self.P_TRACE_MAX)
        self._consec_bad = self._consec_bad + 1 if bad else 0

        if self._consec_bad >= self.LOST_CONSEC_STEPS:
            self._lost_time = time.time()
            t_since_start = self._lost_time - self._node_start
            log_msg = (f'[KIDNAPPING] t={t_since_start:.1f}s since node start | '
                       f'ratio={ratio:.2f} trace(P)={np.trace(P):.2f} | '
                       f'pose=({x_ekf[0]:+.2f}, {x_ekf[1]:+.2f}, '
                       f'{np.degrees(x_ekf[2]):+.0f}\u00b0) | '
                       f'\u2192 WAITING {self.RECOVERY_WAIT_S:.0f}s ...')
            self.get_logger().warn(log_msg)
            self._logfile.write(f"{datetime.now().isoformat()} | t={t_since_start:.1f}s | {log_msg}\n")
            with self._lock:
                self._mode        = self.WAITING
                self._wait_start  = time.time()
                self._loc_buffer  = []
                self._loc_running = False
            self._consec_bad = 0
            self._event("lost", ratio=ratio, pose=x_ekf.tolist())

        self._publish_tf(msg.header.stamp, odom_pose)
        self._publish_beams(msg.header.stamp, z_obs, z_hat, valid)
        self.scan_count += 1
        self.scan_durations.append(time.time() - t0)
        self._record_scan(
            msg.header.stamp, z_obs, z_hat, ratio,
            time.perf_counter() - callback_started, odom_pose)

    # \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 TF & Visualisierung \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _publish_beams(self, stamp, z_obs, z_hat, valid):
        with self._lock:
            x_ekf = self.x_ekf
        if x_ekf is None:
            return
        mx, my, mth = x_ekf

        m_obs = Marker()
        m_obs.header.stamp    = stamp
        m_obs.header.frame_id = 'map'
        m_obs.ns     = 'ekf_beams'
        m_obs.id     = 0
        m_obs.type   = Marker.LINE_LIST
        m_obs.action = Marker.ADD
        m_obs.scale.x = 0.02
        m_obs.pose.orientation.w = 1.0

        m_hat = Marker()
        m_hat.header.stamp    = stamp
        m_hat.header.frame_id = 'map'
        m_hat.ns     = 'ekf_beams'
        m_hat.id     = 1
        m_hat.type   = Marker.SPHERE_LIST
        m_hat.action = Marker.ADD
        m_hat.scale.x = 0.07
        m_hat.scale.y = 0.07
        m_hat.scale.z = 0.07
        m_hat.pose.orientation.w = 1.0
        m_hat.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)

        for i, (d_obs, d_hat, ang) in enumerate(zip(z_obs, z_hat, self.beam_angles)):
            ex = mx + d_obs * np.cos(mth + ang)
            ey = my + d_obs * np.sin(mth + ang)
            color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0) if valid[i] \
                    else ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.7)
            m_obs.points += [Point(x=mx, y=my, z=0.05), Point(x=ex, y=ey, z=0.05)]
            m_obs.colors += [color, color]
            if d_hat < self.MAX_RANGE - 1e-3:
                hx = mx + d_hat * np.cos(mth + ang)
                hy = my + d_hat * np.sin(mth + ang)
                m_hat.points.append(Point(x=hx, y=hy, z=0.1))

        arr = MarkerArray()
        arr.markers.append(m_obs)
        arr.markers.append(m_hat)
        self.beam_pub.publish(arr)

    def _publish_tf(self, stamp, odom_pose=None):
        with self._lock:
            x_ekf = self.x_ekf
        if odom_pose is None:
            odom_pose = self.latest_odom
        if x_ekf is None or odom_pose is None:
            return
        mx, my, mth = x_ekf
        ox, oy, oth = odom_pose
        dth = mth - oth
        xmo = mx - np.cos(dth)*ox + np.sin(dth)*oy
        ymo = my - np.sin(dth)*ox - np.cos(dth)*oy
        t = TransformStamped()
        t.header.stamp        = stamp
        t.header.frame_id     = 'map'
        t.child_frame_id      = 'odom'
        t.transform.translation.x = xmo
        t.transform.translation.y = ymo
        t.transform.translation.z = 0.0
        qx, qy, qz, qw = yaw_to_quat(dth)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)


def load_start_pose(path: Path, bag_id: str):
    if not path.exists():
        return None
    with open(path) as f:
        store = json.load(f)
    if bag_id in store and isinstance(store[bag_id], list):
        return store[bag_id]
    for source in ("manual", "found"):
        poses = store.get(source, {})
        if bag_id in poses:
            return poses[bag_id]
        for key, pose in poses.items():
            normalized = key.replace("\\", "/")
            if normalized.endswith("/" + bag_id) or Path(normalized).name == Path(bag_id).name:
                return pose
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="ROS2 live EKF with local run recording.")
    parser.add_argument("map_yaml", type=Path)
    parser.add_argument("--bag-id", required=True, help="Stable ID, e.g. bags/lab/rosbag2_lab_fast")
    parser.add_argument(
        "--start-poses", type=Path,
        default=Path(__file__).with_name("start_poses.json"))
    parser.add_argument("--run-root", type=Path, default=Path("/tmp/ekf_runs"))
    return parser.parse_args()


def main():
    args = parse_args()
    init_pose = load_start_pose(args.start_poses, args.bag_id)
    if init_pose is None:
        print(f"[startpose] No pose for {args.bag_id}; global localization will be used.")
    else:
        print(f"[startpose] {args.bag_id}: {init_pose}")
    rclpy.init()
    node = EKFLocalizationNode(
        str(args.map_yaml), args.bag_id, args.run_root, init_pose=init_pose)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._logfile.write(f"\n# Session ended {datetime.now().isoformat()}\n")
        run_dir = node.finalize_recording()
        node._logfile.close()
        print(f"[recording] saved: {run_dir}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
