#!/usr/bin/env python3
"""
Causal, accelerated runtime simulation of the ROS2 EKF node.

The bag is replayed in timestamp order without sleeping. Only data that has
already arrived at the current virtual time is available to the EKF. Measured
computation durations are mapped into virtual time so background localization
finishes after the same amount of simulated robot time.

Usage:
    python3 runtime/simulator.py <bag_dir> <map.yaml> [--output result.npz]
"""

from __future__ import annotations

import argparse
import heapq
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

from config import (
    N_BEAMS, MAX_RANGE, SENSOR_QUEUE_DEPTH,
    SIGMA_QD, SIGMA_QTH, SIGMA_R, EPS_H, GATE, SIGMA_P0,
    LOST_VALID_RATIO, LOST_CONSEC_STEPS, P_TRACE_MAX,
    RECOVERY_WAIT_S, GRID_SPACING, N_THETA, FREE_THRESH,
    RECOVERY_P_INIT, RECOVERY_GATE, SCORE_N_EVAL, SCORE_MU,
)


@dataclass(frozen=True)
class RuntimeConfig:
    n_beams: int = N_BEAMS
    max_range: float = MAX_RANGE
    sigma_qd: float = SIGMA_QD
    sigma_qth: float = SIGMA_QTH
    sigma_r: float = SIGMA_R
    eps_h: float = EPS_H
    gate: float = GATE
    sigma_p0: float = SIGMA_P0
    lost_valid_ratio: float = LOST_VALID_RATIO
    lost_consec_steps: int = LOST_CONSEC_STEPS
    p_trace_max: float = P_TRACE_MAX
    recovery_wait_s: float = RECOVERY_WAIT_S
    grid_spacing: float = GRID_SPACING
    n_theta: int = N_THETA
    free_thresh: int = FREE_THRESH
    recovery_p_init: float = RECOVERY_P_INIT
    recovery_gate: float = RECOVERY_GATE
    score_n_eval: int = SCORE_N_EVAL
    score_mu: float = SCORE_MU
    sensor_queue_depth: int = SENSOR_QUEUE_DEPTH


MODE_GLOBAL = 0
MODE_WAITING = 1
MODE_TRACKING = 2
MODE_NAMES = ["GLOBAL_LOC", "WAITING", "TRACKING"]

STATUS_PROCESSED = 0
STATUS_BUFFERED = 1
STATUS_IGNORED_SEARCH = 2
STATUS_NO_ODOM = 3
STATUS_DROPPED_QUEUE = 4
STATUS_NAMES = [
    "processed",
    "buffered",
    "ignored_search",
    "no_odom",
    "dropped_queue",
]


def quat_to_yaw(qx, qy, qz, qw):
    return np.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def stamp_to_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def load_map(yaml_path: Path):
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    pgm_path = yaml_path.parent / meta["image"]
    if not pgm_path.exists():
        pgm_path = yaml_path.with_suffix(".pgm")
    img = np.array(Image.open(pgm_path))
    occ_thresh = float(meta.get("occupied_thresh", 0.65))
    occ_mask = img < int((1.0 - occ_thresh) * 255)
    return img, occ_mask, float(meta["resolution"]), np.asarray(meta["origin"][:2])


def load_bag_events(bag_path: Path, cfg: RuntimeConfig):
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    events = []
    beam_angles = None
    sequence = 0

    with Reader(bag_path) as reader:
        for conn, _, raw in reader.messages():
            if conn.topic not in ("/odom", "/scan"):
                continue
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            # ekf_live.py associates scans and odometry by message header stamp,
            # not by the rosbag storage/arrival timestamp.
            t = stamp_to_seconds(msg.header.stamp)
            if conn.topic == "/odom":
                p = msg.pose.pose.position
                q = msg.pose.pose.orientation
                payload = np.array(
                    [p.x, p.y, quat_to_yaw(q.x, q.y, q.z, q.w)], dtype=float
                )
                kind = "odom"
            else:
                ranges = np.asarray(msg.ranges, dtype=float)
                idx = np.linspace(
                    0, len(ranges), cfg.n_beams, endpoint=False
                ).astype(int)
                if beam_angles is None:
                    all_angles = (
                        msg.angle_min
                        + np.arange(len(ranges), dtype=float) * msg.angle_increment
                    )
                    beam_angles = all_angles[idx]
                ranges[~np.isfinite(ranges)] = cfg.max_range
                ranges[ranges < msg.range_min] = cfg.max_range
                ranges[ranges > cfg.max_range] = cfg.max_range
                payload = ranges[idx].copy()
                kind = "scan"
            events.append((t, sequence, kind, payload))
            sequence += 1

    if not events or beam_angles is None:
        raise RuntimeError("Bag contains no usable /odom and /scan messages.")
    # Match the live node: for equal timestamps, odometry is visible to the scan.
    events.sort(key=lambda event: (event[0], 0 if event[2] == "odom" else 1, event[1]))
    events = [
        (event_t, index, kind, payload)
        for index, (event_t, _, kind, payload) in enumerate(events)
    ]
    return events, np.asarray(beam_angles)


class RuntimeSimulator:
    def __init__(
        self,
        bag_path: Path,
        map_yaml: Path,
        cfg: RuntimeConfig,
        runtime_scale: float = 1.0,
        init_pose=None,
        max_recovery: int = 0,
        progress: bool = True,
    ):
        self.bag_path = bag_path
        self.map_yaml = map_yaml
        self.cfg = cfg
        self.runtime_scale = runtime_scale
        self.max_recovery = max_recovery
        self.recovery_count = 0
        self.progress = progress

        self.map_img, self.occ_mask, self.res, self.origin = load_map(map_yaml)
        self.map_h, self.map_w = self.map_img.shape
        self.events, self.beam_angles = load_bag_events(bag_path, cfg)
        self.start_t = self.events[0][0]
        self.end_t = self.events[-1][0]
        scan_times = np.asarray([event[0] for event in self.events if event[2] == "scan"])
        median_scan_dt = float(np.median(np.diff(scan_times))) if len(scan_times) > 1 else 0.2
        self.max_scan_queue_lag = median_scan_dt * cfg.sensor_queue_depth

        step = self.res * 0.5
        self.ray_dists = np.arange(
            1, int(cfg.max_range / step) + 1, dtype=float
        ) * step
        self.x = None if init_pose is None else np.asarray(init_pose, dtype=float)
        self.P = (
            None
            if init_pose is None
            else np.diag([cfg.sigma_p0**2, cfg.sigma_p0**2, cfg.sigma_p0**2])
        )
        self.Q = np.diag([cfg.sigma_qd**2, cfg.sigma_qd**2, cfg.sigma_qth**2])
        self.I3 = np.eye(3)
        self.latest_odom = None
        self.last_odom = None
        self.mode = MODE_GLOBAL if init_pose is None else MODE_TRACKING
        self.loc_buffer = []
        self.search_running = False
        self.wait_start = None
        self.consec_bad = 0
        self.pending = []
        self.pending_sequence = len(self.events)

        self.records = []
        self.event_log = []
        if init_pose is not None:
            self.event_log.append(
                {
                    "time": self.start_t,
                    "type": "initialpose",
                    "pose": self.x.tolist(),
                }
            )
        self.search_durations = []
        self.track_durations = []

    def _build_hypotheses(self):
        step_px = max(1, int(self.cfg.grid_spacing / self.res))
        free_mask = (self.map_img > self.cfg.free_thresh) & (~self.occ_mask)
        rows, cols = np.where(free_mask)
        selected = (rows % step_px == 0) & (cols % step_px == 0)
        rows, cols = rows[selected], cols[selected]
        xs = self.origin[0] + cols * self.res
        ys = self.origin[1] + (self.map_h - 1 - rows) * self.res
        thetas = np.linspace(-np.pi, np.pi, self.cfg.n_theta, endpoint=False)
        return np.asarray(
            [(x, y, theta) for x, y in zip(xs, ys) for theta in thetas],
            dtype=float,
        )

    def _expected_scan_fast(self, pose):
        x, y, theta = pose
        angles = theta + self.beam_angles
        px = x + np.cos(angles)[:, None] * self.ray_dists[None, :]
        py = y + np.sin(angles)[:, None] * self.ray_dists[None, :]
        cols = ((px - self.origin[0]) / self.res).astype(np.int32)
        rows = (self.map_h - 1 - (py - self.origin[1]) / self.res).astype(np.int32)
        oob = (cols < 0) | (cols >= self.map_w) | (rows < 0) | (rows >= self.map_h)
        cols = np.clip(cols, 0, self.map_w - 1)
        rows = np.clip(rows, 0, self.map_h - 1)
        hit = self.occ_mask[rows, cols] | oob
        first = np.argmax(hit, axis=1)
        return np.where(hit.any(axis=1), self.ray_dists[first], self.cfg.max_range)

    def _raycast(self, x, y, angle):
        step = self.res * 0.5
        cx, cy = np.cos(angle), np.sin(angle)
        for k in range(1, int(self.cfg.max_range / step) + 1):
            distance = k * step
            col = int((x + cx * distance - self.origin[0]) / self.res)
            row = self.map_h - 1 - int(
                (y + cy * distance - self.origin[1]) / self.res
            )
            if (
                col < 0
                or col >= self.map_w
                or row < 0
                or row >= self.map_h
                or self.occ_mask[row, col]
            ):
                return distance
        return self.cfg.max_range

    def _expected_scan(self, pose, beam_angles=None):
        angles = self.beam_angles if beam_angles is None else beam_angles
        x, y, theta = pose
        return np.asarray([self._raycast(x, y, theta + angle) for angle in angles])

    def _compute_h(self, pose, beam_angles):
        h = np.zeros((len(beam_angles), 3))
        eps = self.cfg.eps_h
        for col, delta in enumerate(
            (
                np.array([eps, 0.0, 0.0]),
                np.array([0.0, eps, 0.0]),
                np.array([0.0, 0.0, eps]),
            )
        ):
            h[:, col] = (
                self._expected_scan(pose + delta, beam_angles)
                - self._expected_scan(pose - delta, beam_angles)
            ) / (2.0 * eps)
        return h

    def _pose_free(self, pose):
        col = int((pose[0] - self.origin[0]) / self.res)
        row = self.map_h - 1 - int((pose[1] - self.origin[1]) / self.res)
        return (
            0 <= col < self.map_w
            and 0 <= row < self.map_h
            and self.map_img[row, col] > self.cfg.free_thresh
        )

    @staticmethod
    def _odom_delta(odom_before, odom_after):
        dx = odom_after[0] - odom_before[0]
        dy = odom_after[1] - odom_before[1]
        distance = dx * np.cos(odom_before[2]) + dy * np.sin(odom_before[2])
        dtheta = (odom_after[2] - odom_before[2] + np.pi) % (2 * np.pi) - np.pi
        return distance, dtheta

    @classmethod
    def _propagate_pose(cls, pose, odom_before, odom_after):
        distance, dtheta = cls._odom_delta(odom_before, odom_after)
        theta = pose[2]
        return np.array(
            [
                pose[0] + distance * np.cos(theta),
                pose[1] + distance * np.sin(theta),
                (theta + dtheta + np.pi) % (2 * np.pi) - np.pi,
            ]
        ), distance

    def _predict(self, odom_pose, reject_jumps=False):
        if self.x is None or self.last_odom is None:
            return
        x_new, distance = self._propagate_pose(self.x, self.last_odom, odom_pose)
        dtheta = (odom_pose[2] - self.last_odom[2] + np.pi) % (2 * np.pi) - np.pi
        if reject_jumps and (abs(distance) > 0.3 or abs(dtheta) > 1.0):
            self.last_odom = odom_pose.copy()
            return
        theta = self.x[2]
        f = np.array(
            [
                [1.0, 0.0, -distance * np.sin(theta)],
                [0.0, 1.0, distance * np.cos(theta)],
                [0.0, 0.0, 1.0],
            ]
        )
        self.x = x_new
        self.P = f @ self.P @ f.T + self.Q
        self.last_odom = odom_pose.copy()

    def _score_hypothesis(self, start_pose, buffer):
        pose = start_pose.copy()
        total_valid = 0
        total_innovation = 0.0
        count = 0
        n_eval = max(len(buffer) - 1, 1)
        for previous, current in zip(buffer, buffer[1:]):
            distance, dtheta = self._odom_delta(previous[0], current[0])
            if abs(distance) > 0.3 or abs(dtheta) > 1.0:
                continue
            pose, _ = self._propagate_pose(pose, previous[0], current[0])
            expected = self._expected_scan_fast(pose)
            innovation = np.abs(current[1] - expected)
            valid = (
                (current[1] < self.cfg.max_range - 1e-3)
                & (expected < self.cfg.max_range - 1e-3)
                & (innovation < self.cfg.recovery_gate)
            )
            if valid.any():
                total_innovation += float(innovation[valid].mean())
                count += 1
            total_valid += int(valid.sum())
        mean_innovation = total_innovation / max(count, 1)
        raw = total_valid - self.cfg.score_mu * mean_innovation * n_eval
        return raw / max(n_eval * self.cfg.n_beams, 1)

    def _search(self, buffer):
        started = time.perf_counter()
        hypotheses = self._build_hypotheses()
        best_score = -np.inf
        best_pose = hypotheses[0]
        for hypothesis in hypotheses:
            score = self._score_hypothesis(hypothesis, buffer)
            if score > best_score:
                best_score = score
                best_pose = hypothesis.copy()
        duration = time.perf_counter() - started

        final_pose = best_pose.copy()
        for previous, current in zip(buffer, buffer[1:]):
            candidate, distance = self._propagate_pose(
                final_pose, previous[0], current[0]
            )
            dtheta = (current[0][2] - previous[0][2] + np.pi) % (2 * np.pi) - np.pi
            if abs(distance) <= 0.3 and abs(dtheta) <= 1.0:
                final_pose = candidate
        return final_pose, best_score, duration, best_pose

    def _start_search(self, virtual_t, buffer, reason):
        if not buffer:
            return 0.0
        self.search_running = True
        self.mode = MODE_GLOBAL
        self.loc_buffer = []
        self.event_log.append(
            {
                "time": virtual_t,
                "type": "searching",
                "reason": reason,
                "buffer_scans": len(buffer),
            }
        )
        if reason == "recovery":
            self.recovery_count += 1
        final_pose, score, duration, search_pose = self._search(buffer)
        simulated_duration = duration * self.runtime_scale
        self.search_durations.append(duration)
        heapq.heappush(
            self.pending,
            (
                virtual_t + simulated_duration,
                self.pending_sequence,
                "search_done",
                (
                    final_pose,
                    score,
                    buffer[-1][0].copy(),
                    duration,
                    reason,
                    search_pose,
                ),
            ),
        )
        self.pending_sequence += 1
        return duration

    def _finish_search(self, virtual_t, payload):
        final_pose, score, buffer_end_odom, duration, reason, search_pose = payload
        self.x = final_pose
        self.P = np.diag([self.cfg.recovery_p_init**2] * 3)
        self.last_odom = buffer_end_odom
        self.mode = MODE_TRACKING
        self.search_running = False
        self.wait_start = None
        self.consec_bad = 0
        self.event_log.append(
            {
                "time": virtual_t,
                "type": "initialized" if reason == "initial" else "recovered",
                "score": float(score),
                "duration_s": float(duration),
                "pose": self.x.tolist(),
                "search_pose": search_pose.tolist(),
            }
        )

    def _handle_odom(self, odom_pose):
        self.latest_odom = odom_pose.copy()
        if self.x is not None and self.last_odom is None:
            self.last_odom = odom_pose.copy()

    def _handle_scan(self, arrival_t, virtual_t, z_obs):
        callback_started = time.perf_counter()
        background_duration = 0.0
        status = STATUS_PROCESSED
        ratio = np.nan
        expected = np.full(self.cfg.n_beams, self.cfg.max_range)

        if virtual_t - arrival_t > self.max_scan_queue_lag:
            status = STATUS_DROPPED_QUEUE
        elif self.latest_odom is None:
            # Match ekf_live.py: scans before the first usable odometry sample
            # are ignored rather than added to the recorded EKF trajectory.
            return 0.0
        elif self.mode == MODE_GLOBAL:
            if self.search_running:
                status = STATUS_IGNORED_SEARCH
            else:
                status = STATUS_BUFFERED
                self.loc_buffer.append((self.latest_odom.copy(), z_obs.copy()))
                if len(self.loc_buffer) >= self.cfg.score_n_eval:
                    background_duration += self._start_search(
                        virtual_t, list(self.loc_buffer), "initial"
                    )
        elif self.mode == MODE_WAITING:
            status = STATUS_BUFFERED
            self.loc_buffer.append((self.latest_odom.copy(), z_obs.copy()))
            self._predict(self.latest_odom, reject_jumps=True)
            if virtual_t - self.wait_start >= self.cfg.recovery_wait_s:
                background_duration += self._start_search(
                    virtual_t, list(self.loc_buffer), "recovery"
                )
        else:
            self._predict(self.latest_odom)
            expected = self._expected_scan(self.x)
            innovation = z_obs - expected
            valid = (
                (z_obs < self.cfg.max_range - 1e-3)
                & (expected < self.cfg.max_range - 1e-3)
                & (np.abs(innovation) < self.cfg.gate)
            )
            ratio = float(valid.sum()) / self.cfg.n_beams
            if valid.sum() >= 3:
                h = self._compute_h(self.x, self.beam_angles[valid])
                s = h @ self.P @ h.T + np.eye(int(valid.sum())) * self.cfg.sigma_r**2
                try:
                    gain = self.P @ h.T @ np.linalg.inv(s)
                except np.linalg.LinAlgError:
                    pass
                else:
                    correction = gain @ innovation[valid]
                    correction[:2] = np.clip(correction[:2], -0.3, 0.3)
                    correction[2] = np.clip(correction[2], -0.3, 0.3)
                    self.x = self.x + correction
                    self.x[2] = (self.x[2] + np.pi) % (2 * np.pi) - np.pi
                    self.P = (self.I3 - gain @ h) @ self.P

            bad = ratio < self.cfg.lost_valid_ratio or np.trace(self.P) > self.cfg.p_trace_max
            self.consec_bad = self.consec_bad + 1 if bad else 0
            recovery_allowed = (
                self.max_recovery >= 0
                and (self.max_recovery == 0 or self.recovery_count < self.max_recovery)
            )
            if self.consec_bad >= self.cfg.lost_consec_steps and recovery_allowed:
                self.mode = MODE_WAITING
                self.wait_start = virtual_t
                self.loc_buffer = []
                self.consec_bad = 0
                self.event_log.append(
                    {
                        "time": virtual_t,
                        "type": "lost",
                        "ratio": ratio,
                        "pose": self.x.tolist(),
                    }
                )

        duration = max(0.0, time.perf_counter() - callback_started - background_duration)
        self.track_durations.append(duration)
        pose = np.full(3, np.nan) if self.x is None else self.x.copy()
        covariance = np.full((3, 3), np.nan) if self.P is None else self.P.copy()
        self.records.append(
            {
                "time": arrival_t,
                "callback_start": virtual_t,
                "pose": pose,
                "covariance": covariance,
                "mode": self.mode,
                "status": status,
                "ratio": ratio,
                "z_obs": z_obs.copy(),
                "z_hat": expected.copy(),
                "callback_duration": duration,
            }
        )
        return duration

    def run(self):
        queue = list(self.events)
        heapq.heapify(queue)
        processed = 0
        total = len(queue)
        wall_start = time.perf_counter()
        executor_available = self.start_t

        while queue or self.pending:
            next_callback_t = (
                max(queue[0][0], executor_available) if queue else np.inf
            )
            if self.pending and self.pending[0][0] <= next_callback_t:
                virtual_t, _, kind, payload = heapq.heappop(self.pending)
                if kind == "search_done":
                    self._finish_search(virtual_t, payload)
                continue

            arrival_t, _, kind, payload = heapq.heappop(queue)
            virtual_t = max(arrival_t, executor_available)
            if kind == "odom":
                self._handle_odom(payload)
            else:
                duration = self._handle_scan(arrival_t, virtual_t, payload)
                executor_available = virtual_t + duration * self.runtime_scale
            processed += 1
            if self.progress and processed % max(1, total // 20) == 0:
                pct = 100.0 * processed / total
                print(f"  replay {pct:5.1f}%  t={virtual_t - self.start_t:7.1f}s")

        wall_duration = time.perf_counter() - wall_start
        print(
            f"Simulation complete: {len(self.records)} scans, "
            f"{self.end_t - self.start_t:.1f}s robot time in {wall_duration:.1f}s wall time."
        )
        return self._result(wall_duration)

    def _result(self, wall_duration):
        records = self.records
        return {
            "scan_t": np.asarray([r["time"] for r in records]),
            "poses": np.asarray([r["pose"] for r in records]),
            "covariances": np.asarray([r["covariance"] for r in records]),
            "modes": np.asarray([r["mode"] for r in records], dtype=np.int8),
            "statuses": np.asarray([r["status"] for r in records], dtype=np.int8),
            "valid_ratios": np.asarray([r["ratio"] for r in records]),
            "z_obs": np.asarray([r["z_obs"] for r in records]),
            "z_hat": np.asarray([r["z_hat"] for r in records]),
            "callback_durations": np.asarray(
                [r["callback_duration"] for r in records]
            ),
            "callback_latencies": np.asarray(
                [r["callback_start"] - r["time"] for r in records]
            ),
            "beam_angles": self.beam_angles,
            "map_img": self.map_img,
            "map_origin": self.origin,
            "map_res": np.asarray(self.res),
            "events": self.event_log,
            "wall_duration": wall_duration,
        }


def save_result(result, output_path: Path, bag_path: Path, map_yaml: Path, cfg, runtime_scale):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        scan_t=result["scan_t"],
        poses=result["poses"],
        covariances=result["covariances"],
        modes=result["modes"],
        statuses=result["statuses"],
        valid_ratios=result["valid_ratios"],
        z_obs=result["z_obs"],
        z_hat=result["z_hat"],
        callback_durations=result["callback_durations"],
        callback_latencies=result["callback_latencies"],
        beam_angles=result["beam_angles"],
        map_img=result["map_img"],
        map_origin=result["map_origin"],
        map_res=result["map_res"],
    )
    metadata = {
        "bag_path": str(bag_path.resolve()),
        "map_yaml": str(map_yaml.resolve()),
        "result_file": str(output_path.resolve()),
        "runtime_scale": runtime_scale,
        "scan_odom_policy": "latest_odom_at_or_before_scan_stamp",
        "wall_duration_s": result["wall_duration"],
        "config": asdict(cfg),
        "mode_names": MODE_NAMES,
        "status_names": STATUS_NAMES,
        "events": result["events"],
    }
    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump(metadata, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag_path", type=Path)
    parser.add_argument("map_yaml", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output .npz file (default: runtime/cache/<bag>_runtime.npz)",
    )
    parser.add_argument(
        "--runtime-scale",
        type=float,
        default=1.0,
        help="Scale measured computation time in virtual robot time.",
    )
    parser.add_argument(
        "--init-pose",
        nargs=3,
        type=float,
        metavar=("X", "Y", "YAW"),
        help="Start directly in TRACKING, like a live /initialpose message.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output or Path(__file__).parent / "cache" / (
        args.bag_path.name + "_runtime.npz"
    )
    cfg = RuntimeConfig()
    simulator = RuntimeSimulator(
        args.bag_path,
        args.map_yaml,
        cfg,
        runtime_scale=args.runtime_scale,
        init_pose=args.init_pose,
    )
    result = simulator.run()
    save_result(result, output, args.bag_path, args.map_yaml, cfg, args.runtime_scale)
    print(f"Saved: {output}")
    print(f"Player: python3 runtime/player.py")


if __name__ == "__main__":
    main()
