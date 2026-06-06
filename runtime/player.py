"""
Runtime EKF player using PyQt6 and pyqtgraph.

The upper view shows the map, EKF trail, robot pose, and LiDAR beams. The lower
view shows the valid-beam ratio and event timeline. A QTimer drives playback.

  pip install PyQt6 pyqtgraph
"""

from pathlib import Path
import json
import sys

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QButtonGroup, QProgressBar,
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui  import QColor, QPalette

from config import N_BEAMS, MAX_RANGE, GATE, LOST_VALID_RATIO, RECOVERY_WAIT_S
from simulator import RuntimeConfig, RuntimeSimulator, save_result

pg.setConfigOption("imageAxisOrder", "row-major")
pg.setConfigOption("background", "#1a1a2e")
pg.setConfigOption("foreground", "#aaaacc")

# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════
ROOT         = Path(__file__).parent.parent
MAP_BASE     = ROOT / "maps"
BAG_BASE     = ROOT / "bags"
ROBOT_R      = 0.17

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
START_POSES_FILE = CACHE_DIR / "start_poses.json"

# ══════════════════════════════════════════════════════════════════════
# Bag catalog: environment -> label -> bag path and map
# ══════════════════════════════════════════════════════════════════════
BAGS = {
    "elevator": {
        "map": MAP_BASE / "elevator" / "map.yaml",
        "bags": {
            "slow":         BAG_BASE / "elevator" / "elevator_slow",
            "fast":         BAG_BASE / "elevator" / "elevator_fast",
            "fast disturb": BAG_BASE / "elevator" / "elevator_disturbance",
            "kidnap":       BAG_BASE / "elevator" / "elevator_kidnap_slow",
        },
    },
    "lab": {
        "map": MAP_BASE / "lab" / "map.yaml",
        "bags": {
            "slow":         BAG_BASE / "lab" / "rosbag2_lab_slow",
            "fast":         BAG_BASE / "lab" / "rosbag2_lab_fast",
            "fast disturb": BAG_BASE / "lab" / "rosbag2_lab_fast_disturb",
            "kidnap":       BAG_BASE / "lab" / "rosbag2_lab_fast_kidnap",
        },
    },
    "hall": {
        "map": MAP_BASE / "hall" / "map.yaml",
        "bags": {
            "slow":         BAG_BASE / "hall" / "rosbag2_hall_slow",
            "fast":         BAG_BASE / "hall" / "rosbag2_hall_fast",
            "fast disturb": BAG_BASE / "hall" / "rosbag2_hall_fast_disturb",
        },
    },
}

# ══════════════════════════════════════════════════════════════════════
# Map-only loading (no bag, no EKF)
# ══════════════════════════════════════════════════════════════════════

def load_map_only(map_yaml: Path, bag_path: Path = None) -> dict:
    """Load only the map; no bag access and no EKF computation."""
    import yaml as _yaml
    from PIL import Image as _Image
    with open(map_yaml) as f:
        meta = _yaml.safe_load(f)
    map_res    = meta["resolution"]
    map_origin = np.array(meta["origin"][:2])
    pgm = (map_yaml.parent / meta["image"])
    if not pgm.exists():
        pgm = map_yaml.with_suffix(".pgm")
    map_img = np.array(_Image.open(pgm))
    MAP_H, MAP_W = map_img.shape[:2]
    return dict(
        map_only=True,
        MAP_IMG=map_img,
        MAP_RES=map_res,
        MAP_ORIGIN=map_origin,
        MAP_H=MAP_H,
        MAP_W=MAP_W,
        ROBOT_R_PX=ROBOT_R / map_res,
        bag_path=bag_path,
        map_yaml=map_yaml,
    )

# ══════════════════════════════════════════════════════════════════════
# DATASET LADEN  (Cache → neu berechnen)
# ══════════════════════════════════════════════════════════════════════

class LoadCancelled(Exception):
    pass


def load_dataset(bag_path: Path, map_yaml: Path, force: bool = False,
                 max_recovery: int = 0, init_pose=None,
                 progress_cb=None) -> dict:
    """Laedt oder berechnet eine kausale Runtime-Simulation."""
    def _progress(pct: int, text: str):
        if progress_cb is not None:
            progress_cb(int(pct), text)

    if init_pose is not None:
        force = True
    cache_arr = CACHE_DIR / f"{bag_path.name}_runtime.npz"
    cache_evts = CACHE_DIR / f"{bag_path.name}_runtime.json"

    if force or not cache_arr.exists() or not cache_evts.exists():
        _progress(5, "Loading bag and map")
        simulator = RuntimeSimulator(
            bag_path, map_yaml, RuntimeConfig(), init_pose=init_pose,
            max_recovery=max_recovery, progress=False)
        _progress(20, "Simulating runtime behavior")
        result = simulator.run()
        _progress(90, "Saving runtime cache")
        save_result(result, cache_arr, bag_path, map_yaml, RuntimeConfig(), 1.0)
    else:
        _progress(60, "Loading runtime cache")

    d = np.load(cache_arr)
    with open(cache_evts) as f:
        metadata = json.load(f)

    scan_t = d["scan_t"]
    poses = d["poses"]
    xs, ys, ths = poses[:, 0], poses[:, 1], poses[:, 2]
    Ps = d["covariances"]
    valid_ratios = d["valid_ratios"]
    is_lost = d["modes"] != 2
    z_real = d["z_obs"]
    z_hat = d["z_hat"]
    beam_angles = d["beam_angles"]
    map_img = d["map_img"]
    map_origin = d["map_origin"]
    MAP_RES = float(d["map_res"])

    events = []
    for event in metadata.get("events", []):
        event_t = float(event["time"])
        step = int(np.clip(np.searchsorted(scan_t, event_t), 0, len(scan_t) - 1))
        typ = event.get("type", "")
        if typ in ("initialized", "initialpose"):
            typ = "init"
        pose = event.get("search_pose") if typ == "init" else event.get("pose")
        if pose is None:
            pose = poses[step].tolist()
        events.append((step, event_t, typ, tuple(float(v) for v in pose)))

    t_rel      = scan_t - scan_t[0]

    _progress(100, "Done")
    return dict(
        data=None,
        scan_t=scan_t,
        z_real=z_real,
        z_hat=z_hat,
        beam_angles=beam_angles,
        t_rel=t_rel,
        DURATION=float(scan_t[-1] - scan_t[0]),
        DT_MS=max(1, int(np.mean(np.diff(scan_t)) * 1000)),
        MAP_IMG=map_img,
        MAP_RES=MAP_RES,
        MAP_ORIGIN=map_origin,
        MAP_H=map_img.shape[0],
        MAP_W=map_img.shape[1],
        ROBOT_R_PX=ROBOT_R / MAP_RES,
        N_STEPS=len(scan_t),
        xs=xs, ys=ys, ths=ths, Ps=Ps,
        valid_ratios=valid_ratios,
        is_lost=is_lost,
        events=events,
        cache_arr=cache_arr,
        cache_evts=cache_evts,
        bag_path=bag_path,
    )


def has_cached_result(bag_path: Path) -> bool:
    cache_arr = CACHE_DIR / f"{bag_path.name}_runtime.npz"
    cache_evts = CACHE_DIR / f"{bag_path.name}_runtime.json"
    return cache_arr.exists() and cache_evts.exists()


class DatasetLoadThread(QThread):
    progress = pyqtSignal(int, str)
    loaded = pyqtSignal(object, str)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, bag_path, map_yaml, force, max_recovery,
                 init_pose, pose_source, parent=None):
        super().__init__(parent)
        self.bag_path = bag_path
        self.map_yaml = map_yaml
        self.force = force
        self.max_recovery = max_recovery
        self.init_pose = init_pose
        self.pose_source = pose_source

    def run(self):
        try:
            def _progress(pct, text):
                if self.isInterruptionRequested():
                    raise LoadCancelled()
                self.progress.emit(pct, text)

            ds = load_dataset(
                self.bag_path, self.map_yaml,
                force=self.force,
                max_recovery=self.max_recovery,
                init_pose=self.init_pose,
                progress_cb=_progress)
        except LoadCancelled:
            self.cancelled.emit()
            return
        except Exception:
            import traceback
            self.failed.emit(traceback.format_exc())
            return
        self.loaded.emit(ds, self.pose_source)


# ══════════════════════════════════════════════════════════════════════
# HAUPTFENSTER
# ══════════════════════════════════════════════════════════════════════
class EKFPlayer(QMainWindow):

    def __init__(self, ds: dict):
        super().__init__()
        self.ds = ds
        self.step    = 0
        self.playing = False
        self._cur_bag_path = ds.get("bag_path")
        self._cur_map_yaml = ds.get("map_yaml")
        self._pose_store = self._load_pose_store()
        self._init_pose = None
        self._found_pose = None

        # ── Toggle-Stylesheet ────────────────────────────────────────
        self._tog = (
            "QPushButton{background:#222240;color:#888899;border-radius:4px;"
            "padding:3px 8px;border:1px solid #333366;}"
            "QPushButton:checked{background:#1a3355;color:#44aaff;"
            "border:1px solid #44aaff;}"
            "QPushButton:hover{background:#333355;}")

        self.btn_init_pose = QPushButton("⟳")
        self.btn_init_pose.setCheckable(True)
        self.btn_init_pose.setFixedSize(32, 22)
        self.btn_init_pose.setToolTip("Set a custom start pose with two clicks on the map.")
        self.btn_init_pose.setStyleSheet(
            "QPushButton{background:#1a2a1a;color:#88cc44;border-radius:4px;padding:4px;}"
            "QPushButton:checked{background:#2a4a00;color:#aaff44;"
            "border:1px solid #aaff44;}"
            "QPushButton:hover{background:#2a3a1a;}")
        self.btn_init_pose.toggled.connect(self._on_init_pose_toggle)

        # Bag selector
        self._sel_widget = self._build_selector()

        # Map plot
        self.map_plot = pg.PlotWidget()
        self.map_plot.setAspectLocked(True)
        self.map_plot.hideAxis("left"); self.map_plot.hideAxis("bottom")
        self.map_plot.getViewBox().invertY(False)
        self.map_plot.getViewBox().setMouseMode(pg.ViewBox.PanMode)

        # Diagnostics plot
        self.diag_plot = pg.PlotWidget()
        self.diag_plot.setFixedHeight(160)
        self.diag_plot.setLabel("bottom", "t [s]")
        self.diag_plot.setLabel("left", "valid-beam-ratio")

        # Dynamic map items
        self._pen_hit  = pg.mkPen("#44ee88", width=2.5)
        self._pen_miss = pg.mkPen("#ff3344", width=1.5)

        self.trail_item = pg.PlotCurveItem(pen=pg.mkPen("#44aaff", width=2))
        self.robot_item = pg.ScatterPlotItem(
            size=ds["ROBOT_R_PX"] * 2,
            pen=pg.mkPen("#44aaff"), brush=pg.mkBrush("#44aaff40"))
        self.arrow_item = pg.PlotCurveItem(pen=pg.mkPen("#44aaff", width=2))
        self.beams_curves = [pg.PlotCurveItem(pen=self._pen_hit)
                             for _ in range(N_BEAMS)]
        self.info_label = pg.TextItem("", color="w",
                                      fill=pg.mkBrush("#0f346090"))
        self.info_label.setPos(5, 15)

        for c in self.beams_curves:
            self.map_plot.addItem(c)
        for item in [self.trail_item, self.robot_item,
                     self.arrow_item, self.info_label]:
            self.map_plot.addItem(item)

        # ── Controls ─────────────────────────────────────────────────
        self.btn_ghost = QPushButton("--  Path")
        self.btn_ghost.setCheckable(True); self.btn_ghost.setChecked(True)
        self.btn_ghost.setFixedWidth(90)
        self.btn_ghost.setStyleSheet(self._tog)

        self.btn_beams = QPushButton("★  Beams")
        self.btn_beams.setCheckable(True); self.btn_beams.setChecked(True)
        self.btn_beams.setFixedWidth(90)
        self.btn_beams.setStyleSheet(self._tog)
        self.btn_beams.toggled.connect(self._on_beams_toggle)

        self.btn_trail30 = QPushButton("⌚  30 s")
        self.btn_trail30.setCheckable(True); self.btn_trail30.setChecked(False)
        self.btn_trail30.setFixedWidth(90)
        self.btn_trail30.setStyleSheet(self._tog)
        self.btn_trail30.toggled.connect(lambda _: self._draw(self.step))

        self.btn_play = QPushButton("▶  Play")
        self.btn_play.setFixedWidth(100)
        self.btn_play.setStyleSheet(
            "QPushButton{background:#1a3a1a;color:#00ff88;border-radius:4px;padding:4px;}"
            "QPushButton:hover{background:#2a6a2a;}")
        self.btn_play.clicked.connect(self._on_play)

        self.btn_recomp = QPushButton("⟳  Recompute")
        self.btn_recomp.setFixedWidth(120)
        self.btn_recomp.setStyleSheet(
            "QPushButton{background:#3a1a1a;color:#ffaa44;border-radius:4px;padding:4px;}"
            "QPushButton:hover{background:#6a2a2a;}")
        self.btn_recomp.clicked.connect(self._on_recompute)

        self.btn_search_once = QPushButton("1×  Recovery")
        self.btn_search_once.setFixedWidth(105)
        self.btn_search_once.setCheckable(True)
        self.btn_search_once.setToolTip(
            "Allow exactly one recovery search after LOST. "
            "The initial search without a start pose does not count.")
        self.btn_search_once.setStyleSheet(
            "QPushButton{background:#1a1a3a;color:#aaaaff;border-radius:4px;padding:4px;}"
            "QPushButton:checked{background:#2a2a6a;color:#ffffff;border:1px solid #44aaff;}"
            "QPushButton:hover{background:#2a2a5a;}")

        self.btn_search_none = QPushButton("0×  Recovery")
        self.btn_search_none.setFixedWidth(105)
        self.btn_search_none.setCheckable(True)
        self.btn_search_none.setToolTip(
            "Disable recovery searches after LOST. "
            "If no start pose is selected, the required initial search still runs.")
        self.btn_search_none.setStyleSheet(
            "QPushButton{background:#1a1a3a;color:#aaaaff;border-radius:4px;padding:4px;}"
            "QPushButton:checked{background:#3a1a3a;color:#ffffff;border:1px solid #ff44ff;}"
            "QPushButton:hover{background:#2a2a5a;}")

        self._recovery_grp = QButtonGroup(self)
        self._recovery_grp.setExclusive(False)
        self._recovery_grp.addButton(self.btn_search_once)
        self._recovery_grp.addButton(self.btn_search_none)
        self._recovery_grp.buttonClicked.connect(self._on_recovery_btn)

        self.t_label = QLabel("t = 0.0 s   |   ok")
        self.t_label.setStyleSheet("color:#aaaaff; padding-left:8px;")
        self.progress = QProgressBar()
        self.progress.setFixedWidth(190)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.hide()
        self.progress.setStyleSheet(
            "QProgressBar{background:#151528;color:#ccccee;"
            "border:1px solid #333366;border-radius:4px;text-align:center;}"
            "QProgressBar::chunk{background:#44aaff;border-radius:3px;}")
        self.btn_cancel_load = QPushButton("Abbrechen")
        self.btn_cancel_load.setFixedWidth(90)
        self.btn_cancel_load.hide()
        self.btn_cancel_load.setStyleSheet(
            "QPushButton{background:#3a1a1a;color:#ff7777;"
            "border-radius:4px;padding:4px;border:1px solid #663344;}"
            "QPushButton:hover{background:#5a2525;color:#ffaaaa;}"
            "QPushButton:disabled{background:#222240;color:#777788;border:1px solid #333344;}")
        self.btn_cancel_load.clicked.connect(self._on_cancel_load)
        self._load_thread = None
        self._play_after_load = False

        ctrl = QWidget()
        ctrl_lay = QHBoxLayout(ctrl)
        ctrl_lay.setContentsMargins(4, 2, 4, 2)
        ctrl_lay.addWidget(self.btn_play)
        ctrl_lay.addWidget(self.btn_recomp)
        ctrl_lay.addWidget(self.btn_search_once)
        ctrl_lay.addWidget(self.btn_search_none)
        ctrl_lay.addWidget(self.t_label)
        ctrl_lay.addWidget(self.progress)
        ctrl_lay.addWidget(self.btn_cancel_load)
        ctrl_lay.addStretch()
        ctrl_lay.addWidget(self.btn_ghost)
        ctrl_lay.addWidget(self.btn_beams)
        ctrl_lay.addWidget(self.btn_trail30)

        # ── Layout ───────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.map_plot)
        splitter.addWidget(self.diag_plot)
        splitter.setSizes([580, 160])

        central = QWidget()
        main_lay = QVBoxLayout(central)
        main_lay.setContentsMargins(4, 4, 4, 4)
        main_lay.setSpacing(2)
        main_lay.addWidget(self._sel_widget)
        main_lay.addWidget(splitter)
        main_lay.addWidget(ctrl)
        self.setCentralWidget(central)
        self.resize(1100, 820)

        # ── Timer ────────────────────────────────────────────────────
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)

        # ── Startpose-Modus ──────────────────────────────────────────
        self._init_click1 = None   # (plot_x, plot_y) nach 1. Klick
        self._init_dot   = pg.ScatterPlotItem(
            size=14, pen=pg.mkPen("#ff9900", width=2),
            brush=pg.mkBrush("#ff990060"))
        self._init_arrow = pg.PlotCurveItem(pen=pg.mkPen("#ff9900", width=2.5))
        self._found_dot = pg.ScatterPlotItem(
            size=12, pen=pg.mkPen("#44ddff", width=2),
            brush=pg.mkBrush("#44ddff50"))
        self._found_arrow = pg.PlotCurveItem(pen=pg.mkPen("#44ddff", width=2.0))
        self.map_plot.addItem(self._init_dot)
        self.map_plot.addItem(self._init_arrow)
        self.map_plot.addItem(self._found_dot)
        self.map_plot.addItem(self._found_arrow)
        self.map_plot.scene().sigMouseClicked.connect(self._on_map_click)
        self.map_plot.scene().sigMouseMoved.connect(self._on_map_move)
        self._sync_poses_from_store()

        # Initialer Aufbau
        self._rebuild(ds, initial=True)
        if self._cur_env == "elevator" and self._cur_bag_path == BAGS["elevator"]["bags"]["slow"]:
            self._on_bag_select("elevator", "slow")

    # ── Bag-Selektor bauen ───────────────────────────────────────────
    def _build_selector(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(70)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(3)

        # Zeile 1: Umgebungen
        env_row = QWidget()
        env_lay = QHBoxLayout(env_row)
        env_lay.setContentsMargins(0, 0, 0, 0)
        env_lay.setSpacing(4)
        env_lbl = QLabel("Env:")
        env_lbl.setStyleSheet("color:#888899; font-size:11px;")
        env_lay.addWidget(env_lbl)

        self._env_btns: dict[str, QPushButton] = {}
        self._env_group = QButtonGroup(w)
        self._env_group.setExclusive(True)

        for env in BAGS:
            btn = QPushButton(env)
            btn.setCheckable(True)
            btn.setFixedHeight(22)
            btn.setStyleSheet(self._tog if hasattr(self, "_tog") else (
                "QPushButton{background:#222240;color:#888899;border-radius:4px;"
                "padding:2px 10px;border:1px solid #333366;}"
                "QPushButton:checked{background:#1a3355;color:#44aaff;"
                "border:1px solid #44aaff;}"
                "QPushButton:hover{background:#333355;}"))
            self._env_btns[env] = btn
            self._env_group.addButton(btn)
            env_lay.addWidget(btn)
            btn.clicked.connect(lambda _, e=env: self._on_env_select(e))
        env_lay.addStretch()
        self.btn_use_manual = QPushButton("Custom: none")
        self.btn_use_manual.setCheckable(True)
        self.btn_use_manual.setFixedHeight(22)
        self.btn_use_manual.setMinimumWidth(180)
        self.btn_use_manual.setToolTip("Use this saved custom start pose for recompute.")

        self.btn_use_found = QPushButton("Found: none")
        self.btn_use_found.setCheckable(True)
        self.btn_use_found.setFixedHeight(22)
        self.btn_use_found.setMinimumWidth(190)
        self.btn_use_found.setToolTip("Use the last early search pose for recompute.")

        self.btn_clear_found = QPushButton("x")
        self.btn_clear_found.setFixedSize(24, 22)
        self.btn_clear_found.setToolTip("Clear the found pose for this bag.")
        self.btn_clear_found.clicked.connect(self._on_clear_found_pose)

        self._pose_source_grp = QButtonGroup(w)
        self._pose_source_grp.setExclusive(False)
        self._pose_source_grp.addButton(self.btn_use_manual)
        self._pose_source_grp.addButton(self.btn_use_found)
        self._pose_source_grp.buttonClicked.connect(self._on_pose_source_btn)

        manual_box = QWidget()
        manual_lay = QHBoxLayout(manual_box)
        manual_lay.setContentsMargins(0, 0, 0, 0)
        manual_lay.setSpacing(0)
        manual_lay.addWidget(self.btn_use_manual)
        manual_lay.addWidget(self.btn_init_pose)
        env_lay.addWidget(manual_box)

        found_box = QWidget()
        found_lay = QHBoxLayout(found_box)
        found_lay.setContentsMargins(0, 0, 0, 0)
        found_lay.setSpacing(0)
        found_lay.addWidget(self.btn_use_found)
        found_lay.addWidget(self.btn_clear_found)
        env_lay.addWidget(found_box)

        # Row 2: bag buttons.
        bag_row = QWidget()
        self._bag_row_lay = QHBoxLayout(bag_row)
        self._bag_row_lay.setContentsMargins(0, 0, 0, 0)
        self._bag_row_lay.setSpacing(4)
        bag_lbl = QLabel("Bag:")
        bag_lbl.setStyleSheet("color:#888899; font-size:11px;")
        self._bag_row_lay.addWidget(bag_lbl)
        self._bag_stretch = None

        self._bag_btns: dict[str, QPushButton] = {}
        self._bag_group = QButtonGroup(w)
        self._bag_group.setExclusive(True)

        lay.addWidget(env_row)
        lay.addWidget(bag_row)

        # Default: elevator / slow.
        self._cur_env = "elevator"
        self._env_btns["elevator"].setChecked(True)
        self._populate_bag_row("elevator", default="fast")
        return w

    def _populate_bag_row(self, env: str, default=None):
        # Remove old bag buttons.
        for btn in list(self._bag_btns.values()):
            self._bag_row_lay.removeWidget(btn)
            btn.deleteLater()
        if self._bag_stretch is not None:
            self._bag_row_lay.removeItem(self._bag_stretch)
        self._bag_btns.clear()

        # Clear button group.
        for btn in self._bag_group.buttons():
            self._bag_group.removeButton(btn)

        _tog_small = (
            "QPushButton{background:#222240;color:#888899;border-radius:4px;"
            "padding:2px 10px;border:1px solid #333366;font-size:11px;}"
            "QPushButton:checked{background:#1a3355;color:#44aaff;"
            "border:1px solid #44aaff;}"
            "QPushButton:hover{background:#333355;}")

        for label, bag_path in BAGS[env]["bags"].items():
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedHeight(22)
            btn.setStyleSheet(_tog_small)
            if not has_cached_result(bag_path) and not bag_path.exists():
                btn.setToolTip("No cache found. Add the raw bag under bags/ to recompute.")
            self._bag_btns[label] = btn
            self._bag_group.addButton(btn)
            self._bag_row_lay.addWidget(btn)
            btn.clicked.connect(
                lambda _, e=env, lbl=label: self._on_bag_select(e, lbl))

        # Stretch at the end.
        from PyQt6.QtWidgets import QSpacerItem, QSizePolicy
        self._bag_stretch = QSpacerItem(
            0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._bag_row_lay.addItem(self._bag_stretch)

        # Select the default scenario.
        if default and default in self._bag_btns:
            self._bag_btns[default].setChecked(True)
        elif self._bag_btns:
            next(iter(self._bag_btns.values())).setChecked(True)

    def _on_env_select(self, env: str):
        self._cur_env = env
        self._populate_bag_row(env)
        # Show only the map; no EKF
        map_yaml = BAGS[env]["map"]
        # Use the first configured bag as reference for bag_path only.
        first_bag = next(iter(BAGS[env]["bags"].values()), None)
        ds_map = load_map_only(map_yaml, bag_path=first_bag)
        self._cur_bag_path = first_bag
        self._cur_map_yaml = map_yaml
        # Activate the first bag button and load cache if available.
        for lbl, p in BAGS[env]["bags"].items():
            if lbl not in self._bag_btns:
                continue
            self._bag_btns[lbl].setChecked(True)
            self._on_bag_select(env, lbl)
            return
        # No loadable bag; show only the map
        self._rebuild(ds_map)

    def _on_bag_select(self, env: str, label: str):
        bag_path = BAGS[env]["bags"][label]
        map_yaml = BAGS[env]["map"]
        self.timer.stop(); self.playing = False
        self.btn_play.setText("▶  Play")
        self._cur_env      = env
        self._cur_bag_path = bag_path
        self._cur_map_yaml = map_yaml
        self._init_click1  = None
        self._sync_poses_from_store()
        self._update_recomp_style()
        # Load cache directly when available; otherwise show only the map.
        # Important: bag clicks with a selected start pose must not
        # trigger recompute. load_dataset(..., init_pose=...) sets force=True.
        if has_cached_result(bag_path):
            self._start_dataset_load(
                force=False, use_selected_pose=False, play_after=False)
            return
        # No cache or error; show only the map
        ds_map = load_map_only(map_yaml, bag_path=bag_path)
        self._rebuild(ds_map)

    # Rebuild plots.
    def _reset_plot_views(self, ds: dict):
        vb = self.map_plot.getViewBox()
        vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
        vb.setRange(
            xRange=(0, ds["MAP_W"]),
            yRange=(0, ds["MAP_H"]),
            padding=0.08)

        if ds.get("map_only"):
            self.diag_plot.setXRange(0, 1, padding=0)
            self.diag_plot.setYRange(0, 1.1, padding=0)
            self.diag_plot.setLimits(xMin=0, xMax=1, yMin=0, yMax=1.1)
        else:
            self.diag_plot.setXRange(0, ds["DURATION"], padding=0)
            self.diag_plot.setYRange(0, 1.1, padding=0)
            self.diag_plot.setLimits(
                xMin=0, xMax=ds["DURATION"], yMin=0, yMax=1.1)

    def _rebuild(self, ds: dict, initial: bool = False):
        self.ds   = ds
        self.step = 0

        # -- Map-only mode ──────────────────────────────────────────
        if ds.get("map_only"):
            self.timer.stop(); self.playing = False
            self.btn_play.setText("▶  Play")
            # Clear map plot
            for item in list(self.map_plot.items()):
                if item not in (self.trail_item, self.robot_item,
                                self.arrow_item, self.info_label,
                                self._init_dot, self._init_arrow,
                                self._found_dot, self._found_arrow,
                                *self.beams_curves):
                    self.map_plot.removeItem(item)
            # Hide dynamic items.
            self.trail_item.setData([], [])
            self.robot_item.setData([], [])
            self.arrow_item.setData([], [])
            for c in self.beams_curves:
                c.setData([], [])
            self.info_label.setText("")
            self.t_label.setText("No data - set a start pose, then press Play")
            self.t_label.setStyleSheet("color:#888899; padding-left:8px;")
            # Robot size.
            self.robot_item.setSize(ds["ROBOT_R_PX"] * 2)
            # Show map
            img_item = pg.ImageItem(ds["MAP_IMG"])
            self.map_plot.addItem(img_item)
            img_item.setZValue(-20)
            self._reset_plot_views(ds)
            bag_name = ds["bag_path"].name if ds.get("bag_path") else "–"
            self.setWindowTitle(f"Runtime Simulation Player · {bag_name}  [no data]")
            # Clear diagnostics plot.
            self.diag_plot.clear()
            self._reset_plot_views(ds)
            return

        # Update timer interval.
        self.timer.setInterval(ds["DT_MS"])

        # Clear map plot and recreate only static items
        for item in list(self.map_plot.items()):
            if item not in (self.trail_item, self.robot_item,
                            self.arrow_item, self.info_label,
                            self._init_dot, self._init_arrow,
                            self._found_dot, self._found_arrow,
                            *self.beams_curves):
                self.map_plot.removeItem(item)

        # Update robot size.
        self.robot_item.setSize(ds["ROBOT_R_PX"] * 2)

        # Map
        img_item = pg.ImageItem(ds["MAP_IMG"])
        self.map_plot.addItem(img_item)
        img_item.setZValue(-20)
        self._reset_plot_views(ds)

        # Ghost
        self.ghost = pg.PlotCurveItem(
            self._px(ds["xs"]), self._py(ds["ys"]),
            pen=pg.mkPen("#99aabb", width=1.8, style=Qt.PenStyle.DashLine))
        self.ghost.setZValue(-5)
        self.map_plot.addItem(self.ghost)
        try:
            self.btn_ghost.toggled.disconnect()
        except (TypeError, RuntimeError):
            pass
        self.btn_ghost.toggled.connect(lambda v: self.ghost.setVisible(v))
        self.ghost.setVisible(self.btn_ghost.isChecked())

        # Lost/Recovered Marker
        lost_idx = [e[0] for e in ds["events"] if e[2] == "lost"]
        rec_idx  = [e[0] for e in ds["events"] if e[2] == "recovered"]
        if lost_idx:
            self.map_plot.addItem(pg.ScatterPlotItem(
                self._px(ds["xs"][lost_idx]),
                self._py(ds["ys"][lost_idx]),
                symbol="x", size=14, pen=pg.mkPen("#ff3344", width=2.5)))
        if rec_idx:
            self.map_plot.addItem(pg.ScatterPlotItem(
                self._px(ds["xs"][rec_idx]),
                self._py(ds["ys"][rec_idx]),
                symbol="star", size=16,
                pen=pg.mkPen("#ffcc00", width=1.5),
                brush=pg.mkBrush("#ffcc0080")))

        # Fenster-Titel
        self.setWindowTitle(f"Runtime Simulation Player · {ds['bag_path'].name}")

        # --- Diagnose-Plot neu aufbauen ------------------------------
        self.diag_plot.clear()
        self._reset_plot_views(ds)
        self.diag_plot.setMouseEnabled(x=True, y=False)

        # Scrub via Left-Drag
        _dvb = self.diag_plot.getViewBox()
        def _diag_drag(ev, axis=None, _vb=_dvb):
            if ev.button() == Qt.MouseButton.LeftButton:
                ev.accept()
                pos = _vb.mapSceneToView(ev.scenePos())
                i = int(np.clip(
                    np.searchsorted(ds["t_rel"], float(pos.x())),
                    0, ds["N_STEPS"] - 1))
                self._seek(i)
            else:
                pg.ViewBox.mouseDragEvent(_vb, ev, axis=axis)
        _dvb.mouseDragEvent = _diag_drag

        t_rel = ds["t_rel"]
        vr    = ds["valid_ratios"]
        self.diag_plot.plot(t_rel, vr,
                            pen=pg.mkPen("#44ffaa", width=1.5))
        fill = pg.FillBetweenItem(
            self.diag_plot.plot(t_rel, vr),
            self.diag_plot.plot(t_rel, np.zeros(ds["N_STEPS"])),
            brush=pg.mkBrush("#44ffaa18"))
        self.diag_plot.addItem(fill)
        self.diag_plot.addLine(
            y=LOST_VALID_RATIO,
            pen=pg.mkPen("#ff3344", width=1, style=Qt.PenStyle.DashLine))

        # Farbige Intervalle
        _evmap: dict = {}
        _cur_lost = None
        for e in sorted(ds["events"], key=lambda e: e[1]):
            _, t_e, typ, *_ = e
            t_r = float(t_e - ds["scan_t"][0])
            if typ == "lost":
                _cur_lost = t_r
                _evmap[_cur_lost] = {"lost": t_r}
            elif typ == "searching" and _cur_lost is not None:
                _evmap[_cur_lost]["searching"] = t_r
            elif typ == "recovered" and _cur_lost is not None:
                _evmap[_cur_lost]["recovered"] = t_r
                _cur_lost = None

        for ev in _evmap.values():
            t_lost = ev.get("lost")
            t_srch = ev.get("searching")
            t_rec  = ev.get("recovered")
            if t_lost is None or t_rec is None:
                continue
            t_srch_eff = t_srch if t_srch is not None else t_rec
            if t_srch_eff - t_lost > 0.05:
                reg_red = pg.LinearRegionItem(
                    [t_lost, t_srch_eff], movable=False,
                    brush=pg.mkBrush("#ff334460"),
                    pen=pg.mkPen("#ff334480", width=0.5))
                reg_red.setZValue(-10)
                self.diag_plot.addItem(reg_red)
            if t_srch is not None and t_rec - t_srch > 0.1:
                reg_blue = pg.LinearRegionItem(
                    [t_srch, t_rec], movable=False,
                    brush=pg.mkBrush("#44aaff50"),
                    pen=pg.mkPen("#44aaff70", width=0.5))
                reg_blue.setZValue(-9)
                self.diag_plot.addItem(reg_blue)

        for e in ds["events"]:
            _, t_e, typ, *_ = e
            color = {"lost": "#ff3344", "searching": "#ff8800",
                     "recovered": "#ffcc00"}.get(typ, "#888888")
            self.diag_plot.addLine(
                x=float(t_e - ds["scan_t"][0]),
                pen=pg.mkPen(color, width=1.2, alpha=180))

        self.d_marker = self.diag_plot.addLine(
            x=0, pen=pg.mkPen("#44aaff", width=1.8),
            movable=True, hoverPen=pg.mkPen("#88ccff", width=2.5))
        self.d_marker.sigPositionChanged.connect(self._on_marker_drag)
        try:
            self.diag_plot.scene().sigMouseClicked.disconnect(self._on_diag_click)
        except (TypeError, RuntimeError):
            pass
        self.diag_plot.scene().sigMouseClicked.connect(self._on_diag_click)

        self._draw(0)

    # ── Koordinaten-Hilfsfunktionen ──────────────────────────────────
    def _px(self, x_world):
        return (np.asarray(x_world) - self.ds["MAP_ORIGIN"][0]) / self.ds["MAP_RES"]

    def _py(self, y_world):
        return self.ds["MAP_H"] - (np.asarray(y_world) - self.ds["MAP_ORIGIN"][1]) / self.ds["MAP_RES"]

    # ── Draw ─────────────────────────────────────────────────────────
    def _draw(self, i: int):
        ds = self.ds
        i  = int(np.clip(i, 0, ds["N_STEPS"] - 1))
        self.step = i

        xs, ys, ths = ds["xs"], ds["ys"], ds["ths"]
        t_rel       = ds["t_rel"]
        is_lost     = ds["is_lost"]

        # Trail
        i0 = 0
        if self.btn_trail30.isChecked():
            i0 = max(0, int(np.searchsorted(t_rel, t_rel[i] - 30.0)))
        self.trail_item.setData(self._px(xs[i0:i+1]), self._py(ys[i0:i+1]))

        # Robot
        rx, ry = float(self._px(xs[i])), float(self._py(ys[i]))
        color  = "#ff3344" if is_lost[i] else "#44aaff"
        self.robot_item.setData([rx], [ry])
        self.robot_item.setPen(pg.mkPen(color))
        self.robot_item.setBrush(pg.mkBrush(color + "40"))

        ax = rx + np.cos(ths[i]) * ds["ROBOT_R_PX"] * 2.2
        ay = ry - np.sin(ths[i]) * ds["ROBOT_R_PX"] * 2.2
        self.arrow_item.setData([rx, ax], [ry, ay])
        self.arrow_item.setPen(pg.mkPen(color, width=2))

        # Beams
        z_hat = ds["z_hat"][i]
        z_obs = ds["z_real"][i]
        ba    = ds["beam_angles"]
        for k in range(N_BEAMS):
            r   = z_obs[k]
            a   = ths[i] + ba[k]
            hit = (r < MAX_RANGE - 1e-3 and z_hat[k] < MAX_RANGE - 1e-3
                   and abs(r - z_hat[k]) < GATE)
            dist = r if r < MAX_RANGE - 1e-3 else MAX_RANGE
            bx = rx + np.cos(a) * dist / ds["MAP_RES"]
            by = ry - np.sin(a) * dist / ds["MAP_RES"]
            self.beams_curves[k].setData([rx, bx], [ry, by])
            self.beams_curves[k].setPen(self._pen_hit if hit else self._pen_miss)

        # Diagnose-Marker
        self.d_marker.blockSignals(True)
        self.d_marker.setValue(float(t_rel[i]))
        self.d_marker.blockSignals(False)

        # Info
        Ps = ds["Ps"]
        vr = ds["valid_ratios"]
        self.info_label.setText(
            f"t = {t_rel[i]:.1f} s   "
            f"{'LOST' if is_lost[i] else 'ok'}   "
            f"valid {vr[i]*100:.0f}%   "
            f"σx={np.sqrt(Ps[i][0,0]):.3f}  σy={np.sqrt(Ps[i][1,1]):.3f}"
        )
        self.t_label.setText(
            f"t = {t_rel[i]:.1f} s   |   {'⚠ LOST' if is_lost[i] else 'ok'}"
        )
        self.t_label.setStyleSheet(
            f"color:{'#ff6060' if is_lost[i] else '#aaaaff'}; padding-left:8px;"
        )

    # ── Callbacks ────────────────────────────────────────────────────
    def _on_beams_toggle(self, visible: bool):
        for c in self.beams_curves:
            c.setVisible(visible)

    def _seek(self, i: int):
        self._draw(int(np.clip(i, 0, self.ds["N_STEPS"] - 1)))

    def _on_marker_drag(self):
        t = float(self.d_marker.value())
        i = int(np.clip(
            np.searchsorted(self.ds["t_rel"], t), 0, self.ds["N_STEPS"] - 1))
        self._seek(i)

    def _on_diag_click(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = self.diag_plot.getViewBox().mapSceneToView(event.scenePos())
        i = int(np.clip(
            np.searchsorted(self.ds["t_rel"], float(pos.x())),
            0, self.ds["N_STEPS"] - 1))
        self._seek(i)

    def _on_play(self):
        # Wenn noch kein EKF berechnet: erst laden
        if self.ds.get("map_only"):
            self._load_ekf_now(play_after=True)
            if self.ds.get("map_only"):  # loading failed
                return
        self.playing = not self.playing
        if self.playing:
            self.btn_play.setText("⏸  Pause")
            self.timer.start()
        else:
            self.btn_play.setText("▶  Play")
            self.timer.stop()

    def _tick(self):
        nxt = self.step + 1
        if nxt >= self.ds["N_STEPS"]:
            self.playing = False
            self.btn_play.setText("▶  Play")
            self.timer.stop()
            return
        self._draw(nxt)

    def _cur_max_recovery(self) -> int:
        """Number of allowed recovery searches after LOST; the initial search does not count."""
        if self.btn_search_once.isChecked():
            return 1
        if self.btn_search_none.isChecked():
            return -1
        return 0

    def _set_loading_ui(self, loading: bool):
        for btn in [
            self.btn_play, self.btn_recomp, self.btn_init_pose,
            self.btn_use_manual, self.btn_use_found,
            self.btn_search_once, self.btn_search_none,
        ]:
            btn.setEnabled(not loading)
        self.btn_clear_found.setEnabled(not loading and self._found_pose is not None)
        for btn in getattr(self, "_env_btns", {}).values():
            btn.setEnabled(not loading)
        for btn in getattr(self, "_bag_btns", {}).values():
            btn.setEnabled(not loading)
        if loading:
            self.progress.setValue(0)
            self.progress.show()
            self.btn_cancel_load.setEnabled(True)
            self.btn_cancel_load.show()
        else:
            self.progress.hide()
            self.btn_cancel_load.hide()

    def _on_load_progress(self, pct: int, text: str):
        self.progress.setValue(max(0, min(100, int(pct))))
        self.progress.setFormat(f"{text}  {self.progress.value()}%")
        self.t_label.setText(text)
        self.t_label.setStyleSheet("color:#ffaa44; padding-left:8px;")

    def _start_dataset_load(self, force: bool, use_selected_pose: bool = True,
                            play_after: bool = False):
        bag_path = getattr(self, "_cur_bag_path", self.ds.get("bag_path"))
        map_yaml = getattr(self, "_cur_map_yaml", BAGS[self._cur_env]["map"])
        if bag_path is None:
            self.t_label.setText("No bag selected")
            self.t_label.setStyleSheet("color:#ff6060; padding-left:8px;")
            return
        if self._load_thread is not None and self._load_thread.isRunning():
            return
        self.timer.stop(); self.playing = False
        self.btn_play.setText("▶  Play")
        init_pose, pose_source = self._selected_init_pose() if use_selected_pose else (None, "cache")
        self._play_after_load = play_after
        self._set_loading_ui(True)
        self._on_load_progress(0, "Starting computation")
        self._load_thread = DatasetLoadThread(
            bag_path=bag_path,
            map_yaml=map_yaml,
            force=force,
            max_recovery=self._cur_max_recovery(),
            init_pose=init_pose,
            pose_source=pose_source,
            parent=self)
        self._load_thread.progress.connect(self._on_load_progress)
        self._load_thread.loaded.connect(self._on_dataset_loaded)
        self._load_thread.failed.connect(self._on_dataset_failed)
        self._load_thread.cancelled.connect(self._on_dataset_cancelled)
        self._load_thread.finished.connect(self._load_thread.deleteLater)
        self._load_thread.start()

    def _on_dataset_loaded(self, ds: dict, pose_source: str):
        self._remember_found_pose_from_dataset(ds, pose_source)
        self._rebuild(ds)
        self._refresh_pose_ui()
        self._set_loading_ui(False)
        self._load_thread = None
        if self._play_after_load:
            self._play_after_load = False
            self.playing = True
            self.btn_play.setText("⏸  Pause")
            self.timer.start()

    def _on_dataset_failed(self, traceback_text: str):
        print(traceback_text)
        last_line = traceback_text.strip().splitlines()[-1] if traceback_text.strip() else "unknown"
        self.t_label.setText(f"Error: {last_line}")
        self.t_label.setStyleSheet("color:#ff6060; padding-left:8px;")
        self._set_loading_ui(False)
        self._load_thread = None
        self._play_after_load = False

    def _on_dataset_cancelled(self):
        self.t_label.setText("Computation cancelled")
        self.t_label.setStyleSheet("color:#ffaa44; padding-left:8px;")
        self._set_loading_ui(False)
        self._load_thread = None
        self._play_after_load = False

    def _on_cancel_load(self):
        if self._load_thread is None or not self._load_thread.isRunning():
            return
        self.btn_cancel_load.setEnabled(False)
        self.t_label.setText("Cancelling ...")
        self.t_label.setStyleSheet("color:#ffaa44; padding-left:8px;")
        self._load_thread.requestInterruption()

    def _load_ekf_now(self, play_after: bool = False):
        """Load or compute the EKF result for the current bag and rebuild."""
        self._start_dataset_load(force=False, play_after=play_after)

    def _on_recompute(self):
        bag_path = getattr(self, "_cur_bag_path", self.ds.get("bag_path"))
        if bag_path is None:
            return
        self._start_dataset_load(force=True)

    def _on_recovery_btn(self, clicked_btn):
        """Ensure that only one recovery button is active at a time."""
        for btn in self._recovery_grp.buttons():
            if btn is not clicked_btn:
                btn.setChecked(False)

    def _on_search_once(self):
        pass  # Toggle only; recompute is triggered manually.

    # Start-pose feature.
    def _load_pose_store(self) -> dict:
        if not START_POSES_FILE.exists():
            return {"manual": {}, "found": {}}
        try:
            with open(START_POSES_FILE) as f:
                store = json.load(f)
        except Exception:
            return {"manual": {}, "found": {}}
        return {
            "manual": store.get("manual", {}),
            "found": store.get("found", {}),
        }

    def _save_pose_store(self):
        with open(START_POSES_FILE, "w") as f:
            json.dump(self._pose_store, f, indent=2)

    def _bag_key(self, bag_path=None):
        p = bag_path if bag_path is not None else getattr(self, "_cur_bag_path", None)
        if p is None:
            return None
        try:
            return str(Path(p).resolve())
        except Exception:
            return str(p)

    def _pose_from_store(self, kind: str):
        key = self._bag_key()
        if key is None:
            return None
        pose = self._pose_store.get(kind, {}).get(key)
        if pose is None:
            return None
        return tuple(float(v) for v in pose)

    def _save_pose_for_current_bag(self, kind: str, pose):
        key = self._bag_key()
        if key is None or pose is None:
            return
        self._pose_store.setdefault(kind, {})[key] = [float(v) for v in pose]
        self._save_pose_store()

    def _delete_pose_for_current_bag(self, kind: str):
        key = self._bag_key()
        if key is None:
            return
        poses = self._pose_store.setdefault(kind, {})
        if key in poses:
            del poses[key]
            self._save_pose_store()

    def _world_from_plot(self, px: float, py: float):
        """Plot-Pixelkoordinaten → Weltkoordinaten [m]."""
        ds = self.ds
        x = px * ds["MAP_RES"] + ds["MAP_ORIGIN"][0]
        y = (ds["MAP_H"] - py) * ds["MAP_RES"] + ds["MAP_ORIGIN"][1]
        return float(x), float(y)

    def _plot_from_world(self, pose):
        x, y, theta = pose
        px = (float(x) - self.ds["MAP_ORIGIN"][0]) / self.ds["MAP_RES"]
        py = self.ds["MAP_H"] - (float(y) - self.ds["MAP_ORIGIN"][1]) / self.ds["MAP_RES"]
        return float(px), float(py), float(theta)

    def _set_pose_item(self, dot, arrow, pose, color: str, alpha: str, width: float):
        if pose is None:
            dot.setData([], [])
            arrow.setData([], [])
            return
        px, py, theta = self._plot_from_world(pose)
        arrow_len = self.ds["ROBOT_R_PX"] * 5
        ax = px + np.cos(theta) * arrow_len
        ay = py - np.sin(theta) * arrow_len
        dot.setPen(pg.mkPen(color, width=2))
        dot.setBrush(pg.mkBrush(color + alpha))
        arrow.setPen(pg.mkPen(color, width=width))
        dot.setData([px], [py])
        arrow.setData([px, ax], [py, ay])

    def _fmt_pose(self, pose) -> str:
        if pose is None:
            return "none"
        x, y, th = pose
        return f"x={x:+.2f} y={y:+.2f} th={np.degrees(th):+.0f}deg"

    def _sync_poses_from_store(self):
        self._init_pose = self._pose_from_store("manual")
        self._found_pose = self._pose_from_store("found")
        self._init_click1 = None
        if self._init_pose is None:
            self.btn_use_manual.setChecked(False)
        if self._found_pose is None:
            self.btn_use_found.setChecked(False)
        self._refresh_pose_ui()

    def _update_recomp_style(self):
        init_pose, source = self._selected_init_pose()
        if source == "manual":
            self.btn_recomp.setText("⟳  Custom")
            self.btn_recomp.setStyleSheet(
                "QPushButton{background:#3a2a00;color:#ffdd44;"
                "border-radius:4px;padding:4px;border:1px solid #ffdd44;}"
                "QPushButton:hover{background:#6a4a00;}")
        elif source == "found":
            self.btn_recomp.setText("⟳  Found")
            self.btn_recomp.setStyleSheet(
                "QPushButton{background:#142f3a;color:#88ddff;"
                "border-radius:4px;padding:4px;border:1px solid #44ddff;}"
                "QPushButton:hover{background:#1f4f60;}")
        else:
            self.btn_recomp.setText("⟳  Search")
            self.btn_recomp.setStyleSheet(
                "QPushButton{background:#3a1a1a;color:#ffaa44;"
                "border-radius:4px;padding:4px;}"
                "QPushButton:hover{background:#6a2a2a;}")

    def _refresh_pose_ui(self):
        manual_active = self.btn_use_manual.isChecked() and self._init_pose is not None
        found_active = self.btn_use_found.isChecked() and self._found_pose is not None
        manual_edit = self.btn_init_pose.isChecked()

        manual_color = "#aaff44" if (manual_active or manual_edit) else "#777788"
        found_color = "#44ddff" if found_active else "#777788"
        self.btn_use_manual.setText(f"Custom: {self._fmt_pose(self._init_pose)}")
        self.btn_use_found.setText(f"Found: {self._fmt_pose(self._found_pose)}")
        self.btn_use_manual.setStyleSheet(
            f"QPushButton{{background:#222240;color:{manual_color};"
            "font-size:11px;padding:1px 8px;border:1px solid #333344;"
            "border-right:0;border-top-left-radius:4px;border-bottom-left-radius:4px;"
            "border-top-right-radius:0;border-bottom-right-radius:0;text-align:left;}"
            "QPushButton:hover{background:#2a2a48;}"
            "QPushButton:checked{background:#222240;color:#44aaff;}")
        self.btn_init_pose.setStyleSheet(
            f"QPushButton{{background:#1a2a1a;color:{manual_color};"
            "font-size:16px;padding:0 6px;border:1px solid #333344;"
            "border-top-left-radius:0;border-bottom-left-radius:0;"
            "border-top-right-radius:4px;border-bottom-right-radius:4px;}"
            "QPushButton:hover{background:#2a3a1a;}"
            "QPushButton:checked{background:#2a4a00;color:#aaff44;"
            "border:1px solid #333344;}")
        self.btn_use_found.setStyleSheet(
            f"QPushButton{{background:#222240;color:{found_color};"
            "font-size:11px;padding:1px 8px;border:1px solid #333344;"
            "border-right:0;border-top-left-radius:4px;border-bottom-left-radius:4px;"
            "border-top-right-radius:0;border-bottom-right-radius:0;text-align:left;}"
            "QPushButton:hover{background:#2a2a48;}"
            "QPushButton:checked{background:#222240;color:#44ddff;}")
        clear_color = "#ff7777" if self._found_pose is not None else "#9a5555"
        self.btn_clear_found.setEnabled(self._found_pose is not None and self._load_thread is None)
        self.btn_clear_found.setStyleSheet(
            f"QPushButton{{background:#2a1a1a;color:{clear_color};"
            "font-size:14px;padding:0 6px;border:1px solid #663344;"
            "border-top-left-radius:0;border-bottom-left-radius:0;"
            "border-top-right-radius:4px;border-bottom-right-radius:4px;}"
            "QPushButton:hover{background:#3a202a;color:#ff9999;}"
            "QPushButton:disabled{background:#2a1a1a;color:#9a5555;"
            "border:1px solid #663344;}")

        self._set_pose_item(
            self._init_dot, self._init_arrow, self._init_pose,
            "#aaff44" if (manual_active or manual_edit) else "#777788",
            "70" if (manual_active or manual_edit) else "45",
            2.5 if (manual_active or manual_edit) else 1.5)
        self._set_pose_item(
            self._found_dot, self._found_arrow, self._found_pose,
            "#44ddff" if found_active else "#777788",
            "60" if found_active else "35",
            2.0 if found_active else 1.3)
        self._update_recomp_style()

    def _selected_init_pose(self):
        if self.btn_use_manual.isChecked() and self._init_pose is not None:
            return self._init_pose, "manual"
        if self.btn_use_found.isChecked() and self._found_pose is not None:
            return self._found_pose, "found"
        return None, "search"

    def _remember_found_pose_from_dataset(self, ds: dict, pose_source: str):
        if pose_source == "manual":
            return
        found = None
        t0 = float(ds["scan_t"][0]) if "scan_t" in ds else None
        for ev in ds.get("events", []):
            if len(ev) < 4:
                continue
            typ = ev[2]
            t_rel = float(ev[1]) - t0 if t0 is not None else 0.0
            if typ == "init" and pose_source == "search":
                found = ev[3]
            elif typ == "recovered" and t_rel <= 10.0:
                found = ev[3]
        if found is None:
            return
        self._found_pose = tuple(float(v) for v in found)
        self._save_pose_for_current_bag("found", self._found_pose)

    def _on_pose_source_btn(self, clicked_btn):
        for btn in self._pose_source_grp.buttons():
            if btn is not clicked_btn:
                btn.setChecked(False)
        if clicked_btn is self.btn_use_manual and self._init_pose is None:
            self.btn_init_pose.setChecked(True)
            clicked_btn.setChecked(False)
        if clicked_btn is self.btn_use_found and self._found_pose is None:
            clicked_btn.setChecked(False)
        self._refresh_pose_ui()

    def _on_clear_found_pose(self):
        self._found_pose = None
        self.btn_use_found.setChecked(False)
        self._delete_pose_for_current_bag("found")
        self._refresh_pose_ui()

    def _on_init_pose_toggle(self, checked: bool):
        if not checked:
            # Cancel after one click without a second click: reset the marker.
            if self._init_click1 is not None:
                self._init_click1 = None
            self._refresh_pose_ui()
        else:
            self._refresh_pose_ui()

    def _on_map_click(self, event):
        if not self.btn_init_pose.isChecked():
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        event.accept()
        vb  = self.map_plot.getViewBox()
        pos = vb.mapSceneToView(event.scenePos())
        px, py = float(pos.x()), float(pos.y())

        if self._init_click1 is None:
            # 1. Klick: Position merken, Punkt zeichnen
            self._init_click1 = (px, py)
            self._init_dot.setData([px], [py])
            self._init_arrow.setData([px, px], [py, py])
        else:
            # 2. Klick: Richtung → θ berechnen, Pose speichern
            x1, y1 = self._init_click1
            dx, dy  = px - x1, py - y1
            theta   = float(np.arctan2(-dy, dx))  # Plot-y invertiert
            x_w, y_w = self._world_from_plot(x1, y1)
            self._init_pose = (x_w, y_w, theta)
            self._save_pose_for_current_bag("manual", self._init_pose)
            self.btn_use_manual.setChecked(True)
            self.btn_use_found.setChecked(False)
            # Modus beenden
            self._init_click1 = None
            self.btn_init_pose.setChecked(False)
            self._refresh_pose_ui()

    def _on_map_move(self, scene_pos):
        if not self.btn_init_pose.isChecked() or self._init_click1 is None:
            return
        vb  = self.map_plot.getViewBox()
        pos = vb.mapSceneToView(scene_pos)
        px, py = float(pos.x()), float(pos.y())
        x1, y1 = self._init_click1
        dx, dy  = px - x1, py - y1
        theta   = float(np.arctan2(-dy, dx))
        arrow_len = self.ds["ROBOT_R_PX"] * 5
        ax = x1 + np.cos(theta) * arrow_len
        ay = y1 - np.sin(theta) * arrow_len
        self._init_arrow.setData([x1, ax], [y1, ay])


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    bg = QColor("#1a1a2e")
    fg = QColor("#ccccee")
    alt = QColor("#22223e")
    accent = QColor("#44aaff")
    palette = QPalette()
    for role, color in [
        (QPalette.ColorRole.Window, bg),
        (QPalette.ColorRole.Base, bg),
        (QPalette.ColorRole.AlternateBase, alt),
        (QPalette.ColorRole.WindowText, fg),
        (QPalette.ColorRole.Text, fg),
        (QPalette.ColorRole.BrightText, fg),
        (QPalette.ColorRole.ButtonText, fg),
        (QPalette.ColorRole.Button, alt),
        (QPalette.ColorRole.Highlight, accent),
        (QPalette.ColorRole.HighlightedText, QColor("#ffffff")),
        (QPalette.ColorRole.ToolTipBase, alt),
        (QPalette.ColorRole.ToolTipText, fg),
    ]:
        palette.setColor(role, color)
    app.setPalette(palette)
    app.setStyleSheet(
        "QWidget{background:#1a1a2e; color:#ccccee;}"
        "QSplitter::handle{background:#333355;}"
        "QLabel{background:transparent;}")

    default_bag = BAGS["elevator"]["bags"]["fast"]
    default_map = BAGS["elevator"]["map"]
    if has_cached_result(default_bag):
        default_ds = load_dataset(default_bag, default_map, force=False)
    else:
        default_ds = load_map_only(default_map, bag_path=default_bag)
    win = EKFPlayer(default_ds)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
