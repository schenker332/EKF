"""Shared parameters for the runtime simulator and ROS2 EKF node."""

# LiDAR / sensor
N_BEAMS = 30
MAX_RANGE = 3.5
SENSOR_QUEUE_DEPTH = 5

# EKF noise and update
SIGMA_QD = 0.05
SIGMA_QTH = 0.0893
SIGMA_R = 0.06
EPS_H = 0.035
GATE = 0.157
SIGMA_P0 = 0.05

# Lost detection
LOST_VALID_RATIO = 0.25
LOST_CONSEC_STEPS = 5
P_TRACE_MAX = 5.0

# Global localization / recovery
RECOVERY_WAIT_S = 1.0
GRID_SPACING = 0.40
N_THETA = 16
FREE_THRESH = 220
RECOVERY_P_INIT = 0.3
RECOVERY_GATE = 0.5
SCORE_N_EVAL = 30
SCORE_MU = 0.5
