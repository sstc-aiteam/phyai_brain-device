import os

# =========================
# ARM
# =========================
ARMS = {
    "left": {
        "driver": "ur7e",
        "kwargs": {"ip": "192.168.50.52"},
        "motion": {
            "dt": 0.1,
            "dx": 0.01,
            "dr": 0.1,
            "speed": 0.1,
            "acceleration": 0.1,
            "stop_acceleration": 0.1,
        },
        "poses": {
            "default_joints": [2.075042,-2.163444,-1.657608,-0.67119,1.489796,0.406704],
        },
        "safety": {
            "x_range": (-3.0, 3.0),
            "y_range": (-3.0, 3.0),
            "z_range": (-3.0, 3.0),
        },
        "jog": {
            "linear_speed": 0.05,
            "angular_speed": 0.10,
            "acceleration": 0.10,
            "timeout": 0.2,
        },
        "tool": {
            "tip_offset_tcp": [
                -0.00787,
                0.00212,
                0.18251,
            ],
        },
    },
    
    "right": {
        "driver": "ur5",
        "kwargs": {"ip": "192.168.50.51"},
        "motion": {
            "dt": 0.1,
            "dx": 0.01,
            "dr": 0.1,
            "speed": 0.1,
            "acceleration": 0.1,
            "stop_acceleration": 0.1,
        },
        "poses": {
            "default_joints": [4.299484,-1.741723,1.569382,-1.726387,-1.452985,-1.920663],
        },
        "safety": {
            "x_range": (-3.0, 3.0),
            "y_range": (-3.0, 3.0),
            "z_range": (-3.0, 3.0),
        },
        "jog": {
            "linear_speed": 0.05,
            "angular_speed": 0.10,
            "acceleration": 0.10,
            "timeout": 0.2,
        },
        "tool": {
            "tip_offset_tcp": [
                -0.00787,
                0.00212,
                0.18251,
            ],
        },
    },

    "tm": {
        "driver": "tm12",

        "kwargs": {
            "ip": "192.168.50.77",
        },

        "motion": {
            "speed": 0.20,
            "acceleration": 0.20,
            "dt": 0.10,
        },

        "jog": {
            "linear_speed": 0.2,
            "angular_speed": 0.2,
            "acceleration": 0.2,
            "timeout": 0.20,
        },

        "safety": {
            "x_range": [-1.3, 1.3],
            "y_range": [-1.3, 1.3],
            "z_range": [-0.2, 1.5],
        },
    },
}


# ============================================================
# GRIPPER
# ============================================================

GRIPPERS = {
    "left": {
        "driver": "robotiq_eseries",

        "kwargs": {
            "host": "192.168.50.52",
            "timeout": 5.0,
            "auto_activate": True,
        },

        "motion": {
            "speed": 1.0,
            "force": 0.6,
            "wait": True,
            "timeout": 5.0,
        },

        "arm_name": "left",
    },

    "right": {
        "driver": "robotiq",

        "kwargs": {
            "host": "192.168.50.51",
            "port": 63352,
            "timeout": 1.0,
            "auto_activate": False,
        },

        "motion": {
            "speed": 1.0,
            "force": 0.6,
            "wait": True,
            "timeout": 5.0,
        },

        "arm_name": "right",
    },
}

# ============================================================
# CAMERA
# ============================================================

CAMERAS = {
    "left": {
        "driver": "d405",

        "kwargs": {
            "serial_number": "352122272901",
        },

        "stream": {
            "width": 640,
            "height": 480,
            "fps": 30,
            "enable_color": True,
            "enable_depth": True,
            "align_to": "color",
            "frame_timeout_ms": 3000,
        },


        "mount": {
            "mode": "wrist",

            "T_matrix": [
                [0.9997445529, -0.0222418837, -0.0040159119, -0.0083683932],
                [0.0210420059, 0.8510929681, 0.5245931697, -0.0645389948],
                [-0.0082500259, -0.5245436668, 0.8513435727, 0.0513877522],
                [0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000],
            ],

            "arm_name": "left",
        },
    },

    "right": {
        "driver": "d405",

        "kwargs": {
            "serial_number": "260422275184",
        },

        "stream": {
            "width": 640,
            "height": 480,
            "fps": 30,
            "enable_color": True,
            "enable_depth": True,
            "align_to": "color",
            "frame_timeout_ms": 3000,
        },


        "mount": {
            "mode": "wrist",

            "T_matrix": [
                            [0.7063773241, -0.4874432523, -0.5132544708, 0.0604729413],
                            [0.6519854778, 0.7303641276, 0.2036741956, 0.0506925490],
                            [0.2755830415, -0.4785052946, 0.8337185798, 0.0689100160],
                            [0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000],
                        ],

            "arm_name": "right",
        },
    },

    "middle": {
        "driver": "logitech",

        "kwargs": {
            "device_path": "/dev/video0",
        },

        "stream": {
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "frame_timeout_ms": 3000,
        },

        "mount": {
            "mode": "fixed",

            "T_matrix": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
    },

    "tm": {
        "driver": "tm_eih",

        "kwargs": {
            "ip": "192.168.50.77",
            "port": 15567,
        },


        "stream": {
            "image_size": "1M",
            "fps": None,
            "enable_color": True,
            "enable_depth": False,
            "align_to": None,
            "frame_timeout_ms": 3000,
        },

        "mount": {
            "mode": "wrist",
            "arm_name": "tm",
        },
    },
}


# =========================
# OLLAMA / AI AGENT
# =========================

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_VLM_MODEL = "qwen3-vl:2b-instruct-q4_K_M"

OLLAMA_MODEL = "qwen2.5:3b"

OLLAMA_TIMEOUT = 60
OLLAMA_TEMPERATURE = 0.0

ACTION_VLA_CAMERA_COLOR_ORDER = "BGR"