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
            "default_joints": [
                4.730419635772705,
                -1.1167879861644288,
                -1.8503081798553467,
                0.1962459522434692,
                1.4859864711761475,
                -0.00975352922548467,
            ],

            "intermediate_default_joints": [
                -1.5223515669452112,
                -0.9438229364207764,
                -1.077244758605957,
                -1.4643810552409668,
                1.599887490272522,
                -0.09220248857607061,
            ],
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
                0.0,
                0.0,
                0.2856,
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
            # Agent 可呼叫的 UR5 專用初始位置。
            "initial_joints": [
                3.213695526123047,
                -2.06486159959902,
                1.3513789176940918,
                -1.7703469435321253,
                -1.8406603972064417,
                -1.4885686079608362,
            ],

            "default_joints": [
                -1.3208535353290003,
                -1.674202104608053,
                -1.9512276649475098,
                -0.849548415546753,
                1.5535569190979004,
                0.24494972825050354,
            ],

            "intermediate_default_joints": [
                -1.5223515669452112,
                -0.9438229364207764,
                -1.077244758605957,
                -1.4643810552409668,
                1.599887490272522,
                -0.09220248857607061,
            ],
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
                0.0,
                0.0,
                0.2856,
            ],
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
                [0.9834015286, 0.1778336263, 0.0360088160, -0.0600087453],
                [-0.1778021699, 0.9049471635, 0.3865967144, -0.1021993209],
                [0.0361638198, -0.3865822455, 0.9215456286, -0.1597342145],
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
                [0.0008370449, -0.8396024751, -0.5432006840, 0.0673190519],
                [0.9998809151, 0.0090731972, -0.0124832936, -0.0065876642],
                [0.0154095712, -0.5431255480, 0.8395100858, 0.0514831823],
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
}

# ============================================================
# VISION
# ============================================================

YOLO_MODELS = {
    "object_detector": {
        "driver": "yolo",

        "kwargs": {
            "model_path":"/home/itri/tairos_SSTC/models/yolo26seg-demo.pt",
            "imgsz": 640,
            "conf": 0.5,
            "device": "cuda",
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