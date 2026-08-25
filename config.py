import os

# =========================
# ARM
# =========================
ARMS = {
    "left": {
        "driver": "ur7e",
        "kwargs": {"ip": "192.168.50.76"},
        "motion": {
            "dt": 0.1,
            "dx": 0.01,
            "dr": 0.1,
            "speed": 0.1,
            "acceleration": 0.1,
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
        "kwargs": {"ip": "192.168.50.50"},
        "motion": {
            "dt": 0.1,
            "dx": 0.01,
            "dr": 0.1,
            "speed": 0.1,
            "acceleration": 0.1,
        },
        "poses": {
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
            "host": "192.168.50.76",
            "timeout": 5.0,
            "auto_activate": True,
        },

        "motion": {
            "speed": 255,
            "force": 150,
            "wait": True,
            "timeout": 5.0,
        },

        "arm_name": "left",
    },

    "right": {
        "driver": "robotiq",

        "kwargs": {
            "host": "192.168.50.50",
            "port": 63352,
            "timeout": 1.0,
            "auto_activate": False,
        },

        "motion": {
            "speed": 255,
            "force": 150,
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
    "front": {
        "driver": "d405",

        "kwargs": {
            "serial_number": None,
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
                [0.99977168, -0.01763005, -0.01207349, -0.00687425],
                [0.02135767, 0.84198181, 0.53908300, -0.06019584],
                [0.00066159, -0.53921778, 0.84216610, 0.08289779],
                [0.0, 0.0, 0.0, 1.0],
            ],

            "arm_name": "left",
        },
    },

    "rear": {
        "driver": "usb_camera",

        "kwargs": {
            "device_index": 0,
        },

        "stream": {
            "width": 1280,
            "height": 720,
            "fps": 30,
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
# OLLAMA_MODEL = "qwen3-vl:2b-instruct-q4_K_M"

OLLAMA_MODEL = "qwen2.5:3b"

OLLAMA_TIMEOUT = 60
OLLAMA_TEMPERATURE = 0.0