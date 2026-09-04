"""Recording and Training subsystem.

The subsystem contains three independently evolving features: recording,
replay, and training submission.
"""

from recording_training_replay.recording.routes import record_bp
from recording_training_replay.replay.routes import replay_bp
from recording_training_replay.training.routes import legacy_training_bp, training_bp

BLUEPRINTS = (
    record_bp,
    replay_bp,
    training_bp,
    legacy_training_bp,
)

__all__ = ["BLUEPRINTS"]
