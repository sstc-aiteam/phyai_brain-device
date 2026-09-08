from gr00t.experiment.data_config import OxeDroidDataConfig


class OpenDrawerLeftDataConfig(OxeDroidDataConfig):
    """OXE settings for the UR7e dataset's single left camera."""

    video_keys = ["video.left_image"]
