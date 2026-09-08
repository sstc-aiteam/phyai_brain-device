"""Training-target compatibility registry for recorded dataset formats."""


TRAINING_TARGETS = {
    "lerobot_v2": (
        {"value": "groot_n15", "label": "GR00T N1.5"},
    ),
    "lerobot_v3": (),
}


# 回傳指定 LeRobot 格式目前支援的訓練模型。
def available_training_targets(dataset_format):
    return TRAINING_TARGETS.get(dataset_format, ())


# 驗證並正規化使用者選擇的訓練模型。
def resolve_training_target(dataset_format, training_target=None):
    targets = available_training_targets(dataset_format)
    if not targets:
        if training_target:
            raise ValueError(f"{dataset_format} 目前沒有支援的訓練模型轉換")
        return None
    selected = training_target or targets[0]["value"]
    supported = {target["value"] for target in targets}
    if selected not in supported:
        raise ValueError(
            f"{dataset_format} 不支援 training_target={selected}；"
            f"目前支援：{', '.join(sorted(supported))}"
        )
    return selected
