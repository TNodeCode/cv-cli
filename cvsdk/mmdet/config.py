from pydantic import BaseModel


class TrainingConfig(BaseModel):
    """Model configuration
    """
    model_type: str
    model_name: str
    dataset_dir: str
    train_dir: str
    val_dir: str
    test_dir: str
    dataset_classes: list[str]
    batch_size: int
    epochs: int
    work_dir: str
    annotations_train: str
    annotations_val: str
    annotations_test: str
    optimizer: str
    lr: float
    weight_decay: float
    momentum: float
    augmentations: list[dict]

