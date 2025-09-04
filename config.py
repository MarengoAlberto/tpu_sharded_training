import os
from dataclasses import dataclass
from typing import Iterable


@dataclass
class Config:
    IMG_SIZE: Iterable[int] = (80, 80, 3) #(512, 512, 3)
    EPOCHS: int = 2
    DEBUG_MODE: bool = True
    BACKEND: str = "cpu"
    XLA_CACHE: bool = False
    WORLD_SIZE: int = 1

    BACKBONE_MODEL = "resnet101" # resnet18, resnet50
    FPN_CHANNELS: int = 128 # Number of output channels from the FPN.
    NUM_ANCHORS: int = 16 #12 # Number of anchors

    # #Loss function weightage.
    # For OHEM Loss
    NEG2POS_RATIO: int = 3 #4
    CLS_WEIGHTAGE: float = 1.0 #0.5
    LOC_WEIGHTAGE: float = 2.0 #15.0

    # mAP calculation
    VALID_NMS_THRESHOLD: float     = 0.3
    NMS_THRESHOLD: float           = 0.55
    SCORE_THRESHOLD: float         = 0.05

    # # Logging configurations
    # For tensorboard logging and saving checkpoints.
    root_log_dir: str = os.path.join("Logs_Checkpoints", "Model_logs")
    root_checkpoint_dir: str = os.path.join("Logs_Checkpoints", "Model_checkpoints")

    # Current log and checkpoint version directory
    log_dir: str = "version_0"
    checkpoint_dir: str = "version_0"

    ROOT_DIR = 'data/Dataset/Dataset' # Dataset root directory
    CONTAINER_DATA_DIR: str = "/workspace/data"
    ZIP_URL: str = "https://www.dropbox.com/s/k81ljpmzy3fgtx9/Dataset.zip?dl=1"
    CLASSES: tuple = ("__background__", "Reg-plate")

    BATCH_SIZE: int = 16

    # Number of workers to use for training
    NUM_WORKERS: int = 1 #4

    INIT_LEARING_RATE: float = 5e-4
    MOMENTUM: float = 0.9

    # Amount of additional regularization on the weights values
    WEIGHT_DECAY: float = 5e-4

    # At which epoches should we make a "step" in learning rate (i.e. decrease it in some manner)
    STEP_MILESTONE: Iterable[int] = (20, 40, 60, 80, 100, 125)

    # Multiplier applied to current learning rate at each of STEP_MILESTONE
    GAMMA: float = 0.5
    
    # Minimum LR ratio if cosine decay is applied
    MIN_LR_RATIO: float = 0.01
