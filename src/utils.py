import os
import requests
import zipfile
import random
import numpy as np
import torch
from torchinfo import summary
import cv2
import albumentations as A
from albumentations.augmentations import Normalize
from albumentations.pytorch.transforms import ToTensorV2

from .detector import Detector
from .data import DataEncoder


def set_seeds(rank):
    # fix random seeds
    SEED_VALUE = 42 + rank

    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED_VALUE)
        torch.cuda.manual_seed_all(SEED_VALUE)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def load_model(path, num_classses, training_config):
    """
    For reloading trained model
    Args:
        path: model checkpoint path

    Returns:
        Reinitialized Trained Model
    """

    IMG_SIZE = training_config.IMG_SIZE

    # Load model architecture.
    model = Detector(
        backbone_name=training_config.BACKBONE_MODEL,
        num_classes = num_classses,
        fpn_channels = training_config.FPN_CHANNELS,
        num_anchors=training_config.NUM_ANCHORS
    )

    path = os.path.join(path, "Detector_best.pth")


    # Load trained model's state dict.
    model.load_state_dict(torch.load(path))

    print(summary(model, input_size=(1,)+IMG_SIZE[::-1], row_settings=["var_names"]))

    return model


def _load_default_plate_model(model_version, data_config, training_config):
    # A commonly used license-plate YOLOv8 model hosted publicly.
    # You can replace with a local path to a .pt file if you have one.

    NUM_CLASSES = len(data_config.CLASSES)
    load_version = model_version
    checkpoint_path = os.path.join(training_config.root_checkpoint_dir, load_version)
    # Loading trained model
    model = load_model(checkpoint_path, NUM_CLASSES, training_config=training_config)
    
    encoder = DataEncoder(training_config.IMG_SIZE[:2], data_config.CLASSES)

    common_transforms = A.Compose(
        [A.Resize(height=training_config.IMG_SIZE[0], width=training_config.IMG_SIZE[1], interpolation=4), Normalize(), ToTensorV2()],
        bbox_params=A.BboxParams(format="pascal_voc",
        min_visibility=0.01,            # drop boxes almost fully occluded
        min_area=4.0,                   # drop tiny/degenerate boxes
        check_each_transform=True)
    )
    return model, encoder, common_transforms, training_config


def _predict_plates(model, frame_bgr, encoder, transform, trainer_config):
    """
    Very simple placeholder.
    In your real code, run your plate detector to get a bbox and OCR model to get text.
    Return a list of detections. Coordinates are in (x1,y1,x2,y2) pixel space.
    """
    imH, imW = frame_bgr.shape[:2]
    IMG_SIZE_H, IMG_SIZE_W = trainer_config.IMG_SIZE[0], trainer_config.IMG_SIZE[1]

    ratio_h = imH / IMG_SIZE_H
    ratio_w = imW / IMG_SIZE_W
    
    trans_img = transform(image=frame_bgr)
    predictions = model(trans_img["image"].unsqueeze(0).to(trainer_config.DEVICE))
    loc_pred = predictions[0].squeeze(0).detach()
    cls_pred = predictions[1].squeeze(0).detach()

    # Decode predictions
    decoded_preds = encoder.decode(
        loc_pred,
        cls_pred,
        device=trainer_config.DEVICE,
        nms_threshold=trainer_config.VALID_NMS_THRESHOLD,
        score_threshold=trainer_config.SCORE_THRESHOLD,
    ).cpu()
    
    pred_bbox   = decoded_preds[:,:4]
    pred_conf   = decoded_preds[:,4]
    pred_cls_id = decoded_preds[:,5]

    # Scale bounding boxes size according to original image size
    pred_bbox[:, 0] = np.maximum(0,   (pred_bbox[:, 0] * ratio_w))
    pred_bbox[:, 1] = np.maximum(0,   (pred_bbox[:, 1] * ratio_h))
    pred_bbox[:, 2] = np.minimum(imW, (pred_bbox[:, 2] * ratio_w))
    pred_bbox[:, 3] = np.minimum(imH, (pred_bbox[:, 3] * ratio_h))
    return np.column_stack((pred_bbox, pred_conf))


def _direct_download_url(url: str) -> str:
    # Turn a Dropbox share link into a direct download URL
    if "dropbox.com" in url and "dl=1" not in url:
        if "?" in url:
            return url + "&dl=1"
        return url + "?dl=1"
    return url


def download_and_unzip_zip(url: str, save_dir: str, zip_name: str | None = None, timeout: int = 60):
    """
    Download a ZIP from `url` into `save_dir` and unzip it.
    Skips work if extracted folder already exists.
    Validates the downloaded file is a real ZIP.
    """
    os.makedirs(save_dir, exist_ok=True)
    url = _direct_download_url(url)

    if zip_name is None:
        zip_name = os.path.basename(url.split("?")[0]) or "dataset.zip"

    zip_path = os.path.join(save_dir, zip_name)
    extract_dir = os.path.splitext(zip_path)[0]

    # If already extracted, nothing to do
    if os.path.isdir(extract_dir) and os.listdir(extract_dir):
        print(f"✅ Already extracted: {extract_dir}")
        return extract_dir

    # (Re)download if file missing or not a valid zip
    need_download = True
    if os.path.exists(zip_path):
        # Quick validity check
        try:
            with open(zip_path, "rb") as f:
                magic = f.read(4)
            if magic == b"PK\x03\x04" and zipfile.is_zipfile(zip_path):
                need_download = False
            else:
                print("⚠️ Existing file is not a valid ZIP. Re-downloading...")
        except Exception:
            print("⚠️ Could not read existing file. Re-downloading...")

    if need_download:
        tmp_path = zip_path + ".partial"
        # clean up any partial
        try: os.remove(tmp_path)
        except FileNotFoundError: pass

        print(f"⬇️  Downloading: {url}")
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
        os.replace(tmp_path, zip_path)
        print(f"✅ Downloaded to: {zip_path}")

        # Validate ZIP after download
        with open(zip_path, "rb") as f:
            if f.read(4) != b"PK\x03\x04" or not zipfile.is_zipfile(zip_path):
                raise ValueError(
                    "Downloaded file is not a valid ZIP. "
                    "If this is a Dropbox/Drive link, ensure it's a direct download (e.g., ?dl=1)."
                )

    # Unzip
    print(f"📂 Extracting to: {extract_dir}")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print("✅ Extraction complete")
    return extract_dir
