import os
from tqdm.auto import tqdm
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.augmentations import Normalize
from albumentations.pytorch.transforms import ToTensorV2
import itertools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .data import DataEncoder, list_files_in_directory, load_groundtruths



def prepare_targets(image_dir_path):
    
    # Initialize the `targets` list.
    targets = []
    
    image_paths, boxes, labels, _ = load_groundtruths(image_dir_path, train=False, shuffle=False)
    
    for image_path, bbox, label in zip(image_paths, boxes, labels):
        
        # Convert all converted box cordinates in the image to a torch tensor.
        # shape: (# boxes, 4)
        boxes_tensor  = torch.tensor(bbox, dtype = torch.float)
        # Convert all Class IDs in the image to a torch tensor.
        # shpae: (# boxes)
        labels_tensor = torch.tensor(label, dtype = torch.int)
        
        # Create a dictionary having keys: `boxes` and `labels` with values,
        # `boxes_tensor` and `labels_tensor`
        ground_truth_dict = dict(
                                boxes  = boxes_tensor,
                                labels = labels_tensor,
                                img_id = image_path
                                )
        
        # Append the created dictionary to the `targets` list.
        targets.append(ground_truth_dict)
        
    
    return targets



def prepare_predictions(model, image_dir_path, img_size, classes, device,
                        nms_threshold=0.5,
                        score_threshold=0.95,
                        soft_nms_sigma=0.5):
    
    # Initialize the `preds` list.
    preds = []
    
    model.eval()
    model = model.to(device)

    encoder = DataEncoder(img_size, classes)

    common_transforms = A.Compose(
        [A.Resize(height=img_size[0], width=img_size[1], interpolation=4), Normalize(), ToTensorV2()],
        bbox_params=A.BboxParams(format="pascal_voc",
        min_visibility=0.01,            # drop boxes almost fully occluded
        min_area=4.0,                   # drop tiny/degenerate boxes
        check_each_transform=True)
    )
    # Load the ground-truth image paths, bounding boxes and labels.
    image_paths, _, _, _ = load_groundtruths(image_dir_path, train=False, shuffle=False)
    
    iterator = tqdm(image_paths, dynamic_ncols=True)
    for idx, _ in enumerate(iterator):
        
        per_image_box_coords = []
        per_image_box_labels = []
        per_image_box_scores = []
        
        image_path = image_paths[idx]

        # =======================================================
        # Ground-truth
        # =======================================================
        orig_image = cv2.imread(image_path)[..., ::-1]
        orig_image_cpy = orig_image.copy()

        orig_image = orig_image.astype(np.int32)


        # =======================================================
        # Generate and plot Predictions.
        # =======================================================

        # Resize Image
        img = cv2.resize(orig_image_cpy, (img_size[1], img_size[0]), cv2.INTER_CUBIC)
        img = np.ascontiguousarray(img)

        trans_img = common_transforms(image=img)

        # Rescale ratio
        imH, imW = orig_image.shape[:2]
        IMG_SIZE_H, IMG_SIZE_W = img.shape[:2]

        ratio_h = imH / IMG_SIZE_H
        ratio_w = imW / IMG_SIZE_W

        # Generate predictions
        with torch.no_grad():
            predictions = model(trans_img["image"].unsqueeze(0).to(device))

        loc_pred = predictions[0].squeeze(0)
        cls_pred = predictions[1].squeeze(0)

        # Decode predictions
        decoded_preds = encoder.decode(
            loc_pred,
            cls_pred,
            device=device,
            nms_threshold=nms_threshold,
            score_threshold=score_threshold,
        ).cpu()
        

        pred_bbox   = decoded_preds[:,:4]
        pred_conf   = decoded_preds[:,4]
        pred_cls_id = decoded_preds[:,5]

        # Scale bounding boxes size according to original image size
        pred_bbox[:, 0] = np.maximum(0,   (pred_bbox[:, 0] * ratio_w))
        pred_bbox[:, 1] = np.maximum(0,   (pred_bbox[:, 1] * ratio_h))
        pred_bbox[:, 2] = np.minimum(imW, (pred_bbox[:, 2] * ratio_w))
        pred_bbox[:, 3] = np.minimum(imH, (pred_bbox[:, 3] * ratio_h))


       
        pred_dict = dict(
                        # Predictions boxes in (xmin, ymin, xmax, ymax) format.
                        boxes  = pred_bbox, 
                        # Confidence scores.
                        scores = pred_conf,
                        labels = pred_cls_id.int(),
                        img_id = image_path
        )
        # Append `pred_dict` to `preds` list.
        preds.append(pred_dict)
    
    return preds


def xyxy_to_xywh(box):
    x1,y1,x2,y2 = box
    return [float(x1), float(y1), float(x2-x1), float(y2-y1)]


def build_coco_from_simple(gt_simple, image_sizes, cat_id=1, cat_name="Reg-plate"):
    """
    gt_simple: list of dicts per image, each like:
        {"image_id": 123, "boxes": [[x1,y1,x2,y2], ...], "labels": [cat_id,...]}
        labels optional; assumes single-class if omitted.
    image_sizes: dict {image_id: (width, height)}  (needed by COCO)
    """
    images = []
    annos  = []
    ann_id = 1
    for sample in gt_simple:
        img_id = str(sample["img_id"])
        w,h = image_sizes[img_id]
        images.append({"id": img_id, "width": int(w), "height": int(h), "file_name": str(img_id)})

        boxes = sample.get("boxes", [])
        labels = sample.get("labels", [cat_id]*len(boxes))
        for box, lab in zip(boxes, labels):
            x1,y1,x2,y2 = map(float, box)
            w_box = max(0.0, x2 - x1); h_box = max(0.0, y2 - y1)
            if w_box <= 0 or h_box <= 0:   # skip degenerate
                continue
            annos.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(lab),
                "bbox": [x1, y1, w_box, h_box],  # COCO expects xywh
                "area": w_box * h_box,
                "iscrowd": 0
            })
            ann_id += 1

    categories = [{"id": cat_id, "name": cat_name}]
    return {"images": images, "annotations": annos, "categories": categories, "info": "COCO dataset eval"}


def convert_preds_xyxy_to_coco(pred_simple, cat_id=1):
    """
    pred_simple: list of dicts like {"image_id":..., "boxes":[xyxy...], "scores":[...]}
    Returns COCO detection list.
    """
    coco_dets = []
    for p in pred_simple:
        img_id = str(p["img_id"])
        for box, sc in zip(p["boxes"], p["scores"]):
            coco_dets.append({
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": xyxy_to_xywh(box),   # convert to xywh
                "score": float(sc),
            })
    return coco_dets


def get_image_sizes(targets):
    image_sizes = dict()
    for target in targets:
        image_path = target['img_id']
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)
        width = img.shape[2]
        height = img.shape[3]
        image_sizes[image_path] = (width, height)
    return image_sizes


def coco_eval_from_coco_lists(gt_coco_dict, det_list, max_dets=(1,10,100), iou_type="bbox"):
    # Build COCO API from an in-memory dict (no need to write a file)
    coco_gt = COCO()
    coco_gt.dataset = gt_coco_dict
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(det_list) if len(det_list) else COCO()
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.params.maxDets = list(max_dets)  # e.g. (1,10,100) or (1,10,300)
    # Optionally restrict to a subset of image ids:
    # coco_eval.params.imgIds = [im["id"] for im in gt_coco_dict["images"]]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Return as a dict too
    return {
        "AP@[.5:.95]": coco_eval.stats[0],
        "AP@0.50":     coco_eval.stats[1],
        "AP@0.75":     coco_eval.stats[2],
        "AP_small":    coco_eval.stats[3],
        "AP_medium":   coco_eval.stats[4],
        "AP_large":    coco_eval.stats[5],
        "AR_max=1":    coco_eval.stats[6],
        "AR_max=10":   coco_eval.stats[7],
        "AR_max=100":  coco_eval.stats[8],
        "AR_small":    coco_eval.stats[9],
        "AR_medium":   coco_eval.stats[10],
        "AR_large":    coco_eval.stats[11],
    }
