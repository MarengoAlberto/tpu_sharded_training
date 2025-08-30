import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import albumentations as A
from albumentations.augmentations import Normalize
from albumentations.pytorch.transforms import ToTensorV2

from .data import load_groundtruths
from .encoder import DataEncoder


def draw_bbox(image, boxes, labels=None, conf_scores = None, color=(255, 0, 0), thickness=-1):

    for idx, box in enumerate(boxes):
        xmin = box[0]
        ymin = box[1]
        xmax = box[2]
        ymax = box[3]

        image = np.ascontiguousarray(image)
        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)

        if labels:
            display_text = str(labels[idx])

            (text_width, text_height), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            
            cv2.rectangle(image, (xmin, ymin - int(0.9 * text_height)), (xmin + int(0.4*text_width), ymin), color, -1)


            image = cv2.putText(
                image,
                display_text,
                (xmin, ymin - int(0.3 * text_height)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )
            
        elif conf_scores:
            display_text = str(round(conf_scores[idx], 4)*100) + '%'

            (text_width, text_height), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)

            cv2.rectangle(image, (xmin, ymin - int(0.9 * text_height)), (xmin + int(0.4*text_width), ymin), color, -1)


            image = cv2.putText(
                image,
                display_text,
                (xmin, ymin - int(0.3 * text_height)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )

    return image


def visualize_transformed_data(images, boxes, labels, classes):

    # Define subplots
    plt.figure(figsize=(20, 15))

    ncols = 3
    nrows = int(np.ceil(len(images)/ncols))

    for batch_id, (img_trans, box_trans, labels) in enumerate(zip(images, boxes, labels)):

        plt.subplot(nrows, ncols, batch_id+1)
        plt.title("Transformed Image")

        trans_img = np.transpose(img_trans.numpy(), axes=(1,2,0)).astype(np.uint8)
        trans_boxes = box_trans.numpy().astype(np.int32)
        trans_labels = labels.numpy().astype(np.int32)

        trans_labels_names = [classes[cls_idx] for cls_idx in trans_labels]


        trans_img_ann = draw_bbox(trans_img,
                                  trans_boxes,
                                  labels=trans_labels_names,
                                  color=(0, 255, 0), thickness=2)

        plt.imshow(trans_img_ann)
        plt.axis("off")


    plt.show()


def plot_history(
        train_loss=None,
        val_loss=None,
        val_metric_primary=None,
        val_metric_50=None,
        colors=["blue", "green"],
        loss_legend_loc="upper center",
        acc_legend_loc="upper left",
        fig_size=(15, 10),
        block_plot = False,
):

    plt.rcParams["figure.figsize"] = fig_size
    fig = plt.figure()
    fig.set_facecolor("white")

    # Loss Plots
    plt.subplot(2, 1, 1)

    train_loss_range = range(len(train_loss))
    plt.plot(
        train_loss_range,
        train_loss,
        color=f"tab:{colors[0]}",
        label=f"Train Loss",
    )

    valid_loss_range = range(len(val_loss))
    plt.plot(
        valid_loss_range,
        val_loss,
        color=f"tab:{colors[1]}",
        label=f"Valid Loss",
    )

    plt.ylabel("Loss")
    plt.legend(loc=loss_legend_loc)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True)
    plt.title("Training and Validation Loss")

    # Mean Average Precision Plots
    plt.subplot(2, 1, 2)

    val_metric_primary_range = range(len(val_metric_primary))
    plt.plot(
        val_metric_primary_range,
        val_metric_primary,
        color=f"tab:{colors[0]}",
        label=f"mAP@.50:.95",
    )

    val_metric_50_range = range(len(val_metric_50))
    plt.plot(
        val_metric_50_range,
        val_metric_50,
        color=f"tab:{colors[1]}",
        label=f"mAP@0.50",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Mean Average Precision")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc=acc_legend_loc)
    plt.grid(True)
    plt.title("Validation mAP")

    # fig.savefig("loss_MaP_plot.png")
    plt.show(block=block_plot)

    return


def visualize_predictions(
        model,
        device,
        classes,
        img_size,
        root_dir,
        rows=2,
        columns=3,
        nms_threshold=0.5,
        score_threshold=0.95,
        soft_nms_sigma=0.5,
):

    # Define subplots
    fig, ax = plt.subplots(
        nrows=rows,
        ncols=columns,
        figsize=(20, 40),
    )

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
    image_paths, _, _, _ = load_groundtruths(root_dir, train=False, shuffle=False)

    for idx, axis in enumerate(ax.flat):
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
            preds = model(trans_img["image"].unsqueeze(0).to(device))

        loc_pred = preds[0].squeeze(0)
        cls_pred = preds[1].squeeze(0)

        # Decode predictions
        decoded_preds = encoder.decode(
            loc_pred,
            cls_pred,
            device=device,
            nms_threshold=nms_threshold,
            score_threshold=score_threshold,
        ).cpu().numpy()

        for class_idx, class_name in enumerate(classes):

            # Skip boxes that were classified as background.
            if class_name == "__background__":
                continue

            class_tensor = decoded_preds[np.where(decoded_preds[:, 5] == class_idx)]

            # Scale bounding boxes size according to original image size
            class_tensor[:, 0] = np.maximum(0,   (class_tensor[:, 0] * ratio_w))
            class_tensor[:, 1] = np.maximum(0,   (class_tensor[:, 1] * ratio_h))
            class_tensor[:, 2] = np.minimum(imW, (class_tensor[:, 2] * ratio_w))
            class_tensor[:, 3] = np.minimum(imH, (class_tensor[:, 3] * ratio_h))

            pred_labels = [f"{classes[int(i)]}" for i in class_tensor[:, 5]]

            # Plot Predictions.
            orig_image = draw_bbox(
                orig_image,
                class_tensor.astype(np.int32),
                labels=pred_labels,
                color=(255, 0, 0),
                thickness=2,
            )


        axis.imshow(orig_image)
        axis.axis("off")

    plt.figtext(0.50, 0.9, "Predictions", fontsize=20, color="r", ha="center")

    plt.show(block=False)

    return fig



def visualize_three_examples(
        model,
        device,
        classes,
        img_size,
        root_dir,
        nms_threshold=0.5,
        score_threshold=0.95,
        soft_nms_sigma=0.5,
):

    # Define subplots
    fig, ax = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(20, 40),
    )

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
    image_paths, _, _, _ = load_groundtruths(root_dir, train=False, shuffle=False)

    for idx, axis in enumerate(ax.flat):
        image_path = random.choice(image_paths)

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
            preds = model(trans_img["image"].unsqueeze(0).to(device))

        loc_pred = preds[0].squeeze(0)
        cls_pred = preds[1].squeeze(0)

        # Decode predictions
        decoded_preds = encoder.decode(
            loc_pred,
            cls_pred,
            device=device,
            nms_threshold=nms_threshold,
            score_threshold=score_threshold,
        ).cpu().numpy()

        for class_idx, class_name in enumerate(classes):

            # Skip boxes that were classified as background.
            if class_name == "__background__":
                continue

            class_tensor = decoded_preds[np.where(decoded_preds[:, 5] == class_idx)]

            # Scale bounding boxes size according to original image size
            class_tensor[:, 0] = np.maximum(0,   (class_tensor[:, 0] * ratio_w))
            class_tensor[:, 1] = np.maximum(0,   (class_tensor[:, 1] * ratio_h))
            class_tensor[:, 2] = np.minimum(imW, (class_tensor[:, 2] * ratio_w))
            class_tensor[:, 3] = np.minimum(imH, (class_tensor[:, 3] * ratio_h))

            conf_scores = class_tensor[:, 4].tolist()

            # Plot Predictions.
            orig_image = draw_bbox(
                orig_image,
                class_tensor.astype(np.int32),
                conf_scores=conf_scores,
                color=(255, 0, 0),
                thickness=2,
            )


        axis.imshow(orig_image)
        axis.axis("off")

    plt.figtext(0.50, 0.9, "Predictions", fontsize=20, color="r", ha="center")

    plt.show(block=False)

    return fig
