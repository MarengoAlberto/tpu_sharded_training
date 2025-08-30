import cv2
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from .encoder import DataEncoder


class PlateDataset(Dataset):
    def __init__(self,
                 root_dir,
                 classes,
                 transform=None,
                 is_train=True,
                 input_size=(300, 300, 3),
                 debug=False
    ):
        self.root_dir = os.path.expanduser(root_dir)
        self.classes = classes
        self.transforms = transform
        self.input_size = input_size
        self.is_train = is_train
        self.encoder = DataEncoder(self.input_size[:2], self.classes)

        self.image_paths, self.boxes, self.labels, self.num_samples = load_groundtruths(root_dir, train=is_train, shuffle=is_train)
        if debug:
            if is_train:
                cap = 1000
            else:
                cap = 100
            self.num_samples = min(cap, self.num_samples)
            self.image_paths = self.image_paths[:self.num_samples]
            self.boxes = self.boxes[:self.num_samples]
            self.labels = self.labels[:self.num_samples]
            print(f"Debug mode is activated. Dataset size reduced to {self.num_samples} samples.")

    def __len__(self):
        # Get size of the Dataset.
        return self.num_samples


    def __getitem__(self, idx):

        image_path = self.image_paths[idx]
        indexed_boxes = self.boxes[idx]
        indexed_labels = self.labels[idx]

        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        if self.transforms:
            transformed = self.transforms(image=img, bboxes=indexed_boxes, category_ids=indexed_labels)

        else: # Mandatory transforms to be applied.

            common_transforms = A.Compose(
                                [A.Resize(
                                        height=self.input_size[0], width=self.input_size[1],
                                        interpolation=4
                                       ),
                                ToTensorV2()],
                                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"])
                                )

            transformed = common_transforms(image=img, bboxes=indexed_boxes, category_ids=indexed_labels)


        transformed_img    = transformed["image"]
        transformed_boxes  = transformed["bboxes"]
        transformed_labels = transformed["category_ids"]

        transformed_boxes = torch.tensor(transformed_boxes, dtype=torch.float)
        transformed_labels = torch.tensor(transformed_labels, dtype=torch.int)

        # ===========================================================
        # Generate Encoded bounding boxes and labels
        # ===========================================================

        loc_target, cls_target = self.encoder.encode(transformed_boxes, transformed_labels)

        return transformed_img, transformed_boxes, transformed_labels, loc_target, cls_target


    def collate_fn(self, batch):

        return list(zip(*batch))



def list_files_in_directory(directory_path):
    try:
        entries = os.listdir(directory_path)
        files = [entry for entry in entries if os.path.isfile(os.path.join(directory_path, entry))]
        return files
    except FileNotFoundError:
        print(f"Error: Directory '{directory_path}' not found.")
        return []


def load_groundtruths(root_path, train=True, shuffle=True):
    image_paths = []
    boxes = []
    labels = []

    folder = 'train' if train else 'validation'
    directory = os.path.join(root_path, folder, 'Vehicle registration plate')
    file_names = list_files_in_directory(directory)
    num_samples = len(file_names)
    for image_name in file_names:
        image_id, _ = os.path.splitext(os.path.basename(image_name))
        filepath = os.path.join(root_path, folder, 'Vehicle registration plate', 'Label', image_id+'.txt')

        with open(filepath) as f:
            lines = f.readlines()

        image_paths.append(os.path.join(directory, image_name))
        box = []
        label = []

        for line in lines:
            splited = line.strip().split()[-4:]
            xmin = splited[0]
            ymin = splited[1]
            xmax = splited[2]
            ymax = splited[3]

            class_label = int(1)
            box.append([float(xmin), float(ymin), float(xmax), float(ymax)])
            label.append(class_label)
        boxes.append(box)
        labels.append(label)

    print(f"Total {num_samples} images and {len(boxes)} boxes loaded from: {os.path.relpath(directory, os.getcwd())}")

    # Shuffle or Sort
    if shuffle:
        temp = list(zip(image_paths, boxes, labels))
        random.shuffle(temp)
        image_paths, boxes, labels = zip(*temp)
    else:
        image_paths, boxes, labels = zip(*sorted(zip(image_paths, boxes, labels)))

    image_paths = list(image_paths)
    boxes = list(boxes)
    labels = list(labels)

    return image_paths, boxes, labels, num_samples