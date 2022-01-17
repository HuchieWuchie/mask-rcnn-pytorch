import torch
import os
import numpy as np
from PIL import Image

class InstanceSegmentationDataSet(torch.utils.data.Dataset):
    def __init__(self, root_dir, transforms):

        self.root_dir = root_dir
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root_dir, "rgb"))))
        self.masks = list(sorted(os.listdir(os.path.join(root_dir, "masks"))))
        self.objects = list(sorted(os.listdir(os.path.join(root_dir, "object_labels"))))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,
                    idx: int):
        
        img_path = os.path.join(self.root_dir, "rgb", self.imgs[idx])
        mask_path = os.path.join(self.root_dir, "masks", self.masks[idx])
        object_path = os.path.join(self.root_dir, "object_labels", self.objects[idx])

        img = Image.open(img_path).convert("RGB")

        mask_full_one_channel = np.loadtxt(mask_path).astype(np.uint8)
        mask_full = np.zeros((11, mask_full_one_channel.shape[0], mask_full_one_channel.shape[1])).astype(float)
        objects = np.loadtxt(object_path).astype(np.int32)
        objects = np.reshape(objects, (-1, 5))

        class_ids = []
        bboxes = []

        for obj in objects:
            class_id, x1, y1, x2, y2 = obj
            class_id = class_id + 1
            print(class_id, mask_full.shape)
            mask_full[class_id, y1:y2, x1:x2] = mask_full_one_channel[y1:y2, x1:x2] > 0

            class_ids.append(class_id)
            bboxes.append([x1, y1, x2, y2])
        
        # convert everything into a torch.Tensor
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(class_ids, dtype=torch.int64)
        masks = torch.as_tensor(mask_full, dtype=torch.float32)
        image_id = torch.tensor([idx])
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:,0])

        iscrowd = torch.zeros((len(class_ids),), dtype=torch.int64)

        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # Preprocessing
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target