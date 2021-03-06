import torch
import os
import numpy as np
from PIL import Image
import cv2
import torchvision.transforms

class InstanceSegmentationDataSet(torch.utils.data.Dataset):
    def __init__(self, root_dir, transforms, num_classes, rescale = False):

        self.root_dir = root_dir
        self.transforms = transforms
        self.rescale = rescale
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

        h, w = mask_full_one_channel.shape
        h_scale, w_scale = 1, 1

        if self.rescale:
            
            if h < 1024 or w < 1024:
                h_scale = 1024 / h
                w_scale = 1024 / w
                img = img.resize((1024,1024))
                mask_full_one_channel = cv2.resize(mask_full_one_channel, (1024,1024))

        objects = np.loadtxt(object_path).astype(np.int32)
        objects = np.reshape(objects, (-1, 5))
        no_instances = objects.shape[0]
        mask_full = np.zeros((no_instances, mask_full_one_channel.shape[0], mask_full_one_channel.shape[1])).astype(np.uint8)

        class_ids = []
        bboxes = []

        for i, obj in enumerate(objects):
            class_id, x1, y1, x2, y2 = obj
            x1, y1, x2, y2 = int(x1 * w_scale), int(y1 * h_scale), int(x2 * w_scale), int(y2 * h_scale)
            class_id = class_id + 1
            #mask_full[class_id, y1:y2, x1:x2] = mask_full_one_channel[y1:y2, x1:x2] > 0
            mask_full[i, y1:y2, x1:x2] = mask_full_one_channel[y1:y2, x1:x2] > 0

            class_ids.append(class_id)
            bboxes.append([x1, y1, x2, y2])
        #mask_full = mask_full * 255
        """
        p = 0
        for m in mask_full:
            mc = m.copy()
            
            p += 1
            for b in bboxes:
                x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                mc = cv2.rectangle(mc, (x1, y1), (x2, y2), color = 255, thickness = 2)
            cv2.imwrite("test/" + str(p) + ".jpg", mc)
        """
            
        # convert everything into a torch.Tensor
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(class_ids, dtype=torch.int64)
        masks = torch.as_tensor(mask_full, dtype=torch.uint8)
        
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
        #img = torchvision.transforms.Resize((1024,1024), img)
        return img, target

class InstanceSegmentationDataSetAffordance(torch.utils.data.Dataset):
    def __init__(self, root_dir, transforms, num_classes, rescale = False):

        self.root_dir = root_dir
        self.transforms = transforms
        self.rescale = rescale
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

        objects = np.loadtxt(object_path).astype(np.int32)
        objects = np.reshape(objects, (-1, 5))

        no_instances = 0
        for i, obj in enumerate(objects):
            _, x1, y1, x2, y2 = obj
            m = mask_full_one_channel[y1:y2, x1:x2] > 0
            no_instances += np.unique(m).shape[0]

        mask_full = np.zeros((no_instances, mask_full_one_channel.shape[0], mask_full_one_channel.shape[1])).astype(np.uint8)

        class_ids = []
        bboxes = []

        instance = 0
        for i, obj in enumerate(objects):
            _, x1, y1, x2, y2 = obj

            m = mask_full_one_channel[y1:y2, x1:x2] > 0
            labels = np.unique(m)

            for label in labels:
                m = np.zeros(mask_full_one_channel.shape)
                ix = mask_full_one_channel[y1:y2, x1:x2] == label
                x, y = np.where(ix != 0)
                #print("x, y: ", x, y)
                #print("np where ", np.where(ix))
                try:
                    x1_l, y1_l, x2_l, y2_l = np.min(x) + x1, np.min(y) + y1, np.max(x) + x2, np.max(y) + y2
                except ValueError:
                    print("x: ", x)
                    print("y: ", y)
                    print("ix: ", ix)

                mask_full[instance, y1_l:y2_l, x1_l:x2_l] = m[y1_l:y2_l, x1_l:x2_l] > 0
                
                class_ids.append(label)
                bboxes.append([x1_l, y1_l, x2_l, y2_l])

                instance += 1

        # convert everything into a torch.Tensor
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(class_ids, dtype=torch.int64)
        masks = torch.as_tensor(mask_full, dtype=torch.uint8)
        
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
        #img = torchvision.transforms.Resize((1024,1024), img)
        return img, target