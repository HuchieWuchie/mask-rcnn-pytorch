import numpy as np
from PIL import Image

mask_full_one_channel = np.loadtxt("dataset/test/masks/00_00000097.txt").astype(np.uint8)
mask_full = np.zeros((10, mask_full_one_channel.shape[0], mask_full_one_channel.shape[1])).astype(bool)
objects = np.loadtxt("dataset/test/object_labels/00_00000097.txt").astype(np.int32)

class_ids = []
bboxs = []

for obj in objects:
    class_id, x1, y1, x2, y2 = obj
    mask = np.zeros(mask_full_one_channel.shape).astype(np.uint8)
    mask[y1:y2, x1:x2] = mask_full_one_channel[y1:y2, x1:x2]
    mask[mask != 0] = 1
    mask_full[class_id, y1:y2, x1:x2] = mask[y1:y2, x1:x2]

    class_ids.append(class_id)
    bboxs.append([x1, y1, x2, y2])

"""
print(mask.shape)

obj_ids = np.unique(mask)
obj_ids = obj_ids[1:]
print(obj_ids.shape)
print(obj_ids)

masks = mask == obj_ids[:, None, None]

objects = np.loadtxt(self.image_info[image_id]["path_object"]).astype(np.uint8)
        masks, class_ids = [], []
        objects = objects.reshape((-1, 5))

        for obj in objects:
            class_id, x1, y1, x2, y2 = obj
            mask = np.zeros(mask_full.shape).astype(np.uint8)
            mask[y1:y2, x1:x2] = mask_full[y1:y2, x1:x2]
            mask_hot_encoded = np.eye(10)[mask.flatten()].reshape((mask.shape[0], mask.shape[1], 10))
            masks.append(mask_hot_encoded)
            class_ids.append(class_id)
"""