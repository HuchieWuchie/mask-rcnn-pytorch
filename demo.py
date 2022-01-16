import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
import numpy as np
import cv2

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has 
num_classes = 10

# get the model using our helper function
model = get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)

# let's train it for 10 epochs
num_epochs = 10

model.eval()

img_path = "/workspaces/maskrcnn-pytorch/dataset/val/rgb/ILSVRC2014_train_00022910.jpg"
img = Image.open(img_path).convert("RGB")
img_vis = np.asarray(img).copy()
x = [torchvision.transforms.ToTensor()(img).to(device)]
#print(x.size())
#img = np.asarray(img)
#img = np.moveaxis(img, -1, 0)
#print(img.shape)
#tensor_transform = torchvision.transforms.ToTensor()
#x = tensor_transform(img)
predictions = model(x)[0]
boxes, labels, scores, masks = predictions['boxes'], predictions['labels'], predictions['scores'], predictions['masks']


idx = scores > 0.3
boxes = boxes[idx]
labels = labels[idx] 
scores = scores[idx]
print(scores)
print(labels)

for box in boxes:
    ps = (box[0], box[1])
    pe = (box[2], box[3])
    color = (0, 0, 255)
    thickness = 2
    img_vis = cv2.rectangle(img_vis, ps, pe, color, thickness)
    cv2.imwrite("test.jpg", img_vis)

#print(predictions)
#print(type(predictions))
#print(predictions.keys())
#print(predictions['labels'])