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

OBJ_CLASSES = ('__background__', 'bowl', 'tvm', 'pan', 'hammer', 'knife',
                                    'cup', 'drill', 'racket', 'spatula', 'bottle')

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has 
num_classes = 10

# get the model using our helper function
#model = get_model_instance_segmentation(num_classes)
model = torch.load("model.pth")

# move model to the right device
model.to(device)

# let's train it for 10 epochs
num_epochs = 10

model.eval()

#img_path = "/workspaces/maskrcnn-pytorch/dataset/val/rgb/ILSVRC2014_train_00022910.jpg"
#img_path = "/workspaces/maskrcnn-pytorch/dataset/val/rgb/00_00000116.jpg"
#img_path = "/workspaces/maskrcnn-pytorch/dataset/val/rgb/ILSVRC2014_train_00000876.jpg"
img_path = "/workspaces/maskrcnn-pytorch/dataset/val/rgb/ILSVRC2014_train_00037745.jpg"
#img_path = "/workspaces/maskrcnn-pytorch/rgb.png"
#img_path = "/workspaces/maskrcnn-pytorch/custom.png"
img = Image.open(img_path).convert("RGB")
img_vis = np.asarray(img).copy()
img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
x = [torchvision.transforms.ToTensor()(img).to(device)]

predictions = model(x)[0]
boxes, labels, scores, masks = predictions['boxes'], predictions['labels'], predictions['scores'], predictions['masks']


idx = scores > 0.1
boxes = boxes[idx]
labels = labels[idx] 
scores = scores[idx]
masks = masks[idx].cpu().detach().numpy()
print(scores)
print(labels)


for box, label in zip(boxes, labels):
    ps = (box[0], box[1])
    pe = (box[2], box[3])
    color = (0, 0, 255)
    thickness = 2
    img_vis = cv2.rectangle(img_vis, ps, pe, color, thickness)

    img_vis = cv2.putText(img_vis, str(OBJ_CLASSES[label]), (box[0], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, 2)

colors = [(0,0,205), (34,139,34), (192,192,128), (165, 42, 42), (128, 64, 128),
                (204, 102, 0), (184, 134, 11), (0, 153, 153), (0, 134, 141), (184, 0, 141)]
for label, mask in zip(labels, masks):
    img_vis[mask[0] > 0.01] = colors[label]

cv2.imwrite("test.jpg", img_vis)

#print(predictions)
#print(type(predictions))
#print(predictions.keys())
#print(predictions['labels'])