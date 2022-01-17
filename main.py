import torch
from dataset import InstanceSegmentationDataSet
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import utils
import transforms as T
from engine import train_one_epoch, evaluate
import time
import os


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

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
dataset = InstanceSegmentationDataSet(root_dir = 'dataset/train_and_val', transforms = get_transform(train=True))
data_loader = torch.utils.data.DataLoader(
 dataset, batch_size=1, shuffle=True, num_workers=1,
 collate_fn=utils.collate_fn)
# For Training
images,targets = next(iter(data_loader))
images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]
output = model(images,targets)   # Returns losses and detections
# For inference
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)           # Returns predictions

def main():
    log_folder = "logs"
    time_folder = os.path.join(log_folder, str(round(time.time())))
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    if not os.path.exists(time_folder):
        os.mkdir(time_folder)
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has 
    num_classes = 10
    # use our dataset and defined transformations
    dataset = InstanceSegmentationDataSet(root_dir = 'dataset/train_and_val', transforms = get_transform(train=True))
    dataset_test = InstanceSegmentationDataSet(root_dir = 'dataset/test', transforms = get_transform(train=False))

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=1,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10
    min_val_loss = 9999999

    for epoch in range(num_epochs):
        #train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        val_loss = evaluate(model, data_loader_test, device=device, print_freq = 100)
        print("Mean val loss: ", val_loss)

        if val_loss < min_val_loss:
            print("Model improved, saving parameters...")
            min_val_loss = val_loss
            checkpoint_path = os.path.join(time_folder, str(epoch) + ".pth")
            torch.save(model, checkpoint_path)


    print("That's it!")

main()