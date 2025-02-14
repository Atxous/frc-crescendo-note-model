import pandas as pd
import cv2
import numpy as np
import torch
import torchvision
from PIL import ImageDraw


# hardcoded resize and crop
# images are either 1920 x 1080 or 1080 x 1920
class ResizePlusCrop(object):
    def __init__(self):
        pass

    def __call__(self, imgs):
        width, height = imgs.shape[2], imgs.shape[1]
        if height < width:
            cropped = torchvision.transforms.functional.resize(imgs, (360, 640), antialias=True)        
        else:
            cropped = torchvision.transforms.functional.resize(imgs, (640, 360), antialias=True)
            
        return cropped

class RingDataset(torch.utils.data.Dataset):
    def __init__(self, folder, csv_file, type_bbox : str = "yolo", spatial_transform = None, scaler = 1, transform = None):
        self.folder = folder
        self.csv_file = pd.read_csv(csv_file)
        self.to_tensor = torchvision.transforms.ToTensor()
        self.spatial_transform = spatial_transform
        self.transform = transform
        self.scaler = scaler
        if type_bbox == "yolo":
            self.bbox_fn = load_bbox_for_yolo
        elif type_bbox == "multiple_yolo":
            self.bbox_fn = load_multiple_bbox_for_yolo
        else:
            self.bbox_fn = get_bbox

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, index):
        img = cv2.imread(f"{self.folder}/{self.csv_file['image_name'][index]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.to_tensor(img)
        if self.spatial_transform:
            img = self.spatial_transform(img)
        if self.transform:
            img = self.transform(img)
        return img, self.bbox_fn(index, self.csv_file, self.scaler).clone().detach().float()


def get_image_with_bbox(index, folder, csv_file):
    # get the list of files in the folder
    # read the csv file
    csv_file = pd.read_csv(csv_file)
    img = cv2.imread(f"{folder}/{csv_file['image_name'][index]}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, get_bbox(index, csv_file)

def get_bbox(index, csv_file, scaler):
    x1 = int(csv_file['bbox_x'][index])
    y1 = int(csv_file['bbox_y'][index])
    width = int(csv_file['bbox_width'][index])
    height = int(csv_file['bbox_height'][index])
    return x1 * scaler, y1 * scaler, width * scaler, height * scaler

def load_bbox_for_yolo(idx, csv_file, scaler, total_bbox = 10):
    x1, y1, width, height = get_bbox(idx, csv_file, scaler)
    center = ((x1 + width/2), (y1 + height/2))
    center = (center[0] / (csv_file["image_width"][idx] * scaler), center[1] / (csv_file["image_height"][idx] * scaler))
    bbox = center + (width / (csv_file["image_width"][idx] * scaler), height / (csv_file["image_height"][idx] * scaler), 1.0, 0.0)
    # now expand the bbox to have total_bbox bboxes using torch.zeros, and -1 for the class
    expanded_bbox = torch.zeros((total_bbox, len(bbox)))
    expanded_bbox[0] = torch.tensor(bbox)

    # Set the class of the additional bounding boxes to -1
    expanded_bbox[1:, -1] = -1

    return expanded_bbox

def load_multiple_bbox_for_yolo(idx, csv_file, scaler, total_bbox = 10):
    # Group by image name
    instances = csv_file[csv_file["image_name"] == csv_file["image_name"][idx]]
    expanded_bbox = torch.zeros((total_bbox, 6))
    expanded_bbox[:, -1] = -1
    i = 0
    for _, instance in instances.iterrows():
        x1, y1, width, height = instance["bbox_x"], instance["bbox_y"], instance["bbox_width"], instance["bbox_height"]
        center = ((x1 + width/2), (y1 + height/2))
        center = (center[0] / (instance["image_width"] * scaler), center[1] / (instance["image_height"] * scaler))
        bbox = center + (width / (instance["image_width"] * scaler), height / (instance["image_height"] * scaler), 1.0, 0.0)
        expanded_bbox[i] = torch.tensor(bbox)
        i += 1
    return expanded_bbox


def show_images_with_boxes(input_tensor, output_tensor, classes):
    to_img = torchvision.transforms.ToPILImage()
    for img, predictions in zip(input_tensor, output_tensor):
        img = to_img(img)
        if 0 in predictions.shape: # empty tensor
            display(img)
            continue
        confidences = predictions[..., 4].flatten()
        boxes = (
            predictions[..., :4].contiguous().view(-1, 4)
        )  # only take first four features: x0, y0, w, h
        classes = predictions[..., 5:].contiguous().view(boxes.shape[0], -1)
        boxes[:, ::2] *= img.width
        boxes[:, 1::2] *= img.height
        boxes = (torch.stack([
                    boxes[:, 0] - boxes[:, 2] / 2,
                    boxes[:, 1] - boxes[:, 3] / 2,
                    boxes[:, 0] + boxes[:, 2] / 2,
                    boxes[:, 1] + boxes[:, 3] / 2,
        ], -1, ).cpu().to(torch.int32).numpy())
        for box, confidence, class_ in zip(boxes, confidences, classes):
            if confidence < 0.015:
                continue # don't show boxes with very low confidence
            # make sure the box fits within the picture:
            box = [
                max(0, int(box[0])),
                max(0, int(box[1])),
                min(img.width - 1, int(box[2])),
                min(img.height - 1, int(box[3])),
            ]
            # try:  # either the class is given as the sixth feature
            #     idx = int(class_.item())
            # except ValueError:  # or the 20 softmax probabilities are given as features 6-25
            #     idx = int(torch.max(class_, 0)[1].item())

            # print(idx)
            # try:
            #     class_ = classes[idx-1]  # the first index of torch.max is the argmax.
            # except IndexError: # if the class index does not exist, don't draw anything:
            #     continue


            color = (  # green color when confident, red color when not confident.
                int((1 - (confidence.item())**0.8 ) * 255),
                int((confidence.item())**0.8 * 255),
                0,
            )

            draw = ImageDraw.Draw(img)
            draw.rectangle(box, outline=color)
            draw.text(box[:2], text = "ring", fill=color)

        display(img)