# %%
import torch
import pytorchyolo
import cv2
import torchvision
import matplotlib.pyplot as plt

CLASSES = ["ring"]

model = pytorchyolo.TinyYolov2(9)
img = cv2.imread("video1/frame_0.jpg")
img = cv2.resize(img, (416, 416))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# convert to tensor
img = torch.from_numpy(img)
img = img.permute(2, 0, 1)
img = img.unsqueeze(0) / 255.0

# predict
pred = model(img)
print(pred.shape)

from PIL import ImageDraw

# %%
from nms import non_max_suppression_fast
from img_loader import show_images_with_boxes

# show the image with bounding boxes
show_images_with_boxes(img, pred, CLASSES)


# %%
# compress pred to 2D
pred_boxes = pred.view(-1, 5 + 1)
print(pred_boxes[:, :4])
nms_tensor = non_max_suppression_fast(pred_boxes.detach().cpu().numpy(), 0.5)
# convert to tensor
nms_tensor = torch.from_numpy(nms_tensor)
# %%
show_images_with_boxes(img, pred)
# %%
print(pred)
# %%
from img_loader import RingDataset

dataset = RingDataset("video1", "labels_my-project-name_2024-03-23-07-10-18.csv")
# %%
import cv2
# show the first image with bounding boxes
img, target = dataset[0]
# convert img to cv2 format
img = img.permute(1, 2, 0).numpy()
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# draw the bounding box
x1, y1, x2, y2 = target
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
# show the image
cv2.imshow("image", img)
cv2.waitKey(0)
# %%
print(pred)
# %%
from img_loader import load_bbox_for_yolo, show_images_with_boxes

pred = load_bbox_for_yolo(0, "video1", "labels_my-project-name_2024-03-23-07-10-18.csv")
# %%
import cv2
import torch
img = cv2.imread("video1/frame_0.jpg")
# convert pred to tensor
pred = torch.tensor(pred)
# draw the bounding box
show_images_with_boxes([img], [pred], ["ring"])
# %%
