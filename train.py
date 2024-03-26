# %%
import torch
import pytorchyolo
from loss_fn import YOLOLoss
from img_loader import RingDataset


dataset = RingDataset("video1", "video1_annotations.csv")
img, target = dataset[0]
#%%
model = pytorchyolo.TinyYolov2(9)
optim = torch.optim.NAdam(model.parameters(), lr=0.001)
criterion = YOLOLoss(anchors = model.anchors)
output = model(img.unsqueeze(0), yolo = False)
loss = criterion(output, target.unsqueeze(0))
#%%
print()