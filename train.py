# %%
import torch
import pytorchyolo
import torchvision
from loss_fn import YOLOLoss
from img_loader import RingDataset, ResizePlusCrop
from torch.utils.data import DataLoader, ConcatDataset

# set up our model and training parameters
BATCH_SIZE = 64
EPOCHS = 100
IMG_SCALE = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

qmodel = pytorchyolo.TinyYolov2(8, max_channels = 256)
qmodel.load_pretrained(8)
qmodel.freeze_weights()

qmodel.train()
qmodel.qconfig = torch.quantization.get_default_qconfig('x86')
torch.quantization.prepare_qat(qmodel, inplace=True)
qmodel = qmodel.to(device)

transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(p = 0.2),
    torchvision.transforms.RandomVerticalFlip(p = 0.2),
    torchvision.transforms.RandomApply(torch.nn.ModuleList([
        torchvision.transforms.ColorJitter()
    ]), p=0.2)
])

dataset = RingDataset("video1", "annotations/video1_annotations.csv", spatial_transform = ResizePlusCrop(), scaler = 1/IMG_SCALE, transform= transforms)
dataset2 = RingDataset("video3", "annotations/video3_annotations.csv", spatial_transform = ResizePlusCrop(), scaler = 1/IMG_SCALE, transform= transforms)
dataset3 = RingDataset("video4", "annotations/video4_annotations.csv", spatial_transform = ResizePlusCrop(), scaler = 1/IMG_SCALE, transform= transforms)
dataset4 = RingDataset("video8", "annotations/video8_annotations.csv", type_bbox="multiple_yolo", transform= transforms)
concat_data = ConcatDataset([dataset, dataset2, dataset3])
train_dataloader = DataLoader(concat_data, batch_size = BATCH_SIZE, shuffle = True)
horizontal_train_dataloader = DataLoader(dataset4, batch_size = BATCH_SIZE, shuffle = True)
#%%

def lr_lambda(epoch):
    # LR to be 0.001 * (1/1+0.04*epoch)
    base_lr = 0.01
    factor = 0.04
    return base_lr/(1+factor*epoch)


optim = torch.optim.NAdam(qmodel.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
criterion = YOLOLoss(anchors = qmodel.anchors)
# %%
lowest_loss = 100000
x = qmodel.num_pretrained
for epoch in range(EPOCHS):
    total_loss = 0
    length = len(train_dataloader) + len(horizontal_train_dataloader)
    for img, target in train_dataloader:
        img = img.to(device)
        target = target.to(device)
        optim.zero_grad()
        output = qmodel(img, yolo = False)
        loss = criterion(output, target)
        loss.backward()
        optim.step()
        print(f"Loss: {loss.item()}")
        total_loss += loss.item()
    for img, target in horizontal_train_dataloader:
        img = img.to(device)
        target = target.to(device)
        optim.zero_grad()
        output = qmodel(img, yolo = False)
        loss = criterion(output, target)
        loss.backward()
        optim.step()
        print(f"Loss: {loss.item()}")
        total_loss += loss.item()
    if total_loss/length < lowest_loss:
        lowest_loss = total_loss
        torch.save(qmodel.state_dict(), f"epoch-{epoch}-loss-{total_loss}.pt")
    print(f"---------------\nEpoch: {epoch}, Loss: {total_loss}")
    scheduler.step()
    if (epoch+1) % 10 == 0 and x >= 0:
      qmodel.unfreeze_weights(x)
      x -= 1



# %%
qmodel

# %%
dataset4[0]
# %%
from img_loader import show_images_with_boxes
img, bbox = dataset4[0]
show_images_with_boxes(img.unsqueeze(0), bbox.unsqueeze(0), ["ring"])
# %%
