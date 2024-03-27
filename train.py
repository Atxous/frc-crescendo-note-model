# %%
import torch
import pytorchyolo
from loss_fn import YOLOLoss
from img_loader import RingDataset

BATCH_SIZE = 16
EPOCHS = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = RingDataset("video1", "video1_annotations.csv")
img, target = dataset[0]
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = pytorchyolo.TinyYolov2(9)
model.load_state_dict(torch.load("model.pt"))
model.to(device)
#%%

optim = torch.optim.NAdam(model.parameters(), lr=0.001)
criterion = YOLOLoss(anchors = model.anchors)
output = model(img.unsqueeze(0), yolo = False)
loss = criterion(output, target.unsqueeze(0))

model = model.to(device)
# %%
for epoch in range(EPOCHS):
    for img, target in train_dataloader:
        img = img.to(device)
        target = target.to(device)
        optim.zero_grad()
        output = model(img, yolo = False)
        loss = criterion(output, target)
        loss.backward()
        optim.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

# %%
model.eval()
img = img.to(device)
pred = model(img.unsqueeze(0), yolo = True)
# %%
from img_loader import show_images_with_boxes
show_images_with_boxes(img, pred, ["ring"])
# %%
print(pred)

