# %%
import torch
import pytorchyolo
from loss_fn import YOLOLoss
from img_loader import RingDataset, ResizePlusCrop
from torch.utils.data import DataLoader, ConcatDataset

# set up our model and training parameters
BATCH_SIZE = 32
EPOCHS = 5
IMG_SCALE = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

qmodel = pytorchyolo.QuantizedTinyYolov2(9)
qmodel.load_state_dict(torch.load("tiny-yolov2-12-48-pm.pt"))
qmodel.load_pretrained(3)
qmodel.freeze_weights()

qmodel.train()
qmodel.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare_qat(qmodel, inplace=True)
qmodel = qmodel.to("cpu")
epochquantized_model=torch.quantization.convert(qmodel.eval(), inplace=False)
epochquantized_model.load_state_dict(torch.load("tiny-yolov2-12-48-pm-quantized.pt"))

dataset = RingDataset("video1", "annotations/video1_annotations.csv", spatial_transform = ResizePlusCrop(), scaler = 1/IMG_SCALE)
dataset2 = RingDataset("video3", "annotations/video3_annotations.csv", spatial_transform = ResizePlusCrop(), scaler = 1/IMG_SCALE)
dataset3 = RingDataset("video4", "annotations/video4_annotations.csv", spatial_transform = ResizePlusCrop(), scaler = 1/IMG_SCALE)
concat_data = ConcatDataset([dataset, dataset2, dataset3])
train_dataloader = DataLoader(concat_data, batch_size = BATCH_SIZE, shuffle = True)
#%%
optim = torch.optim.NAdam(qmodel.parameters(), lr=0.00003)
criterion = YOLOLoss(anchors = qmodel.anchors)
# %%
for epoch in range(EPOCHS):
    for img, target in train_dataloader:
        img = img.to(device)
        target = target.to(device)
        optim.zero_grad()
        output = qmodel(img, yolo = False)
        loss = criterion(output, target)
        loss.backward()
        optim.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

# %%
img, target = dataset[0]
qmodel.eval()
img = img.to(device)
model = qmodel.to(device)
pred = model(img.unsqueeze(0), yolo = True)
# %%
from img_loader import show_images_with_boxes
show_images_with_boxes(img.unsqueeze(0), pred, ["ring"])
# %%

# %%


backend = "fbgemm"  # x86 machine
torch.backends.quantized.engine = backend
epochquantized_model.qconfig = torch.quantization.get_default_qconfig(backend)
device = "cpu"
img, target = dataset[0]
epochquantized_model.eval()
img = img.to(device).unsqueeze(0)
model = epochquantized_model.to(device)
img = (img * 255).to(torch.int8)
pred = epochquantized_model(img, yolo = True)
# %%
from img_loader import show_images_with_boxes
show_images_with_boxes(img.unsqueeze(0), pred, ["ring"])
# %%
img, target = dataset[0]
img *= 255
img = img.to(torch.int8)
# %%
print(torch.max(img))
# %%
