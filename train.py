import torch
import pytorchyolo

model = pytorchyolo.TinyYolov2(9)
optim = torch.optim.NAdam(model.parameters(), lr=0.001)