#%%
import torch
import pytorchyolo
from loss_fn import YOLOLoss

qmodel = pytorchyolo.TinyYolov2(9)
qmodel.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare_qat(qmodel, inplace=True)
epochquantized_model=torch.quantization.convert(qmodel.eval(), inplace=False)
epochquantized_model.load_state_dict(torch.load("tiny-yolov2-12-48-pm-quantized.pt"))
# %%
