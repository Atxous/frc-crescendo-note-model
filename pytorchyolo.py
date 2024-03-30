import torch
import numpy as np
import urllib
import os

class SiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.float_fn = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        return self.float_fn.mul(self.sigmoid(x), x)

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation = None, pool = None):
        super(ConvBlock, self).__init__()
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False)
        self.activation = activation
        self.pool = pool

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.pool:
          x = self.pool(x)
        if self.activation:
          x = self.activation(x)
        return x

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = torch.nn.BatchNorm2d(channels)
        self.conv2 = torch.nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = torch.nn.BatchNorm2d(channels)
        self.activation = SiLU()
        self.float_fn = self.float_fn = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        skip = x
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.activation(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.float_fn.add(output, skip)
        output = self.activation(output)
        return output

class QAT(torch.nn.Module):
    def __init__(self, model):
        self.model = model
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        return self.dequant(x)


class TinyYolov2(torch.nn.Module):
    def __init__(
        self,
        depth,
        num_classes = 1,
        max_channels = 1024,
        anchors = ((1.08, 1.19),
                   (3.42, 4.41),
                   (6.63, 11.38),
                   (9.42, 5.11),
                   (16.62, 10.52)),
        activation : str = "swish",
    ):
        super(TinyYolov2, self).__init__()
        self.register_buffer('anchors', torch.tensor(anchors))
        self.num_classes = num_classes
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.slow_pool = torch.nn.MaxPool2d(2, 1)
        self.num_pretrained = 0

        if depth < 3:
            raise ValueError("Depth must be at least 3")

        if activation == "swish":
            self.activation = torch.nn.SiLU()
        elif activation == "leaky_relu":
            self.activation = torch.nn.LeakyReLU(0.1)
        elif activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "gelu":
            self.activation = torch.nn.GELU()
        else:
            raise ValueError(f"Activation function {activation} not supported")

        # Define the network architecture
        self.conv = torch.nn.ModuleList([ConvBlock(3, max_channels // (2 ** (depth - 3)), 3, 1, 1, self.activation, self.pool)])
        for i in range(depth-3, 0, -1):
            self.conv.append(ConvBlock(max_channels // (2 ** i), max_channels // (2 ** (i - 1)), 3, 1, 1, self.activation, self.pool))
            self.conv.append(ResidualBlock(max_channels // (2 ** (i - 1))))
        self.conv[-2] = ConvBlock(max_channels // 2, max_channels, 3, 1, 1, self.activation, self.slow_pool)
        self.conv.append(ConvBlock(max_channels, max_channels, 3, 1, 1, self.activation))
        # self.conv.append(ConvBlock(max_channels, max_channels, 3, 1, 1))
        # self.conv.append(torch.nn.Conv2d(max_channels, len(anchors) * (5 + num_classes), 1, 1, 0))
        self.conv.append(ResidualBlock(max_channels))
        self.conv.append(ConvBlock(max_channels, max_channels, 3, 1, 1, self.activation))
        self.conv.append(ResidualBlock(max_channels))
        self.conv.append(ConvBlock(max_channels, max_channels, 3, 1, 1))
        self.conv.append(torch.nn.Conv2d(max_channels, len(anchors) * (5 + num_classes), 1, 1, 0))
        self.conv = torch.nn.Sequential(*self.conv)

    def forward(self, x, yolo = True):
        x = self.conv(x)
        if yolo:
            return self.yolo(x)
        return x

    def load_pretrained(self, num_layers):
      assert num_layers <= 9, "There cannot be more than 9 layers."
      self.num_pretrained = num_layers
      if not os.path.exists("yolov2-tiny-voc.weights"):
          urllib.request.urlretrieve(
              "https://pjreddie.com/media/files/yolov2-tiny-voc.weights",
              "yolov2-tiny-voc.weights",
          )
      with open("yolov2-tiny-voc.weights", "rb") as file:
        version = np.fromfile(file, count=3, dtype=np.int32)
        seen_so_far = np.fromfile(file, count=1, dtype=np.int32)
        weights = np.fromfile(file, dtype=np.float32)
        idx = 0
        for i in range(num_layers):
            if isinstance(self.conv[i], ConvBlock):
                for module in self.conv[i].children():
                    if isinstance(module, torch.nn.Conv2d):
                        if module.bias is not None:
                            n = module.bias.numel()
                            module.bias.data[:] = torch.from_numpy(weights[idx : idx + n]).view_as(self.conv[i].bias.data)
                            idx += n
                            n = module.weight.numel()
                            module.weight.data[:] = torch.from_numpy(weights[idx : idx + n]).view_as(module.weight.data)
                            idx += n
                        if isinstance(module, torch.nn.BatchNorm2d):
                            n = module.bias.numel()
                            module.bias.data[:] = torch.from_numpy(weights[idx : idx + n]).view_as(module.bias.data)
                            idx += n
                            module.weight.data[:] = torch.from_numpy(weights[idx : idx + n]).view_as(module.weight.data)
                            idx += n
                            module.running_mean.data[:] = torch.from_numpy(weights[idx : idx + n]).view_as(module.running_mean)
                            idx += n
                            module.running_var.data[:] = torch.from_numpy(weights[idx : idx + n]).view_as(module.running_var)
                            idx += n
                        if isinstance(module, torch.nn.Linear):
                            n = module.bias.numel()
                            module.bias.data[:] = torch.from_numpy(weights[idx : idx + n]).view_as(module.bias.data)
                            idx += n
                            n = module.weight.numel()
                            module.weight.data[:] = torch.from_numpy(weights[idx : idx + n]).view_as(module.weight.data)
                            idx += n
            if isinstance(self.conv[i], torch.nn.Conv2d):
                if self.conv[i].bias is not None:
                    n = self.conv[i].bias.numel()
                    self.conv[i].bias.data[:] = torch.from_numpy(weights[idx : idx + n]).view_as(self.conv[i].bias.data)
                    idx += n
                n = self.conv[i].weight.numel()
                self.conv[i].weight.data[:] = torch.from_numpy(weights[idx : idx + n]).view_as(self.conv[i].weight.data)
                idx += n
            if isinstance(self.conv[i], torch.nn.BatchNorm2d):
                n = self.conv[i].bias.numel()
                self.conv[i].bias.data[:] = torch.from_numpy(weights[idx : idx + n]).view_as(self.conv[i].bias.data)
                idx += n
                self.conv[i].weight.data[:] = torch.from_numpy(weights[idx : idx + n]).view_as(self.conv[i].weight.data)
                idx += n
                self.conv[i].running_mean.data[:] = torch.from_numpy(weights[idx : idx + n]).view_as(self.conv[i].running_mean)
                idx += n
                self.conv[i].running_var.data[:] = torch.from_numpy(weights[idx : idx + n]).view_as(self.conv[i].running_var)
                idx += n
            if isinstance(self.conv[i], torch.nn.Linear):
                n = self.conv[i].bias.numel()
                self.conv[i].bias.data[:] = torch.from_numpy(weights[idx : idx + n]).view_as(self.conv[i].bias.data)
                idx += n
                n = self.conv[i].weight.numel()
                self.conv[i].weight.data[:] = torch.from_numpy(weights[idx : idx + n]).view_as(self.conv[i].weight.data)
                idx += n

    def freeze_weights(self):
        for i in range(self.num_pretrained):
            if isinstance(self.conv[i], torch.nn.Conv2d):
                for param in self.conv[i].parameters():
                    param.requires_grad = False
            elif isinstance(self.conv[i], torch.nn.BatchNorm2d):
                for param in self.conv[i].parameters():
                    param.requires_grad = False

    def unfreeze_weights(self, layer_idx):
        for param in self.conv[layer_idx].parameters():
            param.requires_grad = True

    def replace_activation(self):
        self.activation = SiLU()
        for module in self.conv:
            if isinstance(module, ResidualBlock):
                module.activation = SiLU()
                module.quantize_addition()
            if isinstance(module, ConvBlock):
                module.activation = SiLU()

    def yolo(self, x):
        # store og shape
        nB, _, nH, nW = x.shape

        # reshape to batch size, anchors, grid size, classes + 5
        x = x.view(nB, self.anchors.shape[0], -1, nH, nW).permute(0, 1, 3, 4, 2)

        # normalized auxiliary tensors
        anchors = self.anchors.to(dtype = x.dtype, device = x.device)
        range_y, range_x = torch.meshgrid(torch.arange(nH, device = x.device), torch.arange(nW, device = x.device), indexing = 'ij')
        anchor_x, anchor_y = anchors[:, 0], anchors[:, 1]

        # compute boxes.
        x = torch.cat([
            (x[:, :, :, :, 0:1].sigmoid() + range_x[None,None,:,:,None]) / nW,  # X center
            (x[:, :, :, :, 1:2].sigmoid() + range_y[None,None,:,:,None]) / nH,  # Y center
            (x[:, :, :, :, 2:3].exp() * anchor_x[None,:,None,None,None]) / nW,  # Width
            (x[:, :, :, :, 3:4].exp() * anchor_y[None,:,None,None,None]) / nH,  # Height
            x[:, :, :, :, 4:5].sigmoid(), # confidence
            x[:, :, :, :, 5:].softmax(-1), # classes
        ], -1)

        return x # (batch_size, # anchors, height, width, 5+num_classes)

        