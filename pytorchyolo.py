import torch

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False)
        
        
    def forward(self, x):
        return self.bn(self.conv(x))
            
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
        conv = torch.nn.ModuleList([torch.nn.Sequential(ConvBlock(3, max_channels // (2 ** (depth - 3)), 3, 1, 1), 
                                                        self.pool,
                                                        self.activation)])
        for i in range(depth-3, 0, -1):
            conv.append(torch.nn.Sequential(ConvBlock(max_channels // (2 ** i), max_channels // (2 ** (i - 1)), 3, 1, 1),
                                            self.pool,
                                            self.activation))
        conv[5] = torch.nn.Sequential(ConvBlock(max_channels // 4, max_channels // 2, 3, 1, 1),
                                     self.slow_pool,
                                     self.activation)
        conv[6] = torch.nn.Sequential(ConvBlock(max_channels // 2, max_channels, 3, 1, 1),
                                     self.activation)
        conv.append(ConvBlock(max_channels, max_channels, 3, 1, 1))
        conv.append(torch.nn.Conv2d(max_channels, len(anchors) * (5 + num_classes), 1, 1, 0))
        
        self.conv = torch.nn.Sequential(*conv)
        
    def forward(self, x, yolo = True):
        x = self.conv(x)
        if yolo:
            return self.yolo(x)
        return x
    
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
        
        