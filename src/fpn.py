import torch.nn as nn
import torch.nn.functional as F

class Lateral_Connection(nn.Module):
    def __init__(self, channels_in, channels_out, **kwargs):
        super().__init__(**kwargs)
        self.conv1x1 = nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        prev, current = inputs
        fm_size = current.shape[2:]
        up = F.interpolate(prev, size=fm_size, mode='bilinear', antialias=True, align_corners=True)
        current = self.conv1x1(current)
        x = up + current
        
        return x
    

class FPN(nn.Module):
    def __init__(self, block_expansion=1, channels_out=64, **kwargs) :
        super().__init__(**kwargs)

        self.layer_4_out_1x1 = nn.Conv2d(512*block_expansion, channels_out, kernel_size=1, stride=1, padding=0)

        self.p1_3x3 = nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1)
        self.p2_3x3 = nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1)
        self.p3_3x3 = nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1)
        self.p4_3x3 = nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1)

        self.p3_out_lat = Lateral_Connection(256*block_expansion, channels_out)
        self.p2_out_lat = Lateral_Connection(128*block_expansion, channels_out)
        self.p1_out_lat = Lateral_Connection(64*block_expansion, channels_out)


    def forward(self, backbone_outputs):
        layer_1_out, layer_2_out, layer_3_out, layer_4_out = backbone_outputs

        # Apply 1x1 Conv.
        p4_out = self.layer_4_out_1x1(layer_4_out)

        # Apply Lateral connections.
        p3_out = self.p3_out_lat((p4_out, layer_3_out))
        p2_out = self.p2_out_lat((p3_out, layer_2_out))
        p1_out = self.p1_out_lat((p2_out, layer_1_out))

        # Apply an additional 3x3 conv.
        p1_out = F.relu(self.p1_3x3(p1_out), inplace=True)
        p2_out = F.relu(self.p2_3x3(p2_out), inplace=True)
        p3_out = F.relu(self.p3_3x3(p3_out), inplace=True)
        p4_out = F.relu(self.p4_3x3(p4_out), inplace=True)

        outputs = (p1_out, p2_out, p3_out, p4_out)

        return outputs

    
