import torch
import torch.nn as nn
from .backbone import Backbone
from .fpn import FPN

class DetectorHead(nn.Module):
    def __init__(self, fpn_channels=64, num_anchors=4, num_classes=2, localization=True, **kwargs):
        super().__init__(**kwargs)
        self.fpn_channels = fpn_channels
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.localization = localization

        if self.localization:
            self.output_head_nodes = 4
            self.head = self._make_head(self.fpn_channels, self.num_anchors*self.output_head_nodes)

        else:
            self.output_head_nodes = self.num_classes
            self.head = self._make_head(self.fpn_channels, self.num_anchors*self.output_head_nodes)


    def forward(self, feature_maps):

        preds = []

        for feature_map in feature_maps:
            pred = self.head(feature_map)
            pred = pred.permute(0,2,3,1).reshape(pred.shape[0], -1, self.output_head_nodes)
            preds.append(pred)

        preds = torch.cat(preds, dim=1)

        return preds


    @staticmethod
    def _make_head(fpn_channels, final_op_channels):
        layers = []

        for _ in range(4):
            layers.append(nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(fpn_channels, final_op_channels, kernel_size=3, stride=1, padding=1))

        return nn.Sequential(*layers)
    


class Detector(nn.Module):
    def __init__(self, backbone_name="resnet18", num_classes=2, fpn_channels=64, num_anchors=9, **kwargs) -> None:
        super().__init__(**kwargs)

        train_backbone = kwargs.get("train_backbone", False)
        self.backbone = Backbone(backbone_name, train_backbone=train_backbone, return_interm_layers=True)

        if backbone_name == "resnet18":
            self.block_expansion = 1
        else:
            self.block_expansion = 4

        self.num_classes = num_classes
        self.fpn_channels = fpn_channels
        self.num_anchors = num_anchors

        self.fpn = FPN(block_expansion=self.block_expansion, channels_out=self.fpn_channels)

        self.embedding_1 = nn.Sequential(
                                        nn.Conv2d(512*self.block_expansion, self.fpn_channels, kernel_size=3, stride=2, padding=1),
                                        nn.ReLU(inplace=True)
                                        )
        self.embedding_2 = nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, stride=2, padding=1)

        

        self.loc_head = DetectorHead(fpn_channels=self.fpn_channels, num_anchors=self.num_anchors,\
                                    num_classes=self.num_classes, localization=True)
        
        self.cls_head = DetectorHead(fpn_channels=self.fpn_channels, num_anchors=self.num_anchors,\
                                    num_classes=self.num_classes, localization=False)
  
    def forward(self, inputs):

        # Get layers from feature extractor.
        x = self.backbone(inputs)
        
        layer1_output = x['0']
        layer2_output = x['1']
        layer3_output = x['2']
        layer4_output = x['3']

        # Get output from FPN model class.
        p1_out, p2_out, p3_out, p4_out = self.fpn((layer1_output, layer2_output, layer3_output, layer4_output))

        # Get outputs from embedding layers.
        embedding1_out = self.embedding_1(layer4_output)
        embedding2_out = self.embedding_2(embedding1_out)

        # Get localization and classifictaion heads.
        loc_out = self.loc_head((p1_out, p2_out, p3_out, p4_out, embedding1_out, embedding2_out))
        cls_out = self.cls_head((p1_out, p2_out, p3_out, p4_out, embedding1_out, embedding2_out))
        
        return loc_out, cls_out



