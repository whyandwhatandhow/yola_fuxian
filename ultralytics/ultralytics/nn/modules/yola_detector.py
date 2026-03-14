import torch
import torch.nn as nn

from ultralytics.nn.modules.yola_iiblock import IIBlock


class YOLA(nn.Module):

    def __init__(self, backbone, neck, head):

        super().__init__()

        self.iim = IIBlock()

        self.backbone = backbone
        self.neck = neck
        self.head = head

        self.consistency_loss = nn.SmoothL1Loss()

    def forward(self, x):

        x, feats = self.iim(x)

        x = self.backbone(x)

        x = self.neck(x)

        pred = self.head(x)

        return pred, feats