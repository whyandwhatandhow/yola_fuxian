import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ReflectedConvolution(nn.Module):

    def __init__(self, kernel_nums=8, kernel_size=3):
        super().__init__()

        self.kernel_nums = kernel_nums
        self.kernel_size = kernel_size

        self.rg_bn = nn.BatchNorm2d(kernel_nums)
        self.gb_bn = nn.BatchNorm2d(kernel_nums)
        self.rb_bn = nn.BatchNorm2d(kernel_nums)

        self.filter = nn.Parameter(
            torch.randn(kernel_nums, 1, kernel_size, kernel_size)
        )

        nn.init.kaiming_normal_(self.filter)

    def mean_constraint(self, kernel):
        bs, cin, kw, kh = kernel.shape
        kernel_mean = torch.mean(kernel.view(bs, -1), dim=1, keepdim=True)
        kernel = kernel.view(bs, -1) - kernel_mean
        return kernel.view(bs, cin, kw, kh)

    def forward(self, img):

        log_img = torch.log(img + 1e-7)

        red = log_img[:, 0:1]
        green = log_img[:, 1:2]
        blue = log_img[:, 2:3]

        filt = self.mean_constraint(self.filter)
        pad = self.kernel_size // 2

        rg = F.conv2d(red, filt, padding=pad) - F.conv2d(green, filt, padding=pad)
        gb = F.conv2d(green, filt, padding=pad) - F.conv2d(blue, filt, padding=pad)
        rb = F.conv2d(red, filt, padding=pad) - F.conv2d(blue, filt, padding=pad)

        rg = self.rg_bn(rg)
        gb = self.gb_bn(gb)
        rb = self.rb_bn(rb)

        return torch.cat([rg, gb, rb], dim=1)


class IIBlock(nn.Module):

    def __init__(self, kernel_nums=8, kernel_size=3, Gtheta=[0.6, 0.8], ii_beta=1.0):
        super().__init__()

        self.Gtheta = Gtheta
        self.ii_beta = ii_beta
        self.ii_loss = None

        self.feat_projector = nn.Sequential(
            nn.Conv2d(3, 24, 3, 1, 1),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
        )

        self.fuse_net = nn.Sequential(
            nn.Conv2d(48, 32, 3, 1, 1, groups=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
        )

        self.iim = ReflectedConvolution(kernel_nums, kernel_size)

    def forward(self, x):

        x_gma = torch.pow(x, np.random.uniform(self.Gtheta[0], self.Gtheta[1]))
        # gamma = torch.empty(1).uniform_(self.Gtheta[0], self.Gtheta[1]).to(x.device)
        # x_gma = torch.pow(x, gamma)
        x_gma = torch.clamp(x_gma, 0, 1)

        feat_ii = self.iim(x)
        feat_ii_gma = self.iim(x_gma)
        if self.training:
            diff = feat_ii - feat_ii_gma
            beta = float(self.ii_beta)
            self.ii_loss = torch.where(diff.abs() <= beta, 0.5 * diff**2, diff.abs() - 0.5 * beta).mean()
        else:
            self.ii_loss = None

        feats = self.feat_projector(x)

        feats_ = torch.cat((feats, feat_ii), dim=1)

        x_out = self.fuse_net(feats_)

        return x_out