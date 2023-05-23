from collections import OrderedDict

from functools import partial
from typing import Dict, List
# import tensorflow as tf
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from src.resnet_backbone import resnet50
from src.wider_resnet import WiderResNetA2
from da_att import PAM_Module, CAM_Module


# 初始化权重
def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class downsample(nn.Module):
    def __init__(self, in_channel, out_channel, change_channel=True):
        super(downsample, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)   # 池化向上取整?
        self.conv_1 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1) # 左右边分支的1*1卷积
        self.conv_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=2, padding=1) # 卷积向下取整
        self.change_channel = change_channel
        if self.change_channel:
            self.conv_3 = nn.Conv2d(in_channel * 4, out_channel, kernel_size=1, stride=1)        # concat之后转通道的1*1卷积

    def forward(self, x):
        identity = x
        out_1 = self.conv_1(self.maxpool(x))
        out_2 = self.conv_2(self.conv_1(identity))
        out = torch.cat([out_1, out_2], dim=1)
        if self.change_channel:
            out = self.conv_3(out)
        return out


# First_Block
class First_Block(nn.Module):  # 包括mod1和mod2
    def __init__(self, in_channel=3, out_channel=64):
        super(First_Block, self).__init__()
        self.layer1 = nn.Sequential(                  # mod1
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
        )
        self.layer2 = downsample(in_channel=64, out_channel=64, change_channel=False)  # mod2:本来out_channel应该等于128，但downsample 会自动concat通道

    def forward(self, x):
        return self.layer2(self.layer1(x))


# 以下是ASPP分支的代码


class Block1(nn.Module):          # block1
    def __init__(self, in_channel=1024, out_channel=256):
        super(Block1, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class Block_Dilconv(nn.Module):          # 三个膨胀卷积
    def __init__(self, dilation, in_channel=1024, out_channel=256):
        super(Block_Dilconv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class Block_Pooling(nn.Sequential):         # 平均池化层
    def __init__(self, in_channel=1024, out_channel=256):
        super(Block_Pooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        # x = self.layer(x)
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, atrous_rates: List[int], in_channel=1024, out_channel=512, mid_channel=1280):
        super(ASPP, self).__init__()
        # self.first_block = First_Block(in_channel=3, out_channel=64)
        modules = []
        modules.append(Block1(in_channel, in_channel//4))          # 在列表中加入block1 in_channel =2048  in_channel//4=256
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(Block_Dilconv(rate, in_channel, in_channel//4))    # 在列表中加入膨胀卷积
        modules.append(Block_Pooling(in_channel, in_channel//4))              # 平均池化层
        self.convs = nn.ModuleList(modules)

        self.layer =nn.Conv2d(mid_channel, out_channel, kernel_size=1, stride=1)           # 经过ASPP之后改变通道数

    def forward(self, x):
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.layer(res)


# 以下是Da_Attention的代码

class Da_Attention(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Da_Attention, self).__init__()
        # in_channels=128,out_channel=128
        self.pam_conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, bias=False),# 1*1的卷积
                                    nn.BatchNorm2d(out_channel),
                                    nn.ReLU())

        self.cam_conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, bias=False),
                                    nn.BatchNorm2d(out_channel),
                                    nn.ReLU())

        self.sa = PAM_Module(out_channel)  # 空间注意力模块
        self.sc = CAM_Module(out_channel)  # 通道注意力模块

        self.after_pam_conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False), # 经过注意力之后的卷积
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU())
        self.after_cam_conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU())

    def forward(self, x):
        # 空间注意力
        pam_feature = self.pam_conv(x)
        pam_feature = self.sa(pam_feature)
        # 通道注意力
        cam_feature = self.cam_conv(x)
        cam_feature = self.sc(cam_feature)

        # 经过注意力后用3*3卷积
        pa_conv = self.after_pam_conv(pam_feature)
        ca_conv = self.after_cam_conv(cam_feature)

        # 两个注意力模块结果相加
        feat_sum = pa_conv+ca_conv

        return feat_sum


# mlp模块
class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.):
        super().__init__()
        # hidden_dim = int((in_dim+out_dim)*2/3)
        # hidden_dim = in_dim * 4
        hidden_dim = int((in_dim / 2))
        self.net = nn.Sequential(
            # 由此可以看出 FeedForward 的输入和输出维度是一致的
            # nn.LayerNorm(dim),
            nn.Linear(in_dim, hidden_dim),
            # 激活函数
            nn.GELU(),
            # 防止过拟合
            nn.Dropout(dropout),
            # 重复上述过程
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        identity = x
        identity = identity.permute(0, 2, 3, 1)                # 把矩阵拉平
        identity = self.net(identity)
        identity = identity.permute(0, 3, 1, 2)

        return x+identity                                      # skip-connection


class Mlp_Fusion(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = MLPBlock(in_dim, out_dim)
        # self.is_first_stage = is_first_stage
        # if is_first_stage:
        #     self.conv_1 = nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=True)
        #     self.maxpool_1 = downsample(128, 256)     # in_channel=128, out_channel=256

    def forward(self, x):  # x来自Feature_Fusion融合模块
        out1 = self.mlp(x)
        # qkv = out1.reshape(B, C, H, W).permute(0,1,2,3)
        # qkv = out1.permute(0, 1, 2, 3).contiguous().view(B, C, H//2, W//2)
        # qkv = out
        # k = v = q = qkv
        # # attn = (q @ k.transpose(0, 1))               # 指定行列进行转置,指定0（表示行）和1（表示列）进行转换
        # B, C, H, W = k.shape
        # k = k.view(B, C, H, W).permute(0, 1, 3, 2)     # transpose 一直有错 改为permute
        # attn = torch.matmul(q, k)
        # attn = attn.softmax(dim=-1)
        # out = torch.matmul(attn, v)                          # out1 = attn @ v
        out2 = out1
        return out1, out2
# 以下是Feature_Fusion融合模块的代码


class Feature_Fusion(nn.Module):
    def __init__(self, in_channel, out_channel, is_maxpool=False):
        super(Feature_Fusion, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)  #为改变Da_Attention分支的通道
        self.maxpool = is_maxpool
        if is_maxpool:                              # 第一次融合通道增加，高宽减半
            self.maxpool_1 = downsample(in_channel, out_channel)     # in_channel=128, out_channel=256
            self.maxpool_2 = downsample(in_channel, out_channel)     # in_channel=128, out_channel=256
        # if is_first_stage:                              # 第一次融合通道增加，高宽减半
        #     self.conv = downsample(in_channel, out_channel)     # in_channel=128, out_channel=256
        self.fusion = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)  # 最后的特征融合

    def forward(self, x1, x2):                       # x1为Da_Attention分支的结果，x2为wider_resnet分支的结果
        # B, C, H, W = x2.shape                        # x2.shape = 256*128*128
        # if self.first_stage:
        #     H = H//2
        #     W = W//2
        if self.maxpool:
            out1 = self.maxpool_2(x2) + self.maxpool_1(self.conv(x1))
        else:
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            out1 = x2 + self.conv(x1)
            out1 = self.fusion(out1)
        out2 = out1
        return out1, out2


# 以下是cross-attenntion的模块代码
# class Cross_Attention(nn.Module):
#     def __init__(self):
#         super(Cross_Attention, self).__init__()
#
#     def forward(self, x1, x2):    # x1来自经过融合模块的输出结果,x2来自Da_Attention的输出结果
#         q = k = x1
#         v = x2
#         B, C, H, W = k.shape
#         k = k.view(B, C, H, W).permute(0, 1, 3, 2)
#         attn = (q @ k)
#         attn = attn.softmax(dim=-1)
#         out1 = torch.matmul(attn, v)                          # out1 = attn @ v
#         out2 = out1
#         return out1, out2


# (第一版融合模块尝试-参数量太大)

# class Feature_Fusion(nn.Module):
#     def __init__(self, in_channel, out_channel, is_maxpool=False):
#         super(Feature_Fusion, self).__init__()
#         self.conv= nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)  #为改变Da_Attention分支的通道
#         self.maxpool = is_maxpool
#         if is_maxpool:                              # 第一次融合通道增加，高宽减半
#             self.maxpool_1 = nn.MaxPool2d(3, stride=2, padding=1)
#             self.maxpool_2 = nn.MaxPool2d(3, stride=2, padding=1)
#
#     def forward(self, x1, x2):                       # x1为Da_Attention分支的结果，x2为wider_resnet分支的结果
#         if self.maxpool:
#             out1 = self.maxpool_2(x2) * self.maxpool_1(self.conv(x1))
#         else:
#             x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
#             out1 = x2 * self.conv(x1)
#         out2 = out1
#         return out1, out2

# 以下是最终的解码器


class ResidualConvUnit(nn.Module):             # 第一种RCU模块
    """Residual convolution module."""

    def __init__(self, in_channel, out_channel):
        """Init.

        Args:
            features (int): number of features
        """
        super(ResidualConvUnit, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass.

        Args:
            x1,x2 (tensor): input

        Returns:
            tensor: output
        """
        identity = x
        out = self.relu(identity)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):       # 第一种decoder_Feature_fusion模块
    """Feature fusion block."""

    def __init__(self, in_channel, out_channel, is_upsample=0):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(in_channel, out_channel)        # self.resConfUnit1 = ResidualConvUnit(in_channel, in_channel)
        self.resConfUnit2 = ResidualConvUnit(out_channel, out_channel)        # self.resConfUnit2 = ResidualConvUnit(out_channel, out_channel)
        self.is_upsample = is_upsample
        self.conv1_1 = nn.Conv2d(                                     # 改变通道的1*1卷积
            512, 256, kernel_size=1, stride=1, bias=True
        )
        self.conv1_2 = nn.Conv2d(                                     # 改变通道的1*1卷积
            256, 128, kernel_size=1, stride=1, bias=True
        )
        self.conv1_3 = nn.Conv2d(
            128, 64, kernel_size=1, stride=1, bias=True
        )

    def forward(self, x1, x2):  # x1为深层特征，x2为浅层特征(Feature_Fusion模块的输出)
        """Forward pass.

        Returns:
            tensor: output
        """
        if self.is_upsample == 2:
            x2 = self.conv1_1(x2)                      # 浅层特征x2改变通道：512->256
            out_1 = self.resConfUnit1(x1)
            output = out_1 + x2                        # 通道已经变成了256
            output = self.resConfUnit2(output)
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)  # output tensor = 256,128,128
        if self.is_upsample == 1:
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
            out_1 = self.resConfUnit1(x1)
            output = out_1 + x2
            output = self.resConfUnit2(output)
            output = self.conv1_2(output)                          # channel：256->128
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)  # output tensor = 128,128,128
        if self.is_upsample == 0:
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
            out_1 = self.resConfUnit1(x1)
            output = out_1 + x2
            output = self.resConfUnit2(output)
            output = self.conv1_3(output)                          # channel:128->64
        return output


class DAFModel(nn.Module):
    def __init__(self, classifier, num_classes, in_channel=1024, out_channel=512):
        super(DAFModel, self).__init__()
        # backbone 模块
        # self.backbone = backbone

        # First_Block    in_channel=3, out_channel=64
        self.first = First_Block()
        # ASPP模块
        self.classifier = classifier
        # wide_resnet分支
        wide_resnet = WiderResNetA2(structure=[3, 6, 3], classes=17, dilation=True)
        wide_resnet = torch.nn.DataParallel(wide_resnet)
        wide_resnet = wide_resnet.module
        self.mod3 = wide_resnet.mod3
        self.mod4 = wide_resnet.mod4
        self.mod5 = wide_resnet.mod5
        self.pool2 = wide_resnet.pool2
        self.pool3 = wide_resnet.pool3
        del wide_resnet
        # 三次注意力模块
        self.da_att1 = Da_Attention(in_channel=128, out_channel=128)
        self.da_att2 = Da_Attention(in_channel=256, out_channel=256)
        self.da_att3 = Da_Attention(in_channel=512, out_channel=512)
        # 融合模块
        self.fusion_1 = Feature_Fusion(in_channel=128, out_channel=256, is_maxpool=True)
        self.mlp_fusion_1 = Mlp_Fusion(in_dim=256, out_dim=256)
        # self.cross_att1 = Cross_Attention()
        self.fusion_2 = Feature_Fusion(in_channel=256, out_channel=512)
        self.mlp_fusion_2 = Mlp_Fusion(in_dim=512, out_dim=512)
        # self.cross_att2 = Cross_Attention()
        # decoder 模块

        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 4, kernel_size=1, stride=1, bias=False),         # channel:1024->256
            nn.BatchNorm2d(in_channel//4),
            nn.ReLU()
        )

        self.refinenet_3 = FeatureFusionBlock(out_channel//2, out_channel//2, is_upsample=2)   # in_channel//2=512 ,out_channel//2=256
        self.refinenet_2 = FeatureFusionBlock(out_channel//2, out_channel//2, is_upsample=1)   # in_channel//4=256 ,out_channel//4=128
        self.refinenet_1 = FeatureFusionBlock(in_channel//8, in_channel//8)   # in_channel//8=128 ,out_channel//8=64

        self.conv_end = nn.Conv2d(out_channel//8, num_classes, kernel_size=1, stride=1)                # out_channel//6=64 最后一个卷积

        initialize_weights(self.conv_end)

    def forward(self, x):
        x = self.first(x)
        identity = x           # x为注意力分支feature identity为mod分支feature

        # 第一次融合
        att_out_1 = self.da_att1(x)
        mod_out_1 = self.mod3(identity)
        att_out_2, mod_out_2 = self.fusion_1(att_out_1, mod_out_1)
        mlp_att_out_2, mlp_mod_out_2 = self.mlp_fusion_1(att_out_2)
        # cross_att_out_1, cross_mod_out_1 = self.cross_att1(mlp_att_out_2, att_out_2)
        att_out_2 = self.da_att2(mlp_att_out_2)
        mod_out_2 = self.mod4(mlp_mod_out_2)
        # 第二次融合
        att_out_3, mod_out_3 = self.fusion_2(att_out_2, mod_out_2)
        mlp_att_out_3, mlp_mod_out_3 = self.mlp_fusion_2(att_out_3)
        # cross_att_out_2, cross_mod_out_2 = self.cross_att2(mlp_att_out_3, att_out_3)
        att_out = self.da_att3(mlp_att_out_3)
        mod_out = self.mod5(mlp_mod_out_3)
        # 第三次在融合
        aspp_out = self.classifier(mod_out)
        out = torch.cat([att_out, aspp_out], dim=1)
        # Decoder模块
        fusion_high_3 = self.layer(out)                # change channel:1024->256
        fusion_high_2 = self.refinenet_3(fusion_high_3, att_out_3)
        fusion_high_1 = self.refinenet_2(fusion_high_2, att_out_2)
        output = self.refinenet_1(fusion_high_1, att_out_1)
        output = self.conv_end(output)

        return {"out": output}
        # return x


def dafmodel_wideresnet(num_classes=21, atrous_rates=[12, 24, 36], in_channel=1024, out_channel=512, mid_channel=1280 ,
                       pretrain_backbone=True):

    out_inplanes = 1024
    classifier = ASPP(atrous_rates=atrous_rates, in_channel=out_inplanes, out_channel=out_channel, mid_channel=mid_channel)
    model = DAFModel(classifier=classifier, num_classes=num_classes, in_channel=in_channel, out_channel=out_channel)

    # 加载vit分支的权重（未成功）
    # model_dict = model.vit.state_dict()
    # pretrained_dict = model.vit.torch.load("vit_base_patch16_224_in21k.pth", map_location='cpu')
    # temp = {}
    # for k, v in pretrained_dict.items():
    #     try:
    #         if np.shape(model_dict[k]) == np.shape(v):
    #             temp[k] = v
    #     except:
    #         pass
    # model_dict.update(temp)
    # model.vit.load_state_dict(model_dict)

    return model


# if __name__ == "__main__":
#     import numpy as np
#     a = DepVModel(21)
#
#     # print(a)
#     im = torch.randn(2, 3, 448, 448)
#     print(a(im))
