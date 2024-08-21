from typing import List

import torch
from torch import nn

from timm.models.swin_transformer_v2 import SwinTransformerV2
from model_utils import IntermediateLayerGetter
from deeplab import DeepLabHeadV3Plus, DeepLabV3Plus
import resnet as resnet

attn_label = True


class CustomSwinTransformerV2(SwinTransformerV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.pointwise_conv = nn.Conv2d(self.embed_dim, self.embed_dim // 2, kernel_size=1)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
            x = self.avgpool(x)
            x = self.pointwise_conv(x)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class CombinedBackbone(nn.Module):
    def __init__(self, resnet, swin_transformer):
        super(CombinedBackbone, self).__init__()
        # ResNet layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Swin Transformer
        self.swin_transformer = swin_transformer

    def forward(self, x):
        # ResNet forward
        x_resnext = self.conv1(x)
        x_resnext = self.bn1(x_resnext)
        x_resnext = self.relu(x_resnext)
        x_resnext = self.maxpool(x_resnext)
        low_level = self.layer1(x_resnext)
        x_resnet = self.layer2(low_level)
        x_resnet = self.layer3(x_resnet)
        x_resnet = self.layer4(x_resnet)
        # Swin Transformer forward
        x_swin = self.swin_transformer.forward_features(x)
        # x_swin_attn = self.swin_transformer.forward_features(x).stages[0].attn()

        # x_patch = self.swin_transformer.patch_embed(x)
        # x_downsample = self.swin_transformer.stages[0].downsample(x_patch)
        # print("x_downsample_shape :", x_downsample.shape)
        # x_attn = self.swin_transformer.stages[0].blocks[0].attn(x_downsample[0].squeeze())
        # print("x_attn_shape :", x_attn.shape)
        # print(self.swin_transformer.stages[0].blocks[0])
        # print(self.swin_transformer.stages[0].blocks[0].attn(x1))

        # Adjust the spatial dimensions of Swin features to match ResNet features
        x_swin = nn.functional.interpolate(x_swin, size=low_level.shape[2:], mode='bilinear', align_corners=False)
        # Concatenate features along the channel dimension
        combined_low_level = torch.cat([low_level, x_swin], dim=1)
        return {
            "low_level": combined_low_level,
            "out": x_resnet
        }


def CSegNet():
    replace_stride_with_dilation = [False, True, True]
    aspp_dilate: List[int] = [12, 24, 36]

    inplanes = 2048
    low_level_planes = 256
    num_classes = 1

    resnext = resnet.__dict__['resnext50_32x4d'](pretrained=True,
                                                    replace_stride_with_dilation=replace_stride_with_dilation)

    swin_model = CustomSwinTransformerV2((448, 448), pretrained=True)

    combined_backbone = CombinedBackbone(resnext, swin_model)
    return_layers = {'layer4': 'out', 'layer1': 'low_level'}
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(combined_backbone, return_layers=return_layers)

    model = DeepLabV3Plus(backbone, classifier)
    return model


if __name__ == '__main__':
    model = CSegNet()
    print(model)
