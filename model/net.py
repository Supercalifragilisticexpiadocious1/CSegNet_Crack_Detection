from typing import List

import torch
from torch import nn
import timm

from resnet import Bottleneck

from utils import IntermediateLayerGetter
from deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
import resnet

attn_label = True


def _segm_resnext(name, backbone_name, num_classes, output_stride, pretrained_backbone):

    replace_stride_with_dilation = [False, True, True]
    aspp_dilate: List[int] = [12, 24, 36]
    
    print(backbone_name)

    inplanes = 2048
    low_level_planes = 256

    resnext = resnet.__dict__['resnext50_32x4d'](pretrained=pretrained_backbone,
                                                    replace_stride_with_dilation=replace_stride_with_dilation)
    swin_model = timm.create_model("swinv2_cr_tiny_ns_224", pretrained=True)
    combined_backbone = CombinedBackbone(resnext, swin_model)
    return_layers = {'layer4': 'out', 'layer1': 'low_level'}
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(combined_backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)

    return model