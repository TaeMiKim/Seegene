import threading
import sys
import math
import os
from urllib.request import urlretrieve
import torch
import torch.nn as nn
from torch.nn import functional as F

sys.path.append("..")

from deepmil.decision_pooling import WildCatPoolDecision, ClassWisePooling
thread_lock = threading.Lock()  # lock for threads to protect the instruction that cause randomness and make them
# thread-safe.

import reproducibility
import constants

from deepmil.SearchAttention import SA

BatchNorm2d = nn.BatchNorm2d

# DEFAULT SEGMENTATION PARAMETERS ###########################
INNER_FEATURES = 256 
OUT_FEATURES = 512 
# ###########################################################

ALIGN_CORNERS = True

__all__ = ['resnet18', 'resnet50', 'resnet101']


model_urls = {
    'resnet18': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet18-imagenet.pth',
    'resnet50': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet50-imagenet.pth',
    'resnet101': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth'
}


def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding.
    :param in_planes:
    :param out_planes:
    :param stride:
    :return:
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class WildCatClassifierHead(nn.Module):
    """
    A WILDCAT type classifier head.
    """
    def __init__(self, inplans, modalities, num_classes, kmax=0.5, kmin=None, alpha=0.6, dropout=0.0):
        super(WildCatClassifierHead, self).__init__()

        self.to_modalities = nn.Conv2d(inplans, num_classes * modalities, kernel_size=1, bias=True)
        self.to_maps = ClassWisePooling(num_classes, modalities)
        self.wildcat = WildCatPoolDecision(kmax=kmax, kmin=kmin, alpha=alpha, dropout=dropout)

    def forward(self, x, seed=None, prngs_cuda=None):

        modalities = self.to_modalities(x)
        maps = self.to_maps(modalities)
        scores = self.wildcat(x=maps, seed=seed, prngs_cuda=prngs_cuda)

        return scores, maps


class MaskHead(nn.Module):
    """
    Class that pulls the mask from feature maps.
    """
    def __init__(self, inplans, modalities, nbr_masks):
        """

        :param inplans: int. number of input features.
        :param modalities: int. number of modalities.
        :param nbr_masks: int. number of masks to pull.
        """
        super(MaskHead, self).__init__()

        self.to_modalities = nn.Conv2d(inplans,
                                       nbr_masks * modalities,
                                       kernel_size=1,
                                       bias=True
                                       )
        self.to_masks = ClassWisePooling(nbr_masks, modalities)

    def forward(self, x):
        """
        The forward function.
        :param x: input tensor fetaure maps.
        :return:
        """
        modalities = self.to_modalities(x)
        masks = self.to_masks(modalities)

        return masks


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class RF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RF, self).__init__()
        self.relu = nn.ReLU()

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1, dilation=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )

        self.conv_cat = BasicConv2d(5*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3, x4), dim=1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation model, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x                                                                                  


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 dataset_name=constants.GLAS,
                 sigma=0.5,
                 w=8,
                 num_classes=2,
                 scale=(0.5, 0.5),
                 modalities=4,
                 kmax=0.5,
                 kmin=None,
                 alpha=0.6,
                 dropout=0.0,
                 set_side_cl=False
                 ):
        
        self.dataset_name = dataset_name
        self.set_side_cl = set_side_cl
        # classifier stuff
        cnd = isinstance(scale, tuple) or isinstance(scale, list)
        cnd = cnd or isinstance(scale, float)
        msg = "`scale` should be a tuple, or a list, or a float with " \
              "values in ]0, 1]. You provided {} .... [NOT " \
              "OK]".format(scale)
        assert cnd, msg

        if isinstance(scale, tuple) or isinstance(scale, list):
            msg = "`scale[0]` (height) should be > 0 and <= 1. " \
                  "You provided `{}` ... [NOT OK]".format(scale[0])
            assert 0 < scale[0] <= 1, msg
            msg = "`scale[1]` (width) should be > 0 and <= 1. " \
                  "You provided `{}` .... [NOT OK]".format(scale[1])
            assert 0 < scale[0] <= 1, msg
        elif isinstance(scale, float):
            msg = "`scale` should be > 0, <= 1. You provided `{}` .... " \
                  "[NOT OK]".format(scale)
            assert 0 < scale <= 1, msg
            scale = (scale, scale)

        self.scale = scale
        self.num_classes = num_classes

        self.inplanes = 128
        super(ResNet, self).__init__()

        # Encoder

        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.rfb8_1 = RF(512, 32)
        self.rfb16_1 = RF(1024, 32)
        self.rfb32_1 = RF(2048, 32)

        self.agg1 = aggregation(32)

        self.SA = SA()

        # Find out the size of the output.
        if isinstance(self.layer4[-1], Bottleneck):
            in_channel4 = self.layer1[-1].bn3.weight.size()[0]
            in_channel8 = self.layer2[-1].bn3.weight.size()[0]
            in_channel16 = self.layer3[-1].bn3.weight.size()[0]
            in_channel32 = self.layer4[-1].bn3.weight.size()[0]
        elif isinstance(self.layer4[-1], BasicBlock):
            in_channel4 = self.layer1[-1].bn2.weight.size()[0]
            in_channel8 = self.layer2[-1].bn2.weight.size()[0]
            in_channel16 = self.layer3[-1].bn2.weight.size()[0]
            in_channel32 = self.layer4[-1].bn2.weight.size()[0]
        else:
            raise ValueError("Supported class .... [NOT OK]")

        print(in_channel32, in_channel16, in_channel8, in_channel4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # =================  SEGMENTOR =========================================
        self.sigma = sigma

        self.const2 = torch.tensor([w], requires_grad=False).float()
        self.register_buffer("w", self.const2)

        assert not self.set_side_cl

        self.pull_mask = None

        # =================================================================
        # ================================ CLASSIFIER =====================
        self.mask_head = WildCatClassifierHead(512,
                                        modalities,
                                        num_classes=num_classes,
                                        kmax=kmax,
                                        kmin=kmin,
                                        alpha=alpha,
                                        dropout=dropout
                                        )

        self.side_cl = None
        
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, glabels, code=None, mask_c=None, seed=None, prngs_cuda=None):
        
        mask, cl_scores_seg, cams = self.segment(x=x, glabels=glabels, seed=seed, prngs_cuda=prngs_cuda)

        mask, x_pos, x_neg = self.get_mask_xpos_xneg(x, mask)
        _, scores_pos, _ = self.segment(x=x_pos, glabels=glabels, seed=seed, prngs_cuda=prngs_cuda)
        _, scores_neg, _ = self.segment(x=x_neg, glabels=glabels, seed=seed, prngs_cuda=prngs_cuda)
        
        return scores_pos, scores_neg, mask, cl_scores_seg, cams

    def get_mask_xpos_xneg(self, x, mask_c):
       
        mask = self.get_pseudo_binary_mask(mask_c)
        x_pos, x_neg = self.apply_mask(x, mask)

        return mask, x_pos, x_neg

    def segment(self, x, glabels, seed=None, prngs_cuda=None):  
        b, _, h, w = x.shape  # [B, 3, h, w]

        x_0 = self.relu1(self.bn1(self.conv1(x)))    # [B, 64, h/2, w/2]
        x_1 = self.relu2(self.bn2(self.conv2(x_0)))  # [B, 64, h/2, w/2]
        x_2 = self.relu3(self.bn3(self.conv3(x_1)))  # [B, 128, h/2, w/2]
        x_3 = self.maxpool(x_2)   # [B, 128, h/4, w/4]
        x_4 = self.layer1(x_3)    # [B, 256, h/4, w/4]
        x_8 = self.layer2(x_4)    # [B, 512, h/8, w/8]
        x_16 = self.layer3(x_8)   # [B, 1024, h/16, w/16]
        x_32 = self.layer4(x_16)  # [B, 2048, h/32, w/32]

        x_8_1 = self.rfb8_1(x_8)    # [B, 32, h/8, w/8]
        x_16_1 = self.rfb16_1(x_16) # [B, 32, h/16, w/16]
        x_32_1 = self.rfb32_1(x_32) # [B. 32, h/32, w/32]

        attention_map = self.agg1(x_32_1, x_16_1, x_8_1)  

        x_8_2 = self.SA(attention_map.sigmoid(), x_8) 
            
        scores, maps = self.mask_head(x=x_8_2, seed=seed, prngs_cuda=prngs_cuda)

        # compute M+
        prob = F.softmax(scores, dim=1)
        mpositive = torch.zeros((b, 1, maps.size()[2], maps.size()[3]),
                                dtype=maps.dtype,
                                layout=maps.layout,
                                device=maps.device
                                )

        for i in range(b):  # for each sample
            for j in range(prob.size()[1]):  # sum the: prob(class) * mask(class)
                mpositive[i] = mpositive[i] + prob[i, j] * maps[i, j, :, :]

        mpos_inter = F.interpolate(input=mpositive,
                                    size=(h, w),
                                    mode='bilinear',
                                    align_corners=ALIGN_CORNERS
                                    )

        cams = F.interpolate(input=maps,
                                size=(h, w),
                                mode='bilinear',
                                align_corners=ALIGN_CORNERS
                                ) 

        return mpos_inter, scores, cams


    def get_pseudo_binary_mask(self, x):
        x = (x - x.min()) / (x.max() - x.min())
        
        return torch.sigmoid(self.w * (x - self.sigma))


    def apply_mask(self, x, mask):
        mask_expd = mask.expand_as(x)
        x_pos = x * mask_expd
        x_neg = x * (1 - mask_expd)

        return x_pos, x_neg


def load_url(url, model_dir='../pretrained', map_location=torch.device('cpu')):
    """
    Download pre-trained models.
    :param url: str, url of the pre-trained model.
    :param model_dir: str, path to the temporary folder where the pre-trained models will be saved.
    :param map_location: a function, torch.device, string, or dict specifying how to remap storage locations.
    :return: torch.load() output. Loaded dict state.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet18']), strict=False)
        if model.side_cl is not None:
            model.side_cl.load_state_dict(load_url(model_urls['resnet18']),
                                          strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet50']), strict=False)
        if model.side_cl is not None:
            model.side_cl.load_state_dict(load_url(model_urls['resnet50']),
                                          strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet101']), strict=False)
        if model.side_cl is not None:
            model.side_cl.load_state_dict(load_url(model_urls['resnet101']),
                                          strict=False)
    return model


def test_resnet():

    c = 1
    model = resnet18(
        pretrained=True, dataset_name=constants.CAMELYON16P512, num_classes=c)
    print("Testing {}".format(model.__class__.__name__))
    model.train()
    print("Num. parameters: {}".format(sum([p.numel() for p in model.parameters()])))
    cuda = "0"
    print("cuda:{}".format(cuda))
    print("DEVICE BEFORE: ", torch.cuda.current_device())
    DEVICE = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        pass

    print("DEVICE AFTER: ", torch.cuda.current_device())
    model.to(DEVICE)

    b = 20
    glabels = torch.randint(low=0, high=c+1, size=(b,), device=DEVICE,
                            dtype=torch.long)
    x = torch.randn(b, 3, 480, 480)
    x = x.to(DEVICE)
    scores_pos, scores_neg, mask, cl_scores_seg, cams = model(x, glabels=glabels)
    print(x.size(), mask.size(), cams.size(), cl_scores_seg.shape)
    ce = nn.CrossEntropyLoss(reduction="mean")
    loss = ce(cl_scores_seg, glabels)
    loss.backward()
    print(list(model.side_cl.parameters())[0].requires_grad,
          list(model.side_cl.parameters())[0].grad,
          list(model.mask_head.parameters())[0].requires_grad,
          list(model.mask_head.parameters())[0].grad
          )


if __name__ == "__main__":
    import sys

    test_resnet()
