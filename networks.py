# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

################################################################################
def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun

################################################################################

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)

        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # print(self.enc_1)
        # print(self.enc_2)
        # print(self.enc_3)

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):


        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

################################################################################

class PartialConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(PartialConv, self).__init__()
        self.input_conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))

        nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        output = self.input_conv(input * mask)

        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask

class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu', conv_bias=False):
        super(PCBActiv, self).__init__()
        if sample == 'down-5':
            self.conv = PartialConv(in_ch, out_ch, kernel_size=5, stride=2, padding=2, bias=conv_bias)
        elif sample == 'down-7':
            self.conv = PartialConv(in_ch, out_ch, kernel_size=7, stride=2, padding=3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=conv_bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)

        if activ == 'relu':
            self.activ = nn.ReLU()
        elif activ == 'leaky':
            self.activ = nn.LeakyReLU(0.2)

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)

        if hasattr(self, 'bn'):
            h = self.bn(h)

        if hasattr(self, 'activation'):
            h = self.activ(h)

        return h, h_mask

class PConvUNet(nn.Module):
    def __init__(self, layer_size=7, in_ch=3, upsampling_mode='nearest'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = PCBActiv(in_ch, out_ch=64, bn=False, sample='down-7')
        self.enc_2 = PCBActiv(64, out_ch=128, sample='down-5')
        self.enc_3 = PCBActiv(128, out_ch=256, sample='down-5')
        self.enc_4 = PCBActiv(256, out_ch=512, sample='down-3')

        for i in range(4, self.layer_size):
            name = 'enc_{:d}'.format(i+1)
            setattr(self, name, PCBActiv(512, 512, sample='down-3'))

        for i in range(4, self.layer_size):
            name = 'dec_{:d}'.format(i+1)
            setattr(self, name, PCBActiv(512+512, 512, activ='leaky'))

        self.dec_4 = PCBActiv(512+256, 256, activ='leaky')
        self.dec_3 = PCBActiv(256+128, 128, activ='leaky')
        self.dec_2 = PCBActiv(128+ 64,  64, activ='leaky')
        self.dec_1 = PCBActiv(64+in_ch, in_ch, bn=False, activ=None, conv_bias=True)

    def forward(self, input, input_mask):
        h_dict = {}
        h_mask_dict = {}

        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask

        h_key_prev = 'h_0'

        for i in range(1, self.layer_size+1):
            l_key = 'enc_{:d}'.format(i)

            h_key = 'h_{:d}'.format(i)
            h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(h_dict[h_key_prev], h_mask_dict[h_key_prev])
            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]

        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i-1)
            dec_l_key = 'dec_{:d}'.format(i)

            h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode)
            h_mask = F.interpolate(h_mask, scale_factor=2, mode='nearest')

            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)

            h, h_mask = getattr(self, dec_l_key)(h, h_mask)

        return h, h_mask

################################################################################

# if __name__ == '__main__':
    vgg16 = VGG16FeatureExtractor()
    x = torch.FloatTensor( np.random.random((1, 3, 224, 224))) # (batch_size, channels, width, height)
    out = vgg16(x)

    # model = PConvUNet()
    # input = torch.FloatTensor( np.random.random((1, 3, 256, 256))) # (batch_size, channels, height, width)
    # mask = torch.FloatTensor( np.random.random((1, 3, 256, 256))) # (batch_size, channels, height, width)
    #
    # output, output_mask = model(input, mask)

    # print(output.size())
    # print(output_mask.size())

    # size = (1, 3, 5, 5)
    # input = torch.ones(size)
    # input_mask = torch.ones(size)
    # input_mask[:, :, 2:, :][:, :, :, 2:] = 0
    #
    # conv = PartialConv(3, 3, 3, 1, 1)
    # l1 = nn.L1Loss()
    # input.requires_grad = True
    #
    # output, output_mask = conv(input, input_mask)
    # loss = l1(output, torch.randn(1, 3, 5, 5))
    # loss.backward()
