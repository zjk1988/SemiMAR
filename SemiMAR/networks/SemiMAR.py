import torch
import torch.nn as nn
import functools
from copy import deepcopy, copy
from .blocks import ConvolutionBlock, ResidualBlock, FullyConnectedBlock
from ..utils import print_model, FunctionModel


class Encoder(nn.Module):
    def __init__(self, input_ch, base_ch, num_down, num_residual, res_norm='instance', down_norm='instance'):
        super(Encoder, self).__init__()

        self.conv0 = ConvolutionBlock(
            in_channels=input_ch, out_channels=base_ch, kernel_size=7, stride=1,
            padding=3, pad='reflect', norm=down_norm, activ='relu')
        
        output_ch = base_ch
        for i in range(1, num_down+1):
            m = ConvolutionBlock(
                in_channels=output_ch, out_channels=output_ch * 2, kernel_size=4,
                stride=2, padding=1, pad='reflect', norm=down_norm, activ='relu')
            setattr(self, "conv{}".format(i), m)
            output_ch *= 2

        for i in range(num_residual):
            setattr(self, "res{}".format(i),
                ResidualBlock(output_ch, pad='reflect', norm=res_norm, activ='relu'))

        self.layers = [getattr(self, "conv{}".format(i)) for i in range(num_down+1)] + \
            [getattr(self, "res{}".format(i)) for i in range(num_residual)]
        
    def forward(self, x):
        sides = []
        for layer in self.layers:
            x = layer(x)
            sides.append(x)
        return x, sides[::-1]


class Decoder(nn.Module):
    def __init__(self, output_ch, base_ch, num_up, num_residual, num_sides, res_norm='instance', up_norm='layer', fuse=False):
        super(Decoder, self).__init__()
        input_ch = base_ch * 2 ** num_up
        input_chs = []

        for i in range(num_residual):
            setattr(self, "res{}".format(i),
                ResidualBlock(input_ch, pad='reflect', norm=res_norm, activ='lrelu'))
            input_chs.append(input_ch)

        for i in range(num_up):
            m = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                ConvolutionBlock(
                    in_channels=input_ch, out_channels=input_ch // 2, kernel_size=5,
                    stride=1, padding=2, pad='reflect', norm=up_norm, activ='lrelu'))
            setattr(self, "conv{}".format(i), m)
            input_chs.append(input_ch)
            input_ch //= 2

        m = ConvolutionBlock(
            in_channels=base_ch, out_channels=output_ch, kernel_size=7,
            stride=1, padding=3, pad='reflect', norm='none', activ='tanh')
        setattr(self, "conv{}".format(num_up), m)
        input_chs.append(base_ch)
        
        self.layers = [getattr(self, "res{}".format(i)) for i in range(num_residual)] + \
            [getattr(self, "conv{}".format(i)) for i in range(num_up + 1)]

        # If true, fuse (concat and conv) the side features with decoder features
        # Otherwise, directly add artifact feature with decoder features
        if fuse:
            input_chs = input_chs[-num_sides:]
            for i in range(num_sides):
                setattr(self, "fuse{}".format(i),
                    nn.Conv2d(input_chs[i] * 2, input_chs[i], 1))
            self.fuse = lambda x, y, i: getattr(self, "fuse{}".format(i))(torch.cat((x, y), 1))
        else:
            self.fuse = lambda x, y, i: x + y

    def forward(self, x, sides=[]):
        m, n = len(self.layers), len(sides)
        assert m >= n, "Invalid side inputs"

        for i in range(m - n):
            x = self.layers[i](x)

        for i, j in enumerate(range(m - n, m)):
            x = self.fuse(x, sides[i], i)
            x = self.layers[j](x)
        return x


class ADN(nn.Module):
    """
    Image with artifact is denoted as low quality image
    Image without artifact is denoted as high quality image
    """

    def __init__(self, input_ch=1, base_ch=64, num_down=3, num_residual=3, num_sides="all",
        res_norm='instance', down_norm='instance', up_norm='layer', fuse=True, shared_decoder=False):
        super(ADN, self).__init__()

        self.n = num_down + num_residual + 1 if num_sides == "all" else num_sides
        self.encoderA = Encoder(input_ch, base_ch, num_down, num_residual, res_norm, down_norm)
        self.decoderA = Decoder(input_ch, base_ch, num_down, num_residual, self.n, res_norm, up_norm, fuse)

    def forward1(self, x_low, x_high, gt):
        # _, sidesA = self.encoder_artA(A)  # encode artifact
        # codeA, _ = self.encoderA(A)  # encode low quality image
        # AA = self.decoderA(codeA, sidesA[-self.n:]) # decode image with artifact (low quality)
        
        # _, sidesB = self.encoder_artB(B)  # encode artifact
        # codeB, _ = self.encoderB(B)  # encode low quality image
        # BB = self.decoderB(codeB, sidesB[-self.n:]) # decode image with artifact (low quality)

        # AB = self.decoderB(codeA, sidesB[-self.n:])
        # BA = self.decoderA(codeB, sidesA[-self.n:])

        # _, sidesA = self.encoder_artA(BA)  # encode artifact
        # codeA, _ = self.encoderA(A)  # encode low quality image

        # _, sidesB = self.encoder_artB(AB)  # encode artifact
        # codeB, _ = self.encoderB(B)  # encode low quality image
        
        # BAB = self.decoderB(codeA, sidesB[-self.n:])
        # ABA = self.decoderA(codeB, sidesA[-self.n:])

        # # _, sidesA = self.encoder_artA(A)  # encode artifact
        # codeA, _ = self.encoderA(A)  # encode low quality image
        # AA = self.decoderA(codeA) # decode image with artifact (low quality)
        
        # # _, sidesB = self.encoder_artB(B)  # encode artifact
        # codeB, _ = self.encoderB(B)  # encode low quality image
        # BB = self.decoderB(codeB) # decode image with artifact (low quality)

        # AB = self.decoderB(codeA)
        # BA = self.decoderA(codeB)

        # # _, sidesA = self.encoder_artA(BA)  # encode artifact
        # codeA, _ = self.encoderA(A)  # encode low quality image

        # # _, sidesB = self.encoder_artB(AB)  # encode artifact
        # codeB, _ = self.encoderB(B)  # encode low quality image
        
        # BAB = self.decoderB(codeA)
        # ABA = self.decoderA(codeB)

        codeA, _negtive1 = self.encoderA(x_low)  # encode low quality image
        l_h = self.decoderA(codeA) # decode image with artifact (low quality)
        noise1 = x_low-l_h

        h_l = x_high+noise1

        codeA, _negtive = self.encoderA(x_high)  # encode low quality image
        h_h = self.decoderA(codeA) # decode image with artifact (low quality)

        codeA, _ = self.encoderA(h_l)  # encode low quality image
        h_l_h = self.decoderA(codeA) # decode image with artifact (low quality)
        noise2 = h_l-h_l_h

        l_h_l = l_h+noise2

        code,_anchor=self.encoderA(l_h)
        _, _positive = self.encoderA(gt)

        return l_h,h_l,h_h,l_h_l,h_l_h,noise1,noise2,_negtive,_negtive1,_positive,_anchor

    def forward2(self, x_low):
        codeA, _negtive = self.encoderA(x_low)  # encode low quality image
        l_h = self.decoderA(codeA) # 
        return l_h

    def forward_lh(self, x_low):
        code, _ = self.encoder_low(x_low)  # encode low quality image
        y = self.decoder(code)
        return y

    def forward_hl(self, x_low, x_high):
        _, sides = self.encoder_art(x_low)  # encode artifact
        code, _ = self.encoder_high(x_high) # encode high quality image
        y = self.decoder_art(code, sides[-self.n:])  # decode image with artifact (low quality)
        return y


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator
    
    This class is adopted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) is str:
            norm_layer = {
                "layer": nn.LayerNorm,
                "instance": nn.InstanceNorm2d,
                "batch": nn.BatchNorm2d,
              "none": None}[norm_layer]

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)] + \
                ([norm_layer(ndf * nf_mult)] if norm_layer else []) + [nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)] + \
            ([norm_layer(ndf * nf_mult)] if norm_layer else []) + [nn.LeakyReLU(0.2, True)]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class ContrastLoss(nn.Module):
    def __init__(self):

        super(ContrastLoss, self).__init__()
        self.l1 = nn.L1Loss().cuda()

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            d_an = self.l1(a_vgg[i], n_vgg[i].detach())
            contrastive = d_ap / (d_an + 1e-7)

            loss += contrastive
        return loss