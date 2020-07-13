import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(Conv2DBatchNormRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels= out_channels, kernel_size=kernel_size, stride= stride, padding=padding)

        self.batchnorm = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        outputs = self.relu()

        return outputs

class FeatureMap_convolution(nn.Module):
    def __init__(self):
        super(FeatureMap_convolution, self).__init__()

        #Block 1
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 3, 64, 3, 2, 1, 1, False
        self.cbnr_1 = Conv2DBatchNormRelu(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

        # block 2
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 64, 64, 3, 1, 1, 1, False
        self.cbnr_2 = Conv2DBatchNormRelu(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

        # block 3
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 64, 128, 3, 1, 1, 1, False
        self.cbnr_3 = Conv2DBatchNormRelu(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

        #block 4 : Maxpooling
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=1)

    def forward(self, x):
        x = self.cbnr_1(x)
        x = self.cbnr_2(x)
        x = self.cbnr_3(x)
        outputs = self.maxpool(x)

        return outputs

class ResidualBlockPSP(nn.Sequential):
    def __init__(self, n_block, in_channels, mid_channels, out_channels, stride, dilation):
        super(ResidualBlockPSP, self).__init__()

        #bottleNeckPSP
        self.add_module('block_1', bottleNeckPSP(in_channels, mid_channels, out_channels, stride, dilation))
        for i in range(n_block - 1):
            self.add_module('block' + str(i + 2), bottleNeckIdentifyPSP(out_channels, mid_channels, stride, dilation))

class Conv2DBacthNorm(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, stride, padding, dilation, bias):
        super(Conv2DBacthNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(in_channels=out_channels)

    def forward(self, x):
        x = self.conv(x)
        outputs = self.batchnorm(x)
        return outputs

class bottleNeckPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation):
        super(bottleNeckPSP, self).__init__()
        self.cbnr_1 = Conv2DBatchNormRelu(in_channels=in_channels, mid_channels=mid_channels,kernel_size=1,
                                        stride =stride,padding= dilation,dilation= 1, bias=False)
        self.cbnr_2 = Conv2DBatchNormRelu(in_channels=mid_channels, mid_channels=mid_channels,kernel_size=3,
                                        stride =stride, padding=dilation, dilation= 1, bias=False)
        self.cbn_1 = Conv2DBacthNorm(in_channels=mid_channels, out_channels=out_channels, kernel_size=1,
                                     stride = 1, padding=0, dilation=1, bias=False)

        #skip connection

        self.cbn_residual = Conv2DBacthNorm(in_channels, out_channels, kernel_size=1, stride=stride, padding = 0, dilation=1, bias = 0)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.cbn_1(self.cbnr_2(self.cbnr_1(x)))
        residual = self.cbn_residual(x)

        return self.relu(conv, residual)

class bottleNeckIdentifyPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, stride, dilation):
        super(bottleNeckIdentifyPSP, self).__init__()
        self.cbnr_1 = Conv2DBatchNormRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.cbnr_2 = Conv2DBatchNormRelu(mid_channels, mid_channels, kernel_size=3, stride=1, padding=dilation, dilation=1, bias=False)
        self.cbn_1 = Conv2DBacthNorm(mid_channels, in_channels, kernel_size=1, stride = 1, padding=0, dilation=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.cbn_1(self.cbnr_2(self.cbnr_1(x)))
        residual = x
        return self.relu(conv + residual)

class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes, height, width):
        super(PyramidPooling, self).__init__()

        self.height = height
        self.width = width
        out_channels = int(in_channels/len(pool_sizes))

        #pool_size = (6,3,2,1)

        self.avpool_1 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[0])
        self.cbnr_1 = Conv2DBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.avpool_2 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[1])
        self.cbnr_2 = Conv2DBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.avpool_3 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[2])
        self.cbnr_3 = Conv2DBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.avpool_4 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[3])
        self.cbnr_5 = Conv2DBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

    def forward(self, x):
        out1 = self.cbnr_1(self.avpool_1(x))
        out1 = F.interpolate(out1, size=(self.height, self.width), mode='bilinear', align_corners=True)

        out2 = self.cbnr_2(self.avpool_2(x))
        out2 = F.interpolate(out2, size=(self.height, self.width), mode='bilinear', align_corners=True)

        out3 = self.cbnr_3(self.avpool_3(x))
        out3 = F.interpolate(out3, size=(self.height, self.width), mode='bilinear', align_corners=True)

        out4 = self.cbnr_4(self.avpool_4(x))
        out4 = F.interpolate(out4, size=(self.height, self.width), mode='bilinear', align_corners=True)

        output = torch.cat([x, out1, out2, out3, out4], dim=1)
        return output

class DecoderPSPFeature(nn.Module):
    def __init__(self, height, width, n_classes):
        super(DecoderPSPFeature, self).__init__()
        self.height = height
        self.width = width
        self.cbnr = Conv2DBatchNormRelu(in_channels=4096, out_channels= 512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.dropout = nn.Dropout(p=0.1)
        self.classification = nn.Conv2d(in_channels=512, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cbnr(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = F.interpolate(x, size=(self.height, self.width), mode='bilinear', align_corners=True)

        return output

class AuxilirayPSPLayers(nn.Module):
    def __init__(self, height, width, n_classes):
        super(AuxilirayPSPLayers, self).__init__()
        self.cbnr = Conv2DBatchNormRelu(in_channels=1024, out_channels= 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.dropout = nn.Dropout(p=0.1)
        self.classification = nn.Conv2d(in_channels=256, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cbnr(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = F.interpolate(x, size=(self.height, self.width), mode='bilinear', align_corners=True)

        return output


if __name__ == "__main__" :
    x = torch.randn(1,3, 475, 475)
    feature_conv = FeatureMap_convolution()
    outputs = feature_conv(x)
    print(outputs.shape)
