import torch
import torch.nn as nn

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


if __name__ == "__main__" :
    x = torch.randn(1,3, 475, 475)
    feature_conv = FeatureMap_convolution()
    outputs = feature_conv(x)
    print(outputs.shape)