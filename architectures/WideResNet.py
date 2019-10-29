import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, nFin, nFout, stride=1, dropout_rate=0):
        super(ResBlock, self).__init__()
        self.equalInOut = (nFin == nFout and stride == 1)

        self.conv_block = nn.Sequential(
            nn.Conv2d(nFin, nFout, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(nFout),
            nn.ReLU(inplace=True),
            nn.Conv2d(nFout, nFout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(nFout),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or nFin != nFout:
            self.shortcut = nn.Sequential(
                nn.Conv2d(nFin, nFout, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(nFout),
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.conv_block(x) + self.shortcut(identity))
        return out

class WideResNet(nn.Module):
    def __init__(self, opt):
        depth = opt['depth']
        widen_factor = opt['widen_factor']
        self.use_relu = opt['userelu']

        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = int((depth-4)/6)
        k = widen_factor


        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(nStages[0]),
            nn.ReLU(inplace=True)
        )


        self.layer1 = self._wide_layer(ResBlock, nStages[1], n, stride=2)
        self.layer2 = self._wide_layer(ResBlock, nStages[2], n, stride=2)
        self.layer3 = self._wide_layer(ResBlock, nStages[3], n, stride=2)

        # self.bn = nn.BatchNorm2d(nStages[3], momentum=0.9)
        # self.bn = nn.BatchNorm2d(nStages[3])

        # if 'SS' in opt:
        #     self.SS = True
        #     self.layer_SS = self._wide_layer(ResBlock, nStages[3], n, stride=2)
        #     # self.bn_SS = nn.BatchNorm2d(nStages[3], momentum=0.9)
        #     self.bn_SS = nn.BatchNorm2d(nStages[3])
        # else:
        #     self.SS = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _wide_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        # layers.append(nn.BatchNorm2d(planes))
        # layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        out = self.conv1(x)
        # import pdb; pdb.set_trace()
        out = self.layer1(out)
        # import pdb; pdb.set_trace()
        out = self.layer2(out)
        out = self.layer3(out)

        # import pdb; pdb.set_trace()
        out = F.avg_pool2d(out, 10)
        out = out.view(out.size(0), -1)
        # import pdb; pdb.set_trace()
        return out

def create_model(opt):
    return WideResNet(opt)
