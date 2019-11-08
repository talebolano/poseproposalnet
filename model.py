import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class AddionalLayer(nn.Module):

    def __init__(self, ch):
        super(AddionalLayer, self).__init__()
        #initialW = initializers.HeNormal(scale=1.0)
        #with self.init_scope():
        self.conv1 = nn.Conv2d(
                2048, ch, 3, 1, padding=3 // 2,bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(
                ch, ch, 3, 1, padding=3 // 2)
        self.relu = nn.LeakyReLU(inplace=True,negative_slope=0.1)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x

class poseprosalnet(nn.Module):
    def __init__(self,keypoint_names,edges,local_grid_size,insize,pretrained=False):
        super(poseprosalnet,self).__init__()
        feature = resnet50(pretrained=pretrained)
        #self.insize = insize
        self.keypoint_names = keypoint_names
        self.edges = edges
        self.local_grid_size = local_grid_size
        self.insize = insize
        # 定义网络###############################################################################
        self.feature = nn.Sequential(
            feature.conv1,
            feature.bn1,
            feature.relu,
            feature.maxpool,

            feature.layer1,
            feature.layer2,
            feature.layer3,
            feature.layer4
        )
        self.addionallayer = AddionalLayer(ch=512)

        self.lastconv = nn.Conv2d(512,6 * len(self.keypoint_names) +
                                        self.local_grid_size[0] * self.local_grid_size[1] * len(self.edges),
                                        kernel_size=1, stride=1, padding=1 // 2)
        self.lastact = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        ##########################################################################
        self.outsize = self.get_outsize()
    def get_outsize(self):
        inp = torch.zeros(1,3,self.insize[0],self.insize[1])
        out = self.forward(inp)
        _,_,h,w = out[0].shape
        return h,w

    def forward(self, x):
        K = len(self.keypoint_names)
        B = x.shape[0]
        #outW, outH = self.outsize

        x = self.feature(x)
        x = self.addionallayer(x)
        x = self.lastconv(x)
        feature_map = self.lastact(x)
        _, _, outH, outW = feature_map.shape

        resp = feature_map[:, 0 * K:1 * K, :, :]
        conf = feature_map[:, 1 * K:2 * K, :, :]
        x = feature_map[:, 2 * K:3 * K, :, :]
        y = feature_map[:, 3 * K:4 * K, :, :]
        w = feature_map[:, 4 * K:5 * K, :, :]
        h = feature_map[:, 5 * K:6 * K, :, :]

        e = feature_map[:, 6 * K:, :, :].reshape((
            B,
            len(self.edges),
            self.local_grid_size[1], self.local_grid_size[0],
            outH, outW
        ))


        return resp, conf, x, y, w, h, e


if __name__=="__main__":
    net = poseprosalnet()
    rand = torch.randn(1,3,224,224)
    print(net(rand).shape)


