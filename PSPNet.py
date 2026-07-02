import torch
from torch import nn
from torch.nn import functional as F
from efficientnet import efficientnet_b0
from ViG import pvig_ti_256_gelu


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.interpolate(input=x, size=(h, w), mode='bilinear', align_corners=True)
        return self.conv(p)





class PSPNet_efficientnet(nn.Module):
    def __init__(self, n_classes=10, sizes=(1, 2, 3, 6), psp_size=40):
        super(PSPNet_efficientnet, self).__init__()
        self.encoder = efficientnet_b0(num_classes=8)
        self.psp = PSPModule(psp_size, 512, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(512, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, n_classes)

        self.drop_2 = nn.Dropout2d(p=0.15)
        # self.final = nn.Conv2d(64, n_classes, kernel_size=1),


        # self.classifier = nn.Sequential(
        #     nn.Linear(deep_features_size, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, n_classes)
        # )

    def forward(self, x):
        x = self.encoder.extract_features(x, encoder_depth=2)
        p = self.psp(x)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        # p = self.drop_2(p)

        # auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

        return p, x



class PSPNet_vig(nn.Module):
    def __init__(self, n_classes=10, sizes=(1, 2, 3, 6), psp_size=96):
        super(PSPNet_vig, self).__init__()
        encoder = pvig_ti_256_gelu(8)
        self.layer0 = encoder.stem
        self.pos_embed = encoder.pos_embed
        self.layer1, self.layer2, self.layer3, self.layer4 = encoder.get_stages()
        self.psp = PSPModule(psp_size, 512, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(512, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, n_classes)

        self.drop_2 = nn.Dropout2d(p=0.15)
        # self.final = nn.Conv2d(64, n_classes, kernel_size=1),


        # self.classifier = nn.Sequential(
        #     nn.Linear(deep_features_size, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, n_classes)
        # )

    def forward(self, x):
        x = self.layer0(x) + self.pos_embed
        x = self.layer1(x)
        x = self.layer2(x)
        p = self.psp(x)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        # p = self.drop_2(p)

        # auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

        return p, x


class PSPNet_vig1(nn.Module):
    def __init__(self, n_classes=10, sizes=(1, 2, 3, 6), psp_size=96):
        super(PSPNet_vig1, self).__init__()
        encoder = pvig_ti_256_gelu(8)
        self.layer0 = encoder.stem
        self.pos_embed = encoder.pos_embed
        self.layer1, self.layer2, self.layer3, self.layer4 = encoder.get_stages()
        self.psp = PSPModule(psp_size, 512, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(512, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, n_classes)

        self.drop_2 = nn.Dropout2d(p=0.15)
        # self.final = nn.Conv2d(64, n_classes, kernel_size=1),


        # self.classifier = nn.Sequential(
        #     nn.Linear(deep_features_size, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, n_classes)
        # )

    def forward(self, x):
        x = self.layer0(x) + self.pos_embed
        x = self.layer1(x)
        x = self.layer2(x)
        p = self.psp(x)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        # p = self.drop_2(p)

        # auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

        return p


if __name__ == "__main__":
    model = PSPNet_vig(n_classes=10).cuda()
    model.eval()
    img = torch.rand(1, 12, 256, 256).cuda()
    with torch.no_grad():
        out = model(img)
    print(out[0].shape, out[1].shape)




