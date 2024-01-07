import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class NeuralNetwork(nn.Module):
    def __init__(self, numberClass):
        _resnet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(_resnet50.children())[:-1])
        self.flattten = nn.Flatten(2)
        self.final_layer = nn.Linear(2048, numberClass)

    def forward(self, x):
        return self.final_layer(torch.squeeze(self.flattten(self.backbone(x)), dim=-1))


class UNETNetwork(nn.Module):
    def __init__(self, numberClass):
        super().__init__()
        _resnet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = create_feature_extractor(
            _resnet50,
            {
                "relu": "feat1",
                "layer1": "feat2",
                "layer2": "feat3",
                "layer3": "feat4",
                "layer4": "feat5",
            },
        )

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # self.upsampling_2x = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsampling_2x_bilinear = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv5 = nn.Conv2d(
            in_channels=2048, out_channels=1024, kernel_size=3, padding=1
        )
        self.conv6 = nn.Conv2d(
            in_channels=1024, out_channels=512, kernel_size=3, padding=1
        )
        self.conv7 = nn.Conv2d(
            in_channels=512, out_channels=256, kernel_size=3, padding=1
        )
        self.conv8 = nn.Conv2d(
            in_channels=256, out_channels=128, kernel_size=3, padding=1
        )
        self.convfinal = nn.Conv2d(
            in_channels=128, out_channels=numberClass, kernel_size=1
        )

    def forward(self, x):
        backbone_output = self.backbone(x)
        feat2, feat3, feat4, feat5 = (
            backbone_output["feat2"],
            backbone_output["feat3"],
            backbone_output["feat4"],
            backbone_output["feat5"],
        )
        feat4to6 = self.upsampling_2x_bilinear(self.conv5(feat5).relu())
        feat3to7 = self.upsampling_2x_bilinear(self.conv6(feat4 + feat4to6).relu())
        feat2to8 = self.upsampling_2x_bilinear(self.conv7(feat3 + feat3to7).relu())
        featout = self.upsampling_2x_bilinear(self.conv8(feat2 + feat2to8).relu())
        return self.upsampling_2x_bilinear(self.convfinal(featout))


# class UNETNetwork(nn.Module):
#     def __init__(self, numberClass):
#         super().__init__()
#         _resnet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
#         self.backbone = create_feature_extractor(
#             _resnet50,
#             {
#                 "relu": "feat1",
#                 "layer1": "feat2",
#                 "layer2": "feat3",
#                 "layer3": "feat4",
#                 "layer4": "feat5",
#             },
#         )
#         # self.upsampling_2x = nn.UpsamplingNearest2d(scale_factor=2)
#         self.upsampling_2x_bilinear = nn.UpsamplingBilinear2d(scale_factor=2)
#         self.conv5 = nn.Conv2d(
#             in_channels=2048, out_channels=1024, kernel_size=3, padding=1
#         )
#         self.conv6 = nn.Conv2d(
#             in_channels=1024, out_channels=512, kernel_size=3, padding=1
#         )
#         self.conv7 = nn.Conv2d(
#             in_channels=512, out_channels=256, kernel_size=3, padding=1
#         )
#         self.conv8 = nn.Conv2d(
#             in_channels=256, out_channels=128, kernel_size=3, padding=1
#         )
#         self.convfinal = nn.Conv2d(
#             in_channels=128, out_channels=numberClass, kernel_size=1
#         )

#     def forward(self, x):
#         backbone_output = self.backbone(x)
#         feat2, feat3, feat4, feat5 = (
#             backbone_output["feat2"],
#             backbone_output["feat3"],
#             backbone_output["feat4"],
#             backbone_output["feat5"],
#         )
#         feat4to6 = self.upsampling_2x_bilinear(self.conv5(feat5).relu())
#         feat3to7 = self.upsampling_2x_bilinear(self.conv6(feat4 + feat4to6).relu())
#         feat2to8 = self.upsampling_2x_bilinear(self.conv7(feat3 + feat3to7).relu())
#         featout = self.upsampling_2x_bilinear(self.conv8(feat2 + feat2to8).relu())
#         return self.upsampling_2x_bilinear(self.convfinal(featout))
