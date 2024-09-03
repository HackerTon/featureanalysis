import torch
import math
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights, resnet34, ResNet34_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms.functional import resize
from enum import Enum


class BackboneType(Enum):
    RESNET34 = 1
    RESNET50 = 2


class UNETNetwork(nn.Module):
    def __init__(self, numberClass):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = create_feature_extractor(
            backbone,
            {
                "layer1": "feat2",
                "layer2": "feat3",
                "layer3": "feat4",
                "layer4": "feat5",
            },
        )

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.upsampling_2x_bilinear = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv5 = nn.Conv2d(
            in_channels=2048,
            out_channels=1024,
            kernel_size=3,
            padding=1,
        )
        self.conv6 = nn.Conv2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=3,
            padding=1,
        )
        self.conv7 = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=3,
            padding=1,
        )
        self.conv8 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.convfinal = nn.Conv2d(
            in_channels=128,
            out_channels=numberClass,
            kernel_size=1,
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


class FPNNetwork_new(nn.Module):
    def __init__(self, numberClass):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = create_feature_extractor(
            backbone,
            {
                "layer1": "feat2",
                "layer2": "feat3",
                "layer3": "feat4",
                "layer4": "feat5",
            },
        )

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.upsampling_2x_bilinear = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsampling_4x_bilinear = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsampling_8x_bilinear = nn.UpsamplingBilinear2d(scale_factor=8)

        self.conv5 = nn.Conv2d(
            in_channels=2048,
            out_channels=1024,
            kernel_size=3,
            padding=1,
        )
        self.conv6 = nn.Conv2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=3,
            padding=1,
        )
        self.conv7 = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=3,
            padding=1,
        )
        self.conv8 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.convfinal = nn.Conv2d(
            in_channels=128,
            out_channels=numberClass,
            kernel_size=1,
        )

        self.conv5_3x3_1 = nn.Conv2d(
            in_channels=1024,
            out_channels=128,
            kernel_size=3,
            padding="same",
        )
        self.conv4_3x3_1 = nn.Conv2d(
            in_channels=512,
            out_channels=128,
            kernel_size=3,
            padding="same",
        )
        self.conv3_3x3_1 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            padding="same",
        )
        self.conv2_3x3_1 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding="same",
        )
        self.final_conv = nn.Conv2d(
            in_channels=512,
            out_channels=numberClass,
            kernel_size=1,
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

        conv5_prediction = self.conv5_3x3_1(feat4to6).relu()
        conv4_prediction = self.conv4_3x3_1(feat3to7).relu()
        conv3_prediction = self.conv3_3x3_1(feat2to8).relu()
        conv2_prediction = self.conv2_3x3_1(featout)

        final_prediction_5 = self.upsampling_8x_bilinear(conv5_prediction)
        final_prediction_4 = self.upsampling_4x_bilinear(conv4_prediction)
        final_prediction_3 = self.upsampling_2x_bilinear(conv3_prediction)
        final_prediction_2 = conv2_prediction

        concatenated_prediction = torch.concatenate(
            [
                final_prediction_5,
                final_prediction_4,
                final_prediction_3,
                final_prediction_2,
            ],
            dim=1,
        )

        return self.upsampling_2x_bilinear(self.final_conv(concatenated_prediction))


class FPNNetwork(nn.Module):
    def __init__(self, numberClass):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = create_feature_extractor(
            backbone,
            {
                "layer1": "feat2",
                "layer2": "feat3",
                "layer3": "feat4",
                "layer4": "feat5",
            },
        )

        with torch.no_grad():
            outputs_prediction = self.backbone(torch.rand([1, 3, 256, 256])).values()
            backbone_dimensions = [output.size(1) for output in outputs_prediction]

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.upsampling_2x_bilinear = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsampling_4x_bilinear = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsampling_8x_bilinear = nn.UpsamplingBilinear2d(scale_factor=8)
        self.conv5_1x1 = nn.Conv2d(
            in_channels=backbone_dimensions[-1],
            out_channels=256,
            kernel_size=1,
        )
        self.conv5_3x3_1 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.conv5_3x3_2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.conv4_1x1 = nn.Conv2d(
            in_channels=backbone_dimensions[-2],
            out_channels=256,
            kernel_size=1,
        )
        self.conv4_3x3_1 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.conv4_3x3_2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.conv3_1x1 = nn.Conv2d(
            in_channels=backbone_dimensions[-3],
            out_channels=256,
            kernel_size=1,
        )
        self.conv3_3x3_1 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.conv3_3x3_2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.conv2_1x1 = nn.Conv2d(
            in_channels=backbone_dimensions[-4],
            out_channels=256,
            kernel_size=1,
        )
        self.conv2_3x3_1 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.conv2_3x3_2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.final_conv = nn.Conv2d(
            in_channels=512,
            out_channels=numberClass,
            kernel_size=1,
        )

    def forward(self, x):
        backbone_output = self.backbone(x)
        feat2, feat3, feat4, feat5 = (
            backbone_output["feat2"],
            backbone_output["feat3"],
            backbone_output["feat4"],
            backbone_output["feat5"],
        )

        conv5_mid = self.conv5_1x1(feat5).relu()
        conv5_prediction = self.conv5_3x3_1(conv5_mid).relu()
        conv5_prediction = self.conv5_3x3_2(conv5_prediction).relu()

        conv4_lateral = self.conv4_1x1(feat4).relu()
        conv4_mid = conv4_lateral + self.upsampling_2x_bilinear(conv5_mid)
        conv4_prediction = self.conv4_3x3_1(conv4_mid).relu()
        conv4_prediction = self.conv4_3x3_2(conv4_prediction).relu()

        conv3_lateral = self.conv3_1x1(feat3).relu()
        conv3_mid = conv3_lateral + self.upsampling_2x_bilinear(conv4_mid)
        conv3_prediction = self.conv3_3x3_1(conv3_mid).relu()
        conv3_prediction = self.conv3_3x3_2(conv3_prediction).relu()

        conv2_lateral = self.conv2_1x1(feat2).relu()
        conv2_mid = conv2_lateral + self.upsampling_2x_bilinear(conv3_mid)
        conv2_prediction = self.conv2_3x3_1(conv2_mid).relu()
        conv2_prediction = self.conv2_3x3_2(conv2_prediction).relu()

        final_prediction_5 = self.upsampling_8x_bilinear(conv5_prediction)
        final_prediction_4 = self.upsampling_4x_bilinear(conv4_prediction)
        final_prediction_3 = self.upsampling_2x_bilinear(conv3_prediction)
        final_prediction_2 = conv2_prediction

        concatenated_prediction = torch.concatenate(
            [
                final_prediction_5,
                final_prediction_4,
                final_prediction_3,
                final_prediction_2,
            ],
            dim=1,
        )

        concatenated_prediction = self.final_conv(concatenated_prediction)
        return self.upsampling_4x_bilinear(concatenated_prediction)


class MultiNet(nn.Module):
    def __init__(self, numberClass, backboneType: BackboneType):
        super().__init__()

        if backboneType == BackboneType.RESNET34:
            backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            self.backbone = create_feature_extractor(
                backbone,
                {
                    "layer1": "feat2",
                    "layer2": "feat3",
                    "layer3": "feat4",
                    "layer4": "feat5",
                },
            )
        elif backboneType == BackboneType.RESNET50:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.backbone = create_feature_extractor(
                backbone,
                {
                    "layer1": "feat2",
                    "layer2": "feat3",
                    "layer3": "feat4",
                    "layer4": "feat5",
                },
            )
        else:
            raise Exception(f"No {backboneType}")

        with torch.no_grad():
            outputs_prediction = self.backbone(torch.rand([1, 3, 256, 256])).values()
            backbone_dimensions = [output.size(1) for output in outputs_prediction]

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.upsampling_2x_bilinear = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsampling_4x_bilinear = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsampling_8x_bilinear = nn.UpsamplingBilinear2d(scale_factor=8)
        self.conv5_1x1 = nn.Conv2d(
            in_channels=backbone_dimensions[-1],
            out_channels=512,
            kernel_size=1,
        )
        self.conv5_3x3_1 = nn.Conv2d(
            in_channels=512,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.conv5_3x3_2 = nn.Conv2d(
            in_channels=128,
            out_channels=numberClass,
            kernel_size=3,
            padding=1,
        )
        self.conv4_1x1 = nn.Conv2d(
            in_channels=backbone_dimensions[-2],
            out_channels=512,
            kernel_size=1,
        )
        self.conv4_3x3_1 = nn.Conv2d(
            in_channels=512,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.conv4_3x3_2 = nn.Conv2d(
            in_channels=128,
            out_channels=numberClass,
            kernel_size=3,
            padding=1,
        )
        self.conv3_1x1 = nn.Conv2d(
            in_channels=backbone_dimensions[-3],
            out_channels=512,
            kernel_size=1,
        )
        self.conv3_3x3_1 = nn.Conv2d(
            in_channels=512,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.conv3_3x3_2 = nn.Conv2d(
            in_channels=128,
            out_channels=numberClass,
            kernel_size=3,
            padding=1,
        )
        self.conv2_1x1 = nn.Conv2d(
            in_channels=backbone_dimensions[-4],
            out_channels=512,
            kernel_size=1,
        )
        self.conv2_3x3_1 = nn.Conv2d(
            in_channels=512,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.conv2_3x3_2 = nn.Conv2d(
            in_channels=128,
            out_channels=numberClass,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        backbone_output = self.backbone(x)
        feat2, feat3, feat4, feat5 = (
            backbone_output["feat2"],
            backbone_output["feat3"],
            backbone_output["feat4"],
            backbone_output["feat5"],
        )

        conv5_mid = self.conv5_1x1(feat5).relu()
        conv5_prediction = self.conv5_3x3_1(conv5_mid).relu()
        conv5_prediction = self.conv5_3x3_2(conv5_prediction)

        conv4_lateral = self.conv4_1x1(feat4).relu()
        conv4_mid = conv4_lateral + self.upsampling_2x_bilinear(conv5_mid)
        conv4_prediction = self.conv4_3x3_1(conv4_mid).relu()
        conv4_prediction = self.conv4_3x3_2(conv4_prediction)

        conv3_lateral = self.conv3_1x1(feat3).relu()
        conv3_mid = conv3_lateral + self.upsampling_2x_bilinear(conv4_mid)
        conv3_prediction = self.conv3_3x3_1(conv3_mid).relu()
        conv3_prediction = self.conv3_3x3_2(conv3_prediction)

        conv2_lateral = self.conv2_1x1(feat2).relu()
        conv2_mid = conv2_lateral + self.upsampling_2x_bilinear(conv3_mid)
        conv2_prediction = self.conv2_3x3_1(conv2_mid).relu()
        conv2_prediction = self.conv2_3x3_2(conv2_prediction)

        final_prediction_5 = self.upsampling_8x_bilinear(conv5_prediction)
        final_prediction_4 = self.upsampling_4x_bilinear(conv4_prediction)
        final_prediction_3 = self.upsampling_2x_bilinear(conv3_prediction)
        final_prediction_2 = conv2_prediction

        return self.upsampling_4x_bilinear(
            final_prediction_5
            + final_prediction_4
            + final_prediction_3
            + final_prediction_2
        )


class MultiNetV2(nn.Module):
    def __init__(self, numberClass, backboneType: BackboneType):
        super().__init__()
        if backboneType == BackboneType.RESNET34:
            backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            self.backbone = create_feature_extractor(
                backbone,
                {
                    "layer1": "feat2",
                    "layer2": "feat3",
                    "layer3": "feat4",
                    "layer4": "feat5",
                },
            )
        elif backboneType == BackboneType.RESNET50:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.backbone = create_feature_extractor(
                backbone,
                {
                    "layer1": "feat2",
                    "layer2": "feat3",
                    "layer3": "feat4",
                    "layer4": "feat5",
                },
            )
        else:
            raise Exception(f"No {backboneType}")

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.upsampling_2x_bilinear = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsampling_4x_bilinear = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsampling_8x_bilinear = nn.UpsamplingBilinear2d(scale_factor=8)
        self.conv5 = nn.Conv2d(
            in_channels=2048,
            out_channels=1024,
            kernel_size=3,
            padding=1,
        )
        self.conv5_m = nn.Conv2d(
            in_channels=1024,
            out_channels=numberClass,
            kernel_size=3,
            padding=1,
        )
        self.conv6 = nn.Conv2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=3,
            padding=1,
        )
        self.conv6_m = nn.Conv2d(
            in_channels=512,
            out_channels=numberClass,
            kernel_size=3,
            padding=1,
        )
        self.conv7 = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=3,
            padding=1,
        )
        self.conv7_m = nn.Conv2d(
            in_channels=256,
            out_channels=numberClass,
            kernel_size=3,
            padding=1,
        )
        self.conv8 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.conv8_m = nn.Conv2d(
            in_channels=128,
            out_channels=numberClass,
            kernel_size=1,
            padding="same",
        )
        self.convfinal = nn.Conv2d(
            in_channels=128,
            out_channels=numberClass,
            kernel_size=1,
        )

    def forward(self, x):
        backbone_output = self.backbone(x)
        feat2, feat3, feat4, feat5 = (
            backbone_output["feat2"],
            backbone_output["feat3"],
            backbone_output["feat4"],
            backbone_output["feat5"],
        )

        featoutput1 = self.upsampling_2x_bilinear(self.conv5(feat5).relu())
        featoutput2 = self.upsampling_2x_bilinear(
            self.conv6(feat4 + featoutput1).relu()
        )
        featoutput3 = self.upsampling_2x_bilinear(
            self.conv7(feat3 + featoutput2).relu()
        )
        featoutput4 = self.upsampling_2x_bilinear(
            self.conv8(feat2 + featoutput3).relu()
        )

        # featoutput1 = self.upsampling_8x_bilinear(self.conv5_m(featoutput1))
        # featoutput2 = self.upsampling_4x_bilinear(self.conv6_m(featoutput2))
        # featoutput3 = self.upsampling_2x_bilinear(self.conv7_m(featoutput3))
        featoutput4 = self.conv8_m(featoutput4)

        # sum_output = featoutput1 + featoutput2 + featoutput3 + featoutput4
        return featoutput4


class PositionalEncoding(nn.Module):
    def __init__(self, max_length: int, embedding_length: int):
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_length, 2)
            * (-math.log(10000.0) / embedding_length)
        )
        pe = torch.zeros(max_length, 1, embedding_length)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(0)]


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        h_w_size: int,
        channel_width: int,
        num_of_heads: int = 1,
    ):
        super().__init__()
        self.h_w_size = h_w_size
        self.channel_width = channel_width
        self.position_encoding = PositionalEncoding(h_w_size, channel_width)
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=channel_width,
            num_heads=num_of_heads,
        )
        self.Q = nn.Linear(channel_width, channel_width)
        self.K = nn.Linear(channel_width, channel_width)
        self.V = nn.Linear(channel_width, channel_width)

    def forward(self, x: torch.Tensor, need_weights: bool = True) -> torch.Tensor:
        """
        x, [B, T, E]
        """

        x = self.position_encoding(x)
        weighted_K = self.K(x)
        weighted_Q = self.Q(x)
        weighted_V = self.V(x)
        attended_matrix, weights = self.attention(
            weighted_Q,
            weighted_K,
            weighted_V,
            need_weights=need_weights,
        )

        return attended_matrix


class MultiNetWithAttention(nn.Module):
    def __init__(self, numberClass, backboneType: BackboneType):
        super().__init__()

        if backboneType == BackboneType.RESNET34:
            backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            self.backbone = create_feature_extractor(
                backbone,
                {
                    "layer1": "feat2",
                    "layer2": "feat3",
                    "layer3": "feat4",
                    "layer4": "feat5",
                },
            )
        elif backboneType == BackboneType.RESNET50:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.backbone = create_feature_extractor(
                backbone,
                {
                    "layer1": "feat2",
                    "layer2": "feat3",
                    "layer3": "feat4",
                    "layer4": "feat5",
                },
            )
        else:
            raise Exception(f"No {backboneType}")

        with torch.no_grad():
            outputs_prediction = self.backbone(torch.rand([1, 3, 256, 256])).values()
            backbone_dimensions = [output.size(1) for output in outputs_prediction]

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.attention = SelfAttentionBlock(
            128 * 128,
            256,
            num_of_heads=4,
        )
        self.conv1 = nn.Conv2d(
            in_channels=backbone_dimensions[1],
            out_channels=256,
            kernel_size=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=backbone_dimensions[2],
            out_channels=256,
            kernel_size=1,
        )
        self.conv3 = nn.Conv2d(
            in_channels=backbone_dimensions[3],
            out_channels=256,
            kernel_size=1,
        )

        self.classifier_conv = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            padding="same",
        )

        self.final_classifier_conv = nn.Conv2d(
            in_channels=256,
            out_channels=numberClass,
            kernel_size=3,
            padding="same",
        )

        self.conv5 = nn.Conv2d(
            in_channels=backbone_dimensions[-1],
            out_channels=256,
            kernel_size=3,
            padding=1,
        )
        self.conv4 = nn.Conv2d(
            in_channels=backbone_dimensions[-2],
            out_channels=256,
            kernel_size=3,
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            in_channels=backbone_dimensions[-3],
            out_channels=256,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=backbone_dimensions[-4],
            out_channels=256,
            kernel_size=3,
            padding=1,
        )

    def resize_factor(self, x: torch.Tensor, factor: int = 1) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        return resize(x, [h * factor, w * factor])

    def forward(self, x):
        backbone_output = self.backbone(x)
        feat2, feat3, feat4, feat5 = (
            backbone_output["feat2"],
            backbone_output["feat3"],
            backbone_output["feat4"],
            backbone_output["feat5"],
        )

        # featoutput1 = self.resize_factor(self.conv5(feat5).relu(), 2)
        # print(featoutput1.shape)
        # print(feat4.shape)
        # featoutput2 = self.resize_factor(self.conv4(feat4 + featoutput1).relu(), 2)
        # featoutput3 = self.resize_factor(self.conv3(feat3 + featoutput2).relu(), 2)
        # featoutput4 = self.resize_factor(self.conv2(feat2 + featoutput3).relu(), 2)

        # print(featoutput1.shape)
        # print(featoutput2.shape)
        # print(featoutput3.shape)
        # print(featoutput4.shape)

        # feat3_resized_128 = resize(feat3, [128, 128])
        # feat4_resized_128 = resize(feat4, [128, 128])
        # feat5_resized_128 = resize(feat5, [128, 128])
        # feat3_transformed = self.conv1(feat3_resized_128)
        # feat4_transformed = self.conv2(feat4_resized_128)
        # feat5_transformed = self.conv3(feat5_resized_128)

        # (
        #     batch_size,
        #     channel,
        #     height,
        #     width,
        # ) = feat2.size()

        # feat2 = feat2.permute([0, 2, 3, 1])
        # feat3_transformed = feat3_transformed.permute([0, 2, 3, 1])
        # feat4_transformed = feat4_transformed.permute([0, 2, 3, 1])
        # feat5_transformed = feat5_transformed.permute([0, 2, 3, 1])

        # # Concantenated only the token dimension
        # # [B, HxW * 4, E]
        # concatenated = torch.cat(
        #     [
        #         # feat2.view([batch_size, height * width, channel]),
        #         # feat3_transformed.view([batch_size, height * width, channel]),
        #         # feat4_transformed.view([batch_size, height * width, channel]),
        #         feat5_transformed.view([batch_size, height * width, channel]),
        #     ],
        #     dim=1,
        # )
        # self_attended = self.attention(concatenated)
        # # Get only the first [256, 128, 128] from the attended self
        # self_attended = (
        #     self_attended[:, 0 : (128 * 128), :]
        #     .permute([0, 2, 1])
        #     .view([batch_size, 256, 128, 128])
        # )
        # classified = self.classifier_conv(self_attended).relu()
        # classified = self.final_classifier_conv(classified).relu()
        # # return resize(classified, [512, 512])
        # return classified


# Modify UNET to follow FPN style
if __name__ == "__main__":
    model = FPNNetwork_new(3)
    with torch.no_grad():
        output = model(torch.rand([1, 3, 512, 512]))
        print(output.shape)
