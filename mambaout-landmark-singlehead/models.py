# models.py
import torch
import torch.nn as nn
import timm


class conv2D_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv2D = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=out_ch),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=out_ch),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        return self.conv2D(x)


class decoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = conv2D_block(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNet2D(nn.Module):
    """
    MambaOut UNet for landmark heatmap detection.

    forward(x) -> heatmap [B, num_classes, H, W]
    """

    def __init__(self, in_ch=1, num_classes=2, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(
            'mambaout_base.in1k', pretrained=pretrained,
            features_only=True, in_chans=in_ch
        )
        encoder_channels = self.encoder.feature_info.channels()

        self.decoder4 = decoderBlock(encoder_channels[3], encoder_channels[2], encoder_channels[2])
        self.decoder3 = decoderBlock(encoder_channels[2], encoder_channels[1], encoder_channels[1])
        self.decoder2 = decoderBlock(encoder_channels[1], encoder_channels[0], encoder_channels[0])

        self.UpSample2D_1 = nn.ConvTranspose2d(encoder_channels[0], 96, 2, stride=2)
        self.Conv2D_1 = conv2D_block(96, 64)
        self.UpSample2D_2 = nn.ConvTranspose2d(64, 24, 2, stride=2)

        self.heatmap_head = nn.Conv2d(24, num_classes, kernel_size=1)

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.encoder(x)
        features = [f.permute(0, 3, 1, 2).contiguous() for f in features]  # NHWC -> NCHW

        e1, e2, e3, e4 = features

        d1 = self.decoder4(e4, e3)
        d2 = self.decoder3(d1, e2)
        d3 = self.decoder2(d2, e1)
        d4 = self.UpSample2D_1(d3)
        d5 = self.Conv2D_1(d4)
        d6 = self.UpSample2D_2(d5)

        return self.heatmap_head(d6)  # [B, num_classes, H, W]