import torch
import torch.nn as nn

import config


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs) if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act
            else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )  # stays at same size
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1),
            ]
        )  # /2
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
        )  # stays at same size
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features*4, num_features*2, down=False, kernel_size=3, stride=2,
                          padding=1, output_padding=1),
                ConvBlock(num_features*2, num_features*1, down=False, kernel_size=3, stride=2,
                          padding=1, output_padding=1)
            ]
        )  # x 2
        self.last = nn.Conv2d(num_features*1, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):
        history_x = [x]  # we store the values at each step for skip connections

        x = self.initial(x)

        for layer in self.down_blocks:
            history_x.append(x)
            x = layer(x)

        history_x.append(x)

        # no need to store anything anymore
        x = self.res_blocks(x)

        history_x = history_x[::-1]
        if config.SKIP_CONNECTION == 2:
            x = x + history_x[0]

        for i, layer in enumerate(self.up_blocks):
            x = layer(x)
            if config.SKIP_CONNECTION == 2:
                x = x + history_x[i+1]

        x = self.last(x)
        if config.SKIP_CONNECTION:
            x = x + history_x[-1]

        return torch.tanh(x)


def test():
    img_channels = 3
    img_size = 512
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = Generator(img_channels, 9)
    print(gen)
    print(gen(x).shape)


if __name__ == "__main__":
    test()
