from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConvBlock(nn.Module):
    """单个深度可分离卷积块: DW Conv -> BN -> ReLU -> PW Conv -> BN -> ReLU"""

    def __init__(self, channels: int):
        super(DSConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(channels, channels,
                                   kernel_size=(3, 3), padding=(1, 1),
                                   stride=(1, 1), groups=channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


class DSCNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_ds_layers: int = 4,
        channels: int = 64,
    ):
        """
        纯特征提取器，输出 (B, T//4, channels) 维特征。
        分类层由 wekws 框架的 classifier 模块负责。

        Args:
            input_dim: 输入特征维度（如 80 维 fbank）
            num_ds_layers: 深度可分离卷积层数
            channels:  卷积通道数
        """
        super(DSCNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = channels  # 暴露给外部，方便框架获取

        # 第一层标准卷积：时间和频率都做 stride=2 下采样
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=channels,
                               kernel_size=(4, 10), padding=(1, 4),
                               stride=(2, 2))
        self.bn0 = nn.BatchNorm2d(channels)
        # N 层深度可分离卷积（各自独立权重）
        self.ds_layers = nn.ModuleList([
            DSConvBlock(channels) for _ in range(num_ds_layers)
        ])
        # 只在时间维做 MaxPool，进一步下采样时间
        self.pooling = nn.MaxPool2d(kernel_size=(2, 1))
        # 在频率维做自适应平均池化，压成 1，保留时间维
        self.freq_pool = nn.AdaptiveAvgPool2d((None, 1))

    def forward(
        self,
        x: torch.Tensor,
        in_cache: torch.Tensor = torch.zeros(0, 0, 0, dtype=torch.float)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D) 输入特征，如 fbank
            in_cache: 预留缓存接口（当前未使用）
        Returns:
            output: (B, T//4, channels) 逐帧特征
            out_cache: 空缓存占位
        """
        # (B, T, D) -> (B, 1, T, D)  当作单通道图像
        x = x.unsqueeze(1)
        # conv1: (B, 1, T, D) -> (B, C, T//2, D//2)
        x = self.conv1(x)
        x = self.bn0(x)
        x = F.relu(x)
        # 各层独立权重的 DS Conv
        for ds_layer in self.ds_layers:
            x = ds_layer(x)
        # pooling: 时间维减半 -> (B, C, T//4, D//2)
        x = self.pooling(x)
        # 频率维池化到 1 -> (B, C, T//4, 1)
        x = self.freq_pool(x)
        # 去掉频率维 -> (B, C, T//4)
        x = x.squeeze(3)
        # 转为 (B, T//4, C)
        x = x.permute(0, 2, 1)

        out_cache = torch.zeros(0, 0, 0, dtype=x.dtype, device=x.device)
        return x, out_cache


if __name__ == "__main__":
    dscnn = DSCNN(input_dim=80)
    # 测试不同长度输入
    for T in [100, 200, 43]:
        x = torch.zeros(4, T, 80)  # (B, T, D)
        y, cache = dscnn(x)
        print(f'input: {tuple(x.shape)}  ->  output: {tuple(y.shape)}')