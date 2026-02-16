from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSCNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
    ):
        """
        Args:
            input_dim: 输入特征维度（如 80 维 fbank）
            out_dim:   输出维度（类别数）
        """
        super(DSCNN, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim

        # 第一层标准卷积：时间和频率都做 stride=2 下采样
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=(4, 10), padding=(1, 4),
                               stride=(2, 2))
        # 4 层深度可分离卷积（same padding，不改变尺寸）
        self.depthwise = nn.Conv2d(in_channels=64, out_channels=64,
                                   kernel_size=(3, 3), padding=(1, 1),
                                   stride=(1, 1), groups=64)
        self.pointwise = nn.Conv2d(in_channels=64, out_channels=64,
                                   kernel_size=1)
        # 只在时间维做 MaxPool，进一步下采样时间
        self.pooling = nn.MaxPool2d(kernel_size=(2, 1))
        # 在频率维做自适应平均池化，压成 1，保留时间维
        self.freq_pool = nn.AdaptiveAvgPool2d((None, 1))
        # 逐帧线性层：64 通道 -> out_dim
        self.fc1 = nn.Linear(in_features=64, out_features=out_dim)

    def dw_conv(self, x):
        x = self.depthwise(x)
        x = F.relu(x)
        x = self.pointwise(x)
        x = F.relu(x)
        return x

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
            output: (B, T', out_dim) 逐帧输出，T' = T // 4
            out_cache: 空缓存占位
        """
        # (B, T, D) -> (B, 1, T, D)  当作单通道图像
        x = x.unsqueeze(1)
        # conv1: (B, 1, T, D) -> (B, 64, T//2, D//2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dw_conv(x)
        x = self.dw_conv(x)
        x = self.dw_conv(x)
        x = self.dw_conv(x)
        # pooling: 时间维减半 -> (B, 64, T//4, D//2)
        x = self.pooling(x)
        # 频率维池化到 1 -> (B, 64, T//4, 1)
        x = self.freq_pool(x)
        # 去掉频率维 -> (B, 64, T//4)
        x = x.squeeze(3)
        # 转为 (B, T//4, 64)
        x = x.permute(0, 2, 1)
        # 逐帧分类 -> (B, T//4, out_dim)
        x = self.fc1(x)

        out_cache = torch.zeros(0, 0, 0, dtype=x.dtype, device=x.device)
        return x, out_cache


if __name__ == "__main__":
    dscnn = DSCNN(input_dim=80, out_dim=2)
    # 测试不同长度输入
    for T in [100, 200, 43]:
        x = torch.zeros(4, T, 80)  # (B, T, D)
        y, cache = dscnn(x)
        print(f'input: {tuple(x.shape)}  ->  output: {tuple(y.shape)}')