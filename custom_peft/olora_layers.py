import torch
import torch.nn as nn
import math

class OLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r, r_sum=0, lora_alpha=1, dropout=0.0, fan_in_fan_out=False):
        super().__init__()
        self.in_features = in_features   # ✅ 添加这行
        self.out_features = out_features # ✅ 添加这行
        self.r = r
        self.r_sum = r_sum
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        self.fan_in_fan_out = fan_in_fan_out

        # 正确设置 A/B 的维度
        self.lora_A = nn.Parameter(torch.zeros((r_sum, in_features)))  # 注意这里是 r_sum
        self.lora_B = nn.Parameter(torch.zeros((out_features, r_sum)))
        self.loranew_A = nn.Parameter(torch.zeros((r, in_features)))
        self.loranew_B = nn.Parameter(torch.zeros((out_features, r)))

        self.dropout = nn.Dropout(dropout)

        # 初始化
        nn.init.zeros_(self.lora_A)  # 之前任务训练好的部分，默认初始化为0，真实训练完才拼接进去
        nn.init.zeros_(self.lora_B)
        nn.init.kaiming_uniform_(self.loranew_A, a=math.sqrt(5))
        nn.init.zeros_(self.loranew_B)

    def forward(self, x):
        result = self.dropout(x)
        update = (
            (self.lora_B @ self.lora_A + self.loranew_B @ self.loranew_A) * self.scaling
        )

        if self.fan_in_fan_out:
            update = update.transpose(0, 1)

        return x + result @ update.T

