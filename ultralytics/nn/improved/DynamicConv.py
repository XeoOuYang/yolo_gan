import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import CondConv2d

class DynamicConv(nn.Module):
    """动态卷积层，使用条件卷积（CondConv2d）实现。"""
    def __init__(self, in_features, out_features, kernel_size=1, stride=1, padding='', dilation=1, groups=1, bias=False, num_experts=4):
        """
        初始化动态卷积层。
        参数：
        in_features : 输入特征通道数
        out_features : 输出特征通道数
        kernel_size : 卷积核大小
        stride : 步长
        padding : 填充
        dilation : 膨胀系数
        groups : 组数
        bias : 是否使用偏置
        num_experts : 专家数量（用于CondConv2d）
        """
        super().__init__()
        # 路由层，用于计算每个专家的权重
        self.routing = nn.Linear(in_features, num_experts)
        # 条件卷积层，实现动态卷积
        self.cond_conv = CondConv2d(in_features, out_features, kernel_size, stride, padding, dilation, groups, bias, num_experts)

    def forward(self, x):
        """前向传播函数，实现动态路由和条件卷积的应用。"""
        # 先对输入进行全局平均池化，并展平
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)
        # 计算路由权重
        routing_weights = torch.sigmoid(self.routing(pooled_inputs))
        # 应用条件卷积
        x = self.cond_conv(x, routing_weights)
        return x