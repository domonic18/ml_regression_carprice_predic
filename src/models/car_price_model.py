#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn


class CarPriceModel(nn.Module):
    """
    二手车价格预测模型
    """
    def __init__(self, in_features=13, out_features=1):
        """
        初始化模型
        
        Args:
            in_features (int): 输入特征维度
            out_features (int): 输出维度
        """
        super(CarPriceModel, self).__init__()
        self.linear1 = nn.Linear(in_features, 64)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)

        self.linear2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(32)

        self.linear3 = nn.Linear(32, out_features)

    def forward(self, x):
        """
        前向传播
        """
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x 