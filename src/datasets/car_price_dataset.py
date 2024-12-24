import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CarPriceDataset(Dataset):
    """二手车数据集类"""
    def __init__(self, X, y):
        """
        初始化数据集
        
        Args:
            X: 特征数据
            y: 目标变量
        """
        self.X = X
        self.y = y.to_numpy() if not isinstance(y, np.ndarray) else y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx], dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.float32))

def create_data_loaders(X_train, X_test, y_train, y_test, batch_size_train=12, batch_size_test=32):
    """
    创建数据加载器
    
    Args:
        X_train: 训练集特征
        X_test: 测试集特征
        y_train: 训练集标签
        y_test: 测试集标签
        batch_size_train: 训练批次大小
        batch_size_test: 测试批次大小
        
    Returns:
        tuple: (训练数据加载器, 测试数据加载器)
    """

    # 在 create_data_loaders 函数中，确保 y 值被重塑为 2D 张量
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)


    train_dataset = CarPriceDataset(X_train, y_train)
    test_dataset = CarPriceDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
    
    return train_loader, test_loader 