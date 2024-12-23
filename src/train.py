import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 配置日志
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_and_analyze_data(file_path):
    """加载并分析数据"""
    logging.info("开始加载数据...")
    df = pd.read_csv(file_path)
    
    logging.info(f"数据形状: {df.shape}")
    logging.info("\n数据类型信息:\n{df.dtypes}")
    logging.info("\n数据统计信息:\n{df.describe()}")
    
    X = df.drop(['price'], axis=1)
    y = df['price']
    
    return X, y

def prepare_data(X, y, test_size=0.2, random_state=0):
    """准备训练集和测试集"""
    logging.info("开始数据集划分...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 数据标准化
    X_train_np = np.array(X_train)
    X_test_np = np.array(X_test)
    
    mean = X_train_np.mean(axis=0)
    std = X_train_np.std(axis=0) + 1e-9
    
    X_train_normalized = (X_train_np - mean) / std
    X_test_normalized = (X_test_np - mean) / std
    
    logging.info(f"训练集形状: {X_train_normalized.shape}")
    logging.info(f"测试集形状: {X_test_normalized.shape}")
    
    # # 添加：对目标值进行标准化
    # y_train_np = np.array(y_train)
    # y_test_np = np.array(y_test)
    
    # y_mean = y_train_np.mean()
    # y_std = y_train_np.std() + 1e-9
    
    # y_train_normalized = (y_train_np - y_mean) / y_std
    # y_test_normalized = (y_test_np - y_mean) / y_std

    # logging.info(f"训练集形状: {X_train_normalized.shape}")
    # logging.info(f"测试集形状: {X_test_normalized.shape}")

    # return X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized
    return X_train_normalized, X_test_normalized, y_train, y_test

# 定义模型
class Model(nn.Module):
    def __init__(self, in_features=13, out_features=1):
        super(Model, self).__init__()
        # 定义3层的全链接神经网络
        self.linear1 = nn.Linear(in_features, 64)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)

        self.linear2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(32)

        self.linear3 = nn.Linear(32, out_features)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x
    
class CarPriceDataset(Dataset):
    """二手车数据集类"""
    def __init__(self, X, y):
        self.X = X
        self.y = y.to_numpy() if not isinstance(y, np.ndarray) else y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx], dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.float32))

def create_data_loaders(X_train, X_test, y_train, y_test, batch_size_train=12, batch_size_test=32):
    """创建数据加载器"""
    train_dataset = CarPriceDataset(X_train, y_train)
    test_dataset = CarPriceDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
    
    return train_loader, test_loader

def evaluate_model(model, dataloader, loss_fn, device):
    """评估模型"""
    model.eval()
    losses = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)  # 将数据移到正确的设备上
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            losses.append(loss.item())
    
    return round(sum(losses) / len(losses), 5)

def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs, device):
    """训练模型"""
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)  # 将数据移到正确的设备上
            if batch_idx == 0:
                logging.info(f"特征范围: {X.min().item():.4f} 到 {X.max().item():.4f}")
                logging.info(f"标签范围: {y.min().item():.4f} 到 {y.max().item():.4f}")
            
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            if batch_idx % 100 == 0:
                logging.info(f'Epoch {epoch+1}/{epochs} - Batch {batch_idx} - Loss: {loss.item():.5f}')
        
        # 计算当前轮次的训练和测试损失
        train_loss = evaluate_model(model, train_loader, loss_fn, device)
        test_loss = evaluate_model(model, test_loader, loss_fn, device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        logging.info(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.5f} - Test Loss: {test_loss:.5f}')
    
    return train_losses, test_losses

def main():
    # 设置日志
    setup_logging()
    import os
    import matplotlib.pyplot as plt


    # 加载和分析数据
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, 'data', 'used_car_train_20200313_handled.csv')
    X, y = load_and_analyze_data(file_path)
    
    # 准备数据
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    
    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(X_train, X_test, y_train, y_test)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'使用设备: {device}')


    # 初始化模型并移至 GPU
    model = Model(in_features=X_train.shape[1], out_features=1)
    model = model.to(device)
    logging.info(f"模型结构:\n{model}")
    
    # 设置训练参数
    epochs = 100
    learning_rate = 1e-4
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 训练模型
    train_losses, test_losses = train_model(
        model, train_loader, test_loader, loss_fn, optimizer, epochs, device
    )
    
    logging.info("训练完成！")

    # 绘制损失曲线
    plt.plot(train_losses, label='Train Loss', c='red')
    plt.plot(test_losses, label='Test Loss', c='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.show()

    # 保存损失函数曲线到文件
    plt.savefig('loss_curve.png')


    # 保存模型
    torch.save(model, 'model.pth')
    return model, train_losses, test_losses

if __name__ == "__main__":
    main()