import torch
import logging
import matplotlib.pyplot as plt
# from tqdm import tqdm

class ModelTrainer:
    """
    模型训练器类
    """
    def __init__(self, model, loss_fn, optimizer, device):
        """
        初始化训练器
        
        Args:
            model: 神经网络模型
            loss_fn: 损失函数
            optimizer: 优化器
            device: 训练设备
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def evaluate_model(self, dataloader):
        """
        评估模型
        
        Args:
            dataloader: 数据加载器
            
        Returns:
            float: 平均损失值
        """
        self.model.eval()
        losses = []
        
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                y_pred = self.model(X)
                loss = self.loss_fn(y_pred, y)
                losses.append(loss.item())
        
        return round(sum(losses) / len(losses), 5)

    def train(self, train_loader, test_loader, epochs, progress_callback=None):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            epochs: 训练轮数
            progress_callback: 进度回调函数
        """
        train_losses = []
        test_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            epoch_losses = []
            
            for batch_idx, (X, y) in enumerate(train_loader):
                X, y = X.to(self.device), y.to(self.device)
                
                y_pred = self.model(X)
                loss = self.loss_fn(y_pred, y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
            
            train_loss = self.evaluate_model(train_loader)
            test_loss = self.evaluate_model(test_loader)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            # 使用回调函数更新进度
            if progress_callback:
                progress_callback(epoch, train_loss, test_loss)
            
            logging.info(f'Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.5f}, test_loss: {test_loss:.5f}')
        
        return train_losses, test_losses

    def save_model(self, path):
        """
        保存模型参数
        
        Args:
            path: 模型保存路径
        """
        # 保存模型参数而不是整个模型
        torch.save(self.model.state_dict(), path)

    def plot_losses(self, train_losses, test_losses, save_path=None):
        """
        绘制损失曲线
        
        Args:
            train_losses: 训练损失列表
            test_losses: 测试损失列表
            save_path: 图像保存路径
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', c='red')
        plt.plot(test_losses, label='Test Loss', c='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Testing Loss')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.show() 