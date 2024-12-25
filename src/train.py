import os
import torch
import torch.nn as nn
from data_processing.data_processor import DataProcessor
from models.car_price_model import CarPriceModel
from datasets.car_price_dataset import create_data_loaders
from training.trainer import ModelTrainer
from utils.logger import setup_logging


def train_car_price_model(data_input: str, 
                         model_save_path: str = 'model.pth',
                         plot_save_path: str = 'loss_curve.png',
                         epochs: int = 10000,
                         learning_rate: float = 1e-4,
                         progress_callback=None):
    """
    训练二手车价格预测模型的主函数
    
    Args:
        data_path (str): 数据文件路径
        model_save_path (str): 模型保存路径
        plot_save_path (str): 损失曲线图保存路径
        epochs (int): 训练轮数
        learning_rate (float): 学习率
        progress_callback (callable): 进度回调函数
        
    Returns:
        tuple: (训练好的模型, 训练损失列表, 测试损失列表)
    """
    # 设置日志
    setup_logging()
    
    # 数据处理
    processor = DataProcessor()
    
    # 根据输入类型处理数据
    if isinstance(data_input, str):
        # 如果输入是文件路径
        X, y = processor.load_and_analyze_data(data_input)
    else:
        # 如果输入是DataFrame
        X, y = processor.load_and_analyze_data(data_input)

    X_train, X_test, y_train, y_test = processor.prepare_data(X, y)
    
    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(X_train, X_test, y_train, y_test)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    model = CarPriceModel(in_features=X_train.shape[1], out_features=1)
    model = model.to(device)
    
    # 设置训练参数
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 创建训练器并训练模型
    trainer = ModelTrainer(model, loss_fn, optimizer, device)
    train_losses, test_losses = trainer.train(
        train_loader, 
        test_loader, 
        epochs,
        progress_callback=progress_callback
    )
    
    # 保存模型和损失曲线
    trainer.save_model(model_save_path)
    trainer.plot_losses(train_losses, test_losses, plot_save_path)
    
    return model, train_losses, test_losses, processor

if __name__ == "__main__":
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, 'data', 'used_car_train_20200313_cleaned.csv')
    model, train_losses, test_losses = train_car_price_model(data_path)